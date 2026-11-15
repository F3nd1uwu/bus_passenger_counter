from datetime import datetime, timezone
import tempfile
from pathlib import Path
import unittest
import json
from dataclasses import asdict

from config.settings import Settings
from core.types import CrossingEvent
from database.local_store import LocalStore
from core.counter_engine import Identity


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "state.db"
        self.store = LocalStore(self.path)
        self.event = CrossingEvent.create(Settings(), datetime.now(timezone.utc), "IN", 1, "person", True, "COUNTED")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_restart_preserves_checkpoint_and_pending_event(self):
        self.store.save("source", {"value": 1}, [self.event])
        self.store.close()
        self.store = LocalStore(self.path)
        self.assertEqual(self.store.load()["payload"], {"value": 1})
        self.assertEqual(self.store.pending()[0]["event_id"], self.event.event_id)

    def test_duplicate_event_rolls_back_whole_checkpoint(self):
        self.store.save("source", {"value": 1}, [self.event])
        with self.assertRaises(Exception):
            self.store.save("source", {"value": 2}, [self.event])
        self.assertEqual(self.store.load()["payload"], {"value": 1})
        self.assertEqual(self.store.pending_count(), 1)

    def test_invalid_state_cannot_commit_events(self):
        with self.assertRaises(ValueError):
            self.store.save("source", {"value": float("nan")}, [self.event])
        self.assertIsNone(self.store.load())
        self.assertEqual(self.store.pending_count(), 0)

    def test_acknowledgement_removes_only_delivered_events(self):
        self.store.save("source", {}, [self.event])
        self.store.acknowledge(self.store.pending())
        self.assertEqual(self.store.pending_count(), 0)
        self.assertIsNotNone(self.store.load())

    def test_source_mismatch_does_not_overwrite_state(self):
        self.store.save("one", {"value": 1})
        with self.assertRaises(ValueError):
            self.store.save("two", {"value": 2})
        self.assertEqual(self.store.load()["source_key"], "one")

    def test_two_workers_cannot_share_local_state(self):
        with self.assertRaises(RuntimeError):
            LocalStore(self.path)

    def test_wire_checkpoint_retains_exact_json_and_legacy_recovery_format(self):
        payload = {'engine': {'day': '2026-11-15', 'identities': {'person': [0.1, 0.2]}},
                   'bus_id': "BUS_'А", 'camera_id': 'DOOR_1', 'next_frame': 99, 'completed': False}
        self.store.save('source', payload, [self.event])
        wire = self.store.load_for_delivery()
        self.assertEqual(json.loads(wire['payload_json']), payload)
        self.assertEqual(self.store.load()['payload'], payload)
        self.assertEqual(wire['payload_meta'], {'day': '2026-11-15', 'bus_id': "BUS_'А", 'camera_id': 'DOOR_1'})

    def test_explicit_revision_cannot_replace_an_already_delivered_state(self):
        self.store.save('source', {'value': 1}, revision=5)
        for revision in [4, 5]:
            with self.assertRaisesRegex(ValueError, 'advance'):
                self.store.save('source', {'value': 2}, [self.event], revision=revision)
        self.assertEqual(self.store.load()['payload'], {'value': 1})
        self.assertEqual(self.store.pending_count(), 0)


class IdentitySnapshotTests(unittest.TestCase):
    def test_snapshot_preserves_dataclass_fields_and_isolates_mutable_features(self):
        identity = Identity('person', [0.1, 0.2], 42, last_in=40, staff=True, gallery=[[0.3, 0.4]])
        state = identity.snapshot()
        self.assertEqual(state, asdict(identity))
        identity.embedding[0] = 9
        identity.gallery[0][0] = 8
        identity.gallery.append([7, 7])
        self.assertEqual(state['embedding'], [0.1, 0.2])
        self.assertEqual(state['gallery'], [[0.3, 0.4]])
        state['gallery'][0][1] = 6
        self.assertEqual(identity.gallery[0][1], 0.4)
