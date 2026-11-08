from datetime import datetime, timezone
import tempfile
from pathlib import Path
import unittest

from config.settings import Settings
from core.types import CrossingEvent
from database.local_store import LocalStore


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
