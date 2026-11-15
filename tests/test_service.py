from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
from unittest.mock import patch
import json

import psycopg2

from core.service import synchronize
from database.local_store import LocalStore


class SynchronizationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / 'state.db'
        self.store = LocalStore(self.path)
        self.addCleanup(lambda: self.store.close())
        self.payload = {'engine': {'day': '2026-11-15', 'identities': {'x': [0.1, 0.2]}},
                        'bus_id': 'BUS_A', 'camera_id': 'DOOR_A', 'next_frame': 12}
        self.db = Mock()

    def test_new_empty_store_does_not_loop_or_call_postgres(self):
        with tempfile.TemporaryDirectory() as folder:
            store = LocalStore(Path(folder) / "state.db")
            try:
                db = Mock()
                synchronize(store, db)
                db.deliver.assert_not_called()
            finally:
                store.close()

    def test_idle_sync_sends_a_checkpoint_once_per_database_instance(self):
        self.store.save('source', self.payload)
        for _ in range(5):
            synchronize(self.store, self.db)
        self.assertEqual(self.db.deliver.call_count, 1)
        checkpoint = self.db.deliver.call_args.args[1]
        self.assertEqual(json.loads(checkpoint['payload_json']), self.payload)
        self.assertEqual(checkpoint['payload_meta'], {'day': '2026-11-15', 'bus_id': 'BUS_A', 'camera_id': 'DOOR_A'})
        other_database = Mock()
        synchronize(self.store, other_database)
        other_database.deliver.assert_called_once()

    def test_new_revision_is_delivered_after_idle_sync(self):
        self.store.save('source', self.payload)
        synchronize(self.store, self.db)
        self.payload['next_frame'] = 20
        self.store.save('source', self.payload)
        synchronize(self.store, self.db)
        self.assertEqual(self.db.deliver.call_count, 2)
        self.assertEqual(json.loads(self.db.deliver.call_args.args[1]['payload_json'])['next_frame'], 20)

    def test_remote_failure_is_retried_instead_of_marked_synchronized(self):
        self.store.save('source', self.payload)
        self.db.deliver.side_effect = [psycopg2.OperationalError('test outage'), None]
        with self.assertRaises(psycopg2.OperationalError):
            synchronize(self.store, self.db)
        synchronize(self.store, self.db)
        synchronize(self.store, self.db)
        self.assertEqual(self.db.deliver.call_count, 2)

    def test_lost_local_acknowledgement_retries_remote_commit(self):
        self.store.save('source', self.payload)
        with patch.object(self.store, 'acknowledge', side_effect=RuntimeError('test local commit failure')):
            with self.assertRaises(RuntimeError):
                synchronize(self.store, self.db)
        synchronize(self.store, self.db)
        self.assertEqual(self.db.deliver.call_count, 2)

    def test_restart_retries_even_if_previous_process_sent_the_revision(self):
        self.store.save('source', self.payload)
        synchronize(self.store, self.db)
        self.store.close()
        self.store = LocalStore(self.path)
        synchronize(self.store, self.db)
        self.assertEqual(self.db.deliver.call_count, 2)

    def test_checkpoint_cannot_overtake_a_multi_batch_outbox(self):
        from config.settings import Settings
        from core.types import CrossingEvent
        from datetime import datetime, timezone
        events = [CrossingEvent.create(Settings(), datetime.now(timezone.utc), 'IN', index, str(index), True, 'COUNTED')
                  for index in range(501)]
        self.store.save('source', self.payload, events)
        synchronize(self.store, self.db)
        self.assertEqual(self.db.deliver.call_count, 2)
        first, last = [call.args for call in self.db.deliver.call_args_list]
        self.assertEqual(len(first[0]), 500)
        self.assertIsNone(first[1])
        self.assertEqual(len(last[0]), 1)
        self.assertEqual(json.loads(last[1]['payload_json']), self.payload)
        self.assertEqual(self.store.pending_count(), 0)
