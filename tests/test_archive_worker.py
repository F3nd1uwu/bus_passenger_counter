from dataclasses import replace
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from config.settings import Settings
from tools import archive_protocol as protocol
from tools.archive_worker import ReviewRequired, deliver_completed, execute, freeze_recipe


class ArchiveWorkerTests(unittest.TestCase):
    def test_recipe_freezes_assets_but_never_writes_database_credentials(self):
        with tempfile.TemporaryDirectory() as folder:
            asset = Path(folder)/'model.bin'
            asset.write_bytes(b'first')
            cfg = replace(Settings(), DB_PASS='private-connection-password', MODEL_PATH=str(asset),
                          REID_MODEL_PATH=str(asset), TRACKER_PATH=str(asset), CLOCK_MODEL_PATH=str(asset))
            path = Path(folder)/'recipe.json'
            freeze_recipe(cfg, path)
            self.assertNotIn('private-connection-password', path.read_text())
            self.assertFalse(any(key.startswith('DB_') for key in protocol.read(path)))
            freeze_recipe(replace(cfg, CPU_THREADS=2, DETECTOR_CPU_THREADS=2), path)
            with self.assertRaises(ReviewRequired):
                freeze_recipe(replace(cfg, CONFIDENCE=.33), path)
            asset.write_bytes(b'changed')
            with self.assertRaises(ReviewRequired):
                freeze_recipe(cfg, path)

    def test_completed_means_final_delivery_succeeded(self):
        store = Mock(load=lambda:dict(payload=dict(completed=True)))
        db = Mock()
        with patch('database.local_store.LocalStore', return_value=store), patch('database.db_client.DatabaseClient', return_value=db), patch('core.service.synchronize') as sync:
            self.assertTrue(deliver_completed(Settings()))
            sync.assert_called_once_with(store, db)
            store.close.assert_called_once()
            db.close.assert_called_once()

    def test_database_failure_does_not_report_success(self):
        store = Mock(load=lambda:dict(payload=dict(completed=True)))
        with patch('database.local_store.LocalStore', return_value=store), patch('database.db_client.DatabaseClient'), patch('core.service.synchronize', side_effect=RuntimeError('offline')):
            with self.assertRaises(RuntimeError):
                deliver_completed(Settings())
        store.close.assert_called_once()

    def test_partial_recording_cannot_be_marked_completed(self):
        store = Mock(load=lambda:dict(payload=dict(completed=False)))
        with patch('database.local_store.LocalStore', return_value=store), patch('database.db_client.DatabaseClient'), patch('core.service.synchronize') as sync:
            self.assertFalse(deliver_completed(Settings()))
            sync.assert_not_called()

    def test_worker_without_current_lease_does_not_start_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            protocol.write(root/'lease.json', dict(token='other', expires=time.time()+60))
            execute(dict(root=folder, attempt_dir=str(root/'old'), token='old'))
            result = protocol.read(root/'old'/'result.json')
            self.assertEqual(result['outcome'], 'interrupted')
            self.assertEqual(result['token'], 'old')

    def test_busy_worker_lock_prevents_second_attempt(self):
        from tools.archive_resources import ProcessLock
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            lock = ProcessLock(root/'worker.lock')
            try:
                execute(dict(root=folder, attempt_dir=str(root/'next'), token='new'))
            finally:
                lock.close()
            self.assertEqual(protocol.read(root/'next'/'result.json')['outcome'], 'retry')


if __name__ == '__main__':
    unittest.main()
