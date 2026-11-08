from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from core.service import synchronize
from database.local_store import LocalStore


class SynchronizationTests(unittest.TestCase):
    def test_new_empty_store_does_not_loop_or_call_postgres(self):
        with tempfile.TemporaryDirectory() as folder:
            store = LocalStore(Path(folder) / "state.db")
            try:
                db = Mock()
                synchronize(store, db)
                db.deliver.assert_not_called()
            finally:
                store.close()
