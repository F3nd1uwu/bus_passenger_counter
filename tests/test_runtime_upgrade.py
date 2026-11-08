from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from config.settings import Settings
from core.service import configuration_key
from core.types import CrossingEvent
from database.local_store import LocalStore
from tools.upgrade_runtime import upgrade


class RuntimeUpgradeTests(unittest.TestCase):
    def setUp(self):
        # Migration fingerprints bytes; it never runs the historical detector.
        # Keep that fixture local instead of requiring an obsolete private model.
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        legacy_model = Path(directory.name) / "legacy-detector.pt"
        legacy_model.write_bytes(b"historical detector fingerprint fixture")
        original_path = Settings.path

        def fixture_path(cfg, value):
            return legacy_model if value == "models/yolo11n.pt" else original_path(cfg, value)

        patcher = patch.object(Settings, "path", fixture_path)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_upgrade_preserves_cursor_identities_events_and_original_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            cfg = Settings(STATE_PATH=str(path), VIDEO_SOURCE="rtsp://example.invalid/live", SOURCE_KIND="stream")
            legacy = replace(cfg, MODEL_PATH="models/yolo11n.pt",
                             REID_MODEL_PATH="models/person_reid.torchscript",
                             TRACKER_PATH="bytetrack_custom.yaml", TRACKING_MODE="iou",
                             CONFIDENCE=.25, POSE_ANCHOR="bbox")
            old_key = configuration_key(legacy)
            payload = {"bus_id": cfg.BUS_ID, "camera_id": cfg.CAMERA_ID, "next_frame": 5673,
                       "completed": False, "engine": {"total_in": 4, "identities": ["saved-person"]}}
            event = CrossingEvent.create(cfg, datetime.now(timezone.utc), "IN", 1, "saved-person", True, "COUNTED")
            store = LocalStore(path)
            store.save(old_key, payload, [event], revision=10)
            store.close()
            upgrade(cfg)
            store = LocalStore(path)
            try:
                saved = store.load()
                self.assertEqual(saved["source_key"], configuration_key(cfg))
                self.assertEqual(saved["revision"], 11)
                for key, value in payload.items():
                    self.assertEqual(saved["payload"][key], value)
                self.assertEqual(store.pending()[0], event.to_dict())
            finally:
                store.close()
            backups = list(path.parent.glob(path.name + ".before-onnx-*"))
            self.assertEqual(len(backups), 1)
            backup = LocalStore(backups[0])
            try:
                self.assertEqual(backup.load()["source_key"], old_key)
                self.assertEqual(backup.load()["payload"], payload)
                self.assertEqual(backup.pending()[0], event.to_dict())
            finally:
                backup.close()
            upgrade(cfg)
            self.assertEqual([p for p in path.parent.glob(path.name + ".before-onnx-*")
                              if not p.name.endswith(".lock")], backups)

    def test_unknown_configuration_is_left_intact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            cfg = Settings(STATE_PATH=str(path), VIDEO_SOURCE="rtsp://example.invalid/live", SOURCE_KIND="stream")
            store = LocalStore(path)
            store.save("unrecognised", {"next_frame": 5673})
            store.close()
            with self.assertRaisesRegex(ValueError, "Unrecognised legacy"):
                upgrade(cfg)
            store = LocalStore(path)
            try:
                self.assertEqual(store.load()["source_key"], "unrecognised")
                self.assertEqual(store.load()["payload"], {"next_frame": 5673})
                self.assertEqual(store.load()["revision"], 1)
            finally:
                store.close()
            self.assertEqual(list(path.parent.glob(path.name + ".before-onnx-*")), [])
