from dataclasses import replace
import unittest

from config.profiles import apply_profile
from config.settings import Settings


class ProfileTests(unittest.TestCase):
    def test_profile_preserves_source_storage_and_database(self):
        cfg = Settings(VIDEO_SOURCE="private_recording.mp4", STATE_PATH="data/separate.sqlite3", DB_HOST="local-db")
        result = apply_profile(cfg, "test2")
        self.assertEqual((result.VIDEO_SOURCE, result.STATE_PATH, result.DB_HOST),
                         (cfg.VIDEO_SOURCE, cfg.STATE_PATH, cfg.DB_HOST))
        self.assertEqual(result.CONFIDENCE, .30)
        self.assertEqual(result.STAFF_MODE, "model")
        self.assertEqual(result.STAFF_DECISION_DELAY_SEC, 10)
        self.assertEqual((result.TRACKER_DRIFT_THRESHOLD, result.Y_CABIN), (.6, .57))
        self.assertEqual((cfg.TRACKER_DRIFT_THRESHOLD, cfg.Y_CABIN), (0, .60))
        self.assertEqual(replace(result, STAFF_MODE="disabled").STAFF_MODE, "disabled")

    def test_absent_profile_preserves_existing_installation(self):
        cfg = Settings()
        self.assertIs(apply_profile(cfg, None), cfg)

    def test_unknown_profile_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_profile(Settings(), "../unrelated")
