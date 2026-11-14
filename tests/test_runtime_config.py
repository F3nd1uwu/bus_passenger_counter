from dataclasses import replace
import unittest
from unittest.mock import patch

from config.settings import Settings
from core.service import configuration_key


class RuntimeConfigurationTests(unittest.TestCase):
    def test_cpu_worker_count_does_not_invalidate_hourly_state(self):
        cfg = Settings(VIDEO_SOURCE="rtsp://example.invalid/live", SOURCE_KIND="stream")
        self.assertEqual(configuration_key(cfg), configuration_key(replace(cfg, CPU_THREADS=1)))

    def test_changing_tracking_algorithm_changes_state_key(self):
        cfg = Settings(VIDEO_SOURCE="rtsp://example.invalid/live", SOURCE_KIND="stream")
        legacy = replace(cfg, TRACKING_MODE="iou", TRACKER_PATH="bytetrack_custom.yaml")
        self.assertNotEqual(configuration_key(cfg), configuration_key(legacy))

    def test_enabled_drift_guard_changes_state_key(self):
        cfg = Settings(VIDEO_SOURCE="rtsp://example.invalid/live", SOURCE_KIND="stream")
        self.assertNotEqual(configuration_key(cfg), configuration_key(replace(cfg, TRACKER_DRIFT_THRESHOLD=.6)))

    def test_disabled_drift_guard_preserves_the_previous_fingerprint(self):
        from dataclasses import asdict
        cfg = Settings(VIDEO_SOURCE="rtsp://example.invalid/live", SOURCE_KIND="stream")
        current = configuration_key(cfg)
        previous = asdict(cfg)
        previous.pop('TRACKER_DRIFT_THRESHOLD')
        with patch('core.service.asdict', return_value=previous):
            self.assertEqual(configuration_key(cfg), current)

    def test_child_filter_changes_state_key_only_when_enabled(self):
        cfg = Settings(VIDEO_SOURCE="rtsp://example.invalid/live", SOURCE_KIND="stream")
        self.assertEqual(configuration_key(cfg), configuration_key(replace(cfg, CHILD_DECISION_DELAY_SEC=20.)))
        self.assertNotEqual(configuration_key(cfg), configuration_key(replace(cfg, CHILD_MODE='model')))
