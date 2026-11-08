from dataclasses import replace
import unittest

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
