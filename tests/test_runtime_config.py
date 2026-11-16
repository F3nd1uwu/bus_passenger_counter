from dataclasses import replace
import unittest
from unittest.mock import patch

from config.settings import Settings
from core.service import configuration_key


class RuntimeConfigurationTests(unittest.TestCase):
    def test_default_detector_backend_preserves_old_checkpoint_key(self):
        from dataclasses import asdict
        cfg = Settings(TIME_SOURCE='pts', VIDEO_SOURCE='data/test.mp4', SOURCE_KIND='file')
        previous = asdict(cfg)
        previous.pop('CPU_DETECTOR_BACKEND')
        previous.pop('DETECTOR_CPU_THREADS')
        current = configuration_key(cfg)
        with patch('core.service.asdict', return_value=previous):
            self.assertEqual(configuration_key(cfg), current)

    def test_backend_change_requires_new_state_but_thread_count_does_not(self):
        cfg = Settings(TIME_SOURCE='pts', VIDEO_SOURCE='data/test.mp4', SOURCE_KIND='file',
                       MODEL_PATH='models/candidates/person_test2_v2.onnx')
        self.assertEqual(configuration_key(cfg), configuration_key(replace(cfg, DETECTOR_CPU_THREADS=2)))
        self.assertNotEqual(configuration_key(cfg), configuration_key(replace(cfg, CPU_DETECTOR_BACKEND='openvino')))

    def test_camera_file_rename_and_review_note_do_not_reset_hourly_memory(self):
        import json
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first, second = root / 'first.mp4', root / 'renamed.mp4'
            first.write_bytes(b'same recording')
            second.write_bytes(first.read_bytes())
            clock_path = root / 'clock.json'
            clock = dict(source_sha256='content-hash', model_sha256='ocr-hash', timezone='Europe/Saratov',
                         roi=[0, 0, 1, 1], end_frame=10, segments=[dict(frame=0, pts=0., offset=1000)])
            clock_path.write_text(json.dumps(clock), encoding='utf-8')
            cfg = Settings(VIDEO_SOURCE=str(first), CLOCK_MAP_PATH=str(clock_path))
            key = configuration_key(cfg)
            self.assertEqual(key, configuration_key(replace(cfg, VIDEO_SOURCE=str(second), SOURCE_KIND='file')))
            clock['note'] = 'review notes may be clarified without changing time'
            clock_path.write_text(json.dumps(clock), encoding='utf-8')
            self.assertEqual(key, configuration_key(cfg))
            clock['segments'][0]['offset'] += 3600
            clock_path.write_text(json.dumps(clock), encoding='utf-8')
            self.assertNotEqual(key, configuration_key(cfg))

    def test_cpu_worker_count_does_not_invalidate_hourly_state(self):
        cfg = Settings(TIME_SOURCE="pts", VIDEO_SOURCE="data/test.mp4", SOURCE_KIND="file")
        self.assertEqual(configuration_key(cfg), configuration_key(replace(cfg, CPU_THREADS=1)))

    def test_changing_tracking_algorithm_changes_state_key(self):
        cfg = Settings(TIME_SOURCE="pts", VIDEO_SOURCE="data/test.mp4", SOURCE_KIND="file")
        legacy = replace(cfg, TRACKING_MODE="iou", TRACKER_PATH="bytetrack_custom.yaml")
        self.assertNotEqual(configuration_key(cfg), configuration_key(legacy))

    def test_enabled_drift_guard_changes_state_key(self):
        cfg = Settings(TIME_SOURCE="pts", VIDEO_SOURCE="data/test.mp4", SOURCE_KIND="file")
        self.assertNotEqual(configuration_key(cfg), configuration_key(replace(cfg, TRACKER_DRIFT_THRESHOLD=.6)))

    def test_disabled_drift_guard_preserves_the_previous_fingerprint(self):
        from dataclasses import asdict
        cfg = Settings(TIME_SOURCE="pts", VIDEO_SOURCE="data/test.mp4", SOURCE_KIND="file")
        current = configuration_key(cfg)
        previous = asdict(cfg)
        previous.pop('TRACKER_DRIFT_THRESHOLD')
        with patch('core.service.asdict', return_value=previous):
            self.assertEqual(configuration_key(cfg), current)

    def test_child_filter_changes_state_key_only_when_enabled(self):
        cfg = Settings(TIME_SOURCE="pts", VIDEO_SOURCE="data/test.mp4", SOURCE_KIND="file")
        self.assertEqual(configuration_key(cfg), configuration_key(replace(cfg, CHILD_DECISION_DELAY_SEC=20.)))
        self.assertNotEqual(configuration_key(cfg), configuration_key(replace(cfg, CHILD_MODE='model')))
