from pathlib import Path
import unittest

import numpy as np

from config.settings import ROOT, Settings
from core.detector import PersonDetector
from core.reid import ReIDEncoder
from core.vest_classifier import person_crop


class CropTests(unittest.TestCase):
    def test_boxes_outside_image_do_not_wrap_with_negative_slices(self):
        frame = np.zeros((100, 100, 3), np.uint8)
        for bbox in [(-100, -100, -1, -1), (110, 110, 200, 200), (50, 50, 10, 10)]:
            self.assertIsNone(person_crop(frame, bbox))
        self.assertEqual(person_crop(frame, (-10, -10, 60, 80)).shape, (80, 60, 3))


@unittest.skipUnless((ROOT / "models/person_reid.torchscript").is_file(), "Run python -m tools.prepare_reid")
class ModelTests(unittest.TestCase):
    def test_real_reid_encoder_returns_stable_unit_features(self):
        encoder = ReIDEncoder(ROOT / "models/person_reid.torchscript")
        frame = np.random.default_rng(5).integers(0, 256, (256, 128, 3), dtype=np.uint8)
        first = encoder.encode(frame, (0, 0, 128, 256))
        second = encoder.encode(frame, (0, 0, 128, 256))
        self.assertEqual(first.shape, (512,))
        self.assertTrue(np.isfinite(first).all())
        self.assertAlmostEqual(float(np.linalg.norm(first)), 1, places=5)
        np.testing.assert_allclose(first, second)

    def test_full_detector_starts_with_disabled_vest_filter(self):
        detector = PersonDetector(Settings(STAFF_MODE="disabled"))
        self.assertIsNotNone(detector.reid)
        self.assertIsNone(detector.vest)
        detector.reset()

    def test_enabled_staff_filter_requires_weights(self):
        with self.assertRaisesRegex(ValueError, "VEST_MODEL_PATH"):
            PersonDetector(Settings(STAFF_MODE="model", VEST_MODEL_PATH=""))

    @unittest.skipUnless((ROOT / "models/person_reid.onnx").is_file(), "Prepare CPU ONNX models")
    def test_onnx_batch_keeps_crop_order_and_matches_torchscript(self):
        encoder = ReIDEncoder(ROOT / "models/person_reid.onnx", threads=2)
        reference = ReIDEncoder(ROOT / "models/person_reid.torchscript")
        frame = np.random.default_rng(21).integers(0, 256, (256, 256, 3), dtype=np.uint8)
        boxes = [(0, 0, 128, 256), (-100, -100, -1, -1), (128, 0, 256, 256)]
        batch = encoder.encode_many(frame, boxes)
        self.assertIsNone(batch[1])
        for index in (0, 2):
            expected = reference.encode(frame, boxes[index])
            self.assertEqual(batch[index].shape, (512,))
            self.assertGreater(float(np.dot(batch[index], expected)), .99999)

    def test_detection_only_appearance_mode_handles_empty_frames(self):
        detector = PersonDetector(Settings(STAFF_MODE="disabled"), detection_only=True)
        frame = np.zeros((480, 640, 3), np.uint8)
        self.assertEqual(detector.detect(frame), [])
        self.assertEqual(detector.detect(frame), [])
