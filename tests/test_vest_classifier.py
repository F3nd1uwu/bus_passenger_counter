from types import SimpleNamespace
import unittest

import numpy as np

from core.vest_classifier import VestClassifier, garment_tensor, person_crop, vest_tensor, spatial_consensus


class FakeSession:
    def __init__(self, output):
        self.output = output

    def get_inputs(self):
        return [SimpleNamespace(name="images")]

    def run(self, _, inputs):
        assert inputs["images"].dtype == np.float32
        return [self.output]


class VestTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((80, 40, 3), np.uint8)

    def model(self, output, garment=True):
        result = VestClassifier.__new__(VestClassifier)
        result.session = FakeSession(np.asarray(output, np.float32))
        result.members = None
        result.garment_detector = garment
        result.size = 32
        return result

    def test_models_receive_luminance_not_hue(self):
        blue = np.zeros((80, 40, 3), np.uint8)
        blue[:, :, 0] = 255  # OpenCV luminance 29, same as the gray image.
        gray = np.full_like(blue, 29)
        np.testing.assert_array_equal(vest_tensor(blue), vest_tensor(gray))
        np.testing.assert_array_equal(garment_tensor(blue, 32)[0], garment_tensor(gray, 32)[0])

    def test_invalid_or_tiny_boxes_are_unknown(self):
        for box in [(0, 0, float("nan"), 50), (0, 0, 40, float("inf")), (0, 0, 10, 10)]:
            self.assertIsNone(person_crop(self.frame, box))

    def test_classifier_softmax_handles_large_logits(self):
        model = self.model([[1000, 1002]], garment=False)
        self.assertAlmostEqual(model.predict(self.frame, (0, 0, 40, 80)), .880797, places=5)

    def test_garment_in_letterbox_padding_is_not_evidence(self):
        # Content occupies x=8..24. High score at x=2 belongs to padding.
        output = [[[2, 16], [16, 16], [4, 12], [12, 12], [.99, .40]]]
        model = self.model(output)
        self.assertAlmostEqual(model.predict(self.frame, (0, 0, 40, 80)), .40, places=5)

    def test_end_to_end_and_dense_detector_outputs_agree(self):
        dense = self.model([[[16], [16], [12], [12], [.75]]])
        end_to_end = self.model([[[10, 10, 22, 22, .75, 0]]])
        for model in [dense, end_to_end]:
            self.assertAlmostEqual(model.predict(self.frame, (0, 0, 40, 80)), .75)

    def test_malformed_model_output_fails_explicitly(self):
        for output in [np.zeros((2, 5, 1)), np.full((1, 5, 1), np.nan), [[[0, 0, 20, 20, .9, 1]]]]:
            with self.assertRaises(RuntimeError):
                self.model(output).predict(self.frame, (0, 0, 40, 80))

    def test_ensemble_requires_both_models_to_agree(self):
        first = self.model([[[16], [16], [12], [12], [.90]]])
        second = self.model([[[16], [16], [12], [12], [.20]]])
        ensemble = VestClassifier.__new__(VestClassifier)
        ensemble.members = [first, second]
        self.assertAlmostEqual(ensemble.predict(self.frame, (0, 0, 40, 80)), .20, places=5)
        self.assertIsNone(ensemble.predict(self.frame, (0, 0, 5, 5)))

    def test_spatial_ensemble_rejects_different_garment_locations(self):
        first = self.model([[[10, 0, 22, 12, .9, 0]]])
        second = self.model([[[10, 20, 22, 32, .95, 0]]])
        ensemble = VestClassifier.__new__(VestClassifier)
        ensemble.members, ensemble.minimum_iou = [first, second], .3
        self.assertEqual(ensemble.predict(self.frame, (0, 0, 40, 80)), 0)

    def test_spatial_consensus_uses_best_matching_pair_not_highest_unrelated_scores(self):
        first = np.array([[0, 0, 10, 10, .95], [20, 20, 30, 30, .7]])
        second = np.array([[40, 40, 50, 50, .9], [21, 21, 31, 31, .8]])
        self.assertAlmostEqual(spatial_consensus(first, second, .3), .7)
        self.assertEqual(spatial_consensus(first[:0], second, .3), 0)
