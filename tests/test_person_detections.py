import unittest
import numpy as np

from core.person_detections import distinct_person_indices


class PersonDetectionTests(unittest.TestCase):
    def select(self, boxes, features):
        return distinct_person_indices(np.asarray(boxes, np.float32), np.asarray(features, np.float32)).tolist()

    def test_partial_crop_of_the_same_body_is_removed(self):
        boxes = [[10, 0, 110, 200, .9, 0], [12, 5, 108, 150, .8, 0]]
        self.assertEqual(self.select(boxes, [[1, 0], [.9, .43589]]), [0])

    def test_overlapping_people_with_different_appearance_remain(self):
        boxes = [[10, 0, 110, 200, .9, 0], [12, 5, 108, 150, .8, 0]]
        self.assertEqual(self.select(boxes, [[1, 0], [0, 1]]), [0, 1])

    def test_child_in_front_of_adult_is_not_a_duplicate_crop(self):
        boxes = [[10, 0, 110, 200, .9, 0], [15, 60, 105, 200, .8, 0]]
        self.assertEqual(self.select(boxes, [[1, 0], [1, 0]]), [0, 1])

    def test_side_by_side_lookalikes_remain_separate(self):
        boxes = [[10, 0, 110, 200, .9, 0], [50, 0, 150, 200, .8, 0]]
        self.assertEqual(self.select(boxes, [[1, 0], [1, 0]]), [0, 1])

    def test_full_body_preferred_over_higher_confidence_torso_crop(self):
        boxes = [[10, 0, 110, 200, .6, 0], [12, 5, 108, 150, .9, 0]]
        self.assertEqual(self.select(boxes, [[1, 0], [1, 0]]), [0])

    def test_loose_box_with_different_boundaries_does_not_suppress_person(self):
        boxes = [[0, 0, 150, 250, .6, 0], [12, 25, 108, 180, .9, 0]]
        self.assertEqual(self.select(boxes, [[1, 0], [1, 0]]), [0, 1])
