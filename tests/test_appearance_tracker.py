from types import SimpleNamespace
import unittest

import numpy as np
from ultralytics.trackers.bot_sort import BOTrack
from ultralytics.engine.results import Boxes

from core.appearance_tracker import AppearanceTracker


class AppearanceTrackerTests(unittest.TestCase):
    def setUp(self):
        args = SimpleNamespace(track_high_thresh=.25, track_low_thresh=.1, new_track_thresh=.25,
                               track_buffer=60, match_thresh=.8, fuse_score=True, gmc_method="none",
                               proximity_thresh=.1, appearance_thresh=.8, with_reid=True, model="auto")
        self.tracker = AppearanceTracker(args, minimum_similarity=.5)

    @staticmethod
    def track(vector):
        return BOTrack(np.array([50, 50, 20, 20, 0], np.float32), 1.0, 0, np.array(vector, np.float32))

    def test_overlapping_boxes_cannot_steal_a_different_appearance(self):
        old, new = self.track([1, 0]), self.track([0, 1])
        distances = self.tracker.get_dists([old], [new])
        self.assertGreater(distances[0, 0], self.tracker.args.match_thresh)

    def test_same_appearance_remains_matchable(self):
        old, new = self.track([1, 0]), self.track([.99, .01])
        distances = self.tracker.get_dists([old], [new])
        self.assertLess(distances[0, 0], self.tracker.args.match_thresh)

    def feed(self, vector, confidence=.9, bbox=(40, 40, 60, 60)):
        boxes = Boxes(np.array([[*bbox, confidence, 0]], np.float32), (100, 100))
        return self.tracker.update(boxes, feats=np.asarray([vector], np.float32))

    def test_gradually_changing_crop_keeps_one_trajectory(self):
        self.tracker.minimum_similarity = .70
        ids = []
        for angle in np.linspace(0, 1.6, 9):
            result = self.feed([np.cos(angle), np.sin(angle)])
            self.assertEqual(len(result), 1)
            ids.append(int(result[0, 4]))
        self.assertEqual(len(set(ids)), 1)

    def test_lost_track_does_not_take_next_similar_person(self):
        self.tracker.minimum_similarity = .70
        old_id = int(self.feed([1, 0])[0, 4])
        for _ in range(5):
            self.tracker.update(Boxes(np.empty((0, 6), np.float32), (100, 100)))
        self.feed([.8, .6])
        current = self.feed([.8, .6])
        self.assertEqual(len(current), 1)
        self.assertNotEqual(int(current[0, 4]), old_id)

    def test_lost_person_can_return_with_strong_appearance_evidence(self):
        old_id = int(self.feed([1, 0])[0, 4])
        for _ in range(5):
            self.tracker.update(Boxes(np.empty((0, 6), np.float32), (100, 100)))
        current = self.feed([.99, .01])
        self.assertEqual(int(current[0, 4]), old_id)

    def test_low_confidence_box_cannot_bypass_appearance_veto(self):
        old_id = int(self.feed([1, 0])[0, 4])
        self.assertEqual(len(self.feed([0, 1], confidence=.15)), 0)
        self.feed([0, 1])
        current = self.feed([0, 1])
        self.assertNotEqual(int(current[0, 4]), old_id)

    def test_low_confidence_same_person_is_still_tracked(self):
        old_id = int(self.feed([1, 0])[0, 4])
        current = self.feed([.99, .01], confidence=.15)
        self.assertEqual(len(current), 1)
        self.assertEqual(int(current[0, 4]), old_id)

    def test_similar_crop_cannot_teleport_to_disjoint_box(self):
        old_id = int(self.feed([1, 0])[0, 4])
        self.feed([1, 0], bbox=(75, 75, 95, 95))
        current = self.feed([1, 0], bbox=(75, 75, 95, 95))
        self.assertEqual(len(current), 1)
        self.assertNotEqual(int(current[0, 4]), old_id)

    def test_departed_body_fragment_cannot_take_next_person(self):
        old_id = None
        for bottom in [90, 85, 78, 70, 60, 50, 40, 30, 23, 18]:
            current = self.feed([1, 0], bbox=(25, 0, 75, bottom))
            if old_id is None:
                old_id = int(current[0, 4])
        self.feed([.8, .6], bbox=(25, 0, 75, 50))
        current = self.feed([.8, .6], bbox=(25, 0, 75, 60))
        self.assertEqual(len(current), 1)
        self.assertNotEqual(int(current[0, 4]), old_id)
