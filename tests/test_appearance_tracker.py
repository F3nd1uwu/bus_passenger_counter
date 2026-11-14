from types import SimpleNamespace
import unittest

import numpy as np
from ultralytics.trackers.bot_sort import BOTrack
from ultralytics.engine.results import Boxes
from ultralytics.trackers.basetrack import TrackState

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

    def test_lost_crop_cannot_take_an_already_observed_person(self):
        lost, active, detection = [self.track([1, 0]) for _ in range(3)]
        self.tracker.frame_id = 10
        lost.state, lost.frame_id = TrackState.Lost, 6
        active.state, active.frame_id = TrackState.Tracked, 9
        distances = self.tracker.get_dists([lost, active], [detection])
        self.assertEqual(distances[0, 0], 1)
        self.assertLess(distances[1, 0], self.tracker.args.match_thresh)

    def test_unrelated_active_person_does_not_block_lost_track_recovery(self):
        lost, detection = self.track([1, 0]), self.track([1, 0])
        active = self.track([0, 1])
        self.tracker.frame_id = 10
        lost.state, lost.frame_id = TrackState.Lost, 6
        active.state, active.frame_id = TrackState.Tracked, 9
        distances = self.tracker.get_dists([lost, active], [detection])
        self.assertLess(distances[0, 0], self.tracker.args.match_thresh)
        self.assertEqual(distances[1, 0], 1)

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

    def test_recent_turn_survives_four_missing_detections(self):
        self.tracker.minimum_similarity = .70
        old_id = int(self.feed([1, 0])[0, 4])
        # The last crop differs from the old EMA after a gradual turn.
        for angle in np.linspace(.1, 1.3, 10):
            self.assertEqual(int(self.feed([np.cos(angle), np.sin(angle)])[0, 4]), old_id)
        for _ in range(4):
            self.tracker.update(Boxes(np.empty((0, 6), np.float32), (100, 100)))
        current = self.feed([np.cos(1.4), np.sin(1.4)])
        self.assertEqual(int(current[0, 4]), old_id)

    def test_different_appearance_after_short_gap_does_not_reuse_id(self):
        self.tracker.minimum_similarity = .70
        old_id = int(self.feed([1, 0])[0, 4])
        for _ in range(4):
            self.tracker.update(Boxes(np.empty((0, 6), np.float32), (100, 100)))
        self.feed([0, 1])
        self.assertNotEqual(int(self.feed([0, 1])[0, 4]), old_id)

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

    def test_weak_box_after_one_missing_frame_keeps_id(self):
        self.tracker.set_detection_threshold(.35)
        old_id = int(self.feed([1, 0])[0, 4])
        self.tracker.update(Boxes(np.empty((0, 6), np.float32), (100, 100)))
        current = self.feed([.99, .01], confidence=.20)
        self.assertEqual(len(current), 1)
        self.assertEqual(int(current[0, 4]), old_id)


    def test_weak_different_person_after_gap_does_not_reactivate(self):
        self.tracker.set_detection_threshold(.35)
        self.feed([1, 0])
        self.tracker.update(Boxes(np.empty((0, 6), np.float32), (100, 100)))
        self.assertEqual(len(self.feed([0, 1], confidence=.20)), 0)

    def test_weak_detection_never_initializes_new_track(self):
        self.tracker.set_detection_threshold(.35)
        for _ in range(4):
            self.assertEqual(len(self.feed([1, 0], confidence=.30)), 0)

    def test_low_confidence_does_not_revive_long_missing_track(self):
        self.feed([1, 0])
        for _ in range(4):
            self.tracker.update(Boxes(np.empty((0, 6), np.float32), (100, 100)))
        self.assertEqual(len(self.feed([1, 0], confidence=.15)), 0)

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

    def drift_scene(self):
        self.tracker.drift_threshold = .6
        self.tracker.frame_id = 20
        self.tracker._image_shape = (100, 100)
        old = BOTrack(np.array([50, 60, 100, 80, 0], np.float32), .9, 0, np.array([0, 1], np.float32))
        old.track_id, old.frame_id = 1, 19
        old._appearance_history = [(f, np.array([1, 0], np.float32)) for f in range(3, 8)]
        new = BOTrack(np.array([50, 60, 100, 80, 0], np.float32), .4, 0, np.array([0, 1], np.float32))
        peer = self.track([1, 0])
        peer.track_id, peer.frame_id = 2, 19
        peer._appearance_history = [(f, np.array([1, 0], np.float32)) for f in range(15, 20)]
        peer._last_strong_box = np.array([0, 0, 40, 60], np.float32)
        self.tracker.tracked_stracks = [old, peer]
        return old, new, peer

    def test_merged_crop_cannot_gradually_inherit_a_persons_appearance(self):
        old, new, _ = self.drift_scene()
        self.assertEqual(self.tracker.get_dists([old], [new])[0, 0], 1)
        self.assertEqual(self.tracker._drift_rejections[0], {1})

    def test_single_person_bending_does_not_create_an_identity_boundary(self):
        old, new, _ = self.drift_scene()
        self.tracker.tracked_stracks = [old]
        self.assertLess(self.tracker.get_dists([old], [new])[0, 0], self.tracker.args.match_thresh)

    def test_old_or_duplicate_peer_is_not_evidence_of_a_second_person(self):
        for stale in [True, False]:
            with self.subTest(stale=stale):
                old, new, peer = self.drift_scene()
                if stale:
                    peer._appearance_history = [(f, np.array([1, 0], np.float32)) for f in range(1, 6)]
                else:
                    peer.curr_feat = np.array([0, 1], np.float32)
                self.assertLess(self.tracker.get_dists([old], [new])[0, 0], self.tracker.args.match_thresh)

    def test_confident_or_normal_sized_crop_may_change_during_a_turn(self):
        for confident in [True, False]:
            with self.subTest(confident=confident):
                old, new, _ = self.drift_scene()
                if confident:
                    new.score = .9
                else:
                    old._tlwh = new._tlwh = np.array([10, 20, 80, 80], np.float32)
                self.assertLess(self.tracker.get_dists([old], [new])[0, 0], self.tracker.args.match_thresh)

    def test_low_confidence_stage_also_applies_the_drift_veto(self):
        old, new, _ = self.drift_scene()
        new.score = .2
        distances = self.tracker._gate_appearance([old], [new], np.zeros((1, 1)))
        self.assertEqual(distances[0, 0], 1)

    def test_geometry_rejection_does_not_create_a_false_reid_exclusion(self):
        old, new, _ = self.drift_scene()
        old._tlwh = np.array([150, 0, 30, 80], np.float32)
        self.assertEqual(self.tracker.get_dists([old], [new])[0, 0], 1)
        self.assertEqual(self.tracker._drift_rejections, {})

    def test_new_track_retains_the_explicitly_rejected_parent(self):
        self.tracker._drift_rejections = {0: {7}}
        activated = []
        self.tracker._init_new_tracks([0], [self.track([1, 0])], activated)
        self.assertEqual(activated[0]._separated_from, (7,))
        self.tracker.identity_separations = {1: (7,)}
        self.tracker.reset()
        self.assertEqual(self.tracker.identity_separations, {})

    def test_drift_history_is_bounded(self):
        self.tracker.drift_threshold = .6
        for _ in range(100):
            self.feed([1, 0])
        history = self.tracker.tracked_stracks[0]._appearance_history
        self.assertLessEqual(len(history), self.tracker.DRIFT_LAG + 6)
