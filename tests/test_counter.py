from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import unittest

import numpy as np

from config.settings import Settings
from core.counter_engine import PassengerCounterEngine
from core.types import Detection


class CounterTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Settings(STAFF_MODE="model")
        self.engine = PassengerCounterEngine(self.cfg)
        self.frame = np.zeros((100, 100, 3), np.uint8)
        self.start = datetime(2026, 11, 6, 12, tzinfo=timezone.utc)
        self.time = 0

    def feed(self, y, tid=1, embedding=(1, 0, 0), probability=0, x=40, at=None):
        self.time = self.time + 0.1 if at is None else at
        detection = Detection(tid, (x, y - 10, x + 20, y + 10), 0.9,
                              np.array(embedding) if embedding is not None else None, probability)
        return self.engine.update(self.frame, [detection], self.start + timedelta(seconds=self.time))

    def cross(self, tid=1, entering=True, embedding=(1, 0, 0), probability=0):
        events = []
        for y in ([20, 20, 50, 80, 80] if entering else [80, 80, 50, 20, 20]):
            events.extend(self.feed(y, tid, embedding, probability))
        return events

    def test_confirmed_entry_and_exit(self):
        self.assertEqual([(e.event_type, e.counted) for e in self.cross()], [("IN", True)])
        self.assertEqual([(e.event_type, e.counted) for e in self.cross(entering=False)], [("OUT", True)])
        self.assertEqual((self.engine.total_in, self.engine.total_out), (1, 1))

    def test_repeats_in_both_directions_suppressed(self):
        self.cross()
        self.cross(entering=False)
        events = self.cross() + self.cross(entering=False)
        self.assertTrue(all(not e.counted and e.reason == "REPEAT_WITHIN_HOUR" for e in events))
        self.assertEqual((self.engine.total_in, self.engine.total_out), (1, 1))

    def test_hour_is_not_extended_by_a_suppressed_entry(self):
        self.cross()
        self.time = 100
        self.cross(entering=False)
        self.cross()
        self.time = 3601
        self.engine.reset_tracking()
        self.assertTrue(self.cross(tid=50)[0].counted)

    def test_id_switch_during_a_crossing_preserves_origin(self):
        self.feed(20)
        self.feed(20)
        self.feed(50, tid=2)
        self.feed(80, tid=2)
        events = self.feed(80, tid=2)
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0].counted)

    def test_new_tracker_id_after_return_uses_person_identity(self):
        first = self.cross()[0]
        self.engine.reset_tracking()
        again = self.cross(tid=900)[0]
        self.assertEqual(first.identity_id, again.identity_id)
        self.assertFalse(again.counted)

    def test_distinct_people_close_in_time_are_both_counted(self):
        self.cross()
        self.assertTrue(self.cross(tid=2, embedding=(0, 1, 0))[0].counted)
        self.assertEqual(self.engine.total_in, 2)

    def test_simultaneous_lookalikes_cannot_share_identity(self):
        events = []
        for index, y in enumerate([20, 20, 50, 80, 80]):
            detections = [Detection(tid, (x, y - 10, x + 10, y + 10), 1, np.array([1, 0]), 0)
                          for tid, x in [(1, 20), (2, 60)]]
            events += self.engine.update(self.frame, detections, self.start + timedelta(seconds=index / 10))
        self.assertEqual(sum(e.counted for e in events), 2)
        self.assertEqual(len({e.identity_id for e in events}), 2)

    def test_jitter_near_line_and_partial_return_do_not_count(self):
        for y in [20, 20, 45, 52, 48, 55, 20, 20]:
            self.assertEqual(self.feed(y), [])

    def test_single_frame_in_far_zone_is_insufficient(self):
        for y in [20, 20, 80, 50, 20, 20]:
            self.assertEqual(self.feed(y), [])

    def test_first_appearance_in_cabin_is_not_an_entry(self):
        for _ in range(30):
            self.assertEqual(self.feed(80), [])

    def test_stationary_passenger_does_not_become_staff(self):
        for i in range(80):
            self.feed(20, at=i / 10)
        events = self.feed(80) + self.feed(80)
        self.assertTrue(events[0].counted)
        self.assertEqual(events[0].reason, "COUNTED")

    def test_vest_evidence_excludes_staff(self):
        events = self.cross(probability=0.99)
        self.assertEqual(events[0].reason, "STAFF")
        self.assertFalse(events[0].counted)

    def test_one_bad_staff_prediction_does_not_exclude_passenger(self):
        self.feed(20, probability=0.99)
        self.feed(20)
        self.feed(50)
        self.feed(80)
        self.assertTrue(self.feed(80)[0].counted)

    def test_outside_door_does_not_count(self):
        for y in [20, 20, 50, 80, 80]:
            self.assertEqual(self.feed(y, x=95), [])

    def test_missing_features_never_fall_back_to_colour_or_tracker_id(self):
        self.assertEqual(self.cross(embedding=None), [])

    def test_checkpoint_preserves_cooldowns(self):
        self.cross()
        state = json.loads(json.dumps(self.engine.snapshot()))
        self.engine = PassengerCounterEngine(self.cfg, state)
        self.time += 5
        self.assertFalse(self.cross(tid=42)[0].counted)

    def test_checkpoint_in_middle_of_crossing(self):
        self.feed(20)
        self.feed(20)
        self.feed(50)
        self.engine = PassengerCounterEngine(self.cfg, json.loads(json.dumps(self.engine.snapshot())))
        self.feed(80, tid=99)
        event = self.feed(80, tid=99)[0]
        self.assertTrue(event.counted)

    def test_midnight_does_not_reset_hourly_memory(self):
        self.start = datetime(2026, 11, 6, 19, 59, 59, tzinfo=timezone.utc)
        self.cross()
        self.time += 5
        self.engine.reset_tracking()
        self.assertFalse(self.cross(tid=55)[0].counted)
        self.assertEqual(self.engine.day, "2026-11-07")
        self.assertEqual(self.engine.total_in, 0)

    def test_timestamps_must_be_aware_and_monotonic(self):
        self.feed(20)
        with self.assertRaises(ValueError):
            self.engine.update(self.frame, [], datetime(2026, 1, 1))
        with self.assertRaises(ValueError):
            self.engine.update(self.frame, [], self.start)

    def test_invalid_embedding_is_rejected(self):
        for vector in [(0, 0), (float("nan"), 1)]:
            with self.assertRaises(ValueError):
                self.feed(20, embedding=vector)

    def test_geometry_is_validated(self):
        for kwargs in ({"Y_STREET": 0.9}, {"REID_THRESHOLD": float("nan")}, {"MAX_IDENTITIES": 0}):
            with self.assertRaises(ValueError):
                replace(self.cfg, **kwargs).validate()

    def test_unresolved_crossing_is_auditable_at_eof(self):
        self.assertEqual(self.cross(probability=None), [])
        events = self.engine.finish()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].reason, "INSUFFICIENT_MODEL_EVIDENCE")
        self.assertFalse(events[0].counted)
        self.assertEqual(self.engine.finish(), [])

    def test_pending_crossing_survives_restart_and_late_evidence(self):
        self.assertEqual(self.cross(probability=None), [])
        self.engine = PassengerCounterEngine(self.cfg, json.loads(json.dumps(self.engine.snapshot())))
        events = []
        for _ in range(3):
            events += self.feed(80, tid=90)
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0].counted)

    def test_invalid_staff_probability_fails_explicitly(self):
        with self.assertRaises(ValueError):
            self.feed(20, probability=float("nan"))

    def test_unidentified_crossing_is_audited_after_restart(self):
        self.assertEqual(self.cross(embedding=None), [])
        self.engine = PassengerCounterEngine(self.cfg, json.loads(json.dumps(self.engine.snapshot())))
        events = self.engine.update(self.frame, [], self.start + timedelta(seconds=1))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].reason, "INSUFFICIENT_MODEL_EVIDENCE")
        self.assertFalse(events[0].counted)

    def test_explicit_disabled_staff_mode_counts_without_vest_evidence(self):
        self.cfg = replace(self.cfg, STAFF_MODE="disabled")
        self.engine = PassengerCounterEngine(self.cfg)
        event = self.cross(probability=None)[0]
        self.assertTrue(event.counted)
        self.assertEqual(event.reason, "COUNTED")

    def test_camera_exit_establishes_origin_before_bbox_shrinks(self):
        # Measured trajectory of the supplied top-down camera: the first full
        # person box has anchor y=0.652 and then moves towards the street.
        events = []
        for y in [65.2, 64, 62, 60, 58, 50, 38, 35, 32]:
            events.extend(self.feed(y))
        self.assertEqual([(e.event_type, e.counted) for e in events], [("OUT", True)])

    def test_reid_keeps_initial_appearance_after_pose_changes(self):
        first = self.cross()[0]
        for angle in np.linspace(0, np.pi / 2, 90):
            self.feed(80, embedding=(np.cos(angle), np.sin(angle), 0))
        identity = self.engine.identities[first.identity_id]
        self.assertLess(identity.embedding[0], self.cfg.REID_THRESHOLD)
        self.assertLessEqual(len(identity.gallery), self.cfg.REID_GALLERY_SIZE)
        state = json.loads(json.dumps(self.engine.snapshot()))
        self.engine = PassengerCounterEngine(self.cfg, state)
        self.time += 5
        repeated = self.cross(tid=95)[0]
        self.assertEqual(repeated.identity_id, first.identity_id)
        self.assertFalse(repeated.counted)


if __name__ == "__main__":
    unittest.main()
