from dataclasses import replace
from datetime import timedelta
from threading import Event
import unittest

import numpy as np

from config.settings import Settings
from core.child_filter import ChildClassifier, ChildEvidence, ChildObserver
from core.counter_engine import PassengerCounterEngine
from core.types import ChildObservation, Detection
from tests import test_counter as counter_tests


class ChildEvidenceTests(unittest.TestCase):
    def test_three_distinct_positive_samples_are_required(self):
        evidence = ChildEvidence()
        for _ in range(3):
            evidence.observe('child', 'same-frame', 3)
        self.assertFalse(evidence.confirmed)
        evidence.observe('child', 'second', 3)
        evidence.observe('child', 'third', 3)
        self.assertTrue(evidence.confirmed)

    def test_unknown_or_adult_evidence_prevents_exclusion(self):
        for verdict in ['unknown', 'adult']:
            with self.subTest(verdict=verdict):
                evidence = ChildEvidence()
                for i in range(3):
                    evidence.observe('child', str(i), 3)
                evidence.observe(verdict, 'new', 3)
                self.assertFalse(evidence.confirmed)
                if verdict == 'adult':
                    for i in range(3):
                        evidence.observe('child', 'later'+str(i), 3)
                    self.assertFalse(evidence.confirmed)

    def test_sample_memory_is_bounded(self):
        evidence = ChildEvidence()
        for i in range(100):
            evidence.observe('unknown', str(i), 3)
        self.assertEqual(len(evidence.seen_samples), 32)


class ChildModelPolicyTests(unittest.TestCase):
    def model(self, age=5.):
        class Session:
            def __init__(self, output):
                self.output = output
                self.calls = 0
            def run(self, *args):
                self.calls += 1
                return [self.output]
        model = ChildClassifier.__new__(ChildClassifier)
        model.meta = dict(minimum_child_margin=.005, maximum_child_age=7., adult_veto_age=12.)
        model.text = np.zeros((11, 512), np.float32)
        model.text[:4, 0] = 1
        model.text[4:, 1] = 1
        features = np.zeros((2, 512), np.float32)
        features[:, 0] = 1
        model.vision = Session(features)
        model.age = Session(np.array([[age]], np.float32))
        model.vision_mean = model.age_mean = np.zeros(3, np.float32)
        model.vision_std = model.age_std = np.ones(3, np.float32)
        model.empty_face = np.zeros((3, 384, 384), np.float32)
        return model

    def test_conflicting_views_are_unknown_without_slow_age_inference(self):
        model = self.model()
        model.vision.output[1] = 0
        model.vision.output[1, 1] = 1
        self.assertEqual(model.predict(np.zeros((100, 60, 3), np.uint8)), 'unknown')
        self.assertEqual(model.age.calls, 0)

    def test_age_borderline_and_invalid_outputs_are_unknown(self):
        for age, expected in [(5., 'child'), (7., 'unknown'), (12., 'adult'), (-1., 'unknown'), (float('nan'), 'unknown')]:
            with self.subTest(age=age):
                self.assertEqual(self.model(age).predict(np.zeros((100, 60, 3), np.uint8)), expected)

    def test_invalid_vision_output_does_not_exclude_anyone(self):
        model = self.model()
        model.vision.output[:] = np.nan
        self.assertEqual(model.predict(np.zeros((100, 60, 3), np.uint8)), 'unknown')


class ChildCounterTests(unittest.TestCase):
    def setUp(self):
        fixture = counter_tests.CounterTests()
        fixture.setUp()
        self.fixture = fixture
        self.cfg = replace(fixture.cfg, STAFF_MODE='disabled', CHILD_MODE='model')
        fixture.cfg = self.cfg
        fixture.engine = PassengerCounterEngine(self.cfg)

    def observations(self, verdict='child', embedding=(1, 0, 0)):
        return [ChildObservation(1, verdict, str(i), np.array(embedding)) for i in range(3)]

    def test_uncertain_age_counts_at_eof(self):
        self.assertEqual(self.fixture.cross(), [])
        events = self.fixture.engine.finish()
        self.assertEqual([(e.counted, e.reason) for e in events], [(True, 'COUNTED')])

    def test_late_child_evidence_survives_track_disappearance(self):
        self.fixture.cross()
        engine = self.fixture.engine
        engine.update(self.fixture.frame, [], self.fixture.start + timedelta(seconds=3))
        self.assertIn(1, engine.tracks)
        engine.observe_children(self.observations())
        events = engine.finish()
        self.assertEqual([(e.counted, e.reason) for e in events], [(False, 'CHILD')])
        identity = engine.identities[engine.tracks[1].identity_id]
        self.assertIsNone(identity.last_in)

    def test_child_evidence_is_saved_with_pending_events(self):
        self.fixture.cross()
        self.fixture.engine.observe_children(self.observations())
        restored = PassengerCounterEngine(self.cfg, self.fixture.engine.snapshot())
        self.assertEqual([(e.counted, e.reason) for e in restored.finish()], [(False, 'CHILD')])

    def test_confirmed_child_does_not_increase_in_or_out(self):
        self.fixture.cross()
        self.fixture.engine.observe_children(self.observations())
        events = self.fixture.engine.finish()
        self.fixture.cross(entering=False)
        events += self.fixture.engine.finish()
        self.assertEqual([(e.event_type, e.counted, e.reason) for e in events],
                         [('IN', False, 'CHILD'), ('OUT', False, 'CHILD')])
        self.assertEqual((self.fixture.engine.total_in, self.fixture.engine.total_out), (0, 0))

    def test_confirmed_child_can_be_excluded_without_vest_evidence(self):
        self.fixture.engine = PassengerCounterEngine(replace(self.cfg, STAFF_MODE='model'))
        self.fixture.cross(probability=None)
        self.fixture.engine.observe_children(self.observations())
        self.assertEqual(self.fixture.engine.finish()[0].reason, 'CHILD')

    def test_changed_appearance_rejects_a_late_child_result(self):
        self.fixture.cross()
        engine = self.fixture.engine
        engine.observe_children(self.observations(embedding=(0, 1, 0)))
        self.assertTrue(engine.finish()[0].counted)

    def test_reid_after_short_ttl_does_not_discard_an_older_pending_event(self):
        self.fixture.cross()
        engine = self.fixture.engine
        engine.update(self.fixture.frame, [Detection(2, (40, 70, 60, 90), .9, np.array([1, 0, 0]))],
                      self.fixture.start + timedelta(seconds=3))
        self.assertIn(1, engine.tracks)
        self.assertEqual(len(engine.finish()), 1)

    def test_checkpoint_keeps_old_pending_and_new_origin_for_one_identity(self):
        self.fixture.cross()
        engine = self.fixture.engine
        engine.observe_children(self.observations())
        for second in [3., 3.1]:
            engine.update(self.fixture.frame, [Detection(2, (40, 70, 60, 90), .9, np.array([1, 0, 0]))],
                          self.fixture.start + timedelta(seconds=second))
        saved = engine.snapshot()
        pid = engine.tracks[2].identity_id
        self.assertEqual(saved['continuity'][pid]['track_id'], 2)
        self.assertEqual(saved['continuity'][pid]['zone'], 'cabin')
        self.assertEqual(len(saved['deferred']), 1)
        restored = PassengerCounterEngine(self.cfg, saved)
        self.assertEqual([(e.counted, e.reason) for e in restored.finish()], [(False, 'CHILD')])

    def test_disabled_mode_ignores_age_results(self):
        self.fixture.engine = PassengerCounterEngine(replace(self.cfg, CHILD_MODE='disabled'))
        events = self.fixture.cross()
        self.fixture.engine.observe_children(self.observations())
        self.assertTrue(events[0].counted)


class ChildWorkerTests(unittest.TestCase):
    def test_queue_is_bounded_and_frame_processing_does_not_wait(self):
        started, release = Event(), Event()
        class Classifier:
            def predict(self, crop):
                started.set()
                release.wait(3)
                return 'unknown'
        observer = ChildObserver(Classifier())
        try:
            frame = np.zeros((600, 1000, 3), np.uint8)
            detections = [Detection(i, ((i % 5)*180, (i // 5)*140, (i % 5)*180+100, (i // 5)*140+100),
                                    .9, np.array([1., 0, 0])) for i in range(20)]
            for _ in range(3):
                observer.update(frame, detections, .3)
            self.assertTrue(started.wait(1))
            self.assertLessEqual(len(observer.jobs), observer.CAPACITY)
            observer.reset()
            release.set()
            self.assertEqual(observer.drain(), [])
        finally:
            release.set()
            observer.close()

    def test_inference_errors_are_unknown(self):
        class Classifier:
            def predict(self, crop):
                raise RuntimeError('test inference failure')
        observer = ChildObserver(Classifier())
        try:
            for _ in range(3):
                observer.update(np.zeros((100, 100, 3), np.uint8),
                                [Detection(1, (10, 0, 70, 90), .9, np.array([1., 0, 0]))], .3)
            with self.assertLogs('core.child_filter', level='ERROR'):
                updates = observer.drain(timeout=1)
            self.assertEqual(updates[0].verdict, 'unknown')
        finally:
            observer.close()
