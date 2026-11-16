"""Opt-in CPU experiments; importing this module does not change production models."""
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
from dataclasses import replace
import time

from core.detector import PersonDetector
from core.openvino_session import OpenVINOSession


class Timings:
    """Inclusive stage measurements. Nested stages must not be added together."""

    def __init__(self):
        self.seconds = defaultdict(float)
        self.calls = defaultdict(int)

    def wrap(self, name, method):
        @wraps(method)
        def measured(*args, **kwargs):
            tick = time.perf_counter()
            try:
                return method(*args, **kwargs)
            finally:
                self.seconds[name] += time.perf_counter() - tick
                self.calls[name] += 1
        return measured

    def report(self):
        return {name: dict(seconds=value, calls=self.calls[name])
                for name, value in sorted(self.seconds.items())}


def detector_factory(backend='onnx', timings=None, detector_threads=None):
    if backend not in {'onnx', 'openvino'}:
        raise ValueError('Expected onnx or openvino')

    class ExperimentalDetector(PersonDetector):
        def __init__(self, cfg, *args, **kwargs):
            cfg = replace(cfg, CPU_DETECTOR_BACKEND=backend, DETECTOR_CPU_THREADS=detector_threads or 0)
            super().__init__(cfg, *args, **kwargs)
            if timings is None:
                return
            self.model.predict = timings.wrap('person_predict_inclusive', self.model.predict)
            if self.reid is not None:
                self.reid.encode_many = timings.wrap('reid_inclusive', self.reid.encode_many)
            if self.vest is not None:
                self.vest.predict = timings.wrap('vest_inclusive', self.vest.predict)
            if self.appearance_tracker is not None:
                self.appearance_tracker.update = timings.wrap('tracker_inclusive', self.appearance_tracker.update)
            self.detect = timings.wrap('detect_inclusive', self.detect)

    return ExperimentalDetector


@contextmanager
def instrument(backend, timings, detector_threads=None):
    """Patch only inside the explicit benchmark context, including failure cleanup."""
    from unittest.mock import patch
    from contextlib import ExitStack
    from core.counter_engine import PassengerCounterEngine
    from core.video_source import VideoSource
    from database.local_store import LocalStore
    from database.db_client import DatabaseClient
    with ExitStack() as stack:
        for module in ('core.service', 'tools.evaluate_clip'):
            stack.enter_context(patch(module + '.PersonDetector', detector_factory(backend, timings, detector_threads)))
        for cls, method, name in [
            (VideoSource, 'read', 'read_inclusive'),
            (PassengerCounterEngine, 'update', 'counter_inclusive'),
            (PassengerCounterEngine, '_identity', 'identity_inclusive'),
            (PassengerCounterEngine, 'snapshot', 'snapshot_inclusive'),
            (LocalStore, 'save', 'local_save_inclusive'),
            (DatabaseClient, 'deliver', 'postgres_deliver_inclusive'),
        ]:
            stack.enter_context(patch.object(cls, method, timings.wrap(name, getattr(cls, method))))
        yield
