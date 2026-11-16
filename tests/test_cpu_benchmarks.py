"""Optional backend boundaries must preserve production behaviour and data ownership."""
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from core.counter_engine import PassengerCounterEngine
from tools.cpu_backends import OpenVINOSession, Timings, instrument


class CPUBenchmarkTests(unittest.TestCase):
    def test_openvino_is_rejected_for_gpu_or_non_onnx_configuration(self):
        from config.settings import Settings
        from dataclasses import replace
        cfg = Settings(CPU_DETECTOR_BACKEND='openvino', MODEL_PATH='models/candidates/person_test2_v2.onnx')
        cfg.validate()
        for changed in [replace(cfg, DEVICE='0'), replace(cfg, MODEL_PATH='weights.pt'),
                        replace(cfg, CPU_DETECTOR_BACKEND='unknown'), replace(cfg, DETECTOR_CPU_THREADS=-1)]:
            with self.assertRaises(ValueError):
                changed.validate()

    def test_openvino_result_is_not_overwritten_by_next_request(self):
        buffer = np.zeros((1, 5, 2), np.float32)

        class Request:
            def infer(self, inputs):
                buffer[:] += 1

            def get_output_tensor(self, index):
                return SimpleNamespace(data=buffer)

        session = OpenVINOSession.__new__(OpenVINOSession)
        session.request = Request()
        session.outputs = [SimpleNamespace(name='output0')]
        first = session.run(['output0'], {'images': np.zeros(1)})[0]
        second = session.run(None, {'images': np.zeros(1)})[0]
        np.testing.assert_array_equal(first, np.ones_like(buffer))
        np.testing.assert_array_equal(second, 2 * np.ones_like(buffer))
        with self.assertRaises(ValueError):
            session.run(['wrong-output'], {})

    def test_profiling_restores_real_engine_after_exception(self):
        original = PassengerCounterEngine.update
        with self.assertRaisesRegex(RuntimeError, 'intentional'):
            with instrument('onnx', Timings()):
                self.assertIsNot(PassengerCounterEngine.update, original)
                raise RuntimeError('intentional')
        self.assertIs(PassengerCounterEngine.update, original)

    def test_failed_operation_is_still_accounted_for(self):
        timings = Timings()
        def failure():
            raise ValueError('failure')
        with self.assertRaises(ValueError):
            timings.wrap('stage', failure)()
        self.assertEqual(timings.report()['stage']['calls'], 1)
        self.assertGreater(timings.report()['stage']['seconds'], 0)

    def test_openvino_is_not_imported_by_default(self):
        # A clean process can use the ONNX benchmark without optional packages.
        import subprocess
        result = subprocess.run([sys.executable, '-c',
            "import sys; import tools.cpu_backends; assert 'openvino' not in sys.modules"],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
