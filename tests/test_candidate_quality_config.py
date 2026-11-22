import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.benchmark_cpu import controls


class CandidateQualityConfigTests(unittest.TestCase):
    def capture(self, vest):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = root / 'plan.json'
            plan.write_text(json.dumps({'episodes': [{'id': 'one'}]}))
            args = argparse.Namespace(output=root/'result.json', plan=plan, shard_index=0, shards=1,
                                      model=root/'person.onnx', backend='onnx', threads=2,
                                      detector_threads=None, staff_mode='model', vest_model=vest)
            captured = []

            def constructor(cfg):
                captured.append(cfg)
                raise RuntimeError('stop before loading private videos or models')

            with patch('tools.cpu_backends.detector_factory', return_value=constructor):
                with self.assertRaisesRegex(RuntimeError, 'stop before loading'):
                    controls(args)
            return captured[0]

    def test_vest_candidate_reaches_quality_worker(self):
        model = Path('models/candidates/separate_test_vest.onnx')
        self.assertEqual(self.capture(model).VEST_MODEL_PATH, str(model.resolve()))

    def test_unspecified_vest_retains_profile_model(self):
        self.assertEqual(self.capture(None).VEST_MODEL_PATH, 'models/candidates/work_vest_v4.onnx')
