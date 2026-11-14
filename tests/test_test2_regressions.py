"""Opt-in regressions on private, reviewed video caches (never downloaded).

RUN_TEST2_REGRESSION=1 python -m unittest tests.test_test2_regressions
"""
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from config.settings import ROOT, Settings
from config.profiles import apply_profile
from tools.replay_detections import replay_raw
from tools.replay_trace import replay
from tools.check_identity_pairs import evaluate_pairs


@unittest.skipUnless(os.getenv('RUN_TEST2_REGRESSION') == '1', 'Opt-in test2 video regressions')
class Test2RegressionTests(unittest.TestCase):
    def test_known_passages_and_hourly_repeat(self):
        cfg = replace(apply_profile(Settings(), 'test2'), STAFF_MODE='disabled')
        plan = json.loads((ROOT/'reports/test2-2026-11-13/regression_plan.json').read_text('utf-8'))
        for episode in plan['episodes']:
            with self.subTest(episode=episode['id']):
                cache = ROOT/'.local/test2_review/raw_v2'/(episode['id']+'.npz')
                self.assertTrue(cache.is_file(), f'Private ML cache required: {cache}')
                with tempfile.TemporaryDirectory() as folder:
                    output = Path(folder)/'replay.json'
                    result = replay_raw(cache, cfg, output=output)
                    self.assertEqual((result['in'], result['out']),
                                     (episode['manual_in'], episode['manual_out']))
                    self.assertEqual(result['suppressed'], 0)
                    if episode['id'] == 'control_clip_05':
                        with np.load(cache, allow_pickle=False) as data:
                            np.savez_compressed(output.with_suffix('.features.npz'), features=data['features'])
                        repeated = replay(output, cfg)
                        self.assertEqual((repeated[0]['new_in'], repeated[0]['new_out']), (8, 0))
                        self.assertEqual((repeated[1]['new_in'], repeated[1]['new_out']), (0, 0))

    def test_driver_excluded_but_striped_jacket_counted(self):
        cfg = replace(apply_profile(Settings(), 'test2'), CPU_THREADS=2)
        for name, source, counts, staff in [
            ('V01', 'data/test2.mp4', (0, 0), 1),
            ('V03', 'data/test2.mp4', (0, 0), 1),
            ('V05', 'data/test2.mp4', (0, 0), 1),
            ('control_clip_01', 'data/test.mp4', (1, 0), 0),
        ]:
            with self.subTest(episode=name):
                cache = ROOT/'.local/test2_review/raw_v2'/(name+'.npz')
                self.assertTrue(cache.is_file(), f'Private ML cache required: {cache}')
                result = replay_raw(cache, cfg, source=source)
                self.assertEqual((result['in'], result['out']), counts)
                self.assertEqual(sum(e['reason'] == 'STAFF' for e in result['events']), staff)

    def test_foreground_light_panel_puffer_is_not_staff(self):
        # P02 is garment-v3 training/calibration, not an independent accuracy test.
        cfg = replace(apply_profile(Settings(), 'test2'), CPU_THREADS=2)
        cache = ROOT/'.local/test2_review/raw_v2/P02.npz'
        self.assertTrue(cache.is_file(), f'Private ML cache required: {cache}')
        with tempfile.TemporaryDirectory() as folder:
            result = replay_raw(cache, cfg, source='data/test2.mp4', output=Path(folder)/'puffer.json')
        self.assertEqual((result['in'], result['out']), (7, 0))
        self.assertFalse(any(d['staff_confirmed'] for row in result['trace'] for d in row['detections']))
        labels = json.loads((ROOT/'reports/id-continuity-2026-11-14/identity_pairs.json').read_text('utf-8'))
        pair_result = evaluate_pairs(result['trace'], labels)
        for label in ['tracker', 'identity']:
            self.assertEqual(pair_result['summary'][label], {'correct': 5, 'incorrect': 0, 'unresolved': 0})
