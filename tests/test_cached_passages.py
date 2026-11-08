"""Optional association regressions using the supplied raw ML captures.

RUN_CACHED_VIDEO_REGRESSION=1 python -m unittest tests.test_cached_passages
"""
from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from config.settings import ROOT, Settings
from tools.replay_detections import replay_raw
from tools.replay_trace import replay


@unittest.skipUnless(os.getenv('RUN_CACHED_VIDEO_REGRESSION') == '1', 'Opt-in cached video regressions')
class CachedPassageTests(unittest.TestCase):
    def test_manual_passages_and_hourly_replay(self):
        cfg = replace(Settings(), TRACKER_IDENTITY_MIN_SIMILARITY=.70)
        cache_dir = ROOT / 'data/tracking_improvement_v2'
        expected = {'clip_01': (1, 0), 'clip_02': (1, 5), 'clip_03': (0, 4),
                    'clip_04': (3, 2), 'clip_05': (8, 0), 'clip_06': (7, 0),
                    'boardings': (6, 0), 'exit': (0, 1), 'empty': (0, 0)}
        with tempfile.TemporaryDirectory() as folder:
            for name, counts in expected.items():
                with self.subTest(clip=name):
                    cache_path = cache_dir / (name + '.raw.npz')
                    self.assertTrue(cache_path.is_file(), f'Raw ML cache missing: {cache_path}')
                    output = Path(folder) / (name + '.json')
                    result = replay_raw(cache_path, cfg, output=output)
                    self.assertEqual((result['in'], result['out']), counts)
                    self.assertEqual(result['suppressed'], 0)
                    # These passages contain distinct people, including opposite
                    # directions in 02 and 04. Correct totals must not hide ID theft.
                    self.assertEqual(len({e['identity_id'] for e in result['events']}), sum(counts))
                    with np.load(cache_path, allow_pickle=False) as cache:
                        np.savez_compressed(output.with_suffix('.features.npz'), features=cache['features'])
                    repetitions = replay(output, cfg)
                    self.assertEqual((repetitions[0]['new_in'], repetitions[0]['new_out']), counts)
                    self.assertEqual((repetitions[1]['new_in'], repetitions[1]['new_out']), (0, 0))
