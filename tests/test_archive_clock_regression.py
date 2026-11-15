"""Opt-in checks on reviewed private maps and the existing raw ML captures.

RUN_ARCHIVE_REGRESSION=1 requires data/test*.clock.json and the local raw_v2
captures. No source video, person images, features or map observations are public.
"""
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import unittest
from zoneinfo import ZoneInfo

import numpy as np

from config.settings import ROOT, Settings
from config.profiles import apply_profile
from core.camera_clock import ClockMap
from tools.replay_detections import replay_raw


def retime_known_capture(path, clock, output):
    """These frozen raw_v2 captures used 2026-11-09 UTC as a synthetic origin.

    This helper is deliberately specific to that capture set, not a converter
    for arbitrary caches whose timestamp origin is unknown.
    """
    with np.load(path, allow_pickle=False) as cache:
        data = {key: cache[key] for key in cache.files}
    media = data['timestamps'] - datetime(2026, 11, 9, tzinfo=timezone.utc).timestamp()
    if abs(float(media[0]) - int(data['frames'][0]) / clock.payload['fps']) >= 2:
        raise ValueError('Frozen cache timestamp origin is inconsistent')
    frames, stamps, cuts, rows, offsets = [], [], [], [], [0]
    previous = None
    for j, (index, pts) in enumerate(zip(data['frames'], media)):
        index, pts = int(index), float(pts)
        if clock.is_excluded(index, pts):
            continue
        now, _ = clock.at(index, pts)
        frames.append(index)
        stamps.append(now.timestamp())
        cuts.append(previous is not None and clock.crossed_cut(0, 0, previous_pts=previous, pts=pts))
        previous = pts
        lo, hi = data['offsets'][j:j + 2]
        rows.extend(range(lo, hi))
        offsets.append(len(rows))
    np.savez_compressed(output, boxes=data['boxes'][rows], features=data['features'][rows],
                        valid=data['valid'][rows], offsets=np.asarray(offsets), frames=np.asarray(frames),
                        timestamps=np.asarray(stamps), shape=data['shape'], cuts=np.asarray(cuts, bool))


@unittest.skipUnless(os.getenv('RUN_ARCHIVE_REGRESSION') == '1', 'Opt-in prepared camera maps and private ML captures')
class ArchiveClockRegressionTests(unittest.TestCase):
    def clock(self, name):
        cfg = replace(Settings(), VIDEO_SOURCE=f'data/{name}.mp4', CLOCK_MAP_PATH=f'data/{name}.clock.json')
        return ClockMap.load(cfg.path(cfg.CLOCK_MAP_PATH), cfg)

    def test_camera_dates_and_real_person_frames_are_preserved(self):
        for name, expected in [('test', '2026-11-01 05:39:57'), ('test2', '2026-11-09 06:28:10')]:
            clock = self.clock(name)
            stamp, _ = clock.at(0, 0)
            self.assertEqual(stamp.astimezone(ZoneInfo('Europe/Saratov')).strftime('%Y-%m-%d %H:%M:%S'), expected)
        # Low OCR confidence must not hide actual movement after a scene change.
        self.assertFalse(self.clock('test').is_excluded(222686, 8918.082))
        self.assertFalse(self.clock('test2').is_excluded(381850, 15294.867011111111))

    def test_old_manual_totals_survive_camera_time_and_edit_resets(self):
        cfg = replace(apply_profile(Settings(), 'test2'), STAFF_MODE='disabled')
        clock = self.clock('test')
        plan = json.loads((ROOT / 'reports/test2-2026-11-13/regression_plan.json').read_text('utf-8'))
        with tempfile.TemporaryDirectory() as folder:
            for episode in plan['episodes']:
                with self.subTest(episode=episode['id']):
                    path = ROOT / '.local/test2_review/raw_v2' / (episode['id'] + '.npz')
                    meta = json.loads(path.with_suffix('.json').read_text('utf-8'))
                    self.assertEqual(meta['source'], 'data/test.mp4')
                    cache = Path(folder) / (episode['id'] + '.npz')
                    retime_known_capture(path, clock, cache)
                    result = replay_raw(cache, cfg)
                    self.assertEqual((result['in'], result['out']), (episode['manual_in'], episode['manual_out']))

    def test_driver_checks_with_actual_camera_time(self):
        clock = self.clock('test2')
        cfg = replace(apply_profile(Settings(), 'test2'), CPU_THREADS=2)
        with tempfile.TemporaryDirectory() as folder:
            for name in ['V01', 'V03', 'V05']:
                with self.subTest(episode=name):
                    path = ROOT / '.local/test2_review/raw_v2' / (name + '.npz')
                    cache = Path(folder) / (name + '.npz')
                    retime_known_capture(path, clock, cache)
                    result = replay_raw(cache, cfg, source='data/test2.mp4')
                    self.assertEqual((result['in'], result['out']), (0, 0))
                    self.assertEqual(sum(e['reason'] == 'STAFF' for e in result['events']), 1)
