from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from config.settings import Settings
from core.camera_clock import ClockMap
from core.service import configuration_key
from core.video_source import VideoSource
from tools.build_clock_map import MEDIA_CLOCK_POLICY, build, media_clock_segments


class MediaClockFallbackTests(unittest.TestCase):
    def rows(self, offsets):
        origin = datetime(2026, 11, 15, 5, tzinfo=timezone.utc).timestamp()
        return [dict(frame=i * 125, pts=i * 5., timestamp=None if offset is None else origin + i * 5 + offset,
                     offset=None if offset is None else origin + offset, text='test', confidence=.99)
                for i, offset in enumerate(offsets)]

    def clock(self, rows):
        segments, estimated, count = media_clock_segments(rows)
        return ClockMap(dict(format='camera_clock_map_v1', clock_policy=MEDIA_CLOCK_POLICY,
                             end_frame=rows[-1]['frame'] + 1, segments=segments)), estimated, count

    def test_unreadable_clock_advances_by_video_pts_not_one_second_per_frame(self):
        rows = self.rows([0, None, None, 0])
        clock, estimated, count = self.clock(rows)
        first, _ = clock.at(0, 0.)
        next_frame, _ = clock.at(1, .04)
        last, _ = clock.at(375, 15.)
        self.assertAlmostEqual((next_frame - first).total_seconds(), .04)
        self.assertAlmostEqual((last - first).total_seconds(), 15.)
        self.assertEqual(count, 2)
        self.assertEqual(len(estimated), 1)
        self.assertFalse(clock.exclusions)

    def test_readable_clock_after_motion_gap_recalibrates_to_camera_time(self):
        rows = self.rows([0, 0, 0, None, None, 120, 120, 120])
        clock, _, _ = self.clock(rows)
        before, _ = clock.at(0, 0.)
        after, cut = clock.at(625, 25.)
        self.assertEqual((after - before).total_seconds(), 145.)
        self.assertTrue(cut)

    def test_short_overlap_holds_clock_then_catches_up_without_discarding_frames(self):
        rows = self.rows([0] * 20 + [-14] * 20)
        clock, _, _ = self.clock(rows)
        floor = clock.at(2500, 100.)[0]
        self.assertEqual(clock.at(2625, 105.)[0], floor)
        self.assertGreater(clock.at(2900, 116.)[0], floor)
        stamps = [clock.at(i, i / 25)[0] for i in range(1, 4500)]
        self.assertTrue(all(b >= a for a, b in zip(stamps, stamps[1:])))
        self.assertFalse(any(clock.is_excluded(i, i / 25) for i in range(4500)))
        self.assertAlmostEqual(clock.at(4375, 175.)[0].timestamp(), rows[35]['timestamp'])

    def test_recurring_seconds_digit_errors_do_not_block_workday(self):
        rows = self.rows([0 if i % 2 == 0 else -5 for i in range(85)])
        clock, _, count = self.clock(rows)
        self.assertGreaterEqual(count, 42)
        self.assertEqual(len(clock.segments), 1)
        self.assertFalse(clock.exclusions)

    def test_isolated_wrong_hour_does_not_recalibrate_clock(self):
        offsets = [0] * 25
        offsets[10] = 3600
        clock, _, _ = self.clock(self.rows(offsets))
        self.assertEqual(len(clock.segments), 1)

    def test_missing_initial_frame_and_unreadable_tail_use_recording_calibration(self):
        rows = self.rows([None, 0, 0, None])
        clock, estimated, _ = self.clock(rows)
        self.assertEqual((clock.at(375, 15)[0] - clock.at(0, 0)[0]).total_seconds(), 15.)
        self.assertEqual(len(estimated), 2)

    def test_missing_all_camera_dates_cannot_use_server_time(self):
        with self.assertRaisesRegex(ValueError, 'No readable camera'):
            self.clock(self.rows([None] * 10))

    def test_invalid_floor_cannot_hide_backwards_timestamp(self):
        rows = self.rows([0] * 20 + [-14] * 20)
        clock, _, _ = self.clock(rows)
        payload = deepcopy(clock.payload)
        payload['segments'][1]['clock_floor'] -= 1
        with self.assertRaisesRegex(ValueError, 'backwards'):
            ClockMap(payload)
        payload = deepcopy(clock.payload)
        payload['segments'][1]['clock_floor'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'floor'):
            ClockMap(payload)

    def test_floor_is_part_of_resume_configuration_fingerprint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            asset, output = root / 'asset.bin', root / 'clock.json'
            asset.write_bytes(b'isolated asset')
            clock, _, _ = self.clock(self.rows([0] * 20 + [-14] * 20))
            payload = clock.payload
            payload['source_sha256'] = 'isolated-source'
            cfg = replace(Settings(), CLOCK_MAP_PATH=str(output), MODEL_PATH=str(asset),
                          REID_MODEL_PATH=str(asset), TRACKER_PATH=str(asset), CLOCK_MODEL_PATH=str(asset))
            output.write_text(json.dumps(payload), encoding='utf-8')
            first = configuration_key(cfg)
            payload['segments'][1]['clock_floor'] += 1
            output.write_text(json.dumps(payload), encoding='utf-8')
            self.assertNotEqual(first, configuration_key(cfg))

    def test_tolerant_builder_and_reader_keep_every_frame_through_unreadable_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, model, output = root / 'video.avi', root / 'clock.onnx', root / 'map.json'
            model.write_bytes(b'synthetic clock model')
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'FFV1'), 25, (64, 64))
            self.assertTrue(writer.isOpened())
            for i in range(250):
                writer.write(np.full((64, 64, 3), i, np.uint8))
            writer.release()

            class Recognizer:
                def __init__(self, *args):
                    pass

                def read_crop(self, image):
                    index = round(float(image.mean()))
                    if 25 <= index < 200:
                        return 'unreadable', .1
                    return f'15-11-2026 05:00:{index // 25:02d}', .99

            cfg = replace(Settings(), VIDEO_SOURCE=str(source), CLOCK_MODEL_PATH=str(model))
            with patch('tools.build_clock_map.ClockRecognizer', Recognizer):
                payload = build(cfg, output, interval=1, allow_media_fallback=True)
            reader = VideoSource(cfg, clock_map=ClockMap.load(output, cfg))
            frames = []
            try:
                while not reader.eof:
                    packet = reader.read()
                    if packet:
                        frames.append(packet)
            finally:
                reader.close()
            self.assertEqual(len(frames), 250)
            self.assertEqual(payload['clock_policy'], MEDIA_CLOCK_POLICY)
            self.assertAlmostEqual((frames[1].timestamp - frames[0].timestamp).total_seconds(), .04)


if __name__ == '__main__':
    unittest.main()
