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
from core.camera_clock import ClockMap, parse_clock, sha256
from core.video_source import VideoSource
from tools.build_clock_map import build


class CameraClockTests(unittest.TestCase):
    def test_camera_date_and_ocr_separator_variants(self):
        for text in ['01-11-2026 Bc 22:32:38', '0111-2026 BcZ2:32.38', '01112026223238']:
            self.assertEqual(parse_clock(text, 'Europe/Saratov'),
                             datetime(2026, 11, 1, 18, 32, 38, tzinfo=timezone.utc))
        for text in ['01-11-2026 35:00:00', '31-02-2026 12:00:00', '01-11-2026 12:00:97', 'unreadable']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_clock(text, 'Europe/Saratov')

    def test_clock_jump_uses_camera_hour_and_preserves_media_seconds(self):
        clock = ClockMap({'format': 'camera_clock_map_v1', 'end_frame': 100,
                          'segments': [{'frame': 0, 'offset': 100000.0}, {'frame': 10, 'offset': 104000.0}]})
        a, _ = clock.at(9, .9)
        b, cut = clock.at(10, 1.0)
        self.assertGreater((b - a).total_seconds(), 3600)
        self.assertTrue(cut)
        with self.assertRaises(ValueError):
            clock.at(100, 10)
        with self.assertRaises(ValueError):
            ClockMap({'format': 'camera_clock_map_v1', 'end_frame': 100,
                      'segments': [{'frame': 0, 'offset': 100}, {'frame': 10, 'offset': 99}]})

    def test_decoded_pts_takes_precedence_over_seek_frame_hint(self):
        clock = ClockMap(dict(format='camera_clock_map_v1', end_frame=100, segments=[
            dict(frame=0, pts=0., offset=1000.),
            dict(frame=10, pts=2., offset=2000., excluded_transition_start=8,
                 excluded_start_pts=1.8, reviewed=True)]))
        self.assertEqual(clock.at(10, 1.)[0].timestamp(), 1001.)
        self.assertEqual(clock.at(15, 2.1)[0].timestamp(), 2002.1)
        self.assertTrue(clock.is_excluded(5, 1.9))
        self.assertFalse(clock.is_excluded(8, 1.))
        self.assertTrue(clock.crossed_cut(10, 15, previous_pts=1., pts=2.1))

    def test_build_refines_cut_and_reader_detects_it_when_stride_skips_exact_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / 'test.avi'
            # Pixel values encode frame IDs for the recognizer stub. Lossy MJPG
            # round-trips differently across FFmpeg builds; use exact pixels.
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'FFV1'), 10, (64, 64))
            self.assertTrue(writer.isOpened())
            # Visible values stand in for real clock text, including a montage at frame 15.
            for i in range(50):
                writer.write(np.full((64, 64, 3), 20 + i, np.uint8))
            writer.release()
            model = root / 'clock.onnx'
            model.write_bytes(b'test recognizer')
            cfg = replace(Settings(), VIDEO_SOURCE=str(path), CLOCK_MODEL_PATH=str(model), FRAME_STRIDE=2)

            class Recognizer:
                def __init__(self, *args):
                    pass

                def read_crop(self, image):
                    i = round(float(image.mean())) - 20
                    hour = '16' if i == 20 else '06' if i >= 15 else '05'
                    return f'01-11-2026 Bc {hour}:00:0{i // 10}', .99

            output = root / 'clock.json'
            output.with_suffix('.observations.json').write_text('{incomplete cache', encoding='utf-8')
            with patch('tools.build_clock_map.ClockRecognizer', Recognizer):
                result = build(cfg, output, interval=1)
            self.assertEqual([r['frame'] for r in result['segments']], [0, 15])
            cfg = replace(cfg, CLOCK_MAP_PATH=str(output))
            clock = ClockMap.load(output, cfg)
            reader = VideoSource(cfg, clock_map=clock)
            packets = []
            try:
                while not reader.eof:
                    packet = reader.read()
                    if packet:
                        packets.append(packet)
            finally:
                reader.close()
            cuts = [p for p in packets if p.time_discontinuity]
            self.assertEqual([p.index for p in cuts], [16])
            self.assertAlmostEqual(cuts[0].media_seconds, 1.6)
            self.assertEqual(cuts[0].timestamp.hour, 2)  # 06:00 Saratov in UTC
            self.assertEqual(len(packets), 25)
            model.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'recognizer'):
                ClockMap.load(output, cfg)

    def test_camera_map_cannot_be_reused_on_another_video(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video, model, output = root / 'video.avi', root / 'model.onnx', root / 'map.json'
            video.write_bytes(b'video A')
            model.write_bytes(b'model')
            cfg = replace(Settings(), VIDEO_SOURCE=str(video), CLOCK_MODEL_PATH=str(model))
            output.write_text(json.dumps(dict(format='camera_clock_map_v1', end_frame=10,
                                              segments=[dict(frame=0, offset=1000)],
                                              source_sha256=sha256(video), model_sha256=sha256(model),
                                              timezone=cfg.TIMEZONE, roi=list(map(float, cfg.CLOCK_ROI.split(','))))))
            ClockMap.load(output, cfg)
            video.write_bytes(b'video B')
            with self.assertRaisesRegex(ValueError, 'different video'):
                ClockMap.load(output, cfg)

    def test_montage_exclusions_require_review(self):
        payload = dict(format='camera_clock_map_v1', end_frame=100, segments=[
            dict(frame=0, offset=1000), dict(frame=40, offset=2000, excluded_transition_start=35, reviewed=False)])
        with self.assertRaisesRegex(ValueError, 'unreviewed'):
            ClockMap(payload)
        ClockMap(payload, allow_unreviewed=True)
        payload['segments'][1]['reviewed'] = True
        self.assertEqual(ClockMap(payload).exclusions, [(35, 40)])

    def test_no_wall_clock_fallback_when_map_is_missing(self):
        from core.service import run
        with tempfile.TemporaryDirectory() as folder, patch('core.service.PersonDetector') as detector, patch('core.service.DatabaseClient') as database:
            path = Path(folder) / 'local.mp4'
            path.write_bytes(b'not opened before validation')
            with self.assertRaisesRegex(ValueError, 'Build camera time first'):
                run(replace(Settings(), VIDEO_SOURCE=str(path)), preview=True)
            detector.assert_not_called()
            database.assert_not_called()

    def test_review_cannot_accept_a_map_changed_after_rendering(self):
        from tools.review_clock_map import review
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'video.avi'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 10, (64, 64))
            self.assertTrue(writer.isOpened())
            for i in range(40):
                writer.write(np.full((64, 64, 3), i, np.uint8))
            writer.release()
            payload = dict(format='camera_clock_map_v1', source_sha256=sha256(source),
                           fps=10, end_frame=40, segments=[dict(frame=0, pts=0., offset=1000),
                           dict(frame=15, pts=1.5, offset=2000, evidence_review_required=True, reviewed=False)])
            path, output = root / 'map.json', root / 'review'
            original = json.dumps(payload)
            path.write_text(original, encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Export contact'):
                review(source, path, output, accept=True, note='test review')
            review(source, path, output)
            path.write_text(original + '\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'changed'):
                review(source, path, output, accept=True, note='test review')
            path.write_text(original, encoding='utf-8')
            review(source, path, output, accept=True, note='test review')
            saved = json.loads(path.read_text(encoding='utf-8'))
            self.assertTrue(saved['segments'][1]['reviewed'])
            self.assertIn('manifest_sha256', saved['review_record'])
