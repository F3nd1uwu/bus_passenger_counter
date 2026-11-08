from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from config.settings import Settings
from core.video_source import VideoSource, parse_start, source_kind


class VideoSourceTests(unittest.TestCase):
    def test_stream_reconnect_marks_discontinuity_without_treating_it_as_eof(self):
        capture = Mock()
        capture.open.return_value = True
        capture.get.return_value = 25
        capture.read.side_effect = [(False, None), (True, np.zeros((64, 64, 3), np.uint8))]
        cfg = replace(Settings(), VIDEO_SOURCE="rtsp://example.invalid/live")
        reader = VideoSource(cfg)
        with patch("core.video_source.cv2.VideoCapture", return_value=capture), patch.object(reader, "_retry_delay"):
            try:
                self.assertIsNone(reader.read())
                self.assertFalse(reader.eof)
                packet = reader.read()
                self.assertTrue(packet.discontinuity)
                self.assertEqual(capture.open.call_count, 2)
                self.assertIsNotNone(packet.timestamp.tzinfo)
            finally:
                reader.close()

    def test_source_types_and_timezone(self):
        self.assertEqual(source_kind("0"), "camera")
        self.assertEqual(source_kind("rtsp://user:secret@example.invalid/stream"), "stream")
        self.assertEqual(source_kind("D:/data/file.mp4"), "file")
        self.assertEqual(source_kind("0", "file"), "file")
        with self.assertRaises(ValueError):
            parse_start("2026-11-01T05:40:00")

    def test_video_time_eof_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 64))
            self.assertTrue(writer.isOpened())
            for _ in range(10):
                writer.write(np.zeros((64, 64, 3), np.uint8))
            writer.release()
            start = datetime(2026, 11, 1, tzinfo=timezone.utc)
            reader = VideoSource(replace(Settings(), VIDEO_SOURCE=str(path)), start, resume_frame=5)
            frames = []
            try:
                while not reader.eof:
                    packet = reader.read()
                    if packet:
                        frames.append(packet)
                self.assertEqual(len(frames), 5)
                self.assertEqual(frames[0].index, 5)
                self.assertAlmostEqual((frames[0].timestamp - start).total_seconds(), 0.5, places=3)
                self.assertTrue(frames[0].discontinuity)
                self.assertFalse(frames[1].discontinuity)
                self.assertIsNone(reader.read())
                reader.seek(2)
                after_seek = reader.read()
                self.assertEqual(after_seek.index, 2)
                self.assertTrue(after_seek.discontinuity)
                self.assertAlmostEqual((after_seek.timestamp - start).total_seconds(), 0.2, places=3)
                reader.seek(100)
                self.assertEqual(reader.read().index, 9)
                reader.seek(-20)
                self.assertEqual(reader.read().index, 0)
            finally:
                reader.close()
