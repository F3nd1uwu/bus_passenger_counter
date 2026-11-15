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
    def test_network_sources_are_rejected_before_opening_capture(self):
        for source in ("rtsp://user:secret@example.invalid/live", "https://example.invalid/a.mp4", "//server/share/a.mp4"):
            with self.subTest(source=source), patch("core.video_source.cv2.VideoCapture") as capture:
                with self.assertRaisesRegex(ValueError, "Only local") as error:
                    VideoSource(replace(Settings(), VIDEO_SOURCE=source))
                self.assertNotIn("secret", str(error.exception))
                capture.assert_not_called()

    def test_source_types_and_timezone(self):
        self.assertEqual(source_kind("0"), "file")
        with self.assertRaises(ValueError):
            source_kind("rtsp://user:secret@example.invalid/stream")
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
            reader = VideoSource(replace(Settings(TIME_SOURCE="pts"), VIDEO_SOURCE=str(path)), start, resume_frame=5)
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
