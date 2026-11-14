from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from config.settings import Settings
from core.preview_player import PreviewPlayer
from core.service import run
from core.types import Detection


class PreviewPlayerTests(unittest.TestCase):
    def setUp(self):
        self.patches = ExitStack()
        for name in ("namedWindow", "createTrackbar", "imshow", "setTrackbarPos", "destroyAllWindows"):
            self.patches.enter_context(patch(f"core.preview_player.cv2.{name}"))
        self.patches.enter_context(patch("core.preview_player.cv2.getWindowProperty", return_value=1))
        self.player = PreviewPlayer(25, 600, seekable=True)

    def tearDown(self):
        self.patches.close()

    def test_programmatic_progress_never_seeks_but_user_slider_does(self):
        with patch("core.preview_player.cv2.setTrackbarPos", side_effect=lambda _, __, value: self.player._timeline_changed(value)):
            self.player.present(np.zeros((64, 64, 3), np.uint8), 100)
        self.assertIsNone(self.player.take_seek())
        self.player._timeline_changed(30)
        self.assertEqual(self.player.take_seek(), 30)

    def test_arrows_seek_five_seconds_and_clamp_to_file(self):
        self.player.frame_index = 200
        self.player.handle_key(2424832)
        self.assertEqual(self.player.take_seek(), 75)
        self.player.handle_key(2555904)
        self.assertEqual(self.player.take_seek(), 325)
        self.player.frame_index = 10
        self.player.handle_key(65361)
        self.assertEqual(self.player.take_seek(), 0)
        self.player.frame_index = 590
        self.player.handle_key(65363)
        self.assertEqual(self.player.take_seek(), 599)

    def test_seek_while_paused_keeps_pause(self):
        self.player.handle_key(ord(" "))
        self.player.frame_index = 200
        self.player.handle_key(2424832)
        self.assertEqual(self.player.take_seek(), 75)
        self.assertTrue(self.player.paused)

    def test_end_of_file_can_restart_or_seek_back(self):
        self.player.frame_index = 599
        self.player.mark_end()
        self.player.handle_key(2424832)
        self.assertEqual(self.player.take_seek(), 474)
        self.assertTrue(self.player.paused)
        self.player.mark_end()
        self.player.handle_key(ord(" "))
        self.assertEqual(self.player.take_seek(), 0)
        self.assertFalse(self.player.paused)

    def test_live_stream_cannot_seek(self):
        player = PreviewPlayer(60, 0, seekable=False)
        player.handle_key(2555904)
        self.assertIsNone(player.take_seek())
        self.assertFalse(player.has_timeline)


class PreviewSeekIntegrationTests(unittest.TestCase):
    def test_delayed_passage_is_finalized_at_eof_and_preview_limit(self):
        class DetectorStub:
            def __init__(self, *args, **kwargs):
                pass

            def reset(self):
                pass

            def detect(self, frame):
                y = round(float(frame.mean()) / 10) * 10
                return [Detection(1, (40, y-10, 60, y+10), 1, np.array([1, 0]), .1)]

        class PlayerStub:
            paused = False
            seek_request = None
            ended = False

            def __init__(self, *args, **kwargs):
                pass

            def take_seek(self):
                return None

            def poll(self):
                return self.ended

            def mark_end(self):
                self.ended = True
                self.paused = True

            def present(self, *args):
                pass

        with tempfile.TemporaryDirectory() as folder, ExitStack() as mocks:
            path = Path(folder)/'pending.avi'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (100, 100))
            self.assertTrue(writer.isOpened())
            for y in [20, 20, 50, 80, 80, 80]:
                writer.write(np.full((100, 100, 3), y, np.uint8))
            writer.release()
            mocks.enter_context(patch('core.service.PersonDetector', DetectorStub))
            mocks.enter_context(patch('core.service.PreviewPlayer', PlayerStub))
            mocks.enter_context(patch('core.service.DatabaseClient', side_effect=AssertionError('Preview opened database')))
            mocks.enter_context(patch('cv2.destroyAllWindows'))
            shown = []

            def draw(frame, detections, cfg, engine, *args, **kwargs):
                shown.append(engine.total_in)
                return frame

            mocks.enter_context(patch('core.service.draw_preview', side_effect=draw))
            cfg = replace(Settings(), VIDEO_SOURCE=str(path), STAFF_MODE='model',
                          VEST_CONFIRM_FRAMES=2, STAFF_DECISION_DELAY_SEC=10.)
            for show, limit in [(False, None), (True, None), (False, 5)]:
                with self.subTest(show=show, max_frames=limit):
                    shown.clear()
                    result = run(cfg, preview=True, show=show, max_frames=limit)
                    self.assertEqual((result['in'], result['out']), (1, 0))
                    if show:
                        self.assertEqual(shown[-1], 1, 'EOF must redraw the final GUI totals')

    def test_backward_seek_resets_preview_counts_and_does_not_reject_time(self):
        class DetectorStub:
            def __init__(self, *args, **kwargs):
                pass

            def reset(self):
                pass

            def detect(self, frame):
                y = round(float(frame.mean()) / 10) * 10
                return [Detection(1, (40, y - 10, 60, y + 10), 1, np.array([1, 0]), None)]

        with tempfile.TemporaryDirectory() as folder, ExitStack() as mocks:
            path = Path(folder) / "crossing.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (100, 100))
            self.assertTrue(writer.isOpened())
            for y in [20, 20, 50, 80, 80, 80, 80, 80, 80, 80]:
                writer.write(np.full((100, 100, 3), y, np.uint8))
            writer.release()
            callbacks, frames = {}, []
            sought = False

            def trackbar(name, window, position, maximum, callback):
                callbacks["seek"] = callback

            def set_position(name, window, position):
                frames.append(position)
                callbacks["seek"](position)

            def key(delay):
                nonlocal sought
                if frames and frames[-1] == 4 and not sought:
                    sought = True
                    callbacks["seek"](0)
                return -1

            mocks.enter_context(patch("core.service.PersonDetector", DetectorStub))
            mocks.enter_context(patch("core.service.DatabaseClient", side_effect=AssertionError("Preview opened database")))
            for name in ("namedWindow", "imshow", "destroyAllWindows"):
                mocks.enter_context(patch(f"cv2.{name}"))
            mocks.enter_context(patch("cv2.createTrackbar", side_effect=trackbar))
            mocks.enter_context(patch("cv2.setTrackbarPos", side_effect=set_position))
            mocks.enter_context(patch("cv2.waitKeyEx", side_effect=key))
            mocks.enter_context(patch("cv2.getWindowProperty", return_value=1))
            cfg = replace(Settings(), VIDEO_SOURCE=str(path), STAFF_MODE="disabled")
            result = run(cfg, preview=True, show=True, max_frames=10)
            self.assertEqual(frames, list(range(5)) * 2)
            self.assertEqual(result["in"], 1)
            self.assertEqual(result["frames"], 10)
