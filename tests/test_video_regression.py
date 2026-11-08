"""Explicit slow regression on the supplied recording: RUN_VIDEO_REGRESSION=1."""
import os
from pathlib import Path
import tempfile
import unittest

from config.settings import ROOT, Settings
from tools.evaluate_clip import evaluate
from tools.replay_trace import replay


@unittest.skipUnless(os.getenv("RUN_VIDEO_REGRESSION") == "1", "Set RUN_VIDEO_REGRESSION=1 for real-video model checks")
class VideoRegressionTests(unittest.TestCase):
    def check_interval(self, first, last):
        with tempfile.TemporaryDirectory(prefix="video-regression-", dir=ROOT / "data") as folder:
            report_path = Path(folder) / "report.json"
            report = evaluate(Settings(), first, last, report_path)
            return report, replay(report_path, Settings())

    def test_six_boardings_and_no_exits(self):
        report, repetitions = self.check_interval(225350, 226350)
        self.assertEqual((report["in"], report["out"]), (6, 0))
        self.assertEqual((repetitions[1]["new_in"], repetitions[1]["new_out"]), (0, 0))
        self.assertGreaterEqual(repetitions[1]["suppressed"], 6)

    def test_exit_and_hourly_repeat(self):
        report, repetitions = self.check_interval(323700, 324299)
        self.assertEqual((report["in"], report["out"]), (0, 1))
        self.assertEqual(repetitions[1]["new_out"], 0)

    def test_empty_door(self):
        report, _ = self.check_interval(35950, 36099)
        self.assertEqual((report["in"], report["out"]), (0, 0))
