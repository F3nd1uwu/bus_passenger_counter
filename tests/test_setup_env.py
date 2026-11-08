from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from config.settings import ROOT
from tools.setup_env import setup


class SetupEnvironmentTests(unittest.TestCase):
    def test_new_configuration_has_generated_password_and_no_assumed_recording_date(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / ".env.example").write_text((ROOT / ".env.example").read_text(encoding="utf-8"), encoding="utf-8")
            with patch("tools.setup_env.ROOT", root):
                setup()
            values = dict(line.split("=", 1) for line in (root / ".env").read_text(encoding="utf-8").splitlines()
                          if line and not line.startswith("#"))
            self.assertGreaterEqual(len(values["DB_PASS"]), 32)
            self.assertNotIn("replace-with", values["DB_PASS"])
            self.assertEqual(values["VIDEO_START"], "")
            self.assertEqual(values["CONFIDENCE"], "0.35")
            self.assertEqual(values["TRACKER_IDENTITY_MIN_SIMILARITY"], "0.70")

    def test_existing_configuration_is_preserved_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            expected = b"DB_PASS=existing-private-setting\r\nVIDEO_START=custom\r\n"
            (root / ".env").write_bytes(expected)
            with patch("tools.setup_env.ROOT", root):
                setup()
            self.assertEqual((root / ".env").read_bytes(), expected)
