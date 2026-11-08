from pathlib import Path
import tempfile
import unittest

from tools.train_detector import validate_labels


class TrainingDataTests(unittest.TestCase):
    def test_unlabelled_images_are_not_silently_used_as_negatives(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "images/train").mkdir(parents=True)
            (root / "images/train/person.jpg").write_bytes(b"annotation preflight does not decode images")
            with self.assertRaisesRegex(ValueError, "Missing manual annotation"):
                validate_labels(root)

    def test_invalid_boxes_fail_before_starting_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for split in ("train", "val"):
                (root / f"images/{split}").mkdir(parents=True)
                (root / f"labels/{split}").mkdir(parents=True)
                (root / f"images/{split}/person.jpg").write_bytes(b"image")
                (root / f"labels/{split}/person.txt").write_text("0 0.5 0.5 0.2 0.3\n")
            self.assertEqual(validate_labels(root)["train"]["boxes"], 1)
            (root / "labels/val/person.txt").write_text("0 nan 0.5 0.2 0.3\n")
            with self.assertRaisesRegex(ValueError, "Invalid normalized box"):
                validate_labels(root)
