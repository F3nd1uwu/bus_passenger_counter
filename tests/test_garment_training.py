import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from tools.train_garment import validate_dataset


class GarmentTrainingTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name).resolve()
        self.rows = []
        for i, split in enumerate(["train", "val", "test"]):
            for positive in [False, True]:
                image = self.root / "images" / split / f"crop_{int(positive)}.png"
                image.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(image), np.full((64, 32), 20+i*30+int(positive), np.uint8))
                label = self.root / "labels" / split / (image.stem + ".txt")
                label.parent.mkdir(parents=True, exist_ok=True)
                label.write_text("0 .5 .5 .8 .6\n" if positive else "")
                self.rows.append(dict(path=image.relative_to(self.root).as_posix(), split=split,
                                      sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                                      reviewed=True, episode=split,
                                      garment_xyxy_normalized=[.1, .2, .9, .8] if positive else None))
        self.save()

    def save(self):
        (self.root / "annotations.json").write_text(json.dumps({"samples": self.rows}))

    def test_reviewed_dataset_passes(self):
        self.assertEqual(validate_dataset(self.root)["train"], {"images": 2, "boxes": 1})

    def test_modified_label_is_rejected(self):
        (self.root / "labels/train/crop_1.txt").write_text("0 .4 .5 .8 .6\n")
        with self.assertRaisesRegex(ValueError, "differs"):
            validate_dataset(self.root)

    def test_episode_split_leakage_is_rejected(self):
        self.rows[2]["episode"] = "train"
        self.save()
        with self.assertRaisesRegex(ValueError, "leak"):
            validate_dataset(self.root)

    def test_colour_crop_is_rejected_even_if_reviewed(self):
        path = self.root / self.rows[0]["path"]
        image = np.zeros((64, 32, 3), np.uint8)
        image[:, :, 2] = 255
        cv2.imwrite(str(path), image)
        self.rows[0]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.save()
        with self.assertRaisesRegex(ValueError, "grayscale"):
            validate_dataset(self.root)

    def test_adjacent_video_crops_cannot_leak_under_different_episode_names(self):
        for row, frame in [(self.rows[0], 1000), (self.rows[2], 1001)]:
            row.update(source="data/test2.mp4", frame=frame)
        self.save()
        with self.assertRaisesRegex(ValueError, "temporal leakage"):
            validate_dataset(self.root)

    def test_video_source_without_frame_is_not_silently_ignored(self):
        self.rows[0]["source"] = "data/test2.mp4"
        self.save()
        with self.assertRaisesRegex(ValueError, "source frame"):
            validate_dataset(self.root)
