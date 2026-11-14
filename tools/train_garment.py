"""Train the work-garment detector on reviewed grayscale person crops.

Requires annotations.json with samples containing path, split, reviewed,
sha256, episode and garment_xyxy_normalized (null for a passenger negative).
Test images are verified for split leakage but never used to fit/select weights.
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np

from tools.train_detector import validate_labels


def validate_dataset(root):
    root = Path(root).resolve()
    audit = json.loads((root / "annotations.json").read_text(encoding="utf-8"))
    episodes, hashes, paths, positions = {}, {}, set(), {}
    for row in audit["samples"]:
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root) or row.get("reviewed") is not True or row["split"] not in {"train", "val", "test"}:
            raise ValueError("Unreviewed or invalid garment sample")
        if path in paths or row["path"].split("/")[:2] != ["images", row["split"]]:
            raise ValueError("Duplicate image or inconsistent split directory")
        paths.add(path)
        # Earlier exports encode source/frame in the crop name; newer crops
        # can declare them explicitly. Different crop hashes or invented
        # episode labels must not hide adjacent frames in different splits.
        legacy = re.fullmatch(r"(test2|test)_(\d+)_\d+", path.stem)
        if Path(row.get("source", "")).suffix.lower() in {".mp4", ".avi", ".mkv", ".mov"}:
            if type(row.get("frame")) is not int or row["frame"] < 0:
                raise ValueError("Invalid garment source frame")
            position = (Path(row["source"]).stem, row["frame"])
        elif legacy:
            position = (legacy[1], int(legacy[2]))
        else:
            position = None
        if position:
            positions.setdefault(position[0], []).append((position[1], row["split"]))
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Changed reviewed crop: {path.name}")
        for mapping, key in ((episodes, row["episode"]), (hashes, row["sha256"])):
            if mapping.setdefault(key, row["split"]) != row["split"]:
                raise ValueError("Garment episodes/images must not leak across splits")
        image = cv2.imread(str(path))
        if image is None or not np.array_equal(image[:, :, 0], image[:, :, 1]) or not np.array_equal(image[:, :, 1], image[:, :, 2]):
            raise ValueError("Garment detector requires grayscale training crops")
        box = row["garment_xyxy_normalized"]
        expected = []
        if box is not None:
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                raise ValueError("Invalid reviewed garment rectangle")
            expected = [(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1]
        label = root / "labels" / row["split"] / (path.stem + ".txt")
        values = label.read_text(encoding="utf-8").split()
        if expected:
            if len(values) != 5 or values[0] != "0" or not np.allclose(list(map(float, values[1:])), expected, atol=1e-6):
                raise ValueError("YOLO label differs from reviewed garment rectangle")
        elif values:
            raise ValueError("Passenger negative has a garment label")
    actual = {p.resolve() for p in (root / "images").glob("*/*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    if paths != actual:
        raise ValueError("Unreviewed image files are present in the dataset")
    gap = audit.get("minimum_source_gap_frames", 500)
    if type(gap) is not int or gap < 0:
        raise ValueError("Invalid garment source split gap")
    for frames in positions.values():
        ordered = sorted(frames)
        if any(b[0] - a[0] <= gap and a[1] != b[1] for a, b in zip(ordered, ordered[1:])):
            raise ValueError("Garment source frames too close across splits: temporal leakage")
    return validate_labels(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=Path("models/yolo26n.pt"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=.0005)
    parser.add_argument("--freeze", type=int, default=5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--name", default="garment")
    args = parser.parse_args()
    if (not args.weights.is_file() or min(args.epochs, args.threads, args.patience) <= 0
            or args.freeze < 0 or not math.isfinite(args.learning_rate) or not 0 < args.learning_rate < 1):
        parser.error("Existing pretrained weights and positive epochs/threads required")
    root = args.dataset.resolve()
    print("Verified garment labels:", validate_dataset(root), flush=True)
    import torch
    from ultralytics import YOLO
    from ultralytics.utils import YAML
    runtime = root / "data.runtime.yaml"
    YAML.save(runtime, dict(path=str(root), train="images/train", val="images/val", test="images/test", names={0: "staff_vest"}))
    model = YOLO(str(args.weights))

    def started(trainer):
        torch.set_num_threads(args.threads)
        snapshot = {"annotations_sha256": hashlib.sha256((root / "annotations.json").read_bytes()).hexdigest(),
                    "initial_weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
                    "torch_threads": torch.get_num_threads(), "test_used_for_training": False}
        (trainer.save_dir / "dataset_snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    model.add_callback("on_pretrain_routine_start", started)
    model.train(data=str(runtime), imgsz=320, epochs=args.epochs, batch=8, device="cpu", workers=0,
                optimizer="AdamW", lr0=args.learning_rate, lrf=.1, freeze=args.freeze, patience=args.patience, mosaic=.3, close_mosaic=5,
                hsv_h=0, hsv_s=0, hsv_v=.25, degrees=15, translate=.1, scale=.3, fliplr=.5,
                seed=42, project=str(Path("runs/vest_detection").resolve()), name=args.name,
                exist_ok=False, plots=False, amp=False)


if __name__ == "__main__":
    main()
