"""Evaluate detector boxes against manual labels at a fixed operating point.

This measures detection, not passenger-count accuracy. It never trains or changes
working models. Frame-level train/val/test episodes remain separate.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from config.settings import Settings
from core.detector import PersonDetector


def iou(box, targets):
    if not len(targets):
        return np.empty(0)
    first = np.maximum(box[:2], targets[:, :2])
    last = np.minimum(box[2:], targets[:, 2:])
    intersection = np.maximum(last-first, 0).prod(axis=1)
    area = (box[2]-box[0])*(box[3]-box[1])
    areas = (targets[:, 2]-targets[:, 0])*(targets[:, 3]-targets[:, 1])
    return intersection / np.maximum(area+areas-intersection, 1e-9)


def evaluate(model, dataset, splits, confidence=.10, overlap=.50):
    dataset = Path(dataset).resolve()
    audit_path = dataset / "annotations.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    cfg = replace(Settings(), MODEL_PATH=str(Path(model).resolve()), CONFIDENCE=confidence)
    detector = PersonDetector(cfg, detection_only=True)
    summaries = {}
    for split in splits:
        rows = []
        tp = fp = fn = empty = false_alarms = 0
        for name, record in sorted(audit["images"].items()):
            if not name.startswith(f"images/{split}/"):
                continue
            path = dataset / name
            if not record["reviewed"] or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"Unreviewed/changed source: {name}")
            frame = cv2.imread(str(path))
            result = detector.model.predict(frame, classes=[0], conf=confidence, imgsz=cfg.IMAGE_SIZE,
                                            device="cpu", verbose=False)[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            truth = np.asarray(record["boxes_xyxy"], dtype=np.float32).reshape(-1, 4)
            used, matched, extra = set(), [], []
            for index in np.argsort(-scores):
                overlaps = iou(boxes[index], truth)
                for old in used:
                    overlaps[old] = -1
                target = int(np.argmax(overlaps)) if len(overlaps) else -1
                if target >= 0 and overlaps[target] >= overlap:
                    used.add(target)
                    matched.append({"prediction": int(index), "target": target})
                else:
                    extra.append(int(index))
            missing = sorted(set(range(len(truth))) - used)
            tp += len(matched)
            fp += len(extra)
            fn += len(missing)
            if not len(truth):
                empty += 1
                false_alarms += bool(len(boxes))
            rows.append({"image": name, "tp": len(matched), "fp": len(extra), "fn": len(missing),
                         "missing": missing, "extra": extra, "boxes": boxes.tolist(), "scores": scores.tolist()})
        precision = tp / max(tp+fp, 1)
        recall = tp / max(tp+fn, 1)
        summaries[split] = {"images": len(rows), "tp": tp, "fp": fp, "fn": fn,
                            "precision": precision, "recall": recall,
                            "f1": 2*precision*recall/max(precision+recall, 1e-9),
                            "empty_images": empty, "false_alarm_empty_images": false_alarms,
                            "details": rows}
        print(json.dumps({"model": str(model), "split": split,
                          **{k: v for k, v in summaries[split].items() if k != "details"}}), flush=True)
    return {"model": str(model), "confidence": confidence, "iou": overlap,
            "annotation_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(), "splits": summaries}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/bus_people"))
    parser.add_argument("--splits", nargs="+", choices=["val", "test"], default=["val", "test"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=.10)
    parser.add_argument("--iou", type=float, default=.50)
    args = parser.parse_args()
    if not 0 < args.confidence <= 1 or not 0 < args.iou <= 1:
        parser.error("Confidence and matching IoU must be in (0, 1]")
    protected = {args.dataset.resolve() / name for name in
                 ("annotations.json", "manifest.json", "training_snapshot.json")}
    if args.output.suffix.lower() != ".json" or args.output.resolve() in protected:
        parser.error("Choose a separate .json report; dataset annotation files cannot be overwritten")
    results = [evaluate(model, args.dataset, args.splits, args.confidence, args.iou) for model in args.models]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
