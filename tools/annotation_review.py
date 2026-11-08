"""Compile visually reviewed pixel boxes into YOLO labels and review sheets.

This tool does not generate labels with a detector. Each image must have an
explicit reviewed record, including empty records for inspected negatives.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def compile_annotations(dataset):
    dataset = Path(dataset).resolve()
    source = dataset / "annotations.json"
    audit = json.loads(source.read_text(encoding="utf-8"))
    records = audit["images"]
    expected = {p.relative_to(dataset).as_posix() for split in ("train", "val", "test")
                for p in (dataset / "images" / split).glob("*.jpg")}
    if set(records) != expected:
        raise ValueError(f"Every image needs an explicit record; missing={expected-set(records)}, extra={set(records)-expected}")
    for split in ("train", "val", "test"):
        stems = {Path(name).stem for name in expected if name.startswith(f"images/{split}/")}
        orphaned = [p.name for p in (dataset / "labels" / split).glob("*.txt") if p.stem not in stems]
        if orphaned:
            raise ValueError(f"Archive labels for excluded images before compiling {split}: {orphaned}")
    counts = {}
    for name in sorted(expected):
        record = records[name]
        image_path = dataset / name
        image = cv2.imread(str(image_path))
        if image is None or record.get("reviewed") is not True:
            raise ValueError(f"Unreadable or unreviewed image: {name}")
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Image changed since visual annotation: {name}")
        height, width = image.shape[:2]
        lines = []
        for box in record["boxes_xyxy"]:
            x1, y1, x2, y2 = box
            if not np.isfinite(box).all() or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError(f"Invalid pixel box in {name}: {box}")
            lines.append(f"0 {(x1+x2)/(2*width):.8f} {(y1+y2)/(2*height):.8f} {(x2-x1)/width:.8f} {(y2-y1)/height:.8f}")
        split = image_path.parent.name
        label = dataset / "labels" / split / (image_path.stem + ".txt")
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        value = counts.setdefault(split, {"images": 0, "boxes": 0, "negative_images": 0})
        value["images"] += 1
        value["boxes"] += len(lines)
        value["negative_images"] += not lines
    out = dataset / "review" / "labelled"
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        names = sorted(name for name in expected if name.startswith(f"images/{split}/"))
        for start in range(0, len(names), 16):
            canvas = np.full((1560, 1920, 3), 240, np.uint8)
            for i, name in enumerate(names[start:start+16]):
                im = cv2.imread(str(dataset / name))
                for j, box in enumerate(records[name]["boxes_xyxy"]):
                    x1, y1, x2, y2 = box
                    cv2.rectangle(im, (x1, y1), (x2-1, y2-1), (0, 255, 0), 2)
                    cv2.putText(im, str(j+1), (x1+3, min(470, y1+20)), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 0), 2)
                x, y = (i % 4) * 480, (i // 4) * 390
                canvas[y+30:y+390, x:x+480] = cv2.resize(im, (480, 360))
                text = f"{Path(name).stem}: {len(records[name]['boxes_xyxy'])}"
                cv2.putText(canvas, text, (x+5, y+22), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)
            cv2.imwrite(str(out / f"{split}_{start:03d}.jpg"), canvas)
    print(json.dumps(counts, indent=2))
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/bus_people"))
    compile_annotations(parser.parse_args().dataset)
