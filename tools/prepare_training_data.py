"""Export separate recording intervals for manual YOLO person annotation.

No labels are fabricated. In particular, missing labels are NOT negative examples.
Choose whole boarding episodes for each split; do not split neighbouring frames.
"""
import argparse
import json
from pathlib import Path

import cv2


def export(source, output, split, start, end, step):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_file() or start < 0 or end < start or step < 1:
        raise ValueError("Invalid source or frame interval")
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    for item in manifest:
        if item["source"] == str(source) and item["split"] != split and max(start, item["start"]) <= min(end, item["end"]):
            raise ValueError("This interval overlaps another split; neighbouring copies must not leak into validation")
    images = output / "images" / split
    images.mkdir(parents=True, exist_ok=True)
    (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened() or not cap.set(cv2.CAP_PROP_POS_FRAMES, start):
        raise ValueError("Cannot open/seek the source")
    count = 0
    try:
        for frame_index in range(start, end + 1):
            ok, frame = cap.read()
            if not ok:
                break
            if (frame_index - start) % step == 0:
                path = images / f"{source.stem}_{frame_index:08d}.jpg"
                if path.exists():
                    raise ValueError(f"Refusing to overwrite existing annotation image: {path}")
                if not cv2.imwrite(str(path), frame):
                    raise OSError(f"Cannot write {path}")
                count += 1
    finally:
        cap.release()
    manifest.append({"source": str(source), "split": split, "start": start, "end": end,
                     "step": step, "exported": count, "annotation": "required"})
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "data.yaml").write_text(
        f"path: {output.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: person\n",
        encoding="utf-8")
    print(f"Exported {count} unlabelled images to {images}. Add matching YOLO .txt labels before training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/test.mp4"))
    parser.add_argument("--output", type=Path, default=Path("datasets/bus_people"))
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--step", type=int, default=25)
    args = parser.parse_args()
    export(args.source, args.output, args.split, args.start, args.end, args.step)
