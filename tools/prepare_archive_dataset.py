"""Extract an explicit archive annotation plan sequentially; never trust seek indices.

The plan freezes source-frame splits before proposals/review. Detector proposals
are deliberately unreviewed and cannot be compiled by annotation_review.py.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_plan(plan):
    samples = plan['samples']
    gap = plan['minimum_source_gap_frames']
    if type(gap) is not int or gap < 0 or not samples:
        raise ValueError('A nonempty plan and nonnegative temporal gap are required')
    positions = {}
    for row in samples:
        frame, split = row['frame'], row['split']
        if type(frame) is not int or frame < 0 or split not in {'train', 'val', 'test'}:
            raise ValueError('Invalid source frame or split')
        if frame in positions:
            raise ValueError('Duplicate source frame')
        positions[frame] = split
    ordered = sorted(positions)
    if any(b-a <= gap and positions[a] != positions[b] for a, b in zip(ordered, ordered[1:])):
        raise ValueError('Neighbouring frames leak across splits')
    return positions


def verify_reviewed_dataset(root):
    """Verify review, image integrity, source split separation and compiled labels."""
    import numpy as np
    root = Path(root).resolve()
    audit = json.loads((root / 'annotations.json').read_text(encoding='utf-8'))
    records, positions, hashes, counts = audit['images'], {}, {}, {}
    actual = {p.relative_to(root).as_posix() for split in ('train', 'val', 'test')
              for p in (root / 'images' / split).glob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}}
    if actual != set(records):
        raise ValueError('Every dataset image must have exactly one reviewed record')
    for name, row in records.items():
        path = (root / name).resolve()
        if (not path.is_relative_to(root) or row.get('reviewed') is not True
                or row.get('excluded') is True):
            raise ValueError('Unreviewed, excluded or unsafe image')
        split = Path(name).parent.name
        if split not in {'train', 'val', 'test'} or row['split'] != split:
            raise ValueError('Inconsistent split')
        if digest(path) != row['sha256']:
            raise ValueError('Image changed after review')
        if hashes.setdefault(row['sha256'], split) != split:
            raise ValueError('Identical image leaked across splits')
        positions.setdefault(row['source'], []).append(dict(frame=row['frame'], split=split))
        if 'review_sheet' in row:
            sheet = (root / row['review_sheet']).resolve()
            if not sheet.is_relative_to(root) or digest(sheet) != row['review_sheet_sha256']:
                raise ValueError('Changed or missing review sheet')
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError('Unreadable reviewed image')
        height, width = image.shape[:2]
        expected = []
        for box in row['boxes_xyxy']:
            if len(box) != 4 or not np.isfinite(box).all():
                raise ValueError('Invalid reviewed rectangle')
            x, y, X, Y = box
            if not (0 <= x < X <= width and 0 <= y < Y <= height):
                raise ValueError('Reviewed rectangle outside image')
            expected.append([0, (x+X)/(2*width), (y+Y)/(2*height), (X-x)/width, (Y-y)/height])
        label = root / 'labels' / split / (path.stem + '.txt')
        values = [list(map(float, line.split())) for line in label.read_text(encoding='utf-8').splitlines()
                  if line.strip()]
        if (len(values) != len(expected) or any(len(v) != 5 for v in values)
                or (expected and not np.allclose(values, expected, atol=1e-7, rtol=0))):
            raise ValueError('Compiled labels differ from reviewed rectangles')
        count = counts.setdefault(split, dict(images=0, boxes=0))
        count['images'] += 1
        count['boxes'] += len(expected)
    for source_rows in positions.values():
        validate_plan(dict(samples=source_rows, minimum_source_gap_frames=audit['minimum_source_gap_frames']))
    if set(counts) != {'train', 'val', 'test'} or any(not counts[s]['boxes'] for s in ('train', 'val')):
        raise ValueError('Require three nonempty splits and annotated train/val people')
    expected_labels = {f"labels/{Path(name).parent.name}/{Path(name).stem}.txt" for name in records}
    actual_labels = {p.relative_to(root).as_posix() for p in (root / 'labels').glob('*/*.txt')}
    if expected_labels != actual_labels:
        raise ValueError('Orphaned or missing compiled labels')
    return counts


def extract(plan_path, output):
    plan = json.loads(Path(plan_path).read_text(encoding='utf-8'))
    positions = validate_plan(plan)
    source, output = Path(plan['source']).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a new dataset directory to preserve previous annotations')
    if digest(source) != plan['source_sha256']:
        raise ValueError('Source differs from the frozen plan')
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError('Cannot open video')
    output.mkdir(parents=True)
    records = {}
    try:
        for index in range(max(positions)+1):
            if not cap.grab():
                raise ValueError(f'Video ended before planned frame {index}')
            if index not in positions:
                continue
            ok, image = cap.retrieve()
            if not ok:
                raise ValueError(f'Cannot decode planned frame {index}')
            name = f'images/{positions[index]}/{source.stem}_{index:08d}.jpg'
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 97]):
                raise OSError(path)
            records[name] = dict(frame=index, split=positions[index], source=plan['source'],
                                 sha256=digest(path), reviewed=False, boxes_xyxy=[])
            if len(records) % 100 == 0:
                print(f'Extracted {len(records)}/{len(positions)}', flush=True)
    finally:
        cap.release()
    (output / 'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    audit = dict(version=2, policy=plan['policy'], minimum_source_gap_frames=plan['minimum_source_gap_frames'],
                 source_sha256=plan['source_sha256'], extraction='sequential grab/retrieve, zero-based frame index',
                 images=records)
    (output / 'annotations.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    (output / 'data.yaml').write_text(f'path: {output.as_posix()}\ntrain: images/train\nval: images/val\n'
                                     'test: images/test\nnames:\n  0: person\n', encoding='utf-8')
    return audit


def propose(root, weights):
    import torch
    from ultralytics import YOLO
    root = Path(root)
    path = root / 'annotations.json'
    audit = json.loads(path.read_text(encoding='utf-8'))
    if any(row['reviewed'] for row in audit['images'].values()) or 'proposal_weights_sha256' in audit:
        raise ValueError('Refusing to overwrite existing proposals/review')
    torch.set_num_threads(3)
    model = YOLO(str(weights))
    if model.task != 'detect' or model.names.get(0) != 'person':
        raise ValueError('Proposals require a person detector with class 0')
    for index, (name, row) in enumerate(audit['images'].items()):
        image = cv2.imread(str(root / name))
        result = model.predict(image, classes=[0], conf=.12, imgsz=640, device='cpu', verbose=False)[0]
        height, width = image.shape[:2]
        boxes = []
        for box in result.boxes.xyxy.tolist():
            x1, y1, x2, y2 = [round(x) for x in box]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
        row['boxes_xyxy'] = boxes
        row['proposal_confidences'] = result.boxes.conf.tolist()
        if index % 100 == 99:
            print(f'Proposed {index+1}', flush=True)
    audit['proposal_weights_sha256'] = digest(weights)
    path.write_text(json.dumps(audit, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('extract')
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('propose')
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p = sub.add_parser('verify')
    p.add_argument('--dataset', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'extract':
        extract(args.plan, args.output)
    elif args.command == 'propose':
        propose(args.dataset, args.weights)
    else:
        print(json.dumps(verify_reviewed_dataset(args.dataset), indent=2))
