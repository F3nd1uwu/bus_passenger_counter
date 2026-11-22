"""Fixed-threshold raw detection evaluation on reviewed frames, without tracking.

IoU matching is one-to-one and maximizes match count before localization quality.
Frame precision/recall are not passenger counting or identity accuracy.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import lap


def match_boxes(expected, predicted, threshold=.5):
    if not 0 < threshold <= 1:
        raise ValueError('IoU threshold must be in (0, 1]')
    expected = np.asarray(expected, dtype=float).reshape(-1, 4)
    predicted = np.asarray(predicted, dtype=float).reshape(-1, 4)
    for boxes in (expected, predicted):
        if not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= boxes[:, :2]):
            raise ValueError('Expected finite, positive-area xyxy boxes')
    if not len(expected) or not len(predicted):
        return dict(tp=0, fp=len(predicted), fn=len(expected), matched_iou=[], duplicate_fp=0)
    intersection = np.maximum(0, np.minimum(expected[:, None, 2:], predicted[None, :, 2:])
                              - np.maximum(expected[:, None, :2], predicted[None, :, :2])).prod(axis=2)
    area_a = np.prod(expected[:, 2:] - expected[:, :2], axis=1)
    area_b = np.prod(predicted[:, 2:] - predicted[:, :2], axis=1)
    iou = intersection / (area_a[:, None] + area_b[None, :] - intersection)
    eligible = iou >= threshold
    weights = eligible * (min(iou.shape) + 1 + iou)
    _, assignment, _ = lap.lapjv(-weights, extend_cost=True)
    pairs = [(g, int(p)) for g, p in enumerate(assignment) if p >= 0 and eligible[g, p]]
    matched_pred = {p for _, p in pairs}
    return dict(tp=len(pairs), fp=len(predicted)-len(pairs), fn=len(expected)-len(pairs),
                matched_iou=[float(iou[g, p]) for g, p in pairs],
                duplicate_fp=sum(p not in matched_pred and bool(eligible[:, p].any())
                                 for p in range(len(predicted))))


def evaluate(dataset, model_path, split, output, confidence=.3, threads=1):
    from config.settings import Settings
    from core.detector import PersonDetector
    import torch
    torch.set_num_threads(threads)
    dataset, model_path, output = Path(dataset).resolve(), Path(model_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve earlier evaluations; choose a new output')
    audit_path = dataset / 'annotations.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    cfg = replace(Settings(), MODEL_PATH=str(model_path), CPU_THREADS=threads,
                  CONFIDENCE=confidence, TRACKER_PATH='botsort_reid.yaml', TRACKING_MODE='appearance')
    detector = PersonDetector(cfg, detection_only=True)
    rows = []
    started = time.perf_counter()
    try:
        for name, row in sorted(audit['images'].items()):
            if Path(name).parent.name != split:
                continue
            path = (dataset / name).resolve()
            if not path.is_relative_to(dataset) or row.get('reviewed') is not True:
                raise ValueError('Only reviewed local images may be evaluated')
            if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('Reviewed image changed')
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError('Unreadable image')
            result = detector.model.predict(image, classes=[0], conf=confidence, imgsz=640,
                                            device='cpu', verbose=False)[0]
            boxes, scores = result.boxes.xyxy.tolist(), result.boxes.conf.tolist()
            rows.append(dict(image=name, frame=row.get('frame'), source=row.get('source'),
                             expected=row['boxes_xyxy'], predicted=boxes, confidence=scores,
                             iou50=match_boxes(row['boxes_xyxy'], boxes),
                             iou75=match_boxes(row['boxes_xyxy'], boxes, .75)))
    finally:
        detector.close()
    if not rows:
        raise ValueError('Requested split is empty')
    totals = {}
    for key in ('iou50', 'iou75'):
        values = {kind: sum(row[key][kind] for row in rows) for kind in ('tp', 'fp', 'fn', 'duplicate_fp')}
        tp, fp, fn = values['tp'], values['fp'], values['fn']
        values.update(precision=tp/(tp+fp) if tp+fp else 0, recall=tp/(tp+fn) if tp+fn else 0,
                      f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
        totals[key] = values
    report = dict(format='reviewed_raw_detections_v1', split=split, frames=len(rows), confidence=confidence,
                  annotations_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
                  model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                  wall_seconds=time.perf_counter()-started, metrics=totals,
                  scope='Raw person boxes, no ReID deduplication or tracking; not counting accuracy', rows=rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--split', choices=['val', 'test'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--confidence', type=float, default=.3)
    parser.add_argument('--threads', type=int, choices=[1, 2, 3, 4], default=1)
    args = parser.parse_args()
    if not 0 < args.confidence < 1:
        parser.error('confidence must be in (0,1)')
    evaluate(args.dataset, args.model, args.split, args.output, args.confidence, args.threads)
