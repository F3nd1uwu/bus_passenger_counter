"""Evaluate visible vest decisions through the production grayscale classifier."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from core.vest_classifier import VestClassifier, garment_tensor
from tools.train_garment import validate_dataset


def evaluate(dataset, model, split, output, threshold=.2, threads=1):
    dataset, model, output = Path(dataset).resolve(), Path(model).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Choose a new evaluation output')
    validate_dataset(dataset)
    audit_path = dataset / 'annotations.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    classifier = VestClassifier(model, cpu_threads=threads)
    rows = []
    for row in audit['samples']:
        if row['split'] != split:
            continue
        image = cv2.imread(str(dataset / row['path']))
        height, width = image.shape[:2]
        score = classifier.predict(image, (0, 0, width, height))
        positive = row['garment_xyxy_normalized'] is not None
        found = score is not None and score >= threshold
        location_iou = None
        if positive and found:
            detections = classifier._garment_detections(image)
            predicted = detections[np.argmax(detections[:, 4]), :4]
            _, (x, y, X, Y) = garment_tensor(image, classifier.size)
            gt = np.asarray(row['garment_xyxy_normalized']) * [X-x, Y-y, X-x, Y-y] + [x, y, x, y]
            intersection = np.maximum(0, np.minimum(predicted[2:], gt[2:])
                                      - np.maximum(predicted[:2], gt[:2])).prod()
            union = np.prod(gt[2:]-gt[:2]) + np.prod(predicted[2:]-predicted[:2]) - intersection
            location_iou = float(intersection / union) if union > 0 else 0
        rows.append(dict(path=row['path'], source=row['source'], frame=row['frame'],
                         origin=row.get('origin', 'expanded_review'), expected_vest=positive,
                         score=score, predicted_vest=found, highest_score_box_iou=location_iou))
    if not rows:
        raise ValueError('Empty evaluation split')
    def summarize(selected):
        tp = sum(r['expected_vest'] and r['predicted_vest'] for r in selected)
        fp = sum(not r['expected_vest'] and r['predicted_vest'] for r in selected)
        fn = sum(r['expected_vest'] and not r['predicted_vest'] for r in selected)
        tn = len(selected)-tp-fp-fn
        return dict(images=len(selected), tp=tp, fp=fp, fn=fn, tn=tn,
                    precision=tp/(tp+fp) if tp+fp else 0, recall=tp/(tp+fn) if tp+fn else 0,
                    f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
                    false_positive_rate=fp/(fp+tn) if fp+tn else 0,
                    abstentions=sum(r['score'] is None for r in selected),
                    localized_positive_detections=sum((r['highest_score_box_iou'] or 0) >= .5 for r in selected))
    report = dict(format='production_vest_crop_evaluation_v1', split=split, threshold=threshold,
                  model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                  annotations_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
                  metrics=summarize(rows), by_origin={origin: summarize([r for r in rows if r['origin'] == origin])
                                                     for origin in sorted({r['origin'] for r in rows})},
                  scope='Single-crop visible garment decisions, not staff identity or event-level exclusion accuracy',
                  rows=rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2), flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, type=Path)
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--split', choices=['val', 'test'], required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--threshold', type=float, default=.2)
    args = p.parse_args()
    if not 0 < args.threshold < 1:
        p.error('threshold must be in (0,1)')
    evaluate(args.dataset, args.model, args.split, args.output, args.threshold)
