"""Compare manually reviewed same/different people against a saved video trace."""
import argparse
import json
from pathlib import Path

import numpy as np


def evaluate_pairs(trace, annotations, minimum_iou=.5, margin=.1):
    frames = {row['frame']: row['detections'] for row in trace}
    samples = {}
    for sample in annotations['samples']:
        key = sample['id']
        if key in samples:
            raise ValueError('Duplicate identity sample ID')
        box = np.asarray(sample['bbox'], dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
            raise ValueError('Invalid manual person box')
        candidates = []
        for detection in frames.get(sample['frame'], []):
            other = np.asarray(detection['bbox'], dtype=float)
            if (detection['conf'] < annotations.get('minimum_confidence', .3)
                    or other.shape != (4,) or not np.isfinite(other).all()
                    or np.any(other[2:] <= other[:2])):
                continue
            intersection = np.maximum(0, np.minimum(box[2:], other[2:]) - np.maximum(box[:2], other[:2]))
            area = float(np.prod(intersection))
            union = float(np.prod(box[2:] - box[:2]) + np.prod(other[2:] - other[:2]) - area)
            candidates.append((area / union, detection))
        candidates.sort(key=lambda item: -item[0])
        if (not candidates or candidates[0][0] < minimum_iou
                or (len(candidates) > 1 and candidates[0][0] - candidates[1][0] < margin)):
            samples[key] = None
        else:
            samples[key] = candidates[0][1]
    rows = []
    for pair in annotations['pairs']:
        left, right = samples[pair['left']], samples[pair['right']]
        if type(pair['same_person']) is not bool:
            raise ValueError('same_person must be boolean')
        row = dict(pair)
        for field, label in [('id', 'tracker'), ('identity', 'identity')]:
            if left is None or right is None or left.get(field) is None or right.get(field) is None:
                row[label] = 'unresolved'
            else:
                row[label] = 'correct' if (left[field] == right[field]) == pair['same_person'] else 'incorrect'
        rows.append(row)
    return {'pairs': rows, 'summary': {
        label: {status: sum(row[label] == status for row in rows)
                for status in ['correct', 'incorrect', 'unresolved']}
        for label in ['tracker', 'identity']}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('annotations', type=Path)
    args = parser.parse_args()
    report = json.loads(args.trace.read_text('utf-8'))
    labels = json.loads(args.annotations.read_text('utf-8'))
    print(json.dumps(evaluate_pairs(report['trace'], labels), indent=2))
