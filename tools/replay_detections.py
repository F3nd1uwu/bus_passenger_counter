"""Replay cached raw person detections through the actual tracker and counter.

Unlike replay_trace, this reruns association, including unconfirmed detections.
No neural-network inference or database IO is performed.
"""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.utils import YAML

from config.settings import Settings
from core.appearance_tracker import AppearanceTracker
from core.counter_engine import PassengerCounterEngine
from core.types import Detection
from core.person_detections import distinct_person_indices


def replay_raw(path, cfg, *, output=None, tracker_class=AppearanceTracker, suppress_duplicates=True):
    if cfg.STAFF_MODE != 'disabled' or cfg.TRACKING_MODE != 'appearance':
        raise ValueError('Raw replay requires appearance tracking and disabled staff filter')
    cfg.validate()
    with np.load(path, allow_pickle=False) as cache:
        boxes, features, valid = cache['boxes'], cache['features'], cache['valid']
        offsets, indices, timestamps = cache['offsets'], cache['frames'], cache['timestamps']
        shape = tuple(int(x) for x in cache['shape'])
    if (len(offsets) != len(indices) + 1 or len(indices) != len(timestamps)
            or offsets[0] != 0 or offsets[-1] != len(boxes)
            or np.any(np.diff(offsets) < 0) or len(features) != len(boxes) or len(valid) != len(boxes)
            or len(shape) != 2 or min(shape) <= 0):
        raise ValueError('Invalid raw detection cache dimensions')
    args = YAML.load(str(cfg.path(cfg.TRACKER_PATH)))
    args['with_reid'] = True
    tracker = tracker_class(SimpleNamespace(**args), cfg.TRACKER_IDENTITY_MIN_SIMILARITY)
    engine = PassengerCounterEngine(cfg)
    frame = np.zeros((*shape, 3), np.uint8)
    events, trace = [], []
    for j, index in enumerate(indices):
        lo, hi = offsets[j:j + 2]
        mask = valid[lo:hi]
        b, f = boxes[lo:hi][mask], features[lo:hi][mask]
        source_indices = lo + np.flatnonzero(mask)
        if suppress_duplicates:
            keep = distinct_person_indices(b, f)
            b, f, source_indices = b[keep], f[keep], source_indices[keep]
        tracked = tracker.update(Boxes(b, shape), feats=f if len(f) else None)
        detections = [Detection(int(t[4]), tuple(b[int(t[-1]), :4].tolist()),
                                float(b[int(t[-1]), 4]), f[int(t[-1])]) for t in tracked]
        current = engine.update(frame, detections, datetime.fromtimestamp(float(timestamps[j]), timezone.utc))
        events.extend(dict(e.to_dict(), frame=int(index)) for e in current)
        if output:
            trace.append({'frame': int(index), 'detections': [
                {'id': d.track_id, 'bbox': d.bbox, 'conf': d.confidence,
                 'feature': int(source_indices[int(t[-1])]),
                 'zone': engine.tracks[d.track_id].zone if d.track_id in engine.tracks else None,
                 'identity': engine.tracks[d.track_id].identity_id if d.track_id in engine.tracks else None}
                for d, t in zip(detections, tracked)]})
    report = {'cache': str(path), 'frames': len(indices), 'in': engine.total_in, 'out': engine.total_out,
              'suppressed': engine.total_suppressed, 'events': events}
    if output:
        report['trace'] = trace
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--association-similarity', type=float, default=.70)
    args = parser.parse_args()
    result = replay_raw(args.cache, replace(Settings(), TRACKER_IDENTITY_MIN_SIMILARITY=args.association_similarity), output=args.output)
    print(json.dumps({k: v for k, v in result.items() if k not in {'trace', 'events'}}, indent=2))
