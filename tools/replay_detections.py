"""Replay cached raw person detections through the actual tracker and counter.

Unlike replay_trace, this reruns association, including unconfirmed detections.
Person/ReID inference is cached. Optional garment inference reads source frames.
No database IO is performed.
"""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import cv2
from ultralytics.engine.results import Boxes
from ultralytics.utils import YAML

from config.settings import Settings
from core.appearance_tracker import AppearanceTracker
from core.counter_engine import PassengerCounterEngine
from core.types import Detection
from core.person_detections import distinct_person_indices


def replay_raw(path, cfg, *, output=None, tracker_class=AppearanceTracker, suppress_duplicates=True,
               legacy_threshold=False, source=None):
    if cfg.TRACKING_MODE != 'appearance':
        raise ValueError('Raw replay requires appearance tracking')
    if cfg.CHILD_MODE != 'disabled':
        raise ValueError('Child filtering needs actual frames and asynchronous results; use tools.evaluate_clip')
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
    tracker.drift_threshold = cfg.TRACKER_DRIFT_THRESHOLD
    if not legacy_threshold:
        tracker.set_detection_threshold(cfg.CONFIDENCE)
    engine = PassengerCounterEngine(cfg)
    frame = np.zeros((*shape, 3), np.uint8)
    events, trace = [], []
    def event_record(event, emitted_frame, **extra):
        stamp = datetime.fromisoformat(event.detected_at).timestamp()
        pos = int(np.searchsorted(timestamps, stamp))
        nearby = [k for k in (pos-1, pos) if 0 <= k < len(indices)]
        best = min(nearby, key=lambda k: abs(timestamps[k] - stamp)) if nearby else None
        crossed = int(indices[best]) if best is not None and abs(timestamps[best] - stamp) < 1e-4 else None
        return dict(event.to_dict(), frame=emitted_frame, crossing_frame=crossed, **extra)
    reader = None
    vest = None
    ages = {}
    if cfg.STAFF_MODE == "model":
        from core.vest_classifier import VestClassifier
        vest = VestClassifier(cfg.path(cfg.VEST_MODEL_PATH), cfg.DEVICE, cfg.CPU_THREADS)
    if vest is not None or source is not None:
        if source is None or not len(indices) or np.any(np.diff(indices) != 1):
            raise ValueError('Visual replay requires source and consecutive frames')
        reader = cv2.VideoCapture(str(cfg.path(source)))
        if not reader.isOpened():
            raise ValueError('Cannot open visual replay source')
        reader.set(cv2.CAP_PROP_POS_FRAMES, int(indices[0]))
    try:
        for j, index in enumerate(indices):
            lo, hi = offsets[j:j + 2]
            mask = valid[lo:hi] & (boxes[lo:hi, 4] >= (cfg.CONFIDENCE if legacy_threshold else tracker.detection_floor))
            b, f = boxes[lo:hi][mask], features[lo:hi][mask]
            source_indices = lo + np.flatnonzero(mask)
            if suppress_duplicates:
                keep = distinct_person_indices(b, f)
                b, f, source_indices = b[keep], f[keep], source_indices[keep]
            if reader is not None:
                ok, frame = reader.read()
                if not ok or frame.shape[:2] != shape:
                    reader.release()
                    raise ValueError('Visual replay source ended or shape differs')
            tracked = tracker.update(Boxes(b, shape), img=frame if reader is not None else None, feats=f if len(f) else None)
            detections = [Detection(int(t[4]), tuple(b[int(t[-1]), :4].tolist()),
                                    float(b[int(t[-1]), 4]), f[int(t[-1])],
                                    separated_from=tracker.identity_separations.get(int(t[4]), ())) for t in tracked]
            for detection in detections:
                age = ages.get(detection.track_id, 0)
                if (vest is not None and detection.confidence >= cfg.CONFIDENCE
                        and (age < cfg.VEST_CONFIRM_FRAMES or age % cfg.FEATURE_INTERVAL_FRAMES == 0)):
                    detection.staff_probability = vest.predict(frame, detection.bbox)
                ages[detection.track_id] = age + 1
            active = {d.track_id for d in detections}
            ages = {tid: age for tid, age in ages.items() if tid in active}
            current = engine.update(frame, detections, datetime.fromtimestamp(float(timestamps[j]), timezone.utc))
            events.extend(event_record(e, int(index)) for e in current)
            if output:
                trace.append({'frame': int(index), 'detections': [
                    {'id': d.track_id, 'bbox': d.bbox, 'conf': d.confidence,
                     'feature': int(source_indices[int(t[-1])]),
                     'staff_probability': d.staff_probability,
                     'staff_confirmed': engine.tracks[d.track_id].staff.confirmed if d.track_id in engine.tracks else False,
                     'pending': bool(engine.tracks[d.track_id].pending) if d.track_id in engine.tracks else False,
                     'zone': engine.tracks[d.track_id].zone if d.track_id in engine.tracks else None,
                     'identity': engine.tracks[d.track_id].identity_id if d.track_id in engine.tracks else None,
                     'separated_from': d.separated_from}
                    for d, t in zip(detections, tracked)]})
        events.extend(event_record(e, int(indices[-1]) if len(indices) else None, final=True)
                      for e in engine.finish())
    finally:
        if reader is not None:
            reader.release()
    report = {'cache': str(path), 'frames': len(indices), 'in': engine.total_in, 'out': engine.total_out,
              'confidence': cfg.CONFIDENCE, 'anchor_y': cfg.ANCHOR_Y,
              'association_similarity': cfg.TRACKER_IDENTITY_MIN_SIMILARITY,
              'drift_threshold': cfg.TRACKER_DRIFT_THRESHOLD,
              'staff_mode': cfg.STAFF_MODE, 'vest_model': cfg.VEST_MODEL_PATH,
              'vest_threshold': cfg.VEST_THRESHOLD, 'vest_confirm_frames': cfg.VEST_CONFIRM_FRAMES,
              'staff_delay': cfg.STAFF_DECISION_DELAY_SEC,
              'frame_semantics': 'frame=decision emission; crossing_frame=physical transition timestamp',
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
    parser.add_argument('--drift-threshold', type=float, default=0.)
    parser.add_argument('--confidence', type=float, default=.35)
    parser.add_argument('--source', help='Required for garment inference; original video matching cached frame indices')
    parser.add_argument('--vest-model')
    parser.add_argument('--vest-threshold', type=float, default=.5)
    parser.add_argument('--staff-decision-delay', type=float, default=10)
    args = parser.parse_args()
    cfg = replace(Settings(), TRACKER_IDENTITY_MIN_SIMILARITY=args.association_similarity,
                  TRACKER_DRIFT_THRESHOLD=args.drift_threshold,
                  CONFIDENCE=args.confidence, STAFF_MODE='model' if args.vest_model else 'disabled',
                  VEST_MODEL_PATH=args.vest_model or '', VEST_THRESHOLD=args.vest_threshold,
                  STAFF_DECISION_DELAY_SEC=args.staff_decision_delay)
    result = replay_raw(args.cache, cfg, output=args.output, source=args.source)
    print(json.dumps({k: v for k, v in result.items() if k not in {'trace', 'events'}}, indent=2))
