"""Check reviewed temporal identities against hash-verified, full pipeline traces.

Target-only annotations are diagnostic: unmatched predictions are not false positives.
Tracker IDs are scoped to their reset epoch. No inference or database writes occur here.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def iou(left, right):
    a, b = np.asarray(left, float), np.asarray(right, float)
    intersection = float(np.prod(np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))))
    union = float(np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - intersection)
    return intersection / union if union > 0 else 0.


def match_targets(objects, detections, minimum_iou=.5, margin=.1, confidence=.3):
    """Mutual unique best matches; reject merged boxes and close competitors."""
    detections = [d for d in detections if d.get('conf', 0) >= confidence]
    scores = np.array([[iou(o['bbox'], d['bbox']) for d in detections] for o in objects])
    result = [None] * len(objects)
    if not objects or not detections:
        return result
    for index in range(len(objects)):
        order = np.argsort(-scores[index])
        best = int(order[0])
        if scores[index, best] < minimum_iou:
            continue
        if len(order) > 1 and scores[index, best] - scores[index, order[1]] < margin:
            continue
        competitors = np.delete(scores[:, best], index)
        if len(competitors) and scores[index, best] - competitors.max() < margin:
            continue
        result[index] = detections[best]
    return result


def validate_annotations(labels, root):
    if labels.get('format') != 'reviewed_track_sequences_v1' or labels.get('coverage') != 'target_people_only':
        raise ValueError('Expected reviewed target-only temporal annotations')
    root = Path(root).resolve()
    sequence_ids = set()
    for sequence in labels['sequences']:
        if sequence['id'] in sequence_ids:
            raise ValueError('Duplicate sequence ID')
        sequence_ids.add(sequence['id'])
        context = labels['contexts'][sequence['context']]
        if sequence['source'] != context['source']:
            raise ValueError('Sequence/context source mismatch')
        previous = -1
        for row in sequence['samples']:
            frame = row['frame']
            if not context['start'] <= frame <= context['end'] or frame <= previous or row.get('reviewed') is not True:
                raise ValueError('Samples must be reviewed, ordered and within context')
            previous = frame
            path = (root / row['image']).resolve()
            if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('Annotation image path/hash mismatch')
            image = cv2.imread(str(path))
            if image is None or hashlib.sha256(image.tobytes()).hexdigest() != row['decoded_sha256']:
                raise ValueError('Decoded annotation image hash mismatch')
            ids = set()
            for obj in row['objects']:
                pid, box = obj['person_id'], np.asarray(obj['bbox'], float)
                if pid in ids or pid not in sequence['people'] or obj['role'] != sequence['people'][pid]['role']:
                    raise ValueError('Duplicate/unknown identity or inconsistent role')
                if (box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2])
                        or np.any(box < 0) or box[2] > image.shape[1] or box[3] > image.shape[0]):
                    raise ValueError('Invalid person box')
                ids.add(pid)
        for event in sequence.get('events', []):
            if (event['person_id'] not in sequence['people'] or event['direction'] not in ('IN', 'OUT')
                    or not context['start'] <= event['start'] <= event['end'] <= context['end']):
                raise ValueError('Invalid passage interval')


def evaluate_sequence(sequence, trace, context, confidence=.3):
    if (Path(trace['source']).as_posix().lower() != Path(context['source']).as_posix().lower()
            or trace['start_frame'] != context['start'] or trace['end_frame'] != context['end']):
        raise ValueError('Trace source/context mismatch')
    frames = {row['frame']: row for row in trace['trace']}
    if len(frames) != context['end'] - context['start'] + 1:
        raise ValueError('Incomplete or duplicate trace frames')
    observations = []
    for sample in sequence['samples']:
        row = frames[sample['frame']]
        if row.get('decoded_sha256') != sample['decoded_sha256'] or row.get('camera_time') != sample['camera_time']:
            raise ValueError(f"Replay image/time mismatch at frame {sample['frame']}")
        if 'tracker_epoch' not in row:
            raise ValueError('Trace needs reset epochs; rerun evaluate_clip --frame-hashes')
        matches = match_targets(sample['objects'], row['detections'], confidence=confidence)
        for obj, detection in zip(sample['objects'], matches):
            observations.append(dict(frame=sample['frame'], person_id=obj['person_id'],
                                     epoch=row['tracker_epoch'], matched=detection is not None,
                                     track_id=detection.get('id') if detection else None,
                                     identity=detection.get('identity') if detection else None,
                                     staff_confirmed=detection.get('staff_confirmed') if detection else None))
    trajectories = {}
    ownership = defaultdict(set)
    for pid in sequence['people']:
        rows = [row for row in observations if row['person_id'] == pid]
        resolved = [row for row in rows if row['matched']]
        changes = []
        for field in ('track_id', 'identity'):
            previous = None
            for row in rows:
                if not row['matched'] or row[field] is None:
                    continue
                same_namespace = previous is not None and (field == 'identity' or previous['epoch'] == row['epoch'])
                if same_namespace and previous[field] != row[field]:
                    changes.append(dict(field=field, before=previous['frame'], after=row['frame'],
                                        old=previous[field], new=row[field]))
                previous = row
        for row in resolved:
            for field in ('track_id', 'identity'):
                if row[field] is not None:
                    namespace = row['epoch'] if field == 'track_id' else 'persistent_gallery'
                    ownership[(field, namespace, str(row[field]))].add(pid)
        trajectories[pid] = dict(samples=len(rows), matched=len(resolved), unresolved=len(rows)-len(resolved),
                                 track_ids=sorted({(row['epoch'], row['track_id']) for row in resolved if row['track_id'] is not None}),
                                 changes=changes,
                                 staff_confirmed_samples=sum(bool(row['staff_confirmed']) for row in resolved))
    merges = [dict(field=field, epoch=epoch, predicted_id=key, people=sorted(people))
              for (field, epoch, key), people in ownership.items() if len(people) > 1]
    # Passage attribution needs the tracked person AND the manually reviewed time interval.
    # One predicted event cannot verify two manually distinct people.
    eligible = []
    events = trace['events']
    for passage in sequence.get('events', []):
        keys = {(row['epoch'], row['track_id']) for row in observations
                if row['person_id'] == passage['person_id'] and row['matched']}
        identities = {row['identity'] for row in observations
                      if row['person_id'] == passage['person_id'] and row['matched'] and row['identity'] is not None}
        candidates = []
        for index, event in enumerate(events):
            crossing = event.get('crossing_frame')
            if crossing is None or crossing not in frames:
                continue
            key = (frames[crossing]['tracker_epoch'], event['track_id'])
            if (passage['start'] <= crossing <= passage['end']
                    and event['event_type'] == passage['direction']
                    and (key in keys or event.get('identity_id') in identities)):
                candidates.append(index)
        eligible.append(candidates)
    passages = []
    for passage, candidates in zip(sequence.get('events', []), eligible):
        exclusive = [index for index in candidates if sum(index in group for group in eligible) == 1]
        event = events[exclusive[0]] if len(exclusive) == 1 else None
        passages.append(dict(passage, status=('unresolved' if event is None else
                            'correct' if event['counted'] == passage.get('counted', True) else 'wrong_exclusion'),
                             predicted_event=event))
    return dict(sequence=sequence['id'], verified_frames=len(sequence['samples']),
                target_observations=len(observations), matched=sum(row['matched'] for row in observations),
                trajectories=trajectories, shared_predicted_ids=merges, passages=passages, observations=observations,
                resets=[row['frame'] for row in frames.values() if row.get('time_discontinuity')])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('annotations', type=Path)
    parser.add_argument('--traces', type=Path, required=True, help='Directory containing CONTEXT_LABEL.json')
    parser.add_argument('--label', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; use a new report path')
    labels = json.loads(args.annotations.read_text('utf-8'))
    validate_annotations(labels, args.annotations.parent)
    reports = {key: json.loads((args.traces / f'{key}_{args.label}.json').read_text('utf-8')) for key in labels['contexts']}
    result = dict(label=args.label, annotation_sha256=hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
                  scope=labels['scope'], coverage=labels['coverage'],
                  sequences=[evaluate_sequence(s, reports[s['context']], labels['contexts'][s['context']],
                                               labels.get('minimum_confidence', .3)) for s in labels['sequences']])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({s['sequence']: dict(matched=s['matched'], observations=s['target_observations'],
                      shared_ids=len(s['shared_predicted_ids']), passages=[p['status'] for p in s['passages']])
                      for s in result['sequences']}, indent=2))


if __name__ == '__main__':
    main()
