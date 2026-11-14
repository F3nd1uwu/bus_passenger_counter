"""Approximate visual child evidence; unknown passengers remain countable.

Model scores are not a verified age or a calibrated probability of being under 7.
The bounded worker never blocks live frame processing on an age inference.
"""
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field, replace
import hashlib
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from core.types import ChildObservation

log = logging.getLogger(__name__)


@dataclass
class ChildEvidence:
    samples: int = 0
    positive: int = 0
    confirmed: bool = False
    adult_seen: bool = False
    seen_samples: list[str] = field(default_factory=list)

    def observe(self, verdict, sample_id, required):
        if verdict not in {'child', 'adult', 'unknown'} or not isinstance(sample_id, str) or not sample_id:
            raise ValueError('Invalid child observation')
        if sample_id in self.seen_samples:
            return
        self.seen_samples = [*self.seen_samples[-31:], sample_id]
        self.samples += 1
        self.adult_seen = self.adult_seen or verdict == 'adult'
        self.positive = self.positive + 1 if verdict == 'child' and not self.adult_seen else 0
        self.confirmed = self.positive >= required and not self.adult_seen


def _tensor(image, size, mean, std, *, whole):
    height, width = image.shape[:2]
    ratio = min(size / width, size / height) if whole else max(size / width, size / height)
    resized = cv2.resize(image, (max(1, round(width * ratio)), max(1, round(height * ratio))),
                         interpolation=cv2.INTER_CUBIC if size == 224 else cv2.INTER_LINEAR)
    h, w = resized.shape[:2]
    if whole:
        canvas = np.zeros((size, size, 3), np.uint8)
        x, y = (size - w) // 2, (size - h) // 2
        canvas[y:y + h, x:x + w] = resized
    else:
        x, y = (w - size) // 2, (h - size) // 2
        canvas = resized[y:y + size, x:x + size]
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    return np.ascontiguousarray(((rgb - mean) / std).transpose(2, 0, 1))


class ChildClassifier:
    def __init__(self, manifest_path):
        import onnxruntime as ort
        from core.onnx_runtime import cpu_options
        path = Path(manifest_path)
        self.meta = json.loads(path.read_text('utf-8'))
        if self.meta.get('format') != 'approx_child_ensemble_v1':
            raise ValueError('Unsupported child model manifest')
        if (not .005 <= self.meta.get('minimum_child_margin', -1) <= 1
                or self.meta.get('maximum_child_age') != 7.0
                or not 7 <= self.meta.get('adult_veto_age', -1) <= 122):
            raise ValueError('Invalid approximate-child policy')
        sessions = {}
        for name in ['vision', 'age']:
            asset = self.meta[name]
            target = (path.parent / asset['file']).resolve()
            if not target.is_relative_to(path.parent.resolve()):
                raise ValueError('Child model must be inside its manifest directory')
            with target.open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != asset['sha256']:
                    raise ValueError(f'Child {name} model checksum differs')
            sessions[name] = ort.InferenceSession(str(target), sess_options=cpu_options(1),
                                                 providers=['CPUExecutionProvider'])
        self.vision, self.age = sessions['vision'], sessions['age']
        self.text = np.asarray(self.meta['embeddings'], np.float32)
        if self.text.shape != (11, 512) or not np.isfinite(self.text).all():
            raise ValueError('Child text embeddings must be finite [11,512] vectors')
        norms = np.linalg.norm(self.text, axis=1, keepdims=True)
        if (norms < 1e-6).any():
            raise ValueError('Invalid child text embedding')
        self.text /= norms
        self.vision_mean = np.asarray(self.meta['vision_mean'], np.float32)
        self.vision_std = np.asarray(self.meta['vision_std'], np.float32)
        if (self.vision_mean.shape != (3,) or self.vision_std.shape != (3,)
                or not np.isfinite(self.vision_mean).all() or not np.isfinite(self.vision_std).all()
                or (self.vision_std <= 0).any()):
            raise ValueError('Invalid child vision normalization')
        self.age_mean = np.array([.485, .456, .406], np.float32)
        self.age_std = np.array([.229, .224, .225], np.float32)
        self.empty_face = _tensor(np.zeros((384, 384, 3), np.uint8), 384,
                                  self.age_mean, self.age_std, whole=True)
        if (len(self.vision.get_inputs()) != 1 or len(self.age.get_inputs()) != 1
                or self.vision.get_inputs()[0].name != 'pixel_values'
                or self.age.get_inputs()[0].name != 'faces_bodies'
                or self.age.get_inputs()[0].shape != [1, 6, 384, 384]
                or self.vision.get_outputs()[0].shape[-1] != 512
                or self.age.get_outputs()[0].shape != [1, 1]):
            raise ValueError('Child models have incompatible inputs')

    def predict(self, crop):
        views = np.stack([_tensor(crop, 224, self.vision_mean, self.vision_std, whole=whole)
                          for whole in [False, True]])
        features = self.vision.run(None, {'pixel_values': views})[0]
        if features.shape != (2, 512) or not np.isfinite(features).all():
            return 'unknown'
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        if (norms < 1e-6).any():
            return 'unknown'
        scores = features / norms @ self.text.T
        margin = float(np.min(scores[:, :4].max(axis=1) - scores[:, 4:].max(axis=1)))
        if margin < self.meta['minimum_child_margin']:
            return 'unknown'
        body = _tensor(crop, 384, self.age_mean, self.age_std, whole=True)
        estimated = self.age.run(None, {'faces_bodies': np.concatenate([self.empty_face, body])[None]})[0]
        if estimated.size != 1 or not np.isfinite(estimated).all():
            return 'unknown'
        age = float(estimated.reshape(-1)[0])
        if not 0 <= age <= 122:
            return 'unknown'
        if 0 <= age < self.meta['maximum_child_age']:
            return 'child'
        return 'adult' if age >= self.meta['adult_veto_age'] else 'unknown'


class ChildObserver:
    """At most eight queued crops; three distinct sampled crops per new track.

    Results carry the source appearance. The counter checks it again so a late
    result cannot silently label a changed trajectory as a child.
    """
    CAPACITY = 8
    INTERVAL = 5

    def __init__(self, classifier):
        self.classifier = classifier
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='child-filter')
        self.jobs = {}
        self.states = {}
        self.frame = 0
        self.closed = False
        self.stats = dict(submitted=0, completed=0, child=0, adult=0, unknown=0, errors=0, timed_out=0)

    def _collect(self):
        updates = []
        for future, data in list(self.jobs.items()):
            if not future.done():
                continue
            del self.jobs[future]
            tid, sample_id, embedding = data
            if future.cancelled():
                continue
            try:
                verdict = future.result()
                if verdict not in {'child', 'adult', 'unknown'}:
                    raise ValueError('Invalid child classifier verdict')
            except Exception:
                log.exception('Child analysis failed; passenger age remains unknown')
                verdict = 'unknown'
                self.stats['errors'] += 1
            self.stats['completed'] += 1
            self.stats[verdict] += 1
            state = self.states.get(tid)
            if state is not None:
                state['last_verdict'] = verdict
                state['adult'] = state['adult'] or verdict == 'adult'
                if state['adult']:
                    for queued, other in list(self.jobs.items()):
                        if other[0] == tid:
                            queued.cancel()
            updates.append(ChildObservation(tid, verdict, sample_id, embedding))
        return updates

    def update(self, frame, detections, confidence):
        self.frame += 1
        updates = self._collect()
        current = {d.track_id: d.embedding for d in detections}
        updates = [replace(item, verdict='unknown')
                   if item.track_id in current and (current[item.track_id] is None
                       or float(current[item.track_id] @ item.embedding) < .85) else item for item in updates]
        if self.closed:
            return updates
        height, width = frame.shape[:2]
        for detection in detections:
            tid = detection.track_id
            state = self.states.setdefault(tid, {'submitted': 0, 'last': -100, 'seen': self.frame,
                                                 'last_verdict': 'unknown', 'adult': False,
                                                 'observed': 0, 'embedding': None})
            gap = self.frame - state['seen']
            state['seen'] = self.frame
            if detection.confidence < confidence or detection.embedding is None:
                state['observed'] = 0
                continue
            consistent = state['embedding'] is None or float(state['embedding'] @ detection.embedding) >= .8
            state['observed'] = state['observed'] + 1 if consistent and gap <= 1 else 1
            state['embedding'] = detection.embedding.copy()
            if not consistent and state['last_verdict'] == 'child':
                updates.append(ChildObservation(tid, 'unknown', f'drift:{self.frame}:{tid}', detection.embedding.copy()))
                state['last_verdict'] = 'unknown'
            if (state['adult'] or state['observed'] < 3
                    or self.frame - state['last'] < self.INTERVAL or len(self.jobs) >= self.CAPACITY
                    or sum(data[0] == tid for data in self.jobs.values()) >= 3
                    or (state['submitted'] >= 3 and state['last_verdict'] != 'child')):
                continue
            x1, y1, x2, y2 = detection.bbox
            left, top, right, bottom = max(0, int(x1)), max(0, int(y1)), min(width, int(x2)), min(height, int(y2))
            if right - left < 48 or bottom - top < 64 or (right-left)*(bottom-top) > .85*width*height:
                continue
            # Mixed bodies are unsuitable age evidence, even if YOLO is confident.
            crowded = False
            area = (right - left) * (bottom - top)
            for other in detections:
                if other.track_id == tid or other.confidence < confidence or other.embedding is None:
                    continue
                a, b, c, d = other.bbox
                intersection = max(0, min(right, c) - max(left, a)) * max(0, min(bottom, d) - max(top, b))
                other_area = (c-a)*(d-b)
                if (area >= .75 * other_area and intersection / max(1, min(area, other_area)) > .35
                        and float(detection.embedding @ other.embedding) < .85):
                    crowded = True
                    break
            if crowded:
                continue
            crop = frame[top:bottom, left:right].copy()
            sample_id = hashlib.sha256(crop.tobytes()).hexdigest()
            state['last'] = self.frame
            state['submitted'] += 1
            self.stats['submitted'] += 1
            future = self.pool.submit(self.classifier.predict, crop)
            self.jobs[future] = (tid, sample_id, detection.embedding.copy())
        self.states = {tid: value for tid, value in self.states.items()
                       if self.frame - value['seen'] <= 300 or any(data[0] == tid for data in self.jobs.values())}
        return updates

    def drain(self, timeout=5):
        if self.jobs:
            _, remaining = wait(list(self.jobs), timeout=timeout)
            self.stats['timed_out'] += len(remaining)
        return self._collect()

    def reset(self):
        for future in self.jobs:
            future.cancel()
        self.jobs.clear()
        self.states.clear()
        self.frame = 0

    def finish(self):
        updates = self.drain()
        # Late results must not relabel a passage after its EOF decision.
        self.reset()
        return updates

    def close(self):
        self.closed = True
        self.reset()
        self.pool.shutdown(wait=False, cancel_futures=True)
