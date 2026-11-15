"""Offline, auditable camera-clock mapping. No server clock fallback."""
from bisect import bisect_right
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from core.onnx_runtime import cpu_options


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def parse_clock(text, timezone_name):
    # The weekday is deliberately ignored. Corrections apply only inside time.
    date = re.search(r'(?<!\d)(\d{2})[-./]?(\d{2})[-./]?([12]\d{3})', text)
    if not date:
        raise ValueError('Camera date is unreadable')
    tail = text[date.end():].translate(str.maketrans({'O': '0', 'o': '0', 'Z': '2', 'z': '2'}))
    clock = re.search(r'([0-2]\d)[:. +\-]?(\d{2})[:. +\-]?(\d{2})(?!\d)', tail)
    if not clock:
        raise ValueError('Camera time is unreadable')
    day, month, year = map(int, date.groups())
    hour, minute, second = map(int, clock.groups())
    naive = datetime(year, month, day, hour, minute, second)
    zone = ZoneInfo(timezone_name)
    aware = naive.replace(tzinfo=zone)
    if aware.utcoffset() != naive.replace(tzinfo=zone, fold=1).utcoffset():
        raise ValueError('Ambiguous camera timezone transition; review recording')
    if aware.astimezone(timezone.utc).astimezone(zone).replace(tzinfo=None) != naive:
        raise ValueError('Nonexistent camera local time')
    return aware.astimezone(timezone.utc)


class ClockRecognizer:
    def __init__(self, model, roi=(0.0125, 0.0458333333333, 0.2875, 0.0875), threads=2):
        import onnxruntime as ort
        if not Path(model).is_file():
            raise ValueError("Camera clock ONNX model does not exist")
        self.session = ort.InferenceSession(str(model), sess_options=cpu_options(threads),
                                            providers=['CPUExecutionProvider'])
        self.characters = ['blank'] + self.session.get_modelmeta().custom_metadata_map['character'].splitlines() + [' ']
        self.input_name = self.session.get_inputs()[0].name
        self.roi = tuple(roi)
        if len(self.roi) != 4 or not (0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1):
            raise ValueError('CLOCK_ROI requires normalized x1,y1,x2,y2')

    def crop(self, image):
        height, width = image.shape[:2]
        x1, y1, x2, y2 = self.roi
        crop = image[int(y1 * height):math.ceil(y2 * height), int(x1 * width):math.ceil(x2 * width)]
        return crop

    def read(self, image):
        return self.read_crop(self.crop(image))

    def read_crop(self, crop):
        target_width = max(1, math.ceil(48 * crop.shape[1] / crop.shape[0]))
        tensor = np.zeros((1, 3, 48, max(320, target_width)), dtype=np.float32)
        tensor[0, :, :, :target_width] = cv2.resize(crop, (target_width, 48)).transpose(2, 0, 1).astype(np.float32) / 127.5 - 1
        scores = self.session.run(None, {self.input_name: tensor})[0][0]
        ids = scores.argmax(axis=1)
        selected = [i for i, value in enumerate(ids) if value != 0 and (i == 0 or value != ids[i - 1])]
        return (''.join(self.characters[ids[i]] for i in selected),
                float(np.mean([scores[i, ids[i]] for i in selected])) if selected else 0.0)


class ClockMap:
    def __init__(self, payload, *, allow_unreviewed=False):
        if payload.get('format') != 'camera_clock_map_v1':
            raise ValueError('Unsupported camera clock map')
        self.payload = payload
        self.end_frame = int(payload['end_frame'])  # exclusive
        self.segments = payload['segments']
        self.exclusions = [(int(s['excluded_transition_start']), int(s['frame'])) for s in self.segments if 'excluded_transition_start' in s]
        for start, end in self.exclusions:
            if not 0 <= start < end < self.end_frame:
                raise ValueError('Invalid excluded montage interval')
        if not allow_unreviewed and any(s.get('reviewed') is not True for s in self.segments if 'excluded_transition_start' in s):
            raise ValueError('Camera map contains unreviewed montage transitions; inspect tools.review_clock_map output first')
        self.excluded_starts = [a for a, _ in self.exclusions]
        if self.exclusions != sorted(self.exclusions):
            raise ValueError("Montage exclusions must be sorted")
        if not allow_unreviewed and any((row.get('rejected_unconfirmed_jump') or row.get('chronology_review_required')) and row.get('reviewed') is not True for row in payload.get('observations', [])):
            raise ValueError('Camera map has unreviewed OCR readings at edits; inspect tools.review_clock_map output')
        self.starts = [int(segment['frame']) for segment in self.segments]
        if (self.end_frame <= 0 or not self.starts or self.starts[0] != 0 or
                self.starts != sorted(set(self.starts)) or self.starts[-1] >= self.end_frame):
            raise ValueError('Invalid camera clock segment boundaries')
        self.cut_frames = set(self.starts[1:])
        self.pts_starts = [float(s['pts']) for s in self.segments] if all('pts' in s for s in self.segments) else None
        if self.pts_starts is not None and (self.pts_starts[0] != 0 or any(not math.isfinite(v) for v in self.pts_starts) or self.pts_starts != sorted(set(self.pts_starts))):
            raise ValueError('Camera clock PTS boundaries must increase')
        self.time_exclusions = [(float(s['excluded_start_pts']), float(s['pts'])) for s in self.segments if 'excluded_start_pts' in s]
        if any(not math.isfinite(a) or not math.isfinite(b) or a > b for a,b in self.time_exclusions):
            raise ValueError('Invalid camera clock PTS exclusion')
        offsets = [float(s['offset']) for s in self.segments]
        if any(not math.isfinite(v) for v in offsets) or any(b < a for a, b in zip(offsets, offsets[1:])):
            raise ValueError('Camera clock cannot run backwards')

    def at(self, index, pts):
        if not 0 <= index < self.end_frame or not math.isfinite(pts) or pts < 0:
            raise ValueError('Frame outside verified camera clock map')
        segment = bisect_right(self.pts_starts, pts) - 1 if self.pts_starts is not None else bisect_right(self.starts, index) - 1
        return (datetime.fromtimestamp(self.segments[segment]['offset'], timezone.utc) + timedelta(seconds=pts),
                (segment > 0 and abs(pts - self.pts_starts[segment]) < 1e-8) if self.pts_starts is not None else index in self.cut_frames)

    def excluded_until(self, index):
        position = bisect_right(self.excluded_starts, index) - 1
        if position >= 0 and self.exclusions[position][0] <= index < self.exclusions[position][1]:
            return self.exclusions[position][1]
        return None

    def crossed_cut(self, previous, current, *, previous_pts=None, pts=None):
        if self.pts_starts is not None:
            return bisect_right(self.pts_starts, previous_pts) != bisect_right(self.pts_starts, pts)
        return bisect_right(self.starts, previous) != bisect_right(self.starts, current)

    def is_excluded(self, index, pts):
        if self.pts_starts is not None:
            return any(a <= pts < b for a,b in self.time_exclusions)
        return self.excluded_until(index) is not None

    @classmethod
    def load(cls, path, cfg, *, verify_source=True):
        try:
            payload = json.loads(Path(path).read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            raise ValueError('Cannot read camera clock map JSON; build/review it before counting') from exc
        if not isinstance(payload, dict) or not {'timezone', 'roi', 'model_sha256', 'source_sha256'}.issubset(payload):
            raise ValueError('Incomplete camera clock map')
        if payload['timezone'] != cfg.TIMEZONE or payload['roi'] != list(map(float, cfg.CLOCK_ROI.split(','))):
            raise ValueError('Camera clock configuration differs from map')
        if payload['model_sha256'] != sha256(cfg.path(cfg.CLOCK_MODEL_PATH)):
            raise ValueError('Camera clock recognizer differs from map')
        if verify_source and payload['source_sha256'] != sha256(cfg.path(cfg.VIDEO_SOURCE)):
            raise ValueError('Camera clock map belongs to a different video')
        try:
            return cls(payload)
        except (KeyError, TypeError, OverflowError) as exc:
            raise ValueError('Malformed camera clock segments') from exc
