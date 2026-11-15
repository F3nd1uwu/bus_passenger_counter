"""Read the burned-in clock before counting; refine montage boundaries offline."""
import argparse
from dataclasses import replace
import json
import logging
import math
import sqlite3
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
import time

import cv2

from config.settings import Settings
from core.camera_clock import ClockMap, ClockRecognizer, parse_clock, sha256
from core.video_source import source_kind

log = logging.getLogger(__name__)


def scan_crops(source, ranges, end, fps, roi):
    """Decode from frame zero: random MP4 seeks can return a different image.

    Only the tiny clock ROI is retained. Presentation-time fallback matches the
    production reader; all skipped frames still advance the decoder clock.
    """
    merged = []
    for start, finish in sorted(ranges):
        start, finish = max(0, int(start)), min(end - 1, int(finish))
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], finish))
        else:
            merged.append((start, finish))
    if not merged:
        return
    cap = cv2.VideoCapture(str(source))
    previous, position = -1., 0
    try:
        for index in range(merged[-1][1] + 1):
            if not cap.grab():
                raise RuntimeError(f'Clock scan decoding failed at frame {index}')
            pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if not math.isfinite(pts) or pts < 0 or (index and (pts <= 0 or pts <= previous)):
                pts = max(index / fps, previous + 1 / fps)
            previous = pts
            while position < len(merged) and index > merged[position][1]:
                position += 1
            if position >= len(merged):
                break
            if index < merged[position][0]:
                continue
            ok, image = cap.retrieve()
            if not ok:
                raise RuntimeError(f'Cannot retrieve camera clock frame {index}')
            height, width = image.shape[:2]
            x1, y1, x2, y2 = roi
            crop = image[int(y1 * height):math.ceil(y2 * height), int(x1 * width):math.ceil(x2 * width)].copy()
            yield index, pts, crop
    finally:
        cap.release()


def build(cfg, output, *, interval=5.0, end_frame=None):
    source_kind(cfg.VIDEO_SOURCE, cfg.SOURCE_KIND)
    if not 0.25 <= interval <= 10:
        raise ValueError('Clock sampling interval must be 0.25..10 seconds')
    builder_hash = sha256(Path(__file__))
    source = cfg.path(cfg.VIDEO_SOURCE)
    if not source.is_file() or Path(output).resolve() == source.resolve():
        raise ValueError('Expected a local video and a separate map output')
    recognizer = ClockRecognizer(cfg.path(cfg.CLOCK_MODEL_PATH), tuple(map(float, cfg.CLOCK_ROI.split(','))),
                                 min(2, cfg.CPU_THREADS))
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError('Cannot open clock-map input')
    cache_path = Path(output).with_suffix(".observations.json")
    cache_key = dict(decoder='sequential_v2', source_sha256=sha256(source), model_sha256=sha256(cfg.path(cfg.CLOCK_MODEL_PATH)),
                     roi=cfg.CLOCK_ROI, timezone=cfg.TIMEZONE, confidence=cfg.CLOCK_MIN_CONFIDENCE, end_frame=end_frame)
    observations = {}
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached["key"] == cache_key:
            for stored_row in cached['observations']:
                row = {key: stored_row[key] for key in ('frame', 'pts', 'text', 'confidence', 'read_frame')}
                row.update({key: stored_row[key] for key in ('raw_text', 'time_text', 'date_inferred') if key in stored_row})
                try:
                    if row['confidence'] < cfg.CLOCK_MIN_CONFIDENCE:
                        raise ValueError('low confidence')
                    row['timestamp'] = parse_clock(row['text'], cfg.TIMEZONE).timestamp()
                    row['offset'] = row['timestamp'] - row['pts']
                except ValueError:
                    row['timestamp'] = row['offset'] = None
                observations[row['frame']] = row
            log.info("Reusing %s verified-source OCR observations", len(observations))
    started = time.monotonic()
    crop_db = None
    crop_path = Path(output).with_suffix(".clock-crops.sqlite3")
    succeeded = False
    try:
        fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or total <= 0:
            raise ValueError('Video needs valid frame count and FPS')
        end = total if end_frame is None else int(end_frame)
        if not 1 <= end <= total:
            raise ValueError('end-frame must be within video, exclusive')

        first_stamp = observations.get(0, {}).get('timestamp')
        roi = tuple(map(float, cfg.CLOCK_ROI.split(',')))
        fine_ranges = []

        def recognize(index, pts, crop):
            nonlocal first_stamp
            text, score = recognizer.read_crop(crop)
            raw_text, time_text, inferred_date = text, None, False
            try:
                if score < cfg.CLOCK_MIN_CONFIDENCE:
                    raise ValueError('Low OCR confidence')
                stamp = parse_clock(text, cfg.TIMEZONE).timestamp()
                if first_stamp is not None and not -1 <= stamp - first_stamp <= 172800:
                    raise ValueError('Date differs by over two days from first frame')
                if index == 0:
                    first_stamp = stamp
            except ValueError:
                stamp = None
                if first_stamp is not None:
                    time_text, time_score = recognizer.read_crop(crop[:, round(crop.shape[1] * .61):])
                    try:
                        if time_score < cfg.CLOCK_MIN_CONFIDENCE:
                            raise ValueError('Low time-only OCR confidence')
                        reference = datetime.fromtimestamp(first_stamp, timezone.utc).astimezone(ZoneInfo(cfg.TIMEZONE))
                        text = reference.strftime('%d-%m-%Y ') + time_text
                        candidate = parse_clock(text, cfg.TIMEZONE)
                        if candidate.timestamp() < first_stamp - 1:
                            candidate += timedelta(days=1)
                            text = candidate.astimezone(ZoneInfo(cfg.TIMEZONE)).strftime('%d-%m-%Y %H:%M:%S')
                        stamp, score, inferred_date = candidate.timestamp(), time_score, True
                    except ValueError:
                        text = raw_text
            row = dict(frame=index, pts=pts, timestamp=stamp,
                       offset=stamp - pts if stamp is not None else None,
                       text=text, confidence=score, read_frame=index, raw_text=raw_text, time_text=time_text, date_inferred=inferred_date)
            observations[index] = row
            return row

        def probe(index, *, strict=True):
            nonlocal crop_db
            if index not in observations:
                if crop_db is None:
                    crop_path.parent.mkdir(parents=True, exist_ok=True)
                    crop_db = sqlite3.connect(crop_path)
                    crop_db.execute('CREATE TABLE IF NOT EXISTS metadata (id INTEGER PRIMARY KEY CHECK(id=1), source_key TEXT NOT NULL)')
                    key = json.dumps(cache_key, sort_keys=True)
                    existing = crop_db.execute('SELECT source_key FROM metadata WHERE id=1').fetchone()
                    if existing is not None and existing[0] != key:
                        raise ValueError('Clock crop cache belongs to a different source/configuration; choose another map output')
                    crop_db.execute('INSERT OR IGNORE INTO metadata VALUES (1,?)', (key,))
                    crop_db.execute('CREATE TABLE IF NOT EXISTS crops (frame INTEGER PRIMARY KEY, pts REAL NOT NULL, image BLOB NOT NULL)')
                    crop_db.commit()
                    cached_frames = {row[0] for row in crop_db.execute('SELECT frame FROM crops')}
                    log.info('Decoding montage brackets sequentially into disk cache; %s crops already cached', len(cached_frames))
                    # Store cropped clocks on disk, not an unbounded image bank in RAM.
                    count = 0
                    for frame, pts, crop in scan_crops(source, fine_ranges, end, fps, roi):
                        if frame in cached_frames:
                            continue
                        packed = cv2.imencode('.png', crop, [cv2.IMWRITE_PNG_COMPRESSION, 1])[1].tobytes()
                        crop_db.execute('INSERT INTO crops VALUES (?,?,?)', (frame, pts, packed))
                        count += 1
                        if count % 1000 == 0:
                            crop_db.commit()
                        if count % 10000 == 0:
                            log.info('Cached %s clock crops on disk', count)
                    crop_db.commit()
                found = crop_db.execute('SELECT pts,image FROM crops WHERE frame=?', (int(index),)).fetchone()
                if found is None:
                    raise ValueError(f'Frame {index} outside camera-clock refinement brackets; use a shorter interval')
                import numpy as np
                pts, packed = found
                recognize(index, pts, cv2.imdecode(np.frombuffer(packed, np.uint8), cv2.IMREAD_COLOR))
            row = observations[index]
            if strict and row['timestamp'] is None:
                raise ValueError(f'Unreadable camera clock at frame {index}; review ROI/recording')
            return row

        step = max(1, round(interval * fps))
        indices = sorted(set([0, end - 1] + list(range(step, end, step))))
        missing_indices = [index for index in indices if index not in observations or
                           (observations[index]['timestamp'] is None and observations[index].get('time_text') is None)]
        if missing_indices:
            for number, (index, pts, crop) in enumerate(scan_crops(source, [(i,i) for i in missing_indices], end, fps, roi)):
                recognize(index, pts, crop)
                if number % 100 == 0:
                    log.info('Sequential clock scan: %s/%s samples, frame %s', number + 1, len(missing_indices), index)
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(json.dumps(dict(key=cache_key, observations=list(observations.values())), ensure_ascii=False), encoding='utf-8')
        anchors = []
        for number, index in enumerate(indices):
            anchors.append(probe(index, strict=False))
            if number % 100 == 0:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(dict(key=cache_key, observations=list(observations.values())), ensure_ascii=False), encoding='utf-8')
                log.info('Clock scan: %s/%s samples, frame %s', number + 1, len(indices), index)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(dict(key=cache_key, observations=list(observations.values())), ensure_ascii=False), encoding="utf-8")
        if anchors[0]['timestamp'] is None or anchors[-1]['timestamp'] is None:
            raise ValueError("Start/end camera clock unreadable: review ROI or explicitly exclude verified credits with --end-frame")
        for row in anchors:
            if row['timestamp'] is not None and not -1 <= row['timestamp'] - anchors[0]['timestamp'] <= 172800:
                row['timestamp'] = row['offset'] = None
                row['rejected_date_outlier'] = True
        # Remove isolated OCR digit errors only when both neighbours agree.
        original = [dict(row) for row in anchors]
        window = max(1, round(60 / interval))
        for i in range(1, len(anchors) - 1):
            row = original[i]
            if row['offset'] is None:
                continue
            before = [r for r in original[max(0, i - window):i] if r['offset'] is not None]
            after = [r for r in original[i + 1:i + window + 1] if r['offset'] is not None]
            if any(abs(a['offset'] - b['offset']) <= 1.5 and abs(row['offset'] - a['offset']) > 1.5
                   and abs(row['offset'] - b['offset']) > 1.5 for a in before for b in after):
                anchors[i]['timestamp'] = anchors[i]['offset'] = None
                anchors[i]['rejected_ocr_outlier'] = True
        # A wrong hour at a real edit may have no equal-offset neighbours.
        # Keep the longest chronological anchor sequence across the whole file;
        # short rejected runs remain in the audit, long reversals need review.
        import numpy as np
        readable = [row for row in anchors if row['offset'] is not None]
        values = np.asarray([row['offset'] for row in readable], dtype=np.float64)
        lengths = np.full(len(readable), -100000, dtype=np.int32)
        parents = np.full(len(readable), -1, dtype=np.int32)
        lengths[0] = 1
        for i in range(1, len(readable)):
            scores = np.where(values[:i] <= values[i] + 1.5, lengths[:i], -100000)
            parent = int(scores.argmax())
            if scores[parent] > 0:
                lengths[i] = scores[parent] + 1
                parents[i] = parent
        if lengths[-1] <= 0:
            raise ValueError('Start/end camera clocks are inconsistent')
        selected, cursor = set(), len(readable) - 1
        while cursor >= 0:
            selected.add(cursor)
            cursor = int(parents[cursor])
        rejected = []
        for i, row in enumerate(readable):
            if i not in selected:
                row['rejected_chronology_outlier'] = True
                row['timestamp'] = row['offset'] = None
                rejected.append(row['frame'])
        rejection_start = None
        for row in anchors:
            if row.get('rejected_chronology_outlier'):
                rejection_start = row['pts'] if rejection_start is None else rejection_start
            elif row['offset'] is not None and rejection_start is not None:
                if row['pts'] - rejection_start > 30:
                    for suspect in anchors:
                        if suspect.get('rejected_chronology_outlier') and rejection_start <= suspect['pts'] < row['pts']:
                            suspect['chronology_review_required'] = True
                rejection_start = None
        if rejected:
            log.info('Rejected %s nonchronological OCR anchors; retained in audit', len(rejected))
        valid = [row for row in anchors if row['offset'] is not None]
        for left, right in zip(valid, valid[1:]):
            missing = [r for r in anchors if left['frame'] < r['frame'] < right['frame'] and r['offset'] is None]
            if not missing:
                continue
            if right['pts'] - left['pts'] > 300:
                raise ValueError(f"Clock obscured for over 300 seconds near frame {missing[0]['frame']}; review manually")
            if abs(right['offset'] - left['offset']) > 1.5:
                continue  # refine montage between the surrounding readable anchors
            for row in missing:
                row['offset'] = left['offset']
                row['timestamp'] = row['pts'] + left['offset']
                row['inferred_between_consistent_anchors'] = [left['frame'], right['frame']]
        segments = [{'frame': 0, 'pts': 0.0, 'offset': anchors[0]['offset']}]
        anchors = [row for row in anchors if row['offset'] is not None]
        fine_ranges = [(a['frame'] - round(2 * fps), b['frame'] + round(2 * fps))
                       for a,b in zip(anchors, anchors[1:]) if b['offset'] - a['offset'] > 1.5]
        left = anchors[0]
        for right in anchors[1:]:
            delta = right['offset'] - left['offset']
            if delta < -1.5:
                raise ValueError(f'Camera clock moved backwards near frame {right["frame"]}; review OCR/recording')
            if delta <= 1.5:
                left = right
                continue
            def classify(index):
                row = probe(index, strict=False)
                if row['offset'] is None:
                    return None
                distances = [abs(row['offset'] - side['offset']) for side in (left, right)]
                return int(distances[1] < distances[0]) if min(distances) <= 1.5 else None

            # Scan the entire bracket: an isolated OCR error need not be at the
            # actual cut. Binary search alone can mistake it for a dissolve.
            lo, hi = left['frame'], right['frame']
            positions = sorted(set([lo, hi] + list(range(lo, hi, max(1, round(fps / 5))))))
            sides = {index: classify(index) for index in positions}
            old = [i for i in positions if sides[i] == 0]
            new = [i for i in positions if sides[i] == 1]
            supported = len(new) >= 2
            if not supported:
                supported = any(classify(index) == 1 for index in
                                (right['frame'] + shift for shift in (5, 10, 20, 40, -5, -10))
                                if 0 <= index < end and index != right['frame'])
            if not supported:
                if right is anchors[-1]:
                    raise ValueError('Final camera time could not be confirmed')
                right['rejected_unconfirmed_jump'] = True
                right['timestamp'] = right['offset'] = None
                log.warning('Rejected unconfirmed clock OCR at frame %s; retaining in audit', right['frame'])
                continue
            if not old or not new:
                raise ValueError(f"No readable clocks around montage frame {right['frame']}")
            # Mixed old/new glyphs inside a dissolve are explicitly reviewed.
            first_new, last_old = min(new), max(old)
            if last_old < first_new:
                lo, hi = last_old, first_new
                for index in range(lo + 1, hi):
                    sides[index] = classify(index)
                old = [i for i, side in sides.items() if side == 0]
                new = [i for i, side in sides.items() if side == 1]
                first_new, last_old = min(new), max(old)
            if last_old < first_new:
                lo, hi = last_old, first_new
            else:
                lo = max((i for i in old if i < first_new), default=left['frame'])
                hi = min((i for i in new if i > last_old), default=right['frame'])
            if hi - lo > round(5 * fps):
                raise ValueError(f"Montage clock unreadable over 5s near frame {lo}; review manually")
            segment = {'frame': hi, 'pts': probe(hi, strict=False)['pts'], 'offset': right['offset']}
            if hi - lo > 1:
                segment.update(excluded_transition_start=lo + 1, excluded_start_pts=probe(lo, strict=False)['pts'] + 1 / fps, reviewed=False)
            segments.append(segment)
            left = right
            log.info('Clock montage at %s; %s ambiguous frames to review', hi, hi - lo - 1)

        payload = {'format': 'camera_clock_map_v1', 'builder_sha256': builder_hash, 'source_sha256': sha256(source),
                   'model_sha256': sha256(cfg.path(cfg.CLOCK_MODEL_PATH)), 'timezone': cfg.TIMEZONE,
                   'roi': list(map(float, cfg.CLOCK_ROI.split(','))), 'fps': fps, 'total_frames': total,
                   'end_frame': end, 'excluded_tail_frames': total - end, 'sample_interval_seconds': interval,
                   'uncertainty_seconds': 1.5, 'segments': segments,
                   'observations': sorted(observations.values(), key=lambda row: row['frame'])}
        ClockMap(payload, allow_unreviewed=True)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + '.tmp')
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temporary.replace(output)
        log.info('Saved clock map: %s segments, %s observations, %.1fs; %s transitions need visual review', len(segments), len(observations), time.monotonic() - started, sum('excluded_transition_start' in s for s in segments))
        succeeded = True
        return payload
    finally:
        cap.release()
        if crop_db is not None:
            crop_db.close()
        if succeeded and crop_db is not None and crop_path.is_file():
            # This exact private scratch file was created/validated above.
            crop_path.unlink()
        if observations:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(dict(key=cache_key, observations=list(observations.values())), ensure_ascii=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--timezone', default='Europe/Saratov')
    parser.add_argument('--roi', default=Settings().CLOCK_ROI)
    parser.add_argument('--interval', type=float, default=5)
    parser.add_argument('--end-frame', type=int, help='Exclusive end; only exclude visually verified credits/non-camera tail')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    cfg = replace(Settings(), VIDEO_SOURCE=args.source, TIMEZONE=args.timezone, CLOCK_ROI=args.roi)
    try:
        build(cfg, args.output, interval=args.interval, end_frame=args.end_frame)
    except (ValueError, RuntimeError) as exc:
        parser.exit(2, f'Clock map error: {exc}\n')


if __name__ == '__main__':
    main()
