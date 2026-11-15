"""Export montage contact sheets, then explicitly accept visually reviewed gaps."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from core.camera_clock import ClockMap, sha256


def review(source, map_path, output, *, accept=False, note=''):
    source, map_path, output = Path(source), Path(map_path), Path(output)
    payload = json.loads(map_path.read_text(encoding='utf-8'))
    if payload['source_sha256'] != sha256(source):
        raise ValueError('Review video differs from clock map')
    clock = ClockMap(payload, allow_unreviewed=True)
    transitions = [s for s in clock.segments if 'excluded_transition_start' in s]
    suspect = [r for r in payload.get('observations', []) if r.get('rejected_unconfirmed_jump') or r.get('chronology_review_required')]
    items = transitions + suspect
    if accept:
        if not note.strip():
            raise ValueError('Acceptance requires --note describing the visual review')
        for segment in items:
            segment['reviewed'] = True
            segment['review_note'] = note.strip()
        temporary = map_path.with_suffix(map_path.suffix + '.tmp')
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temporary.replace(map_path)
        return len(items)
    output.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source))
    rows = []
    try:
        for number, segment in enumerate(items):
            if 'excluded_transition_start' in segment:
                lo, hi = segment['excluded_transition_start'], segment['frame']
            else:
                lo, hi = max(1, segment['frame'] - round(payload['fps'])), min(clock.end_frame - 1, segment['frame'] + round(payload['fps']))
            tiles = []
            indices = (lo - 1, (lo + hi) // 2, hi)
            if 'excluded_start_pts' in segment:
                begin, finish = segment['excluded_start_pts'], segment['pts']
                times = (max(0., begin - 1 / payload['fps']), (begin + finish) / 2, finish)
            elif 'pts' in segment and 'text' in segment:
                times = (max(0., segment['pts'] - 1), segment['pts'], segment['pts'] + 1)
            else:
                times = (None, None, None)
            for index, target in zip(indices, times):
                if target is None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ok, frame = cap.read()
                else:
                    cap.set(cv2.CAP_PROP_POS_MSEC, max(0., target - 3) * 1000)
                    for attempt in range(round(payload['fps'] * 6) + 100):
                        ok, frame = cap.read()
                        if not ok or cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 >= target - .0001:
                            break
                    else:
                        raise ValueError('Cannot locate decoded PTS for review')
                if not ok:
                    raise ValueError('Cannot read transition review frame')
                tile = np.zeros((265, 320, 3), np.uint8)
                tile[:240] = cv2.resize(frame, (320, 240))
                cv2.putText(tile, f'cut {number + 1} / frame {index}', (5, 258), 0, .5, (255, 255, 255), 1)
                tiles.append(tile)
            rows.append(np.hstack(tiles))
        for offset in range(0, len(rows), 5):
            if not cv2.imwrite(str(output / f'transitions-{offset // 5 + 1:02d}.jpg'), np.vstack(rows[offset:offset + 5])):
                raise RuntimeError('Could not save transition contact sheet')
    finally:
        cap.release()
    return len(items)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--map', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path('data/clock_review'))
    parser.add_argument('--accept-transitions', action='store_true', help='Only after reviewing every transition as a synthetic dissolve without a passenger crossing')
    parser.add_argument('--note', default='')
    args = parser.parse_args()
    count = review(args.source, args.map, args.output, accept=args.accept_transitions, note=args.note)
    print(f'{count} transitions: ' + ('review recorded' if args.accept_transitions else f'contact sheets in {args.output}; inspect video if a crossing is visible'))


if __name__ == '__main__':
    main()
