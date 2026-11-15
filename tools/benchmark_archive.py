"""Compare the real file pipeline with a synthetic hourly gallery on disposable PG.

Only use --postgres-port for an isolated local test DB (test-local-only password).
The source video and every model must be local; no production data is modified.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch
from uuid import uuid4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-root', type=Path, required=True)
    parser.add_argument('--assets-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--start-frame', type=int, default=81271)
    parser.add_argument('--frames', type=int, default=600)
    parser.add_argument('--identities', type=int, default=128)
    parser.add_argument('--postgres-port', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if min(args.frames, args.identities, args.postgres_port) <= 0 or args.start_frame < 0:
        parser.error('Invalid benchmark dimensions')
    root, assets = args.code_root.resolve(), args.assets_root.resolve()
    sys.path.insert(0, str(root))
    from dataclasses import fields, replace
    import cv2
    import numpy as np
    from config.settings import Settings
    from config.profiles import apply_profile
    from core.counter_engine import Identity, PassengerCounterEngine
    from core.service import configuration_key, run
    from database.db_client import DatabaseClient
    from database.local_store import LocalStore

    bus = 'BENCH_' + uuid4().hex[:20]
    cfg = replace(apply_profile(Settings(), 'test2'), VIDEO_SOURCE=str(args.source.resolve()), SOURCE_KIND='file',
                  VIDEO_START='2026-11-01T05:39:57+04:00', DB_HOST='127.0.0.1', DB_PORT=args.postgres_port,
                  DB_NAME='bus_analytics', DB_USER='bus_counter', DB_PASS='test-local-only', CPU_THREADS=2,
                  MODEL_PATH=str(assets / 'models/candidates/person_test2_v2.onnx'),
                  REID_MODEL_PATH=str(assets / 'models/person_reid.onnx'),
                  VEST_MODEL_PATH=str(assets / 'models/candidates/work_vest_v3.onnx'),
                  STAFF_DECISION_DELAY_SEC=10, TRACKER_PATH=str(assets / 'botsort_reid.yaml'))
    if 'TIME_SOURCE' in {f.name for f in fields(cfg)}:
        cfg = replace(cfg, TIME_SOURCE='pts')
    cap = cv2.VideoCapture(str(args.source.resolve()))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    ok, _ = cap.read()
    if not ok:
        raise ValueError('Benchmark source frame unavailable')
    pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    cap.release()
    now = datetime.fromisoformat(cfg.VIDEO_START).timestamp() + pts - .1
    engine = PassengerCounterEngine(cfg)
    engine.day = '2026-11-01'
    engine.last_timestamp = now
    rng = np.random.default_rng(41)
    for index in range(args.identities):
        vectors = rng.normal(size=(9, 512)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        key = f'synthetic-{index}'
        engine.identities[key] = Identity(key, vectors[0].tolist(), now, gallery=vectors[1:].tolist())
    metrics = dict(checkpoints=0, checkpoint_bytes=0, local_saves=0, local_save_seconds=0.)

    class MeasuredDatabase(DatabaseClient):
        def deliver(self, events, checkpoint=None):
            if checkpoint is not None:
                metrics['checkpoints'] += 1
                encoded = checkpoint.get('payload_json') or json.dumps(checkpoint['payload'])
                metrics['checkpoint_bytes'] += len(encoded.encode())
            return super().deliver(events, checkpoint)

    class MeasuredStore(LocalStore):
        def save(self, *a, **kw):
            tick = time.perf_counter()
            result = super().save(*a, **kw)
            metrics['local_saves'] += 1
            metrics['local_save_seconds'] += time.perf_counter() - tick
            return result

    db = DatabaseClient('127.0.0.1', args.postgres_port, 'bus_analytics', 'bus_counter', 'test-local-only')
    source_key = None
    try:
        db.initialize()
        with tempfile.TemporaryDirectory(prefix='archive-bench-') as folder:
            cfg = replace(cfg, STATE_PATH=str(Path(folder) / 'runtime.sqlite3'))
            source_key = configuration_key(cfg)
            store = LocalStore(cfg.STATE_PATH)
            store.save(source_key, dict(bus_id=bus, camera_id=cfg.CAMERA_ID, engine=engine.snapshot(),
                                        next_frame=args.start_frame, completed=False))
            store.close()
            tick = time.perf_counter()
            with patch('core.service.DatabaseClient', MeasuredDatabase), patch('core.service.LocalStore', MeasuredStore):
                result = run(cfg, max_frames=args.frames)
            elapsed = time.perf_counter() - tick
            store = LocalStore(cfg.STATE_PATH)
            state = store.load()['payload']
            store.close()
        report = dict(format='archive_pipeline_benchmark_v1',
                      source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip(),
                      runtime_diff_sha256=hashlib.sha256(subprocess.check_output(['git', 'diff', '--', 'core', 'config', 'database'], cwd=root)).hexdigest(),
                      source_sha256=hashlib.file_digest(args.source.open('rb'), 'sha256').hexdigest(),
                      start_frame=args.start_frame, synthetic_identities=args.identities, result=result,
                      elapsed_seconds=elapsed, processed_fps=result['frames'] / elapsed,
                      next_frame=state['next_frame'], final_identity_count=len(state['engine']['identities']), **metrics)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(report, indent=2))
    finally:
        with db.get_cursor() as cur:
            for table in ('raw_passenger_events', 'daily_passenger_traffic', 'current_day_passenger_traffic'):
                cur.execute(f'DELETE FROM {table} WHERE bus_id=%s', (bus,))
            if source_key:
                cur.execute('DELETE FROM counter_runtime_state WHERE source_key=%s', (source_key,))
        db.close()


if __name__ == '__main__':
    main()
