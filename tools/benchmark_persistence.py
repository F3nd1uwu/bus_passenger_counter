"""Measure checkpoint overhead with synthetic identities and an optional disposable DB.

No models or passenger recordings are loaded. --postgres-port must point only at
an isolated local test PostgreSQL using the test credentials, never production.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from uuid import uuid4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--identities', type=int, default=128)
    parser.add_argument('--gallery-size', type=int, default=8)
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--idle-syncs', type=int, default=5)
    parser.add_argument('--postgres-port', type=int)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.identities, args.iterations, args.idle_syncs) <= 0 or args.gallery_size < 0:
        parser.error('Positive sample counts and a nonnegative gallery size are required')
    root = args.code_root.resolve()
    sys.path.insert(0, str(root))
    import numpy as np
    from config.settings import Settings
    from core.counter_engine import Identity, PassengerCounterEngine
    from core.service import synchronize
    from database.local_store import LocalStore
    from database.db_client import DatabaseClient

    generator = np.random.default_rng(41)
    engine = PassengerCounterEngine(Settings())
    engine.day = '2026-11-15'
    for index in range(args.identities):
        vectors = generator.normal(size=(args.gallery_size + 1, 512)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        identity = Identity(f'person-{index}', vectors[0].tolist(), 1000.0, last_in=999.0,
                            gallery=vectors[1:].tolist())
        engine.identities[identity.identity_id] = identity
    timings = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        state = engine.snapshot()
        timings.append(time.perf_counter() - start)
    state_json = json.dumps(state, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    result = {'code_root': str(root), 'source_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=root).decode().strip(),
              'working_tree_diff_sha256': hashlib.sha256(subprocess.check_output(['git','diff','--','core','database'],cwd=root)).hexdigest(),
              'benchmark_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'python': platform.python_version(), 'platform': platform.platform(),
              'logical_cpus': os.cpu_count(), 'sqlite_version': sqlite3.sqlite_version,
              'identities': args.identities, 'dimensions': 512, 'gallery_size': args.gallery_size,
              'iterations': args.iterations, 'idle_syncs': args.idle_syncs,
              'engine_state_sha256': hashlib.sha256(state_json).hexdigest(),
              'snapshot_seconds': timings, 'snapshot_median_seconds': statistics.median(timings)}

    class MeasuredDatabase(DatabaseClient):
        def __init__(self):
            super().__init__('127.0.0.1', args.postgres_port, 'bus_analytics', 'bus_counter', 'test-local-only')
            self.calls = self.checkpoints = self.json_bytes = 0

        def deliver(self, events, checkpoint=None):
            self.calls += 1
            if checkpoint is not None:
                self.checkpoints += 1
                encoded = checkpoint.get('payload_json')
                if encoded is None:
                    encoded = json.dumps(checkpoint['payload'])  # psycopg2 Json default representation
                self.json_bytes += len(encoded.encode('utf-8'))
            super().deliver(events, checkpoint)

        def wal_position(self):
            with self.get_cursor() as cur:
                cur.execute('SELECT pg_current_wal_insert_lsn()::text')
                return cur.fetchone()[0]

        def wal_difference(self, before):
            with self.get_cursor() as cur:
                cur.execute('SELECT pg_wal_lsn_diff(pg_current_wal_insert_lsn(), %s::pg_lsn)', (before,))
                return int(cur.fetchone()[0])

    tag = uuid4().hex
    bus = 'TEST_BENCH_' + tag[:16]
    payload = {'engine': state, 'bus_id': bus, 'camera_id': 'BENCH', 'next_frame': 50000, 'completed': False}
    source_key = 'benchmark-' + tag
    temporary = root / '.local'
    temporary.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='persistence-bench-', dir=temporary) as folder:
        store = LocalStore(Path(folder) / 'runtime.sqlite3')
        database = None
        try:
            store.save(source_key, payload)
            save_times = []
            for _ in range(args.iterations):
                start = time.perf_counter()
                store.save(source_key, payload)
                save_times.append(time.perf_counter() - start)
            result.update(save_seconds=save_times, save_median_seconds=statistics.median(save_times),
                          checkpoint_json_bytes=len(json.dumps(payload,separators=(',',':')).encode()))
            if args.postgres_port:
                database = MeasuredDatabase()
                database.initialize()
                sync_times = []
                for _ in range(args.iterations):
                    store.save(source_key, payload)
                    start = time.perf_counter()
                    synchronize(store, database)
                    sync_times.append(time.perf_counter()-start)
                result.update(sync_seconds=sync_times, sync_median_seconds=statistics.median(sync_times))
                before = database.wal_position()
                calls, checkpoints, json_bytes = database.calls, database.checkpoints, database.json_bytes
                start = time.perf_counter()
                for _ in range(args.idle_syncs):
                    synchronize(store, database)
                result.update(idle_sync_seconds=time.perf_counter()-start,
                              idle_database_calls=database.calls-calls,
                              idle_checkpoints_sent=database.checkpoints-checkpoints,
                              idle_checkpoint_json_bytes=database.json_bytes-json_bytes,
                              idle_wal_bytes=database.wal_difference(before))
                assert database.load_checkpoint(source_key)['payload'] == payload
                assert store.load()['payload'] == payload
                result['roundtrip_equal'] = True
        finally:
            store.close()
            if database is not None:
                try:
                    with database.get_cursor() as cur:
                        for table in ['daily_passenger_traffic','current_day_passenger_traffic']:
                            cur.execute(f'DELETE FROM {table} WHERE bus_id=%s', (bus,))
                        cur.execute('DELETE FROM counter_runtime_state WHERE source_key=%s', (source_key,))
                finally:
                    database.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
