"""Reproducible, opt-in CPU comparison. Never activates or replaces working weights.

Use a private output directory: detailed evaluation traces contain person data.
OpenVINO is optional; the production requirements do not import it.
"""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time

import cv2
import numpy as np

from config.settings import ROOT, Settings
from config.profiles import apply_profile
from core.camera_clock import write_json_atomic
from core.onnx_runtime import cpu_options
from tools.cpu_backends import OpenVINOSession, Timings, instrument


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def environment():
    versions = {}
    for name in ('numpy', 'onnxruntime', 'openvino', 'opencv-python', 'ultralytics', 'torch'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    cpu = platform.processor()
    if Path('/proc/cpuinfo').is_file():
        cpu = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                    if line.startswith('model name')), cpu)
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                         stderr=subprocess.DEVNULL).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return dict(python=platform.python_version(), os=platform.platform(), cpu=cpu,
                logical_cpus=os.cpu_count(), versions=versions, commit=commit,
                cpu_quota=Path('/sys/fs/cgroup/cpu.max').read_text().strip()
                          if Path('/sys/fs/cgroup/cpu.max').is_file() else None,
                code_sha256={name: digest(ROOT / name) for name in
                             ('tools/benchmark_cpu.py', 'tools/cpu_backends.py', 'core/detector.py',
                              'core/reid.py', 'core/counter_engine.py', 'core/service.py', 'core/openvino_session.py')},
                recorded_at_utc=datetime.now(timezone.utc).isoformat())


def input_tensor(image):
    from ultralytics.data.augment import LetterBox
    image = LetterBox((640, 640), auto=True, stride=32)(image=image)
    return np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255


def prepare(args):
    """Calibrate only on an explicit training directory, never a video test interval."""
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process
    images = args.train_images.resolve()
    if images.name != 'train' or not images.is_dir() or args.samples < 1:
        raise ValueError('--train-images must be an existing directory named train')
    paths = sorted(p for p in images.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    random.Random(42).shuffle(paths)
    paths = paths[:args.samples]
    if len(paths) < min(args.samples, 32):
        raise ValueError('Too few training images for the requested calibration')
    output = args.output.resolve()
    if output.exists() or output.with_suffix('.json').exists():
        raise FileExistsError('Choose a new candidate output')
    if output == args.model.resolve() or output.suffix != '.onnx':
        raise ValueError('Use a separate .onnx candidate')
    output.parent.mkdir(parents=True, exist_ok=True)
    preprocessed = output.with_suffix('.preprocessed.onnx')
    if preprocessed.exists():
        raise FileExistsError(preprocessed)
    quant_pre_process(str(args.model), str(preprocessed), skip_symbolic_shape=True)
    import onnxruntime as ort
    session = ort.InferenceSession(str(preprocessed), sess_options=cpu_options(1), providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    del session

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.rows = iter(paths)

        def get_next(self):
            path = next(self.rows, None)
            if path is None:
                return None
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError('Cannot read calibration image: ' + path.name)
            return {input_name: input_tensor(image)}

    tick = time.perf_counter()
    # Per-channel reduced-range weights also cover AVX2/non-VNNI machines.
    quantize_static(str(preprocessed), str(output), Reader(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    per_channel=True, reduce_range=True, op_types_to_quantize=['Conv'],
                    extra_options={'ActivationSymmetric': False, 'WeightSymmetric': True,
                                   'CalibMaxIntermediateOutputs': 8})
    preprocessed.unlink()
    result = dict(format='cpu_int8_candidate_v1', activated=False, precision='INT8 QDQ Conv, per-channel reduced-range',
                  source_sha256=digest(args.model), candidate_sha256=digest(output),
                  calibration_seed=42, calibration_images=[dict(file=p.relative_to(images).as_posix(), sha256=digest(p)) for p in paths],
                  calibration_split='train only; no validation/test labels or frames used',
                  elapsed_seconds=time.perf_counter() - tick, environment=environment())
    write_json_atomic(output.with_suffix('.json'), result)
    return result


def make_session(backend, model, threads):
    if backend == 'openvino':
        return OpenVINOSession(model, threads)
    import onnxruntime as ort
    return ort.InferenceSession(str(model), sess_options=cpu_options(threads), providers=['CPUExecutionProvider'])


def probe(args):
    """Fixed-input screening; not a full video throughput or accuracy measurement."""
    cv2.setNumThreads(1)
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.inputs, allow_pickle=False) as rows:
        tensors = [rows[k] for k in sorted(rows.files)]
    if not tensors or any(t.ndim != 4 or t.shape[:2] != (1, 3) for t in tensors):
        raise ValueError('Expected an NPZ containing [1,3,H,W] float tensors')
    tick = time.perf_counter()
    session = make_session(args.backend, args.model, args.threads)
    setup = time.perf_counter() - tick
    name = session.get_inputs()[0].name
    for tensor in tensors:
        session.run(None, {name: tensor})
    times = []
    for i in range(args.repeats):
        tick = time.perf_counter()
        output = session.run(None, {name: tensors[i % len(tensors)]})
        times.append(time.perf_counter() - tick)
        if any(not np.isfinite(value).all() for value in output):
            raise ValueError('Nonfinite candidate output')
    result = dict(format='cpu_detector_probe_v1', backend=args.backend, threads=args.threads,
                  model_sha256=digest(args.model), input_sha256=digest(args.inputs),
                  iterations=len(times), setup_seconds=setup, mean_ms=float(np.mean(times) * 1000),
                  median_ms=float(np.median(times) * 1000), p95_ms=float(np.percentile(times, 95) * 1000),
                  environment=environment(), scope='Detector only, warmed; no decoding/ReID/tracker/DB, no accuracy claim')
    write_json_atomic(args.output, result)
    return result


def evaluate(args):
    from tools.evaluate_clip import evaluate as evaluate_clip
    if args.output.exists() or args.output.with_suffix('.timing.json').exists():
        raise FileExistsError(args.output)
    cfg = replace(apply_profile(Settings(), 'test2'), MODEL_PATH=str(args.model.resolve()),
                  VIDEO_SOURCE=str(args.source.resolve()), CLOCK_MAP_PATH=str(args.clock_map.resolve()),
                  CPU_THREADS=args.threads, TIME_SOURCE='camera', FRAME_STRIDE=1,
                  STAFF_MODE=args.staff_mode, CPU_DETECTOR_BACKEND=args.backend,
                  DETECTOR_CPU_THREADS=args.detector_threads or 0)
    timings = Timings()
    with instrument(args.backend, timings, args.detector_threads):
        tick = time.perf_counter()
        result = evaluate_clip(cfg, args.start, args.end, args.output, compact=args.compact)
        total = time.perf_counter() - tick
    counts = {}
    for event in result['events']:
        key = event['event_type'] + ':' + event['reason']
        counts[key] = counts.get(key, 0) + 1
    profile = dict(format='cpu_pipeline_profile_v1', backend=args.backend, threads=args.threads,
                   detector_threads=args.detector_threads or args.threads,
                   source_sha256=digest(args.source), model_sha256=digest(args.model),
                   clock_sha256=digest(args.clock_map), start=args.start, end=args.end,
                   frames=result['frames'], in_count=result['in'], out_count=result['out'],
                   pipeline_seconds=result['elapsed_sec'], inclusive_wall_seconds=total,
                   stages=timings.report(), event_reasons=counts, environment=environment(),
                   scope='Video + ML + tracking/counting, no PostgreSQL. Inclusive stages overlap.',
                   activated=False)
    write_json_atomic(args.output.with_suffix('.timing.json'), profile)
    return profile


def matrix(args):
    """One subprocess per file; compare total batch throughput, not per-file FPS."""
    if args.output.exists():
        raise FileExistsError(args.output)
    jobs = json.loads(args.jobs.read_text('utf8'))
    if not isinstance(jobs, list) or not jobs:
        raise ValueError('Jobs must be a nonempty JSON array')
    from concurrent.futures import ThreadPoolExecutor
    output_dir = args.output.parent / (args.output.stem + '-jobs')
    output_dir.mkdir(parents=True, exist_ok=False)

    def one(item):
        index, job = item
        report = output_dir / f'{index:02d}.json'
        cmd = [os.sys.executable, '-m', 'tools.benchmark_cpu', 'evaluate',
               '--source', job['source'], '--clock-map', job['clock_map'],
               '--start', str(job['start']), '--end', str(job['end']),
               '--model', str(args.model), '--backend', args.backend,
               '--threads', str(args.threads), '--output', str(report), '--compact']
        if args.detector_threads is not None:
            cmd += ['--detector-threads', str(args.detector_threads)]
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
        with report.with_suffix('.log').open('w', encoding='utf8') as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        return json.loads(report.with_suffix('.timing.json').read_text('utf8'))

    tick = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(one, enumerate(jobs)))
    elapsed = time.perf_counter() - tick
    result = dict(format='cpu_batch_throughput_v1', workers=args.workers, threads_per_worker=args.threads,
                  detector_threads=args.detector_threads or args.threads,
                  elapsed_seconds=elapsed, frames=sum(row['frames'] for row in rows),
                  aggregate_fps=sum(row['frames'] for row in rows) / elapsed,
                  jobs=rows, environment=environment(), postgres=False)
    write_json_atomic(args.output, result)
    return result


def profile_database(args):
    """Real persistence timing, restricted to an explicitly disposable local DB."""
    import tempfile
    from uuid import uuid4
    from core.camera_clock import ClockMap
    from core.counter_engine import Identity, PassengerCounterEngine
    from core.service import configuration_key, run
    from database.db_client import DatabaseClient
    from database.local_store import LocalStore
    if args.postgres_port == 5432 or not 1024 <= args.postgres_port <= 65535:
        raise ValueError('Use an explicit nonstandard disposable PostgreSQL port')
    if args.output.exists() or min(args.frames, args.identities) < 1 or args.start < 0:
        raise ValueError('Invalid dimensions or existing output')
    bus = 'CPU_BENCH_' + uuid4().hex[:16]
    cfg = replace(apply_profile(Settings(), 'test2'), MODEL_PATH=str(args.model.resolve()),
                  VIDEO_SOURCE=str(args.source.resolve()), CLOCK_MAP_PATH=str(args.clock_map.resolve()),
                  CPU_THREADS=args.threads, TIME_SOURCE='camera', FRAME_STRIDE=1,
                  CPU_DETECTOR_BACKEND=args.backend, DETECTOR_CPU_THREADS=args.detector_threads or 0,
                  DB_HOST='127.0.0.1', DB_PORT=args.postgres_port, DB_NAME='bus_analytics',
                  DB_USER='bus_counter', DB_PASS='test-local-only', BUS_ID=bus)
    clock = ClockMap.load(cfg.CLOCK_MAP_PATH, cfg)
    stamp, _ = clock.at(args.start, args.start / clock.payload['fps'])
    now = stamp.timestamp() - 1
    engine = PassengerCounterEngine(cfg)
    rng = np.random.default_rng(41)
    for i in range(args.identities):
        vectors = rng.normal(size=(9, 512)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        key = f'synthetic-{i}'
        engine.identities[key] = Identity(key, vectors[0].tolist(), now, gallery=vectors[1:].tolist())
    timings = Timings()
    db = DatabaseClient(cfg.DB_HOST, cfg.DB_PORT, cfg.DB_NAME, cfg.DB_USER, cfg.DB_PASS)
    source_key = None
    try:
        db.initialize()
        with tempfile.TemporaryDirectory(prefix='cpu-profile-') as folder:
            cfg = replace(cfg, STATE_PATH=str(Path(folder) / 'runtime.sqlite3'))
            source_key = configuration_key(cfg)
            store = LocalStore(cfg.STATE_PATH)
            try:
                store.save(source_key, dict(bus_id=bus, camera_id=cfg.CAMERA_ID,
                           engine=engine.snapshot(), next_frame=args.start, completed=False))
            finally:
                store.close()
            tick = time.perf_counter()
            with instrument(args.backend, timings, args.detector_threads):
                result = run(cfg, max_frames=args.frames)
            elapsed = time.perf_counter() - tick
        result = dict(format='cpu_persistent_profile_v1', result=result, elapsed_seconds=elapsed,
                      synthetic_identities=args.identities, stages=timings.report(),
                      backend=args.backend, threads=args.threads,
                      detector_threads=args.detector_threads or args.threads, environment=environment(),
                      model_sha256=digest(args.model), source_sha256=digest(args.source),
                      clock_sha256=digest(args.clock_map), start=args.start, postgres=True,
                      scope='Actual archive pipeline + isolated PostgreSQL; synthetic gallery, no independent accuracy claim')
        write_json_atomic(args.output, result)
        return result
    finally:
        try:
            with db.get_cursor() as cur:
                for table in ('raw_passenger_events', 'daily_passenger_traffic', 'current_day_passenger_traffic'):
                    cur.execute(f'DELETE FROM {table} WHERE bus_id=%s', (bus,))
                if source_key:
                    cur.execute('DELETE FROM counter_runtime_state WHERE source_key=%s', (source_key,))
        finally:
            db.close()


def controls(args):
    """Quality-only suite: reuse models, reset all tracking/counting per episode.

    Reuse avoids recompiling OpenVINO for every short clip. The resulting times
    must not be compared against cold, one-process-per-file throughput runs.
    """
    from unittest.mock import patch
    from core.camera_clock import ClockMap
    from core.video_source import VideoSource
    from tools.cpu_backends import detector_factory
    from tools.evaluate_clip import evaluate as evaluate_clip
    if args.output.exists() or not 0 <= args.shard_index < args.shards:
        raise ValueError('Existing output or invalid shard')
    plan = json.loads(args.plan.read_text('utf8'))['episodes']
    selected = [row for i, row in enumerate(plan) if i % args.shards == args.shard_index]
    if not selected:
        raise ValueError('This shard contains no controls')
    cfg = replace(apply_profile(Settings(), 'test2'), MODEL_PATH=str(args.model.resolve()),
                  CPU_THREADS=args.threads, FRAME_STRIDE=1, TIME_SOURCE='camera', STAFF_MODE=args.staff_mode,
                  CPU_DETECTOR_BACKEND=args.backend, DETECTOR_CPU_THREADS=args.detector_threads or 0)
    detector = detector_factory(args.backend, detector_threads=args.detector_threads)(cfg)
    folder = args.output.parent / args.output.stem
    folder.mkdir(parents=True, exist_ok=False)
    clocks, results = {}, []
    real_predict, real_encode = detector.model.predict, detector.reid.encode_many

    def reader(cfg, *a, **kw):
        source, clock = cfg.path(cfg.VIDEO_SOURCE), cfg.path(cfg.CLOCK_MAP_PATH)
        key = (str(source), source.stat().st_size, source.stat().st_mtime_ns,
               str(clock), clock.stat().st_size, clock.stat().st_mtime_ns)
        if key not in clocks:
            clocks[key] = ClockMap.load(clock, cfg)
        return VideoSource(cfg, *a, **kw, clock_map=clocks[key])

    try:
        for row in selected:
            source = Path(row['source'])
            if source.parent == Path('.'):
                source = Path('data') / source
            current = replace(cfg, VIDEO_SOURCE=str(source),
                              CLOCK_MAP_PATH=str(source.with_suffix('.clock.json')))
            detector.reset()
            detector.cfg = current
            try:
                with patch('tools.evaluate_clip.PersonDetector', lambda *a, **kw: detector), \
                     patch('tools.evaluate_clip.VideoSource', reader):
                    result = evaluate_clip(current, row['start'], row['end'],
                                           folder / (row['id'] + '.json'), compact=True)
                    if args.backend == 'openvino' and not isinstance(detector.model.predictor.model.backend.session, OpenVINOSession):
                        raise RuntimeError('Quality report cannot claim OpenVINO without an active OpenVINO session')
            finally:
                # evaluate_clip wraps these methods for its own timings.
                detector.model.predict, detector.reid.encode_many = real_predict, real_encode
            expected, actual = [row['manual_in'], row['manual_out']], [result['in'], result['out']]
            reasons = sorted({event['reason'] for event in result['events']})
            passed = (expected == actual and (not row.get('required_reason') or row['required_reason'] in reasons)
                      and (not row.get('forbidden_reason') or row['forbidden_reason'] not in reasons))
            results.append(dict(case=row['id'], expected=expected, actual=actual, passed=passed,
                                frames=result['frames'], reasons=reasons))
            report = dict(format='cpu_quality_controls_v1', complete=len(results) == len(selected),
                          backend=args.backend, detector_threads=args.detector_threads or args.threads,
                          other_model_threads=args.threads, staff_mode=args.staff_mode,
                          plan_sha256=digest(args.plan), model_sha256=digest(args.model),
                          shard_index=args.shard_index, shards=args.shards, results=results,
                          scope='Quality only; models reused, engine and tracker reset between episodes')
            write_json_atomic(args.output, report)
            print(f"CONTROL {row['id']} expected={expected} actual={actual} passed={passed}", flush=True)
    finally:
        detector.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--train-images', required=True, type=Path)
    p.add_argument('--samples', default=160, type=int)
    p.set_defaults(function=prepare)
    p = commands.add_parser('probe')
    p.add_argument('--inputs', required=True, type=Path)
    p.add_argument('--repeats', default=60, type=int)
    p.set_defaults(function=probe)
    p = commands.add_parser('evaluate')
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--clock-map', required=True, type=Path)
    p.add_argument('--start', required=True, type=int)
    p.add_argument('--end', required=True, type=int)
    p.add_argument('--compact', action='store_true')
    p.add_argument('--staff-mode', choices=['disabled', 'model'], default='model')
    p.set_defaults(function=evaluate)
    p = commands.add_parser('matrix')
    p.add_argument('--jobs', required=True, type=Path)
    p.add_argument('--workers', type=int, choices=[1, 2], required=True)
    p.set_defaults(function=matrix)
    p = commands.add_parser('profile-db', help='Only an isolated local PostgreSQL with test-local-only password')
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--clock-map', required=True, type=Path)
    p.add_argument('--start', type=int, default=81271)
    p.add_argument('--frames', type=int, default=300)
    p.add_argument('--identities', type=int, default=128)
    p.add_argument('--postgres-port', type=int, required=True)
    p.set_defaults(function=profile_database)
    p = commands.add_parser('controls', help='Known quality controls, reusing ML sessions; no throughput claim')
    p.add_argument('--plan', type=Path, default=ROOT / 'reports/test2-2026-11-13/regression_plan.json')
    p.add_argument('--staff-mode', choices=['disabled', 'model'], default='disabled')
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--shard-index', type=int, default=0)
    p.set_defaults(function=controls)
    for name, p in commands.choices.items():
        p.add_argument('--model', type=Path, default=ROOT / 'models/candidates/person_test2_v2.onnx')
        p.add_argument('--output', type=Path, required=True)
        if name != 'prepare':
            p.add_argument('--backend', choices=['onnx', 'openvino'], default='onnx')
            p.add_argument('--threads', type=int, choices=[1, 2, 4], default=2)
            if name != 'probe':
                p.add_argument('--detector-threads', type=int, choices=[1, 2, 4],
                               help='Override only the person detector; ReID/vest keep --threads')
    args = parser.parse_args()
    if getattr(args, 'repeats', 1) < 1:
        parser.error('repeats must be positive')
    result = args.function(args)
    print(json.dumps(result, indent=2), flush=True)
    if args.command == 'controls' and any(not row['passed'] for row in result['results']):
        parser.exit(1, 'Quality controls changed; keep the working backend unchanged\n')


if __name__ == '__main__':
    main()
