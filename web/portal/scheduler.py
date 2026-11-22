"""Single-host dispatcher: durable queue + bounded isolated worker processes."""
import logging
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
from threading import Event
import time

from django.conf import settings
from django.db import DatabaseError
from django.utils import timezone

from tools import archive_protocol as protocol
from tools.archive_resources import ProcessLock, can_start, cpu_capacity, memory_available_mb, rss_mb, cpu_times, idle_cores
from . import jobs
from .models import VideoUpload
from .uploads import paths

log = logging.getLogger(__name__)


class Scheduler:
    def __init__(self):
        self.root = Path(os.environ.get('ARCHIVE_STATE_DIR', settings.STATE_DIR / 'archive-jobs')).resolve()
        self.max_workers = int(os.environ.get('ARCHIVE_MAX_WORKERS', '2'))
        self.threads = int(os.environ.get('ARCHIVE_THREADS_PER_WORKER', '1'))
        self.memory_mb = int(os.environ.get('ARCHIVE_MEMORY_PER_WORKER_MB', '1400'))
        self.max_attempts = int(os.environ.get('ARCHIVE_MAX_ATTEMPTS', '3'))
        self.cpu = min(cpu_capacity(), float(os.environ.get('ARCHIVE_CPU_BUDGET', '3')))
        self.profile = os.environ.get('ARCHIVE_PROFILE', 'test2')
        self.backend = os.environ.get('ARCHIVE_DETECTOR_BACKEND', 'onnx')
        self.clock_library = os.environ.get('ARCHIVE_CLOCK_LIBRARY', '')
        self.stall_timeout = int(os.environ.get('ARCHIVE_STALL_TIMEOUT_SEC', '300'))
        self.clock_timeout = int(os.environ.get('ARCHIVE_CLOCK_TIMEOUT_SEC', '7200'))
        if (min(self.max_workers, self.threads, self.memory_mb, self.max_attempts) < 1
                or min(self.stall_timeout, self.clock_timeout) < 30
                or not math.isfinite(self.cpu) or self.cpu < self.threads or self.backend not in ('onnx', 'openvino')):
            raise ValueError('Invalid archive CPU/memory/worker configuration')
        self.active = {}
        self.stop = Event()
        self.last_cpu = cpu_times()

    def start(self, job):
        root = self.root / str(job.pk)
        attempt = root / str(job.token)
        token = str(job.token)
        spec = dict(root=str(root), attempt_dir=str(attempt), token=token, source=str(paths(job.upload)[1]),
                    bus=job.bus_id_snapshot, camera=job.camera_id_snapshot,
                    threads=self.threads, profile=self.profile, backend=self.backend, clock_library=self.clock_library)
        protocol.write(root / 'lease.json', dict(token=token, expires=time.monotonic()+60))
        protocol.write(attempt / 'spec.json', spec)
        stream = (attempt / 'worker.log').open('ab')
        env = dict(os.environ, OMP_NUM_THREADS=str(self.threads), MKL_NUM_THREADS=str(self.threads),
                   OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS=str(self.threads), PYTHONUNBUFFERED='1')
        try:
            process = subprocess.Popen([sys.executable, '-m', 'tools.archive_worker', '--spec', str(attempt / 'spec.json')],
                                       env=env, stdout=stream, stderr=subprocess.STDOUT, close_fds=True)
        finally:
            stream.close()
        self.active[job.pk] = dict(job=job, process=process, root=root, attempt=attempt,
                                  last_activity=time.monotonic(), seen_update=None, stall_since=None)
        log.info('Started archive job=%s attempt=%s pid=%s', job.pk, job.attempts, process.pid)

    def collect(self):
        for key, item in list(self.active.items()):
            job, process = item['job'], item['process']
            token = str(job.token)
            protocol.write(item['root'] / 'lease.json', dict(token=token, expires=time.monotonic()+60))
            progress = protocol.read(item['attempt'] / 'progress.json')
            fields = {}
            if progress.get('token') == token:
                if progress.get('updated') != item.get('seen_update'):
                    item['last_activity'] = time.monotonic()
                    item['seen_update'] = progress.get('updated')
                for name in ('processed_frame', 'total_frames', 'counted_in', 'counted_out'):
                    value = progress.get(name)
                    if type(value) is int and value >= 0:
                        fields[name] = value
                value = progress.get('processing_fps', 0)
                if isinstance(value, (float, int)) and math.isfinite(value) and value >= 0:
                    fields['processing_fps'] = value
                if progress.get('phase') in ('starting', 'clock', 'processing', 'syncing'):
                    fields['phase'] = progress['phase']
            jobs.update(job, heartbeat=timezone.now(), **fields)
            if process.poll() is None:
                timeout = self.clock_timeout if fields.get('phase') == 'clock' else self.stall_timeout
                if time.monotonic() - item.get('last_activity', time.monotonic()) > timeout:
                    if item.get('stall_since') is None:
                        item['stall_since'] = time.monotonic()
                        try:
                            process.terminate()
                        except ProcessLookupError:
                            pass
                    elif time.monotonic() - item['stall_since'] > 15:
                        process.kill()
                continue
            result = protocol.read(item['attempt'] / 'result.json')
            if result.get('token') != token or result.get('outcome') not in ('succeeded', 'review', 'failed', 'retry', 'interrupted'):
                result = dict(outcome='retry', error='Обработчик неожиданно завершился. Повторим с сохранённого состояния.')
            if item.get('stall_since') is not None:
                result = dict(outcome='retry', error='Обработчик долго не сообщал о прогрессе. Повторим с сохранённого состояния.')
            jobs.release(job, outcome=result['outcome'], error=result.get('error', ''), max_attempts=self.max_attempts)
            del self.active[key]
            log.info('Finished archive job=%s outcome=%s', key, result['outcome'])

    def tick(self):
        self.collect()
        jobs.recover_interrupted(active_ids=list(self.active))
        # Also pick up files uploaded before this feature was installed.
        for upload in VideoUpload.objects.select_related('bus').filter(status='ready', job__isnull=True)[:100]:
            jobs.enqueue(upload)
        available = memory_available_mb()
        current_cpu = cpu_times()
        free_cpu = idle_cores(self.last_cpu, current_cpu,
                             len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1)
        self.last_cpu = current_cpu
        while can_start([rss_mb(item['process'].pid) for item in self.active.values()],
                        max_workers=self.max_workers, threads=self.threads, memory_mb=self.memory_mb,
                        available_mb=available, cpu=self.cpu):
            if current_cpu and (free_cpu is None or free_cpu < self.threads * .75):
                break
            job = jobs.claim()
            if job is None:
                break
            try:
                self.start(job)
            except OSError:
                log.exception('Cannot spawn archive worker')
                jobs.release(job, outcome='retry', error='Не удалось запустить обработчик.', max_attempts=self.max_attempts)
                break
            available = memory_available_mb()
            if free_cpu is not None:
                free_cpu -= self.threads
        protocol.write(settings.STATE_DIR / 'dispatcher.json', dict(updated=time.time(), active=len(self.active),
            capacity=min(self.max_workers, math.floor(self.cpu/self.threads)), available_memory_mb=round(available)))

    def shutdown(self):
        for item in self.active.values():
            if item['process'].poll() is None:
                try:
                    item['process'].terminate()
                except ProcessLookupError:
                    pass
        deadline = time.monotonic()+45
        while self.active and time.monotonic() < deadline:
            try:
                self.collect()
            except (DatabaseError, OSError):
                pass
            time.sleep(.2)
        for item in self.active.values():
            process = item['process']
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)
            try:
                jobs.release(item['job'], outcome='interrupted', error='Служба остановлена. Продолжим с контрольной точки.')
            except DatabaseError:
                pass  # Next controller recovers this running row after taking the lifetime lock.

    def run(self, once=False):
        lock = ProcessLock(settings.STATE_DIR / 'archive-dispatcher.lock')
        handlers = {}
        try:
            jobs.recover_interrupted()
            for sig in (signal.SIGINT, signal.SIGTERM):
                handlers[sig] = signal.signal(sig, lambda *_: self.stop.set())
            log.info('Archive scheduler: CPU budget %.2f, max %s workers, %s threads and %s MB per worker',
                     self.cpu, self.max_workers, self.threads, self.memory_mb)
            while not self.stop.is_set():
                try:
                    self.tick()
                except (DatabaseError, OSError):
                    log.exception('Archive scheduler temporarily unavailable')
                if once:
                    break
                self.stop.wait(5)
        finally:
            self.shutdown()
            protocol.write(settings.STATE_DIR / 'dispatcher.json', dict(updated=0, active=0, capacity=0))
            lock.close()
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def dispatcher_status():
    result = protocol.read(settings.STATE_DIR / 'dispatcher.json')
    result['online'] = 0 <= time.time() - result.get('updated', 0) < 30
    return result
