"""Queue and admission tests, without inference or production databases."""
from datetime import timedelta
import tempfile
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from django.test import TestCase, override_settings
from django.utils import timezone

from tools import archive_protocol as protocol
from tools.archive_resources import ProcessLock, can_start, idle_cores
from .models import User, Bus, VideoUpload, ProcessingJob
from . import jobs


@override_settings(STORAGES={'staticfiles': {'BACKEND': 'django.contrib.staticfiles.storage.StaticFilesStorage'}})
class QueueTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('employee')
        self.other = User.objects.create_user('other')
        self.admin = User.objects.create_user('admin', role='admin')
        self.bus = Bus.objects.create(bus_id='BUS_A', name='Автобус А')
        self.bus.users.add(self.user)
        self.client.force_login(self.user)

    def upload(self, camera='FRONT'):
        return VideoUpload.objects.create(owner=self.user, bus=self.bus, camera_id=camera,
                   original_name='sample.mp4', size=8, received=8, fingerprint='a'*64, status='ready')

    def test_enqueue_is_idempotent_and_rejects_partial(self):
        upload = self.upload()
        self.assertEqual(jobs.enqueue(upload).pk, jobs.enqueue(upload).pk)
        upload.status = 'uploading'
        with self.assertRaises(ValueError):
            jobs.enqueue(upload)

    def test_claim_is_fifo_and_same_camera_cannot_run_twice(self):
        first = jobs.enqueue(self.upload())
        second = jobs.enqueue(self.upload())
        third = jobs.enqueue(self.upload('BACK'))
        self.assertEqual(jobs.claim().pk, first.pk)
        self.assertEqual(jobs.claim().pk, third.pk)
        self.assertIsNone(jobs.claim())
        jobs.release(ProcessingJob.objects.get(pk=first.pk), outcome='succeeded')
        self.assertEqual(jobs.claim().pk, second.pk)

    def test_claim_preserves_bus_camera_snapshot(self):
        job = jobs.enqueue(self.upload())
        self.bus.bus_id = 'NEW_BUS'
        self.bus.save()
        self.assertEqual(jobs.claim().bus_id_snapshot, 'BUS_A')

    def test_stale_attempt_cannot_update_or_complete_new_attempt(self):
        jobs.enqueue(self.upload())
        old = jobs.claim()
        jobs.recover_interrupted()
        new = jobs.claim()
        self.assertNotEqual(old.token, new.token)
        self.assertEqual(jobs.update(old, counted_in=999), 0)
        self.assertEqual(jobs.release(old, outcome='succeeded'), 0)
        new.refresh_from_db()
        self.assertEqual(new.status, 'running')

    def test_retry_delay_limit_and_shutdown_recovery(self):
        jobs.enqueue(self.upload())
        job = jobs.claim()
        jobs.release(job, outcome='retry', max_attempts=3)
        self.assertIsNone(jobs.claim())
        ProcessingJob.objects.filter(pk=job.pk).update(available_at=timezone.now()-timedelta(seconds=1), attempts=2)
        job = jobs.claim()
        jobs.release(job, outcome='retry', max_attempts=3)
        job.refresh_from_db()
        self.assertEqual(job.status, 'failed')
        ProcessingJob.objects.filter(pk=job.pk).update(status='queued', available_at=timezone.now())
        job = jobs.claim()
        jobs.release(job, outcome='interrupted', max_attempts=3)
        job.refresh_from_db()
        self.assertEqual(job.status, 'retry')

    def test_review_never_restarts_automatically(self):
        jobs.enqueue(self.upload())
        job = jobs.claim()
        jobs.release(job, outcome='review', error='Проверьте время')
        self.assertIsNone(jobs.claim())

    def test_normal_restarts_do_not_exhaust_failure_budget(self):
        job = jobs.enqueue(self.upload())
        for _ in range(5):
            ProcessingJob.objects.filter(pk=job.pk).update(available_at=timezone.now())
            job = jobs.claim()
            jobs.release(job, outcome='interrupted', max_attempts=3)
        job.refresh_from_db()
        self.assertEqual(job.status, 'retry')
        self.assertEqual(job.attempts, 0)

    def test_stale_unspawned_claim_is_recovered_without_touching_active_worker(self):
        first = jobs.enqueue(self.upload('FRONT'))
        second = jobs.enqueue(self.upload('BACK'))
        jobs.claim()
        jobs.claim()
        ProcessingJob.objects.update(heartbeat=timezone.now()-timedelta(seconds=120))
        jobs.recover_interrupted(active_ids=[second.pk])
        first.refresh_from_db()
        second.refresh_from_db()
        self.assertEqual(first.status, 'retry')
        self.assertEqual(second.status, 'running')

    def test_status_endpoint_is_scoped_and_revocation_applies(self):
        upload = self.upload()
        jobs.enqueue(upload)
        url = f'/videos/status/?ids={upload.pk}'
        self.assertEqual(len(self.client.get(url).json()['jobs']), 1)
        self.client.force_login(self.other)
        self.assertEqual(self.client.get(url).json()['jobs'], [])
        self.client.force_login(self.user)
        self.bus.users.clear()
        self.assertEqual(self.client.get(url).json()['jobs'], [])
        self.assertEqual(self.client.get('/videos/status/?ids=not-a-uuid').status_code, 400)

    def test_clock_estimation_note_is_visible_on_owned_record_and_status(self):
        upload = self.upload()
        job = jobs.enqueue(upload)
        job.clock_note = 'Время восстановлено по видео; на склейках возможна погрешность.'
        job.save()
        status = self.client.get(f'/videos/status/?ids={upload.pk}').json()['jobs'][0]
        self.assertEqual(status['clock_note'], job.clock_note)
        self.assertContains(self.client.get(f'/videos/{upload.pk}/'), job.clock_note)
        self.client.force_login(self.other)
        self.assertEqual(self.client.get(f'/videos/status/?ids={upload.pk}').json()['jobs'], [])

    def test_only_admin_can_retry_and_running_completed_jobs_are_protected(self):
        upload = self.upload()
        job = jobs.enqueue(upload)
        url = f'/videos/{upload.pk}/retry/'
        self.assertEqual(self.client.post(url).status_code, 403)
        self.client.force_login(self.admin)
        self.assertEqual(self.client.post(url).status_code, 409)
        job.status = 'review'
        job.save()
        self.assertEqual(self.client.post(url).status_code, 302)
        job.refresh_from_db()
        self.assertEqual(job.status, 'queued')
        job.status = 'succeeded'
        job.save()
        self.assertEqual(self.client.post(url).status_code, 409)

    def test_ui_does_not_display_full_completion_before_database_commit(self):
        job = jobs.enqueue(self.upload())
        job.status, job.processed_frame, job.total_frames = 'running', 100, 100
        self.assertEqual(job.progress_percent, 99)
        job.status = 'succeeded'
        self.assertEqual(job.progress_percent, 100)

    def test_record_detail_csv_and_revocation(self):
        upload = self.upload()
        upload.original_name = '=formula.mp4'
        upload.save()
        job = jobs.enqueue(upload)
        detail, download = f'/videos/{upload.pk}/', f'/videos/{upload.pk}/result.csv'
        self.assertEqual(self.client.get(detail).status_code, 200)
        self.assertEqual(self.client.get(download).status_code, 409)
        job.status, job.counted_in, job.finished_at = 'succeeded', 7, timezone.now()
        job.save()
        result = self.client.get(download)
        self.assertEqual(result.status_code, 200)
        self.assertTrue(result.content.startswith(b'\xef\xbb\xbf'))
        self.assertIn("'=formula.mp4", result.content.decode('utf-8-sig'))
        self.client.force_login(self.other)
        self.assertEqual(self.client.get(detail).status_code, 404)
        self.assertEqual(self.client.get(download).status_code, 404)
        self.client.force_login(self.user)
        self.bus.users.clear()
        self.assertEqual(self.client.get(download).status_code, 404)

    def test_cpu_ram_admission_reserves_starting_processes(self):
        limits = dict(max_workers=2, threads=1, memory_mb=1400, available_mb=4000, cpu=3)
        self.assertTrue(can_start([], **limits))
        self.assertTrue(can_start([100], **limits))
        self.assertFalse(can_start([100, 100], **limits))
        self.assertFalse(can_start([100], **{**limits, 'available_mb': 2000}))
        self.assertFalse(can_start([], **{**limits, 'cpu': .5}))
        self.assertFalse(can_start([], **{**limits, 'available_mb': 1000}))
        self.assertEqual(idle_cores((100, 20), (200, 45), 4), 1)
        self.assertIsNone(idle_cores((100, 20), (100, 20), 4))

    def test_process_lock_and_lease_tokens(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'worker.lock'
            lock = ProcessLock(path)
            try:
                with self.assertRaises(RuntimeError):
                    ProcessLock(path)
            finally:
                lock.close()
            other = ProcessLock(path)
            other.close()
            lease = Path(folder)/'lease.json'
            protocol.write(lease, dict(token='new', expires=1e20))
            self.assertFalse(protocol.lease_valid(lease, 'old'))
            self.assertTrue(protocol.lease_valid(lease, 'new'))
            protocol.write(lease, dict(token='new', expires=0))
            self.assertFalse(protocol.lease_valid(lease, 'new'))

    def test_collection_ignores_result_from_another_attempt(self):
        from .scheduler import Scheduler
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as folder, patch.dict('os.environ', {'ARCHIVE_STATE_DIR':folder}), patch('web.portal.scheduler.cpu_capacity', return_value=4):
            scheduler = Scheduler()
            jobs.enqueue(self.upload())
            job = jobs.claim()
            root = Path(folder)/str(job.pk)
            attempt = root/str(job.token)
            protocol.write(attempt/'result.json', dict(token=str(uuid4()), outcome='succeeded'))
            scheduler.active[job.pk] = dict(job=job, process=Mock(poll=lambda:0), root=root, attempt=attempt)
            scheduler.collect()
            job.refresh_from_db()
            self.assertEqual(job.status, 'retry')

    def test_stalled_worker_is_stopped_then_killed_and_retried(self):
        import time
        from .scheduler import Scheduler
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as folder, patch.dict('os.environ', {'ARCHIVE_STATE_DIR':folder}), patch('web.portal.scheduler.cpu_capacity', return_value=4):
            scheduler = Scheduler()
            jobs.enqueue(self.upload())
            job = jobs.claim()
            root, attempt = Path(folder)/'job', Path(folder)/'job'/'attempt'
            protocol.write(attempt/'progress.json', dict(token=str(job.token), updated=1, phase='processing'))
            process = Mock()
            process.poll.return_value = None
            item = dict(job=job, process=process, root=root, attempt=attempt,
                        last_activity=time.monotonic()-301, seen_update=1, stall_since=None)
            scheduler.active[job.pk] = item
            scheduler.collect()
            process.terminate.assert_called_once()
            item['stall_since'] = time.monotonic()-20
            scheduler.collect()
            process.kill.assert_called_once()
            process.poll.return_value = 0
            scheduler.collect()
            job.refresh_from_db()
            self.assertEqual(job.status, 'retry')
            self.assertIn('прогрессе', job.error)
