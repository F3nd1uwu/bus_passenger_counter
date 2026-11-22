"""Durable FIFO queue in the console DB. Only one local scheduler is supported."""
from datetime import timedelta
from uuid import uuid4

from django.db import transaction
from django.db.models import F
from django.utils import timezone

from .models import ProcessingJob, VideoUpload


def enqueue(upload):
    if upload.status != VideoUpload.Status.READY:
        raise ValueError('Only verified complete uploads may be queued')
    return ProcessingJob.objects.get_or_create(upload=upload, defaults={
        'bus_id_snapshot': upload.bus.bus_id, 'camera_id_snapshot': upload.camera_id})[0]


def claim():
    with transaction.atomic():
        # Acquire SQLite's write lock before the read/claim sequence.
        ProcessingJob.objects.filter(status=ProcessingJob.Status.QUEUED).update(attempts=F('attempts'))
        busy = set(ProcessingJob.objects.filter(status='running').values_list('bus_id_snapshot', 'camera_id_snapshot'))
        for job in ProcessingJob.objects.select_related('upload').filter(
                status__in=['queued', 'retry'], available_at__lte=timezone.now()).order_by('available_at', 'pk'):
            if (job.bus_id_snapshot, job.camera_id_snapshot) in busy:
                continue
            if job.upload.status != VideoUpload.Status.READY:
                continue
            job.status, job.token = ProcessingJob.Status.RUNNING, uuid4()
            job.attempts += 1
            job.heartbeat = timezone.now()
            job.started_at = job.started_at or job.heartbeat
            job.phase, job.error = 'starting', ''
            job.save()
            return job


def update(job, **fields):
    """Fence late updates from superseded attempts."""
    return ProcessingJob.objects.filter(pk=job.pk, token=job.token, status='running').update(
        updated_at=timezone.now(), **fields)


def release(job, *, outcome, error='', max_attempts=3):
    status = {'succeeded': 'succeeded', 'review': 'review', 'failed': 'failed',
              'interrupted': 'retry', 'retry': 'retry'}[outcome]
    if status == 'retry' and outcome != 'interrupted' and job.attempts >= max_attempts:
        status = 'failed'
    extra = {'attempts': max(0, job.attempts-1)} if outcome == 'interrupted' else {}
    return update(job, status=status, error=error[:300], phase=status, **extra,
                  finished_at=timezone.now() if status in ('succeeded', 'review', 'failed') else None,
                  available_at=timezone.now() + timedelta(seconds=min(300, 15 * 2 ** min(job.attempts, 4))))


def recover_interrupted(*, active_ids=None):
    # Called only AFTER the scheduler's lifetime file lock was acquired.
    from django.db.models import Case, When, IntegerField
    rows = ProcessingJob.objects.filter(status='running')
    if active_ids is not None:
        from django.db.models import Q
        rows = rows.exclude(pk__in=active_ids).filter(Q(heartbeat__lt=timezone.now()-timedelta(seconds=60)) | Q(heartbeat__isnull=True))
    rows.update(status='retry', phase='waiting', token=None,
        attempts=Case(When(attempts__gt=0, then=F('attempts')-1), default=0, output_field=IntegerField()),
        available_at=timezone.now(), error='Обработчик был перезапущен. Продолжим с сохранённого состояния.')
