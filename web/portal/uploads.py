"""Private, bounded, resumable archive ingestion. Never starts inference in HTTP."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess

from django.conf import settings
from django.db import transaction
from django.db.models import F, Sum
from django.shortcuts import get_object_or_404
from django.utils import timezone

from .models import Bus, User, VideoUpload
from .services import audit


class UploadError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def visible(user):
    rows = VideoUpload.objects.select_related('bus', 'owner', 'job')
    if not user.is_console_admin:
        rows = rows.filter(owner=user, bus__users=user)
    return rows


def accessible(user, pk):
    return get_object_or_404(visible(user), pk=pk)


def paths(upload):
    root = settings.VIDEO_UPLOAD_DIR.resolve()
    # Only generated UUIDs become filenames, never submitted names or camera IDs.
    return root / f'{upload.pk}.part', root / f'{upload.pk}.mp4'


def payload(upload):
    return dict(id=str(upload.pk), name=upload.original_name, size=upload.size,
                received=upload.received, status=upload.status, status_label=upload.get_status_display(),
                chunk_size=settings.VIDEO_UPLOAD_CHUNK_SIZE, error=upload.error)


def create(user, data):
    name, camera, fingerprint = data.get('name'), data.get('camera_id'), data.get('fingerprint')
    size = data.get('size')
    if (not isinstance(name, str) or not 1 <= len(name) <= 200 or '/' in name or '\\' in name
            or any(ord(c) < 32 for c in name) or Path(name).suffix.lower() != '.mp4'):
        raise UploadError('Выберите отдельный файл MP4 с корректным именем.')
    if type(size) is not int or not 0 < size <= settings.VIDEO_UPLOAD_MAX_BYTES:
        raise UploadError('Размер файла превышает разрешённый предел или файл пустой.')
    if not isinstance(camera, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,32}', camera):
        raise UploadError('Код камеры: до 32 латинских букв, цифр, дефисов и подчёркиваний.')
    if not isinstance(fingerprint, str) or not re.fullmatch('[0-9a-f]{64}', fingerprint):
        raise UploadError('Не удалось проверить выбранный файл. Выберите его повторно.')
    buses = Bus.objects.all() if user.is_console_admin else user.buses.all()
    bus = get_object_or_404(buses, pk=str(data.get('bus', '')) if str(data.get('bus', '')).isdigit() else 0)
    root = settings.VIDEO_UPLOAD_DIR
    root.mkdir(parents=True, exist_ok=True)
    with transaction.atomic():
        # The console uses SQLite. Take its write lock before reading reservations;
        # this also serializes creation requests when the table is initially empty.
        User.objects.filter(pk=user.pk).update(last_login=F('last_login'))
        resume = VideoUpload.objects.filter(owner=user, bus=bus, camera_id=camera,
                   original_name=name, size=size, fingerprint=fingerprint,
                   status__in=[VideoUpload.Status.UPLOADING, VideoUpload.Status.READY]).first()
        if resume:
            return resume
        active = VideoUpload.objects.filter(status=VideoUpload.Status.UPLOADING)
        if active.filter(owner=user).count() >= settings.VIDEO_UPLOAD_MAX_ACTIVE:
            raise UploadError('Сначала завершите или отмените незавершённые загрузки.', 409)
        used = VideoUpload.objects.filter(status__in=['uploading', 'ready']).aggregate(n=Sum('size'))['n'] or 0
        reserved = sum(row.size-row.received for row in active.only('size', 'received'))
        if used + size > settings.VIDEO_UPLOAD_QUOTA_BYTES:
            raise UploadError('Достигнут лимит хранилища видео. Обратитесь к администратору.', 507)
        if shutil.disk_usage(root).free - reserved - size < settings.VIDEO_UPLOAD_MIN_FREE_BYTES:
            raise UploadError('На сервере недостаточно свободного места для этой записи.', 507)
        upload = VideoUpload.objects.create(owner=user, bus=bus, camera_id=camera,
                    original_name=name, size=size, fingerprint=fingerprint)
        part, _ = paths(upload)
        with part.open('xb'):
            pass
        audit(user, 'Начата загрузка видео', upload.pk)
        return upload


def append(user, pk, offset, body):
    upload = accessible(user, pk)
    if not body or len(body) > settings.VIDEO_UPLOAD_CHUNK_SIZE or type(offset) is not int or offset < 0:
        raise UploadError('Некорректная часть файла.')
    with transaction.atomic():
        # A write-before-read locks SQLite; select_for_update alone does not.
        VideoUpload.objects.filter(pk=upload.pk).update(updated_at=timezone.now())
        upload = accessible(user, pk)
        if upload.status != VideoUpload.Status.UPLOADING:
            raise UploadError('Загрузка уже завершена или отменена.', 409)
        part, _ = paths(upload)
        if not part.is_file() or part.stat().st_size < upload.received:
            raise UploadError('Незавершённый файл отсутствует. Отмените загрузку и загрузите заново.', 409)
        if offset < upload.received and offset + len(body) <= upload.received:
            with part.open('rb') as stream:
                stream.seek(offset)
                if stream.read(len(body)) == body:
                    return upload  # Lost HTTP response: retry is idempotent.
            raise UploadError('Часть отличается от уже сохранённой. Выберите исходный файл.', 409)
        if offset != upload.received or offset + len(body) > upload.size:
            raise UploadError('Позиция загрузки изменилась. Продолжите загрузку повторно.', 409)
        if shutil.disk_usage(part.parent).free - len(body) < settings.VIDEO_UPLOAD_MIN_FREE_BYTES:
            raise UploadError('На сервере закончилось свободное место. Загрузка сохранена.', 507)
        with part.open('r+b') as stream:
            # A crash after fsync but before DB commit can leave an uncommitted tail.
            stream.truncate(upload.received)
            stream.seek(upload.received)
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        upload.received += len(body)
        upload.save(update_fields=['received', 'updated_at'])
        return upload


def probe(path):
    try:
        # Force the MP4/MOV demuxer and forbid network protocols/external references.
        # An untrusted file renamed .mp4 must not turn ffprobe into a URL fetcher.
        response = subprocess.run([settings.VIDEO_FFPROBE, '-v', 'error', '-protocol_whitelist', 'file',
            '-f', 'mov', '-enable_drefs', '0', '-use_absolute_path', '0', '-select_streams', 'V:0',
            '-show_entries', 'stream=codec_name,width,height:format=duration,format_name',
            '-of', 'json', str(path)], capture_output=True, timeout=12, check=True)
        data = json.loads(response.stdout)
        stream = data['streams'][0]
        duration = float(data['format']['duration'])
        if (not math.isfinite(duration) or duration <= 0 or int(stream['width']) <= 0
                or int(stream['height']) <= 0 or 'mp4' not in data['format']['format_name'].split(',')):
            raise ValueError('Unsupported video')
        return dict(duration_seconds=duration, width=int(stream['width']), height=int(stream['height']))
    except FileNotFoundError:
        raise UploadError('Проверка видео временно недоступна. Обратитесь к администратору.', 503)
    except (subprocess.SubprocessError, ValueError, KeyError, IndexError, TypeError):
        raise UploadError('Не удалось прочитать MP4. Проверьте файл в видеоплеере и загрузите заново.', 422)


def finish(user, pk):
    accessible(user, pk)
    with transaction.atomic():
        VideoUpload.objects.filter(pk=pk).update(updated_at=timezone.now())
        upload = accessible(user, pk)
        if upload.status == VideoUpload.Status.READY:
            from .jobs import enqueue
            enqueue(upload)
            return upload
        if upload.status != VideoUpload.Status.UPLOADING or upload.received != upload.size:
            raise UploadError('Сначала загрузите файл полностью.', 409)
        part, ready = paths(upload)
        # Recover a completed rename if the process stopped before DB commit.
        source = part if part.is_file() else ready
        if not source.is_file() or source.stat().st_size != upload.size:
            raise UploadError('Размер сохранённого файла не совпадает. Отмените загрузку и повторите.', 409)
        with source.open('rb') as stream:
            first = stream.read(1048576)
            stream.seek(max(0, upload.size - 1048576))
            fingerprint = hashlib.sha256(first + stream.read(1048576)).hexdigest()
        if fingerprint != upload.fingerprint:
            raise UploadError('Начало или конец файла не совпадает с выбранной записью. Загрузите исходный файл заново.', 422)
        metadata = probe(source)
        if source == part:
            if ready.exists():
                raise UploadError('Обнаружен конфликт файлов. Обратитесь к администратору.', 409)
            os.replace(part, ready)
        for key, value in metadata.items():
            setattr(upload, key, value)
        upload.status = VideoUpload.Status.READY
        upload.error = ''
        upload.save(update_fields=['status', 'error', 'duration_seconds', 'width', 'height', 'updated_at'])
        audit(user, 'Загружено видео', upload.pk)
        from .jobs import enqueue
        enqueue(upload)
        return upload


def cancel(user, pk):
    accessible(user, pk)
    with transaction.atomic():
        VideoUpload.objects.filter(pk=pk).update(updated_at=timezone.now())
        upload = accessible(user, pk)
        if upload.status == VideoUpload.Status.CANCELLED:
            return upload
        if upload.status != VideoUpload.Status.UPLOADING:
            raise UploadError('Завершённую запись нельзя отменить как загрузку.', 409)
        part, ready = paths(upload)
        if ready.exists():
            raise UploadError('Файл ожидает завершения проверки. Нажмите «Продолжить».', 409)
        part.unlink(missing_ok=True)
        upload.status = VideoUpload.Status.CANCELLED
        upload.error = ''
        upload.save(update_fields=['status', 'error', 'updated_at'])
        audit(user, 'Отменена загрузка видео', upload.pk)
        return upload
