from django.contrib.auth.models import AbstractUser, UserManager
from django.conf import settings
from django.db import models
from uuid import uuid4
from django.utils import timezone


class ConsoleUserManager(UserManager):
    def create_superuser(self, username, email=None, password=None, **extra_fields):
        extra_fields['role'] = 'admin'
        return super().create_superuser(username, email, password, **extra_fields)


class User(AbstractUser):
    class Role(models.TextChoices):
        EMPLOYEE = 'employee', 'Сотрудник'
        ADMIN = 'admin', 'Администратор'
    role = models.CharField(max_length=12, choices=Role.choices, default=Role.EMPLOYEE)
    objects = ConsoleUserManager()

    @property
    def is_console_admin(self):
        return self.is_superuser or self.role == self.Role.ADMIN


class Bus(models.Model):
    class Mode(models.TextChoices):
        TEST = 'test', 'Тестовый'
        PRODUCTION = 'production', 'Рабочий'
        UNKNOWN = 'unknown', 'Не проверен'
    bus_id = models.CharField(max_length=32, unique=True)
    name = models.CharField(max_length=100)
    mode = models.CharField(max_length=12, choices=Mode.choices, default=Mode.UNKNOWN)
    external_code = models.CharField(max_length=100, blank=True)
    users = models.ManyToManyField(settings.AUTH_USER_MODEL, related_name='buses', blank=True)

    class Meta:
        ordering = ['name', 'bus_id']

    def __str__(self):
        return f'{self.name} · {self.bus_id}'


class ReportExport(models.Model):
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.PROTECT, related_name='exports')
    created_at = models.DateTimeField(auto_now_add=True)
    date_from = models.DateField()
    date_to = models.DateField()
    rows = models.JSONField()
    sha256 = models.CharField(max_length=64)
    row_count = models.PositiveIntegerField()

    class Meta:
        ordering = ['-created_at', '-pk']


class AuditEntry(models.Model):
    actor = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, on_delete=models.SET_NULL)
    actor_name = models.CharField(max_length=150)
    action = models.CharField(max_length=80)
    target = models.CharField(max_length=150, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at', '-pk']


class LoginAttempt(models.Model):
    key = models.CharField(max_length=64, primary_key=True)
    failures = models.PositiveIntegerField(default=0)
    last_failure = models.DateTimeField()


class VideoUpload(models.Model):
    class Status(models.TextChoices):
        UPLOADING = 'uploading', 'Загружается'
        READY = 'ready', 'Загружено'
        CANCELLED = 'cancelled', 'Отменено'

    id = models.UUIDField(primary_key=True, default=uuid4, editable=False)
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.PROTECT)
    bus = models.ForeignKey(Bus, on_delete=models.PROTECT)
    camera_id = models.CharField(max_length=32)
    original_name = models.CharField(max_length=200)
    size = models.PositiveBigIntegerField()
    received = models.PositiveBigIntegerField(default=0)
    # Resume checks the first AND last chunk, not only name, size or modification time.
    fingerprint = models.CharField(max_length=64)
    status = models.CharField(max_length=12, choices=Status.choices, default=Status.UPLOADING)
    duration_seconds = models.FloatField(null=True)
    width = models.PositiveIntegerField(null=True)
    height = models.PositiveIntegerField(null=True)
    error = models.CharField(max_length=200, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        constraints = [models.CheckConstraint(condition=models.Q(received__lte=models.F('size')),
                                               name='upload_received_within_size')]


class ProcessingJob(models.Model):
    class Status(models.TextChoices):
        QUEUED = 'queued', 'В очереди'
        RUNNING = 'running', 'Обрабатывается'
        RETRY = 'retry', 'Ожидает повторного запуска'
        REVIEW = 'review', 'Требуется проверка'
        FAILED = 'failed', 'Ошибка обработки'
        SUCCEEDED = 'succeeded', 'Завершено'

    upload = models.OneToOneField(VideoUpload, on_delete=models.PROTECT, related_name='job')
    bus_id_snapshot = models.CharField(max_length=32)
    camera_id_snapshot = models.CharField(max_length=32)
    status = models.CharField(max_length=12, choices=Status.choices, default=Status.QUEUED, db_index=True)
    phase = models.CharField(max_length=20, default='waiting')
    attempts = models.PositiveIntegerField(default=0)
    token = models.UUIDField(null=True)
    heartbeat = models.DateTimeField(null=True)
    available_at = models.DateTimeField(default=timezone.now)
    started_at = models.DateTimeField(null=True)
    finished_at = models.DateTimeField(null=True)
    updated_at = models.DateTimeField(auto_now=True)
    processed_frame = models.PositiveBigIntegerField(default=0)
    total_frames = models.PositiveBigIntegerField(default=0)
    counted_in = models.PositiveIntegerField(default=0)
    counted_out = models.PositiveIntegerField(default=0)
    processing_fps = models.FloatField(default=0)
    error = models.CharField(max_length=300, blank=True)
    clock_note = models.CharField(max_length=300, blank=True)

    @property
    def progress_percent(self):
        if self.status == self.Status.SUCCEEDED:
            return 100
        return min(99, round(self.processed_frame * 100 / self.total_frames)) if self.total_frames else 0

    @property
    def phase_label(self):
        return {'waiting': 'Ожидает свободного обработчика', 'starting': 'Запуск',
                'clock': 'Проверка экранного времени', 'processing': 'Подсчёт пассажиров',
                'syncing': 'Сохранение результатов в базе'}.get(self.phase, '')
