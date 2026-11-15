from django.contrib.auth.models import AbstractUser, UserManager
from django.conf import settings
from django.db import models


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
