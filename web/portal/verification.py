"""Run with manage_web.py test web.portal.verification (isolated databases)."""
from datetime import timedelta
import hashlib
from unittest.mock import patch
from uuid import uuid4

from django.conf import settings
from django.db import connections, OperationalError
from django.test import Client, TestCase, override_settings
from django.test.utils import CaptureQueriesContext
from django.utils import timezone

from web.analytics.models import CameraDaily, PassengerEvent
from .models import AuditEntry, Bus, ReportExport, User
from .services import csv_bytes

PASSWORD = 'Unit-test-only-Q7!k42'


@override_settings(STORAGES={'default': {'BACKEND': 'django.core.files.storage.FileSystemStorage'},
                            'staticfiles': {'BACKEND': 'django.contrib.staticfiles.storage.StaticFilesStorage'}})
class ConsoleTests(TestCase):
    databases = {'default', 'analytics'}

    @classmethod
    def setUpClass(cls):
        connection = connections['analytics']
        existing = connection.introspection.table_names()
        with connection.schema_editor() as editor:
            for model in [CameraDaily, PassengerEvent]:
                if model._meta.db_table not in existing:
                    editor.create_model(model)
        super().setUpClass()

    @classmethod
    def setUpTestData(cls):
        cls.admin = User.objects.create_user('admin', password=PASSWORD, role=User.Role.ADMIN)
        cls.employee = User.objects.create_user('employee', password=PASSWORD)
        cls.other = User.objects.create_user('other', password=PASSWORD)
        cls.a = Bus.objects.create(bus_id='BUS_A', name='Автобус А', mode='production')
        cls.b = Bus.objects.create(bus_id='BUS_B', name='Автобус Б', mode='production')
        cls.a.users.add(cls.employee)
        cls.b.users.add(cls.other)
        today = timezone.localdate()
        now = timezone.now()
        for bus, incoming in [('BUS_A', 11), ('BUS_B', 29), ('TEST_LEGACY', 7)]:
            CameraDaily.objects.using('analytics').create(traffic_date=today, bus_id=bus,
                camera_id='FRONT', total_in=incoming, total_out=3, updated_at=now)
            PassengerEvent.objects.using('analytics').create(event_id=uuid4(), bus_id=bus, camera_id='FRONT',
                detected_at=now, traffic_date=today, event_type='IN', track_id=1,
                identity_id='sensitive-test-identity', counted=True, reason='COUNTED')

    def setUp(self):
        self.client.force_login(self.employee)

    def filters(self, **extra):
        today = timezone.localdate().isoformat()
        return {'date_from': today, 'date_to': today, 'bus': '', 'mode': 'all', **extra}

    def export(self, client=None):
        return (client or self.client).post('/exports/create/', self.filters())

    def test_anonymous_users_are_redirected_before_analytics_is_read(self):
        self.client.logout()
        with CaptureQueriesContext(connections['analytics']) as captured:
            for url in ['/', '/reports/', '/events/', '/exports/', '/users/', '/buses/', '/audit/']:
                self.assertEqual(self.client.get(url).status_code, 302)
        self.assertEqual(len(captured), 0)

    def test_employee_only_sees_assigned_statistics(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['totals']['counted_in'], 11)
        self.assertNotContains(response, 'Автобус Б')
        self.assertNotContains(response, 'TEST_LEGACY')
        self.assertNotContains(response, 'Администрирование')

    def test_employee_cannot_select_another_bus(self):
        for url in ['/', '/reports/', '/events/']:
            self.assertEqual(self.client.get(url, self.filters(bus='BUS_B')).status_code, 403)
        self.assertEqual(self.client.post('/exports/create/', self.filters(bus='BUS_B')).status_code, 403)
        self.assertEqual(ReportExport.objects.count(), 0)

    def test_employee_cannot_use_admin_endpoints(self):
        for url in ['/users/', '/users/new/', f'/users/{self.admin.pk}/', '/buses/', '/audit/']:
            self.assertEqual(self.client.get(url).status_code, 403)
            self.assertEqual(self.client.post(url, {'role': 'admin'}).status_code, 403)
        self.employee.refresh_from_db()
        self.assertEqual(self.employee.role, 'employee')

    def test_unassigned_employee_sees_no_source_data(self):
        self.employee.buses.clear()
        response = self.client.get('/')
        self.assertEqual(response.context['totals']['counted_in'], 0)
        self.assertContains(response, 'Вам ещё не назначены автобусы')

    def test_admin_can_see_unregistered_legacy_statistics(self):
        self.client.force_login(self.admin)
        self.assertEqual(self.client.get('/').context['totals']['counted_in'], 47)
        self.assertContains(self.client.get('/'), 'TEST_LEGACY')

    def test_production_and_test_filters_are_separate(self):
        self.client.force_login(self.admin)
        working = self.client.get('/', self.filters(mode='production'))
        testing = self.client.get('/', self.filters(mode='test'))
        self.assertEqual(working.context['totals']['counted_in'], 40)
        self.assertEqual(testing.context['totals']['counted_in'], 7)

    def test_period_validation_blocks_reversed_or_oversized_range(self):
        today = timezone.localdate()
        for data in [self.filters(date_from=(today + timedelta(days=1)).isoformat()),
                     self.filters(date_from=(today - timedelta(days=500)).isoformat())]:
            self.assertEqual(self.client.get('/reports/', data).status_code, 400)
            self.assertEqual(self.client.post('/exports/create/', data).status_code, 400)

    def test_events_are_scoped_and_identity_features_are_not_selected(self):
        with CaptureQueriesContext(connections['analytics']) as captured:
            response = self.client.get('/events/')
        self.assertContains(response, 'Автобус А')
        self.assertNotContains(response, 'Автобус Б')
        self.assertNotContains(response, 'sensitive-test-identity')
        text = ' '.join(item['sql'].lower() for item in captured)
        self.assertNotIn('counter_runtime_state', text)
        self.assertNotIn('identity_id', text)
        self.assertNotIn('reid_features', text)

    def test_overview_never_reads_runtime_checkpoints(self):
        with CaptureQueriesContext(connections['analytics']) as captured:
            self.client.get('/')
        self.assertNotIn('counter_runtime_state', ' '.join(row['sql'] for row in captured))

    def test_export_is_scoped_and_hash_matches_download(self):
        self.assertEqual(self.export().status_code, 302)
        report = ReportExport.objects.get()
        self.assertEqual({row['bus_id'] for row in report.rows}, {'BUS_A'})
        download = self.client.get(f'/exports/{report.pk}/download/')
        self.assertEqual(download.status_code, 200)
        self.assertEqual(hashlib.sha256(download.content).hexdigest(), report.sha256)
        self.assertTrue(download.content.startswith(b'\xef\xbb\xbf'))
        self.assertEqual(AuditEntry.objects.filter(action='Скачан проверочный отчёт').count(), 1)

    def test_export_snapshot_does_not_change_with_new_counters(self):
        self.export()
        report = ReportExport.objects.get()
        before = self.client.get(f'/exports/{report.pk}/download/').content
        CameraDaily.objects.using('analytics').filter(bus_id='BUS_A').update(total_in=999)
        after = self.client.get(f'/exports/{report.pk}/download/').content
        self.assertEqual(before, after)

    def test_another_employee_cannot_read_export_even_by_known_id(self):
        self.export()
        report = ReportExport.objects.get()
        self.client.force_login(self.other)
        self.assertNotContains(self.client.get('/exports/'), 'Отчёт №')
        self.assertEqual(self.client.get(f'/exports/{report.pk}/download/').status_code, 403)

    def test_revoked_bus_access_also_blocks_historical_download(self):
        self.export()
        report = ReportExport.objects.get()
        self.employee.buses.clear()
        self.assertEqual(self.client.get(f'/exports/{report.pk}/download/').status_code, 403)

    def test_get_cannot_create_export_or_logout(self):
        self.assertEqual(self.client.get('/exports/create/').status_code, 405)
        self.assertEqual(self.client.get('/logout/').status_code, 405)

    def test_csrf_protects_export_and_user_creation(self):
        guarded = Client(enforce_csrf_checks=True)
        guarded.force_login(self.admin)
        self.assertEqual(guarded.post('/exports/create/', self.filters()).status_code, 403)
        self.assertEqual(guarded.post('/users/new/', {}).status_code, 403)

    def test_disabled_account_cannot_access_with_old_session(self):
        User.objects.filter(pk=self.employee.pk).update(is_active=False)
        self.assertEqual(self.client.get('/').status_code, 302)

    def test_self_admin_access_cannot_be_removed(self):
        self.client.force_login(self.admin)
        response = self.client.post(f'/users/{self.admin.pk}/', {'role': 'employee', 'is_active': 'on'})
        self.assertContains(response, 'Нельзя отключить собственный административный доступ')
        self.admin.refresh_from_db()
        self.assertEqual(self.admin.role, 'admin')

    def test_admin_user_creation_hashes_password_and_grants_scope(self):
        self.client.force_login(self.admin)
        response = self.client.post('/users/new/', {'username': 'new-employee', 'role': 'employee',
             'password1': PASSWORD, 'password2': PASSWORD, 'buses': [self.a.pk], 'is_superuser': 'on'})
        self.assertEqual(response.status_code, 302)
        account = User.objects.get(username='new-employee')
        self.assertTrue(account.check_password(PASSWORD))
        self.assertNotEqual(account.password, PASSWORD)
        self.assertFalse(account.is_superuser)
        self.assertEqual(list(account.buses.values_list('bus_id', flat=True)), ['BUS_A'])

    def test_login_rate_limit_applies_even_with_correct_password_after_failures(self):
        self.client.logout()
        for _ in range(5):
            self.assertEqual(self.client.post('/login/', {'username': 'admin', 'password': 'wrong'}).status_code, 200)
        response = self.client.post('/login/', {'username': 'admin', 'password': PASSWORD})
        self.assertEqual(response.status_code, 429)
        self.assertNotIn('_auth_user_id', self.client.session)

    def test_unknown_and_missing_routes_do_not_leak_analytics(self):
        self.assertEqual(self.client.get('/reports/', self.filters(bus='BUS_B')).status_code, 403)
        self.assertEqual(self.client.get('/exports/999999/download/').status_code, 404)

    def test_analytics_errors_do_not_expose_connection_details(self):
        with patch('web.portal.services.overview', side_effect=OperationalError('sensitive-connection-value')):
            response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Статистика временно недоступна')
        self.assertNotContains(response, 'sensitive-connection-value')

    def test_names_are_html_escaped_and_csv_formulas_are_neutralized(self):
        self.a.name = '<script>alert(1)</script>'
        self.a.save()
        response = self.client.get('/')
        self.assertNotContains(response, '<script>alert(1)</script>')
        self.assertContains(response, '&lt;script&gt;')
        row = {'date': '2026-11-15', 'bus_id': 'BUS_A', 'name': '=HYPERLINK("bad")',
               'mode_label': 'Рабочий', 'counted_in': 1, 'counted_out': 0, 'updated_at': ''}
        self.assertIn("'=HYPERLINK", csv_bytes([row]).decode('utf-8-sig'))

    def test_responses_are_private_and_have_content_security_policy(self):
        response = self.client.get('/')
        self.assertIn('no-store', response['Cache-Control'])
        self.assertIn("frame-ancestors 'none'", response['Content-Security-Policy'])

    def test_router_refuses_accidental_analytics_writes(self):
        with self.assertRaisesRegex(RuntimeError, 'read-only'):
            CameraDaily.objects.create(traffic_date=timezone.localdate(), bus_id='BAD', camera_id='BAD',
                total_in=1, total_out=0, updated_at=timezone.now())
        self.assertNotIn('raw_passenger_events', connections['default'].introspection.table_names())
