"""Isolated upload tests: manage_web.py test web.portal.upload_verification."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch

from django.test import Client, TestCase, override_settings

from .models import Bus, User, VideoUpload
from . import uploads


@override_settings(STORAGES={'staticfiles': {'BACKEND': 'django.contrib.staticfiles.storage.StaticFilesStorage'}})
class UploadTests(TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.options = override_settings(VIDEO_UPLOAD_DIR=Path(self.folder.name), VIDEO_UPLOAD_MIN_FREE_BYTES=0,
                    VIDEO_UPLOAD_MAX_BYTES=1024, VIDEO_UPLOAD_QUOTA_BYTES=4096, VIDEO_UPLOAD_CHUNK_SIZE=4)
        self.options.enable()
        self.addCleanup(self.options.disable)
        self.user = User.objects.create_user('employee')
        self.other = User.objects.create_user('other')
        self.admin = User.objects.create_user('admin', role='admin')
        self.bus = Bus.objects.create(bus_id='BUS_A', name='Автобус А')
        self.secret = Bus.objects.create(bus_id='BUS_B', name='Автобус Б')
        self.bus.users.add(self.user)
        self.client.force_login(self.user)
        self.body = b'12345678'
        self.data = dict(name='day.mp4', size=8, bus=self.bus.pk, camera_id='FRONT',
                         fingerprint=hashlib.sha256(self.body * 2).hexdigest())

    def create(self, **overrides):
        return self.client.post('/videos/create/', json.dumps({**self.data, **overrides}), content_type='application/json')

    def upload(self):
        response = self.create()
        self.assertEqual(response.status_code, 200)
        return VideoUpload.objects.get(pk=response.json()['id'])

    def chunk(self, obj, offset, body):
        return self.client.post(f'/videos/{obj.pk}/chunk/', body, content_type='application/octet-stream',
                                HTTP_UPLOAD_OFFSET=str(offset))

    def finish(self, obj):
        return self.client.post(f'/videos/{obj.pk}/finish/')

    def fill(self, obj):
        self.assertEqual(self.chunk(obj, 0, self.body[:4]).status_code, 200)
        self.assertEqual(self.chunk(obj, 4, self.body[4:]).status_code, 200)

    def test_anonymous_and_csrf_are_required(self):
        self.client.logout()
        self.assertEqual(self.create().status_code, 302)
        strict = Client(enforce_csrf_checks=True)
        strict.force_login(self.user)
        self.assertEqual(strict.post('/videos/create/', json.dumps(self.data), content_type='application/json').status_code, 403)

    def test_employee_cannot_assign_other_bus(self):
        self.assertEqual(self.create(bus=self.secret.pk).status_code, 404)
        self.assertEqual(VideoUpload.objects.count(), 0)

    def test_permissions_and_revocation_apply_to_all_endpoints(self):
        obj = self.upload()
        self.client.force_login(self.other)
        for action in ('chunk', 'finish', 'cancel'):
            self.assertEqual(self.client.post(f'/videos/{obj.pk}/{action}/').status_code, 404)
        self.client.force_login(self.user)
        self.bus.users.clear()
        self.assertEqual(self.chunk(obj, 0, b'1234').status_code, 404)
        self.assertNotContains(self.client.get('/videos/'), obj.original_name)

    def test_admin_can_upload_for_any_bus(self):
        self.client.force_login(self.admin)
        self.assertEqual(self.create(bus=self.secret.pk).status_code, 200)

    def test_names_sizes_and_identifiers_are_checked(self):
        for overrides in [dict(name='../a.mp4'), dict(name='C:\\a.mp4'), dict(name='a.zip'), dict(name='a\n.mp4'),
                          dict(size=0), dict(size=2048), dict(size=True), dict(camera_id='../FRONT'), dict(fingerprint='x')]:
            self.assertEqual(self.create(**overrides).status_code, 400, overrides)
        self.assertEqual(VideoUpload.objects.count(), 0)

    def test_session_resume_and_changed_fingerprint(self):
        obj = self.upload()
        self.chunk(obj, 0, b'1234')
        response = self.create()
        self.assertEqual(response.json()['id'], str(obj.pk))
        self.assertEqual(response.json()['received'], 4)
        self.assertNotEqual(self.create(fingerprint='a' * 64).json()['id'], str(obj.pk))

    def test_chunks_are_bounded_ordered_and_idempotent(self):
        obj = self.upload()
        self.assertEqual(self.chunk(obj, 4, b'5678').status_code, 409)
        self.assertEqual(self.chunk(obj, 0, b'12345').status_code, 413)
        self.assertEqual(self.chunk(obj, 0, b'1234').status_code, 200)
        self.assertEqual(self.chunk(obj, 0, b'1234').status_code, 200)
        self.assertEqual(self.chunk(obj, 0, b'xxxx').status_code, 409)
        self.assertEqual(self.chunk(obj, 4, b'5678').status_code, 200)
        self.assertEqual(self.chunk(obj, 8, b'!').status_code, 409)
        self.assertEqual(uploads.paths(obj)[0].read_bytes(), self.body)

    def test_crash_tail_is_removed_on_retry(self):
        obj = self.upload()
        self.chunk(obj, 0, b'1234')
        with uploads.paths(obj)[0].open('ab') as stream:
            stream.write(b'uncommitted')
        self.assertEqual(self.chunk(obj, 4, b'5678').status_code, 200)
        self.assertEqual(uploads.paths(obj)[0].read_bytes(), self.body)

    def test_missing_or_short_partial_is_not_silently_accepted(self):
        obj = self.upload()
        self.chunk(obj, 0, b'1234')
        uploads.paths(obj)[0].write_bytes(b'1')
        self.assertEqual(self.chunk(obj, 4, b'5678').status_code, 409)

    def test_finish_checks_completeness_and_probe_then_atomically_publishes(self):
        obj = self.upload()
        self.assertEqual(self.finish(obj).status_code, 409)
        self.fill(obj)
        with patch.object(uploads, 'probe', return_value=dict(duration_seconds=5., width=640, height=480)):
            self.assertEqual(self.finish(obj).json()['status'], 'ready')
            self.assertEqual(self.finish(obj).json()['status'], 'ready')
        obj.refresh_from_db()
        self.assertEqual(obj.job.status, 'queued')
        part, ready = uploads.paths(obj)
        self.assertFalse(part.exists())
        self.assertEqual(ready.read_bytes(), self.body)
        self.assertEqual(self.create().json()['id'], str(obj.pk))
        self.assertEqual(self.chunk(obj, 0, b'1234').status_code, 409)

    def test_fingerprint_mismatch_does_not_publish(self):
        obj = self.upload()
        self.fill(obj)
        uploads.paths(obj)[0].write_bytes(b'badbytes')
        self.assertEqual(self.finish(obj).status_code, 422)
        self.assertFalse(uploads.paths(obj)[1].exists())

    def test_invalid_video_or_missing_probe_does_not_publish(self):
        obj = self.upload()
        self.fill(obj)
        for status in (422, 503):
            with patch.object(uploads, 'probe', side_effect=uploads.UploadError('test', status)):
                self.assertEqual(self.finish(obj).status_code, status)
            self.assertFalse(uploads.paths(obj)[1].exists())
        self.assertTrue(uploads.paths(obj)[0].exists())

    def test_rename_before_db_commit_can_be_recovered(self):
        obj = self.upload()
        self.fill(obj)
        part, ready = uploads.paths(obj)
        part.rename(ready)
        with patch.object(uploads, 'probe', return_value=dict(duration_seconds=5., width=640, height=480)):
            self.assertEqual(self.finish(obj).json()['status'], 'ready')

    def test_cancel_only_removes_own_partial_and_releases_reservation(self):
        obj = self.upload()
        self.assertEqual(self.client.post(f'/videos/{obj.pk}/cancel/').status_code, 200)
        self.assertEqual(self.client.post(f'/videos/{obj.pk}/cancel/').status_code, 200)
        self.assertFalse(uploads.paths(obj)[0].exists())
        self.assertEqual(self.chunk(obj, 0, b'1234').status_code, 409)

    def test_disk_quota_and_active_limit(self):
        with override_settings(VIDEO_UPLOAD_QUOTA_BYTES=7):
            self.assertEqual(self.create().status_code, 507)
        with patch.object(uploads.shutil, 'disk_usage', return_value=type('Disk', (), {'free': 7})()):
            self.assertEqual(self.create().status_code, 507)
        obj = self.upload()
        with override_settings(VIDEO_UPLOAD_MAX_ACTIVE=1):
            self.assertEqual(self.create(name='other.mp4').status_code, 409)
            self.assertEqual(self.create().json()['id'], str(obj.pk))

    def test_upload_page_is_scoped_and_names_are_escaped(self):
        self.create(name='<script>.mp4')
        self.assertContains(self.client.get('/videos/'), '&lt;script&gt;.mp4')
        self.client.force_login(self.other)
        self.assertNotContains(self.client.get('/videos/'), '&lt;script&gt;.mp4')

    def test_probe_rejects_invalid_metadata_and_uses_argument_list(self):
        good = {'streams': [{'width': 640, 'height': 480}], 'format': {'duration': '5', 'format_name': 'mov,mp4'}}
        with patch.object(uploads.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, json.dumps(good).encode())) as run:
            self.assertEqual(uploads.probe(Path('private.mp4'))['width'], 640)
            self.assertNotIn('shell', run.call_args.kwargs)
            arguments = run.call_args.args[0]
            self.assertEqual(arguments[arguments.index('-protocol_whitelist')+1], 'file')
            self.assertEqual(arguments[arguments.index('-enable_drefs')+1], '0')
        for change in ['NaN', '-1']:
            good['format']['duration'] = change
            with patch.object(uploads.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, json.dumps(good).encode())):
                with self.assertRaises(uploads.UploadError):
                    uploads.probe(Path('private.mp4'))
