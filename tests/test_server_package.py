import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tools import package_server
from tools.verify_release import verify


class ServerPackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        names = package_server.FILES | set(package_server.TEMPLATES) | {'config/__init__.py', 'core/__init__.py',
                'database/__init__.py', '.env', 'reports/private.json', 'models/candidates/extra.onnx', 'models/cpu_models.json'}
        for name in names:
            self.write(name, b'fixture\n')
        self.write('Dockerfile', b'FROM python:3.12-slim\nCMD ["python", "main.py"]\n')
        for stem in ['person_test2_v2', 'work_vest_v3']:
            data = ('model fixture ' + stem).encode()
            self.write(f'models/candidates/{stem}.onnx', data)
            self.write(f'models/candidates/{stem}.json', json.dumps({'onnx_sha256': hashlib.sha256(data).hexdigest()}).encode())
        self.write('models/cpu_models.json', json.dumps({'reid_onnx_sha256': hashlib.sha256(b'fixture\n').hexdigest()}).encode())
        self.command('init', '--quiet')
        self.command('-c', 'core.autocrlf=false', 'add', '.')
        self.command('-c', 'user.name=Package Tests', '-c', 'user.email=tests@example.invalid',
                     'commit', '--quiet', '-m', 'Create packaging fixture')
        self.patcher = patch.object(package_server, 'ROOT', self.root)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def write(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def command(self, *args):
        subprocess.run(['git', *args], cwd=self.root, check=True, capture_output=True)

    def test_distribution_uses_committed_blobs_and_excludes_private_and_research_files(self):
        self.write('core/__init__.py', b'uncommitted local changes')
        manifest = package_server.build('HEAD', self.root / 'release')
        self.assertEqual((self.root / 'release/core/__init__.py').read_bytes(), b'fixture\n')
        for name in ['.env', 'reports/private.json', 'models/candidates/extra.onnx']:
            self.assertNotIn(name, manifest['files'])
            self.assertFalse((self.root / 'release' / name).exists())
        self.assertEqual(verify(self.root / 'release')['source_commit'], manifest['source_commit'])

    def test_existing_output_is_preserved(self):
        target = self.root / 'existing'
        target.mkdir()
        (target / 'keep.txt').write_bytes(b'keep')
        with self.assertRaises(FileExistsError):
            package_server.build('HEAD', target)
        self.assertEqual((target / 'keep.txt').read_bytes(), b'keep')

    def test_verifier_detects_changed_payload(self):
        target = self.root / 'release'
        package_server.build('HEAD', target)
        (target / 'main.py').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'differs'):
            verify(target)

    def test_verifier_rejects_path_outside_distribution(self):
        target = self.root / 'release'
        manifest = package_server.build('HEAD', target)
        manifest['files']['../outside.txt'] = {'bytes': 0, 'sha256': hashlib.sha256(b'').hexdigest()}
        (target / 'release.json').write_text(json.dumps(manifest), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'path'):
            verify(target)
