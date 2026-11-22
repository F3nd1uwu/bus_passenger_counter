import unittest
from unittest.mock import patch, Mock

from tools.start_portal import compose, start


class PortalStartTests(unittest.TestCase):
    def test_explicit_https_overlay(self):
        self.assertIn('docker-compose.https.yml', compose(True))
        self.assertNotIn('docker-compose.https.yml', compose(False))

    def test_installer_waits_for_services_and_preserves_existing_admin(self):
        commands=[]
        with patch('tools.start_portal.run', side_effect=lambda args: commands.append(args)), patch('tools.setup_env.setup'), patch('tools.setup_web_env.setup'), patch('tools.start_portal.subprocess.run', return_value=Mock(returncode=0)):
            start(build=False)
        self.assertTrue(any(args[-1]=='postgres' and '--wait' in args for args in commands))
        self.assertTrue(any(args[-1]=='archive_scheduler' and '--wait' in args for args in commands))
        self.assertFalse(any('createsuperuser' in args for args in commands))
        self.assertFalse(any('counter_service' in args for args in commands))

    def test_new_install_prompts_for_admin(self):
        commands=[]
        with patch('tools.start_portal.run', side_effect=lambda args: commands.append(args)), patch('tools.setup_env.setup'), patch('tools.setup_web_env.setup'), patch('tools.start_portal.subprocess.run', return_value=Mock(returncode=3)):
            start(build=False)
        self.assertTrue(any('createsuperuser' in args for args in commands))
