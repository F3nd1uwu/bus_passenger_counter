"""Start a loopback-only console with isolated synthetic data."""
import argparse
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys

from tools.setup_web_env import setup as setup_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('Choose a port between 1024 and 65535')
    root = Path(__file__).resolve().parents[1]
    setup_environment()
    state = root / '.local' / 'web-demo'
    state.mkdir(parents=True, exist_ok=True)
    access_file = state / 'demo-access.json'
    if access_file.exists():
        password = json.loads(access_file.read_text(encoding='utf-8'))['password']
    else:
        password = secrets.token_urlsafe(20)
        access_file.write_text(json.dumps({'admin': 'demo-admin', 'employee': 'demo-employee',
                                          'password': password}, indent=2) + '\n', encoding='utf-8')
        access_file.chmod(0o600)
    env = dict(os.environ, WEB_DEMO='1', WEB_STATE_DIR=str(state), WEB_DEBUG='0',
               WEB_ALLOWED_HOSTS='localhost,127.0.0.1,[::1]', WEB_REQUIRE_HTTPS='0',
               WEB_SECURE_COOKIES='0', WEB_DEMO_PASSWORD=password)
    command = [sys.executable, '-B', str(root / 'manage_web.py')]
    for operation in [['migrate', '--noinput'], ['seed_demo'], ['collectstatic', '--noinput']]:
        subprocess.run(command + operation, cwd=root, env=env, check=True)
    env.pop('WEB_DEMO_PASSWORD')
    print(f'\nOpen http://127.0.0.1:{args.port}/\nAdmin: demo-admin\nEmployee: demo-employee\nPassword: {password}\nSynthetic data only. Stop with Ctrl+C.\n', flush=True)
    subprocess.run(command + ['runserver', f'127.0.0.1:{args.port}', '--noreload'], cwd=root, env=env, check=True)


if __name__ == '__main__':
    main()
