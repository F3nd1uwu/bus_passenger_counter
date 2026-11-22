"""Install/start the archive portal from a checkout using Docker Compose."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def compose(https=False):
    files = ['docker-compose.yml', 'docker-compose.web.yml', 'docker-compose.archive.yml']
    if https:
        files.append('docker-compose.https.yml')
    return ['docker', 'compose', *[value for file in files for value in ('-f', file)]]


def run(arguments):
    subprocess.run(arguments, cwd=ROOT, check=True)


def start(*, https=False, build=True):
    run(['docker', 'info', '--format', '{{.OSType}}'])
    run(['docker', 'compose', 'version'])
    from tools.setup_env import setup
    from tools.setup_web_env import setup as setup_web
    setup()
    setup_web()
    command = compose(https)
    run(command + ['config', '--quiet'])
    if build:
        run(['docker', 'build', '-t', 'bus-passenger-counter:archive-base', '.'])
        run(command + ['build', 'web', 'archive_scheduler'])
    run(command + ['up', '-d', '--wait', 'postgres'])
    run(command + ['run', '--rm', '--no-deps', 'web_setup'])
    run(command + ['up', '-d', '--no-deps', '--wait', 'web'])
    check = subprocess.run(command + ['exec', '-T', 'web', 'python', 'manage_web.py', 'shell', '-c',
        "import sys; from django.db.models import Q; from web.portal.models import User; "
        "sys.exit(0 if User.objects.filter(Q(role='admin') | Q(is_superuser=True), is_active=True).exists() else 3)"],
        cwd=ROOT, capture_output=True)
    if check.returncode == 3:
        print('Создайте администратора кабинета. Пароль вводится скрыто.')
        run(command + ['exec', 'web', 'python', 'manage_web.py', 'createsuperuser'])
    elif check.returncode:
        raise RuntimeError('Не удалось проверить аккаунт администратора. Проверьте журнал web.')
    run(command + ['up', '-d', '--wait', 'archive_scheduler'])
    if https:
        run(command + ['up', '-d', 'caddy'])
        print('Кабинет запущен на HTTPS-домене из WEB_DOMAIN.')
    else:
        print(f"Кабинет запущен: http://127.0.0.1:{os.environ.get('WEB_PORT', '8000')}/ . Для удалённого сервера используйте домен или SSH-туннель.")
    print('Войдите в кабинет, добавьте автобус в разделе «Автобусы» и загрузите MP4 в разделе «Видеозаписи».')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--https', action='store_true', help='Use configured WEB_DOMAIN and the HTTPS overlay')
    parser.add_argument('--skip-build', action='store_true', help='Reuse previously built local images')
    args = parser.parse_args()
    try:
        start(https=args.https, build=not args.skip_build)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'Запуск не завершён ({type(error).__name__}). Исправьте причину и повторите команду; существующие данные сохраняются.\n')


if __name__ == '__main__':
    main()
