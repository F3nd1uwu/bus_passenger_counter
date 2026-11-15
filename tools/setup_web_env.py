"""Create a private web configuration without changing the worker's .env."""
from pathlib import Path
import secrets

ROOT = Path(__file__).resolve().parents[1]


def setup():
    target = ROOT / '.env.web'
    if target.exists():
        print('Existing .env.web preserved')
        return
    content = (ROOT / 'web.env.example').read_text('utf-8')
    content = content.replace('generate-a-separate-web-secret', secrets.token_urlsafe(64))
    content = content.replace('generate-a-readonly-database-password', secrets.token_urlsafe(32))
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(content)
    try:
        target.chmod(0o600)
    except OSError:
        pass
    print('Created .env.web; secrets were not printed')


if __name__ == '__main__':
    setup()
