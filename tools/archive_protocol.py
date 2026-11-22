"""Atomic, attempt-scoped scheduler/worker messages (no credentials)."""
import json
import os
from pathlib import Path
import time


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, allow_nan=False)
        stream.flush()
    os.replace(temp, path)


def read(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def lease_valid(path, token):
    lease = read(path)
    return lease.get('token') == token and lease.get('expires', 0) > time.monotonic()
