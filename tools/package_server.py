"""Build a minimal server distribution from committed Git objects only."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from config.settings import ROOT

FILES = {
    '.editorconfig', '.gitattributes', '.dockerignore', 'Dockerfile',
    'main.py', 'test_local.py', 'requirements.txt', 'botsort_reid.yaml',
    'docker-compose.yml', 'data/.gitkeep', 'tools/__init__.py', 'tools/setup_env.py',
    'tools/verify_release.py', 'models/person_reid.onnx', 'models/person_reid.json',
    'models/candidates/person_test2_v2.onnx', 'models/candidates/person_test2_v2.json',
    'models/candidates/work_vest_v4.onnx', 'models/candidates/work_vest_v4.json',
    'third_party/ULTRALYTICS_LICENSE.txt', 'third_party/CAMERA_CLOCK_LICENSE.txt',
    'models/camera_clock.onnx', 'models/camera_clock.json', 'tools/build_clock_map.py', 'tools/review_clock_map.py',
    'Dockerfile.web', 'Dockerfile.web.dockerignore', 'docker-compose.web.yml',
    'docker-compose.https.yml', 'deploy/Caddyfile',
    'Dockerfile.worker', 'Dockerfile.worker.dockerignore', 'docker-compose.archive.yml',
    'tools/archive_protocol.py', 'tools/archive_resources.py', 'tools/archive_worker.py',
    'manage_web.py', 'requirements-web.txt', 'web.env.example',
    'tools/setup_web_env.py', 'tools/setup_web_database.py',
    'tools/start_portal.py',
}
TEMPLATES = {
    'deployment/server/README.md': 'README.md',
    'deployment/server/env.example': '.env.example',
    'deployment/server/gitignore': '.gitignore',
    'deployment/server/THIRD_PARTY_NOTICES.md': 'THIRD_PARTY_NOTICES.md',
    'deployment/server/models.md': 'models/README.md',
    'deployment/server/ci.yml': '.github/workflows/ci.yml',
    'tools/vendor/OSNET_LICENSE.txt': 'third_party/OSNET_LICENSE.txt',
    'docs/WEB.md': 'WEB.md',
    'docs/ARCHIVE_QUEUE.md': 'ARCHIVE_QUEUE.md',
    'docs/CUSTOMER_DEMO.md': 'CUSTOMER_DEMO.md',
    'docs/SERVER_UPDATE.md': 'UPDATE.md',
}


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def build(ref, output):
    commit = git('rev-parse', '--verify', ref + '^{commit}').decode().strip()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Output already exists; choose a new empty distribution path')
    tracked = set(git('ls-tree', '-r', '--name-only', commit).decode().splitlines())
    selected = FILES | {name for name in tracked if name.startswith(('core/', 'config/', 'database/'))
                        and name.endswith(('.py', '.sql', '.json'))}
    selected |= {name for name in tracked if name.startswith('web/')
                 and name.endswith(('.py', '.html', '.css', '.js', '.svg'))
                 and name not in {'web/portal/verification.py', 'web/portal/upload_verification.py', 'web/portal/job_verification.py',
                                  'web/portal/management/commands/seed_demo.py'}}
    required = selected | set(TEMPLATES)
    missing = required - tracked
    if missing:
        raise ValueError('Source commit is missing server inputs: ' + ', '.join(sorted(missing)))
    payload = {name: git('show', f'{commit}:{name}') for name in sorted(selected)}
    for source, target in TEMPLATES.items():
        payload[target] = git('show', f'{commit}:{source}')
    old_cmd = b'CMD ["python", "main.py"]'
    if payload['Dockerfile'].count(old_cmd) != 1:
        raise ValueError('Unexpected Docker CMD; review the server launch profile')
    payload['Dockerfile'] = payload['Dockerfile'].replace(old_cmd, b'CMD ["python", "main.py", "--profile", "test2"]')
    # Verify the deployed models against the committed provenance, not local weights.
    for stem in ['person_test2_v2', 'work_vest_v4']:
        meta = json.loads(payload[f'models/candidates/{stem}.json'])
        if hashlib.sha256(payload[f'models/candidates/{stem}.onnx']).hexdigest() != meta['onnx_sha256']:
            raise ValueError('Server model differs from provenance: ' + stem)
    clock = json.loads(payload['models/camera_clock.json'])
    if hashlib.sha256(payload['models/camera_clock.onnx']).hexdigest() != clock['onnx_sha256']:
        raise ValueError('Camera clock model differs from provenance')
    cpu = json.loads(git('show', f'{commit}:models/cpu_models.json'))
    if hashlib.sha256(payload['models/person_reid.onnx']).hexdigest() != cpu['reid_onnx_sha256']:
        raise ValueError('Server ReID differs from provenance')
    manifest = {
        'format': 'server_distribution_v1', 'source_commit': commit,
        'source_branch': 'diploma', 'deployment_branch': 'server',
        'source_repository': 'https://github.com/F3nd1uwu/bus_passenger_counter',
        'profile': 'test2', 'child_filter': 'disabled; experimental model assets omitted',
        'files': {name: {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
                  for name, data in sorted(payload.items())},
    }
    output.mkdir(parents=True)
    for name, data in payload.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (output / 'release.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8', newline='\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ref', default='HEAD', help='Committed full-version source')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = build(args.ref, args.output)
    print(f"Server package: {len(result['files']) + 1} files, {sum(row['bytes'] for row in result['files'].values()):,} bytes; source {result['source_commit']}")
