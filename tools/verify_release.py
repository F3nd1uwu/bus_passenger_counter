"""Check the server distribution's file hashes without opening a database."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'release.json').read_text('utf-8'))
    if (manifest.get('format') != 'server_distribution_v1'
            or not re.fullmatch(r'[0-9a-f]{40}', manifest.get('source_commit', ''))
            or manifest.get('profile') != 'test2' or not manifest.get('files')):
        raise ValueError('Invalid server release manifest')
    model_names = set()
    for name, expected in manifest['files'].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
            raise ValueError('Invalid release path')
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Missing release file: ' + name)
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if path.stat().st_size != expected['bytes'] or digest != expected['sha256']:
            raise ValueError('Release file differs: ' + name)
        if relative.suffix == '.onnx':
            model_names.add(name)
    if model_names != {'models/person_reid.onnx', 'models/candidates/person_test2_v2.onnx',
                       'models/candidates/work_vest_v3.onnx'}:
        raise ValueError('Unexpected server model set')
    from urllib.parse import unquote
    for name in manifest['files']:
        if not name.endswith('.md'):
            continue
        for target in re.findall(r'\[[^\]]*\]\(([^\s)]+)', (root / name).read_text('utf-8')):
            if target.startswith(('https://', 'http://', 'mailto:', '#')):
                continue
            target = unquote(target.split('#', 1)[0])
            if target and not ((root / name).parent / target).exists():
                raise ValueError('Broken distribution documentation link: ' + name)
    print(f"Release verified: {len(manifest['files'])} file checksums; source {manifest['source_commit']}")
    return manifest


def smoke(root):
    # A synthetic black frame exercises all three local models, no camera or DB.
    sys.path.insert(0, str(Path(root).resolve()))
    import numpy as np
    from config.settings import Settings
    from config.profiles import apply_profile
    from core.detector import PersonDetector
    cfg = apply_profile(Settings(), 'test2')
    detector = PersonDetector(cfg)
    try:
        frame = np.zeros((480, 640, 3), np.uint8)
        detector.detect(frame)
        feature = detector.reid.encode(frame, (160, 100, 320, 450))
        score = detector.vest.predict(frame, (160, 100, 320, 450))
        if feature is None or not np.isfinite(feature).all() or score is None or not 0 <= score <= 1:
            raise RuntimeError('Synthetic ReID or vest inference failed')
    finally:
        detector.close()
    print('Local model smoke passed; staff model enabled, child filter disabled')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--smoke', action='store_true', help='Also load the local models and process a synthetic frame')
    args = parser.parse_args()
    verify(args.root)
    if args.smoke:
        smoke(args.root)
