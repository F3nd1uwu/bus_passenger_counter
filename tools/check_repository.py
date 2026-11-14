"""Check public files, model hashes, audit numbers and local documentation links.

Runs without downloading models or opening a database. Local secrets are compared
without printing their values. Intended for pre-commit review and CI.
"""
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit

from config.settings import ROOT


PUBLIC_MODELS = {
    "models/yolo26n.onnx", "models/person_reid.onnx", "models/person_reid.torchscript",
    "models/candidates/person_manual_v1.onnx",
    "models/candidates/person_test2_v2.onnx", "models/candidates/work_vest_v2.onnx",
    "models/candidates/work_vest_v1.onnx",
    "models/candidates/work_vest_v3.onnx",
    "models/candidates/child_age_v1.onnx", "models/candidates/child_vision_v1.onnx",
}
SECRET_PATTERNS = [
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{50,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
]
CREDENTIAL_URL = re.compile(r"(?:rtsp|rtsps|postgres|postgresql)://[^\s:/]+:[^\s@]+@[^\s\"']+", re.I)


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def repository_files():
    result = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                            cwd=ROOT, check=True, capture_output=True)
    return sorted(set(result.stdout.decode("utf-8").rstrip("\0").split("\0")))


def local_private_values():
    path = ROOT / ".env"
    if not path.is_file():
        return []
    values = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip("\"'")
        if key.strip() == "DB_PASS" and len(value) >= 12 and value not in {
            "test-local-only", "replace-with-a-long-random-password",
        }:
            values.append(value)
        if key.strip() == "VIDEO_SOURCE":
            parsed = urlsplit(value)
            if parsed.password and len(parsed.password) >= 8:
                values.append(unquote(parsed.password))
    return values


def check_files(files):
    private = local_private_values()
    allowed_data = {"data/.gitkeep", "datasets/README.md"}
    for name in files:
        path = Path(name)
        if path.name.startswith(".env") and name != ".env.example":
            raise ValueError(f"Private environment included: {name}")
        if set(path.parts) & {"venv", ".venv", ".idea", ".vscode", "__pycache__", ".local", "runs"}:
            raise ValueError(f"Local-only directory included: {name}")
        if path.parts[0] in {"data", "datasets", "state"} and name not in allowed_data:
            raise ValueError(f"Private recording/state/dataset included: {name}")
        if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".npz", ".zip", ".db", ".sqlite3", ".pem", ".key"}:
            raise ValueError(f"Unexpected private/generated asset: {name}")
        if path.suffix.lower() in {".onnx", ".torchscript", ".pt", ".pth"}:
            if name not in PUBLIC_MODELS:
                raise ValueError(f"Unapproved model asset: {name}")
            continue
        if (ROOT / name).stat().st_size > 20 * 1024 * 1024:
            raise ValueError(f"Unexpected large file: {name}")
        try:
            text = (ROOT / name).read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Unexpected non-text file: {name}") from exc
        # Match across patterns assembled at runtime; never print the matched secret.
        urls = [urlsplit(match.group()) for match in CREDENTIAL_URL.finditer(text)]
        private_url = any(not (name.startswith("tests/") and url.hostname == "example.invalid"
                               and url.username == "user" and url.password == "secret") for url in urls)
        if private_url or any(pattern.search(text) for pattern in SECRET_PATTERNS) or any(value in text for value in private):
            raise ValueError(f"Potential credential in public file: {name}")


def check_models():
    cpu = read_json("models/cpu_models.json")
    reid = read_json("models/person_reid.json")
    candidate = read_json("models/candidates/person_manual_v1.json")
    expected = {
        "models/yolo26n.onnx": cpu["detector_onnx_sha256"],
        "models/person_reid.onnx": cpu["reid_onnx_sha256"],
        "models/person_reid.torchscript": reid["export_sha256"],
        "models/candidates/person_manual_v1.onnx": candidate["onnx_sha256"],
    }
    for stem in ("person_test2_v2", "work_vest_v1", "work_vest_v2", "work_vest_v3"):
        metadata = read_json(f"models/candidates/{stem}.json")
        expected[f"models/candidates/{stem}.onnx"] = metadata["onnx_sha256"]
    child = read_json('models/candidates/child_filter_v1.json')
    if child.get('format') != 'approx_child_ensemble_v1' or child.get('status') != 'experimental':
        raise ValueError('Unexpected child ensemble contract')
    if child['maximum_child_age'] != 7 or child['minimum_child_margin'] < .005:
        raise ValueError('Invalid conservative child policy')
    for name, filename in [('age', 'child_age_v1.onnx'), ('vision', 'child_vision_v1.onnx')]:
        if child[name]['file'] != filename:
            raise ValueError('Unexpected child model filename')
        expected['models/candidates/'+filename] = child[name]['sha256']
    for asset in child['source_assets']:
        if asset['file'].startswith('LICENSE.') and sha256(ROOT/'models/candidates'/asset['file']) != asset['sha256']:
            raise ValueError('Child model license checksum differs')
    for filename, version, fusion in [("work_vest_ensemble", 1, "min"), ("work_vest_spatial", 2, "spatial_min")]:
        bundle = read_json(f"models/candidates/{filename}.json")
        if bundle.get("format") != f"grayscale_garment_ensemble_v{version}" or bundle.get("fusion") != fusion:
            raise ValueError("Unexpected garment ensemble contract")
        if version == 2 and not 0 < bundle.get("minimum_iou", 0) <= 1:
            raise ValueError("Invalid garment spatial consensus threshold")
        if len(bundle["members"]) != 2 or {m["path"] for m in bundle["members"]} != {"work_vest_v1.onnx", "work_vest_v2.onnx"}:
            raise ValueError("Unexpected garment ensemble members")
        for member in bundle["members"]:
            if member["sha256"] != expected["models/candidates/" + member["path"]]:
                raise ValueError("Garment ensemble member checksum mismatch")
    for name, digest in expected.items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Model checksum mismatch: {name}")


def check_audit():
    root = ROOT / "reports/independent-audit-2026-11-08"
    result = json.loads((root / "results.json").read_text(encoding="utf-8"))
    freeze = json.loads((root / "manual_freeze.json").read_text(encoding="utf-8"))
    for cid, digest in freeze["label_sha256"].items():
        if sha256(root / f"labels/frozen/{cid}.json") != digest:
            raise ValueError(f"Frozen manual annotation changed: {cid}")
    with (root / "event_matching.csv").open(encoding="utf-8-sig", newline="") as stream:
        events = list(csv.DictReader(stream, delimiter=";"))
    metric = result["metrics"]
    if len(events) != 53 or len(result["episodes"]) != 16:
        raise ValueError("Audit size changed without a revised protocol")
    for direction in ["ALL", "IN", "OUT"]:
        selected = [e for e in events if direction == "ALL" or e["direction"] == direction]
        tp = sum(e["result"] == "TP" for e in selected)
        fn = sum(e["result"] == "FN" for e in selected)
        m = metric[direction]
        if (m["tp"], m["fp"], m["fn"]) != (tp, 0, fn):
            raise ValueError(f"Audit event reconciliation failed: {direction}")
        if abs(m["recall"] - tp / (tp + fn)) > 1e-12:
            raise ValueError(f"Audit recall mismatch: {direction}")
        if abs(m["f1"] - 2 * tp / (2 * tp + fn)) > 1e-12:
            raise ValueError(f"Audit F1 mismatch: {direction}")
    for direction in ["in", "out"]:
        if sum(e[f"manual_{direction}"] for e in result["episodes"]) != metric[f"manual_{direction}"]:
            raise ValueError(f"Manual totals mismatch: {direction}")
        if sum(e[f"model_{direction}"] for e in result["episodes"]) != metric[f"model_{direction}"]:
            raise ValueError(f"Model totals mismatch: {direction}")
    if sum(e["exact_counts"] for e in result["episodes"]) != metric["exact_episodes"]:
        raise ValueError("Exact episode count mismatch")


def check_countpeople_report():
    root = ROOT / "reports/countpeople-candidate-v1"
    for stem in ("manual_holdout", "manual_staff_holdout"):
        expected = (root / f"{stem}.sha256").read_text(encoding="utf-8").split()[0]
        if sha256(root / f"{stem}.json") != expected:
            raise ValueError("Frozen additional video annotation changed")
    result = read_json("reports/countpeople-candidate-v1/video_comparison.json")
    episodes = result["episodes"]
    if len(episodes) != 33 or len({row["id"] for row in episodes}) != 33:
        raise ValueError("CountPeople video comparison size changed")
    old = [row for row in episodes if row["id"].startswith(("old_", "control_"))]
    new = [row for row in episodes if row not in old]
    if len(old) != 25 or sum(r["manual_in"] + r["manual_out"] for r in new) != 28:
        raise ValueError("CountPeople manual event totals changed")
    for label, summary in result["summary"].items():
        exact = sum((row["models"][label]["in"], row["models"][label]["out"])
                    == (row["manual_in"], row["manual_out"]) for row in old)
        counted = sum(row["models"][label]["in"] + row["models"][label]["out"] for row in new)
        if exact != summary["old_exact_episode_totals"] or counted != summary["new_counted_events"]:
            raise ValueError("CountPeople comparison totals disagree with episodes")
    review = read_json("reports/countpeople-candidate-v1/event_review.json")
    for label, metric in review["eight_calibration_episode_metrics"].items():
        tp, fn, fp = (metric[key] for key in ("TP", "FN", "FP"))
        if tp + fn != 28 or tp + fp != result["summary"][label]["new_counted_events"]:
            raise ValueError("Reviewed CountPeople events do not reconcile")
        if abs(metric["f1"] - 2 * tp / (2 * tp + fn + fp)) > 1e-9:
            raise ValueError("Reviewed CountPeople F1 does not reconcile")
    for csv_name, epochs in (("person_training.csv", 8), ("garment_training.csv", 9)):
        with (root / csv_name).open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != epochs or int(rows[-1]["epoch"]) != epochs:
            raise ValueError("Training epoch count includes a non-training validation row")
    frozen = read_json("reports/test2-2026-11-13/results.json")["profile_sha256"]
    if sha256(ROOT / "reports/test2-2026-11-13/profile_snapshot.json") != frozen["config/profiles/test2.json"]:
        raise ValueError("Historical test2 profile snapshot differs from its reported hash")


def check_identity_report():
    root = ROOT / 'reports/id-continuity-2026-11-14'
    result = read_json('reports/id-continuity-2026-11-14/results.json')
    labels = read_json('reports/id-continuity-2026-11-14/identity_pairs.json')
    if sha256(root/'identity_pairs.json') != result['identity_labels_sha256']:
        raise ValueError('Identity pair labels differ from their reported hash')
    if sha256(root/'profile.json') != result['profile_sha256']:
        raise ValueError('ID report profile snapshot hash differs')
    if len(labels['pairs']) != 5 or sum(p['same_person'] for p in labels['pairs']) != 4:
        raise ValueError('Expected four continuity pairs and one distinct-person pair')
    episodes = result['episodes']
    if len(episodes) != 38 or len({e['id'] for e in episodes}) != 38:
        raise ValueError('Expected 38 unique ID/zone regression episodes')
    controls = [e for e in episodes if e['scope'] == 'known_regression']
    if len(controls) != 25:
        raise ValueError('Expected 25 historical controls')
    for episode in episodes:
        for tag in ['baseline', 'released']:
            row = episode[tag]
            counted = [e for e in row['events'] if e['counted']]
            if (row['in'], row['out']) != tuple(sum(e['event_type'] == d for e in counted) for d in ['IN', 'OUT']):
                raise ValueError('Reported totals do not reconcile with saved crossing events')
        if episode in controls and (episode['released']['in'], episode['released']['out']) != (episode['manual_in'], episode['manual_out']):
            raise ValueError('An old control regressed in the ID report')
    for tag in ['baseline', 'released']:
        pairs = result['identity_pairs'][tag]['pairs']
        if [{k: p[k] for k in ['left', 'right', 'same_person']} for p in pairs] != labels['pairs']:
            raise ValueError('Reported identity pairs differ from manual pair definitions')
        for label in ['tracker', 'identity']:
            totals = {s: sum(p[label] == s for p in pairs) for s in ['correct', 'incorrect', 'unresolved']}
            expected = {'correct': 5, 'incorrect': 0, 'unresolved': 0} if tag == 'released' else {'correct': 4, 'incorrect': 1, 'unresolved': 0}
            if totals != expected or totals != result['identity_pairs'][tag]['summary'][label]:
                raise ValueError('Identity pair summaries do not reconcile')
    for name, digest in result['models_sha256'].items():
        if sha256(ROOT/name) != digest:
            raise ValueError('Identity report model checksum differs')


def check_child_report():
    result = read_json('reports/child-filter-2026-11-14/results.json')
    if result['status'] != 'experimental_not_validated_for_child_exclusion':
        raise ValueError('Child report must describe its experimental validation status')
    counted, excluded = 0, 0
    stats = {key: 0 for key in result['summary']['age_jobs']}
    for row in result['episodes']:
        events = row['events']
        totals = tuple(sum(e['counted'] and e['event_type'] == d for e in events) for d in ['IN', 'OUT'])
        if totals != (row['in'], row['out']) or totals != (row['baseline_in'], row['baseline_out']):
            raise ValueError('Child report counts differ from events or preserved baseline')
        counted += sum(totals)
        excluded += sum(e['reason'] == 'CHILD' for e in events)
        if any(e['counted'] for e in events if e['reason'] == 'CHILD'):
            raise ValueError('Child exclusions cannot increase passenger totals')
        for key, value in row['child_analysis'].items():
            stats[key] += value
    summary = result['summary']
    if (counted != summary['counted_passages'] or excluded != summary['child_excluded_events']
            or stats != summary['age_jobs'] or not summary['all_baseline_totals_preserved']):
        raise ValueError('Child report summary does not reconcile')
    if summary['child_precision'] is not None or summary['child_recall'] is not None:
        raise ValueError('Child accuracy cannot be reported without verified labels')
    for name, digest in result['models_sha256'].items():
        if sha256(ROOT/name) != digest:
            raise ValueError('Child report model checksum differs')


def check_release_report():
    result = read_json('reports/release-2026-11-14/verification.json')
    manifest_path = ROOT/'reports/release-2026-11-14/server_manifest.json'
    manifest = read_json('reports/release-2026-11-14/server_manifest.json')
    server = result['server']
    if (manifest.get('format') != 'server_distribution_v1' or manifest.get('profile') != 'test2'
            or manifest['source_commit'] != server['source_commit']
            or sha256(manifest_path) != server['manifest_sha256']
            or len(manifest['files']) + 1 != server['file_count']
            or sum(row['bytes'] for row in manifest['files'].values()) != server['payload_bytes']):
        raise ValueError('Server release manifest and verification differ')
    models = {name for name in manifest['files'] if name.endswith('.onnx')}
    if models != {'models/person_reid.onnx', 'models/candidates/person_test2_v2.onnx',
                  'models/candidates/work_vest_v3.onnx'}:
        raise ValueError('Unexpected deployed model set')
    for name in models:
        if sha256(ROOT/name) != manifest['files'][name]['sha256']:
            raise ValueError('Release report model differs from the active weights')
    for key in ['tests', 'clean_clone']:
        row = result[key]
        discovered = row.get('discovered', row.get('tests_discovered'))
        if row['failed'] or row['passed'] + row['skipped'] != discovered:
            raise ValueError('Release test totals do not reconcile')


def check_links(files):
    for name in files:
        if not name.endswith(".md"):
            continue
        if name.startswith('deployment/server/'):
            # Template links are relative to the generated distribution root.
            # tools.verify_release validates them after packaging.
            continue
        path = ROOT / name
        text = path.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^\]]*\]\(([^\s)]+)", text):
            if target.startswith(("https://", "http://", "mailto:", "#")):
                continue
            target = unquote(target.split("#", 1)[0])
            if target and not (path.parent / target).exists():
                raise ValueError(f"Broken local documentation link in {name}: {target}")


def main():
    files = repository_files()
    check_files(files)
    check_models()
    check_audit()
    check_countpeople_report()
    check_identity_report()
    check_child_report()
    check_release_report()
    check_links(files)
    print(f"Repository checks passed: {len(files)} files, {len(PUBLIC_MODELS)} model checksums, audit reconciliation, documentation links.")


if __name__ == "__main__":
    main()
