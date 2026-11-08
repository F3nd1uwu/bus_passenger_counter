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


def check_links(files):
    for name in files:
        if not name.endswith(".md"):
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
    check_links(files)
    print(f"Repository checks passed: {len(files)} files, four model checksums, audit reconciliation, documentation links.")


if __name__ == "__main__":
    main()
