"""Upgrade the known legacy model profile without replaying counted video.

Refuses unknown old configurations. Retains cursor, counters, identities and the
entire outbox; SQLite backup is made before the checkpoint header is changed.
"""
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

from config.settings import Settings
from core.service import configuration_key
from database.local_store import LocalStore
from tools.prepare_cpu_models import digest


def upgrade(cfg):
    path = cfg.path(cfg.STATE_PATH)
    if not path.is_file():
        print("No legacy runtime file; the new profile will start normally")
        return
    legacy = replace(cfg, MODEL_PATH="models/yolo11n.pt", REID_MODEL_PATH="models/person_reid.torchscript",
                     TRACKER_PATH="bytetrack_custom.yaml", TRACKING_MODE="iou", CONFIDENCE=.25, POSE_ANCHOR="bbox")
    manifest = json.loads(cfg.path("models/cpu_models.json").read_text(encoding="utf-8"))
    if (manifest["reid_source_sha256"] != digest(legacy.path(legacy.REID_MODEL_PATH))
            or manifest["reid_onnx_sha256"] != digest(cfg.path(cfg.REID_MODEL_PATH))):
        raise ValueError("ReID encoder equivalence was not established; refusing to change state")
    store = LocalStore(path)
    try:
        saved = store.load()
        if saved is None or saved["source_key"] == configuration_key(cfg):
            print("Runtime already matches this configuration")
            return
        if saved["source_key"] != configuration_key(legacy):
            raise ValueError("Unrecognised legacy configuration/source: runtime left unchanged")
        payload = saved["payload"]
        if payload["bus_id"] != cfg.BUS_ID or payload["camera_id"] != cfg.CAMERA_ID:
            raise ValueError("Camera/bus mismatch; runtime left unchanged")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = path.with_name(path.name + ".before-onnx-" + stamp)
        destination = sqlite3.connect(backup)
        try:
            store.conn.backup(destination)
        finally:
            destination.close()
        new_key = configuration_key(cfg)
        payload["model_upgrade"] = {"previous_source_key": saved["source_key"], "at": stamp,
                                    "backup": backup.name, "reid_equivalence_manifest": "models/cpu_models.json"}
        with store.conn:
            store.conn.execute("UPDATE checkpoint SET source_key=?, revision=revision+1, payload=? WHERE id=1",
                               (new_key, json.dumps(payload, allow_nan=False)))
        print(f"Runtime upgraded: frame {payload['next_frame']}; queued events {store.pending_count()}; backup {backup.name}")
    finally:
        store.close()


if __name__ == "__main__":
    upgrade(Settings.from_env().validate())
