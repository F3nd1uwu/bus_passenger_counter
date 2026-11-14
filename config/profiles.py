"""Explicit camera-specific vision presets; database/source settings stay local."""
from dataclasses import replace
import json

from config.settings import ROOT

PROFILE_NAMES = ("test2",)
VISION_FIELDS = {
    "MODEL_PATH", "REID_MODEL_PATH", "VEST_MODEL_PATH", "STAFF_MODE", "VEST_THRESHOLD",
    "STAFF_DECISION_DELAY_SEC", "VEST_CONFIRM_FRAMES", "CONFIDENCE", "IMAGE_SIZE",
    "TRACKING_MODE", "TRACKER_PATH", "TRACKER_IDENTITY_MIN_SIMILARITY", "CPU_THREADS",
    "TRACKER_DRIFT_THRESHOLD",
    "Y_CABIN",
}


def apply_profile(cfg, name):
    if name is None:
        return cfg
    if name not in PROFILE_NAMES:
        raise ValueError("Unknown vision profile")
    path = ROOT / "config" / "profiles" / (name + ".json")
    values = json.loads(path.read_text(encoding="utf-8"))
    if not set(values) <= VISION_FIELDS:
        raise ValueError("Vision profile contains non-vision settings")
    for key, value in values.items():
        if type(value) is not type(getattr(cfg, key)):
            raise ValueError(f"Invalid profile value type: {key}")
    return replace(cfg, **values).validate()
