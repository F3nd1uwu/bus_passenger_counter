"""Validated configuration. Geometry uses fractions of the actual frame size."""
from dataclasses import dataclass, fields
import math
import os
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "bus_analytics"
    DB_USER: str = "bus_counter"
    DB_PASS: str = ""
    DB_SSLMODE: str = "prefer"
    BUS_ID: str = "BUS_01"
    CAMERA_ID: str = "CAM_DOOR_01"
    TIMEZONE: str = "Europe/Saratov"
    VIDEO_SOURCE: str = "data/test.mp4"
    SOURCE_KIND: str = "auto"
    VIDEO_START: str = ""
    MODEL_PATH: str = "models/yolo26n.onnx"
    REID_MODEL_PATH: str = "models/person_reid.onnx"
    VEST_MODEL_PATH: str = ""
    STAFF_MODE: str = "disabled"
    CHILD_MODE: str = "disabled"
    CHILD_MODEL_PATH: str = "models/candidates/child_filter_v1.json"
    CHILD_CONFIRM_SAMPLES: int = 3
    CHILD_DECISION_DELAY_SEC: float = 10.0
    TRACKER_PATH: str = "botsort_reid.yaml"
    TRACKING_MODE: str = "appearance"
    TRACKER_IDENTITY_MIN_SIMILARITY: float = 0.5
    TRACKER_DRIFT_THRESHOLD: float = 0.0
    STATE_PATH: str = "data/runtime.sqlite3"
    DEVICE: str = "cpu"
    CPU_THREADS: int = 4
    PERSON_CLASS_ID: int = 0
    CONFIDENCE: float = 0.10
    IMAGE_SIZE: int = 640
    ANCHOR_Y: float = 0.5
    POSE_ANCHOR: str = "bbox"
    POSE_ANCHOR_CONFIDENCE: float = 0.35
    Y_STREET: float = 0.38
    Y_LINE: float = 0.50
    Y_CABIN: float = 0.60
    DOOR_X_MIN: float = 0.04
    DOOR_X_MAX: float = 0.90
    ENTER_FROM: str = "street"
    ZONE_CONFIRM_FRAMES: int = 2
    MIN_TRACK_FRAMES: int = 3
    TRACK_TTL_SEC: float = 2.0
    DUPLICATE_INTERVAL_SEC: float = 3600.0
    REID_THRESHOLD: float = 0.85
    REID_MARGIN: float = 0.06
    REID_GALLERY_SIZE: int = 8
    REID_GALLERY_SIMILARITY: float = 0.98
    MAX_IDENTITIES: int = 10000
    VEST_THRESHOLD: float = 0.90
    VEST_CONFIRM_FRAMES: int = 3
    STAFF_DECISION_DELAY_SEC: float = 10.0
    FEATURE_INTERVAL_FRAMES: int = 5
    RECONNECT_DELAY_SEC: float = 2.0
    OPEN_TIMEOUT_MS: int = 5000
    READ_TIMEOUT_MS: int = 5000
    SYNC_INTERVAL_SEC: float = 5.0
    CHECKPOINT_INTERVAL_SEC: float = 1.0
    REPLAY_FRAMES: int = 90

    @classmethod
    def from_env(cls):
        defaults = cls()
        values = {}
        for field in fields(cls):
            value = os.environ.get(field.name)
            if value is not None:
                try:
                    values[field.name] = type(getattr(defaults, field.name))(value)
                except ValueError as exc:
                    raise ValueError(f"Invalid {field.name}") from exc
        return cls(**values)

    def path(self, value):
        path = Path(value).expanduser()
        return path if path.is_absolute() else ROOT / path

    def validate(self):
        ZoneInfo(self.TIMEZONE)
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{field.name} must be finite")
        if not 0 <= self.Y_STREET < self.Y_LINE < self.Y_CABIN <= 1:
            raise ValueError("Expected 0 <= Y_STREET < Y_LINE < Y_CABIN <= 1")
        if not 0 <= self.DOOR_X_MIN < self.DOOR_X_MAX <= 1:
            raise ValueError("Invalid door horizontal bounds")
        for name in ("CONFIDENCE", "REID_THRESHOLD", "REID_GALLERY_SIMILARITY", "VEST_THRESHOLD", "POSE_ANCHOR_CONFIDENCE", "TRACKER_IDENTITY_MIN_SIMILARITY"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if not 0 <= self.ANCHOR_Y <= 1 or not 0 <= self.REID_MARGIN < 1:
            raise ValueError("Invalid ANCHOR_Y or REID_MARGIN")
        if not 0 <= self.TRACKER_DRIFT_THRESHOLD <= 1:
            raise ValueError("TRACKER_DRIFT_THRESHOLD must be in [0, 1]")
        if self.STAFF_DECISION_DELAY_SEC < 0:
            raise ValueError("STAFF_DECISION_DELAY_SEC must be nonnegative")
        if self.CHILD_MODE not in {"disabled", "model"}:
            raise ValueError("CHILD_MODE must be disabled or model")
        if self.CHILD_CONFIRM_SAMPLES < 3 or self.CHILD_DECISION_DELAY_SEC < 0:
            raise ValueError("Child filter requires at least three samples and a nonnegative delay")
        for name in ("DB_PORT", "IMAGE_SIZE", "ZONE_CONFIRM_FRAMES", "MIN_TRACK_FRAMES",
                     "TRACK_TTL_SEC", "DUPLICATE_INTERVAL_SEC", "MAX_IDENTITIES",
                     "REID_GALLERY_SIZE",
                     "CPU_THREADS",
                     "VEST_CONFIRM_FRAMES", "FEATURE_INTERVAL_FRAMES", "RECONNECT_DELAY_SEC",
                     "OPEN_TIMEOUT_MS", "READ_TIMEOUT_MS", "SYNC_INTERVAL_SEC", "CHECKPOINT_INTERVAL_SEC", "REPLAY_FRAMES"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.SOURCE_KIND not in {"auto", "file", "stream", "camera"}:
            raise ValueError("SOURCE_KIND must be auto, file, stream or camera")
        if self.ENTER_FROM not in {"street", "cabin"}:
            raise ValueError("ENTER_FROM must be street or cabin")
        if self.STAFF_MODE not in {"disabled", "model"}:
            raise ValueError("STAFF_MODE must be disabled or model")
        if self.POSE_ANCHOR not in {"head", "bbox"}:
            raise ValueError("POSE_ANCHOR must be head or bbox")
        if self.TRACKING_MODE not in {"iou", "appearance"}:
            raise ValueError("TRACKING_MODE must be iou or appearance")
        if not self.BUS_ID or not self.CAMERA_ID or max(len(self.BUS_ID), len(self.CAMERA_ID)) > 32:
            raise ValueError("BUS_ID and CAMERA_ID must contain 1..32 characters")
        if self.PERSON_CLASS_ID < 0:
            raise ValueError("PERSON_CLASS_ID must be nonnegative")
        return self
