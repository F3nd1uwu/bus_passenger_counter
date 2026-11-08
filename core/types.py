from dataclasses import asdict, dataclass
from datetime import datetime
from uuid import uuid4

import numpy as np


@dataclass
class Detection:
    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float = 1.0
    embedding: np.ndarray | None = None
    staff_probability: float | None = None
    anchor: tuple[float, float] | None = None
    anchor_required: bool = False


@dataclass(frozen=True)
class CrossingEvent:
    event_id: str
    bus_id: str
    camera_id: str
    detected_at: str
    traffic_date: str
    event_type: str
    track_id: int
    identity_id: str
    counted: bool
    reason: str

    @classmethod
    def create(cls, cfg, now: datetime, direction, track_id, identity_id, counted, reason):
        from zoneinfo import ZoneInfo
        return cls(str(uuid4()), cfg.BUS_ID, cfg.CAMERA_ID, now.isoformat(),
                   now.astimezone(ZoneInfo(cfg.TIMEZONE)).date().isoformat(), direction,
                   track_id, identity_id, counted, reason)

    def to_dict(self):
        return asdict(self)
