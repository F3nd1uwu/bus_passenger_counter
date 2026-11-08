"""Replay real saved detections/features across fresh tracker IDs, without ML IO."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np

from config.settings import Settings
from core.counter_engine import PassengerCounterEngine
from core.types import Detection


def replay(report_path, cfg, repeats=2):
    report_path = Path(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    features = np.load(report_path.with_suffix(".features.npz"))["features"]
    engine = PassengerCounterEngine(cfg)
    frame = np.zeros((480, 640, 3), np.uint8)
    start = datetime(2026, 11, 1, tzinfo=timezone.utc)
    summaries = []
    for repeat in range(repeats):
        engine.reset_tracking()
        before = (engine.total_in, engine.total_out)
        events = []
        for index, row in enumerate(report["trace"]):
            detections = [Detection(d["id"] + repeat * 100000, tuple(d["bbox"]), d["conf"],
                                    features[d["feature"]] if d["feature"] is not None else None,
                                    anchor=d.get("anchor"), anchor_required=d.get("anchor_required", False))
                          for d in row["detections"]]
            events.extend(engine.update(frame, detections, start + timedelta(seconds=(repeat * (len(report["trace"]) + 25) + index) / 25)))
        summary = {"pass": repeat + 1, "new_in": engine.total_in - before[0], "new_out": engine.total_out - before[1],
                   "suppressed": sum(not event.counted for event in events), "identities": len(engine.identities)}
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    replay(args.report, Settings.from_env(), args.repeats)
