"""Reproducible, database-free diagnosis of a source frame interval."""
import argparse
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time

import cv2
import numpy as np

from config.settings import Settings
from core.counter_engine import PassengerCounterEngine
from core.detector import PersonDetector
from core.video_source import VideoSource, source_kind
from core.visualization import draw_preview, PreviewWriter


def evaluate(cfg, start, end, output, *, video=False):
    cfg.validate()
    if source_kind(cfg.VIDEO_SOURCE, cfg.SOURCE_KIND) != "file" or start < 0 or end < start:
        raise ValueError("Evaluation requires a file and 0 <= start <= end")
    output = Path(output).resolve()
    if output.suffix.lower() != ".json":
        raise ValueError("Evaluation output must be a .json report")
    if video and output.with_suffix(".mp4") == cfg.path(cfg.VIDEO_SOURCE).resolve():
        raise ValueError("Annotated output must differ from the input video")
    output.parent.mkdir(parents=True, exist_ok=True)
    detector = PersonDetector(cfg)
    engine = PassengerCounterEngine(cfg)
    reader = VideoSource(cfg, datetime(2026, 11, 1, tzinfo=timezone.utc), resume_frame=start)
    stages = defaultdict(list)
    current = {}
    inference_method = "predict" if detector.appearance_tracker is not None else "track"
    original_track = getattr(detector.model, inference_method)
    feature_method = "encode_many" if detector.appearance_tracker is not None else "encode"
    original_encode = getattr(detector.reid, feature_method)

    def timed_track(*args, **kwargs):
        tick = time.perf_counter()
        value = original_track(*args, **kwargs)
        current["yolo"] = time.perf_counter() - tick
        return value

    def timed_encode(*args, **kwargs):
        tick = time.perf_counter()
        value = original_encode(*args, **kwargs)
        current["reid"] = current.get("reid", 0) + time.perf_counter() - tick
        return value

    setattr(detector.model, inference_method, timed_track)
    setattr(detector.reid, feature_method, timed_encode)
    trace, tracks, all_events = [], {}, []
    feature_indices, feature_bank = {}, []
    writer = None
    wall = time.perf_counter()
    try:
        while reader.index <= end:
            current.clear()
            tick = time.perf_counter()
            packet = reader.read()
            read_seconds = time.perf_counter() - tick
            if packet is None:
                break
            tick = time.perf_counter()
            detections = detector.detect(packet.image)
            detect_seconds = time.perf_counter() - tick
            tick = time.perf_counter()
            events = engine.update(packet.image, detections, packet.timestamp)
            count_seconds = time.perf_counter() - tick
            row = {"frame": packet.index, "detections": [], "events": []}
            for detection in detections:
                x1, y1, x2, y2 = detection.bbox
                anchor = (detection.anchor[1] if detection.anchor is not None else
                          y1 + (y2 - y1) * cfg.ANCHOR_Y) / packet.image.shape[0]
                track = engine.tracks.get(detection.track_id)
                feature_index = None
                if detection.embedding is not None:
                    vector = np.asarray(detection.embedding, np.float32)
                    key = vector.tobytes()
                    if key not in feature_indices:
                        feature_indices[key] = len(feature_bank)
                        feature_bank.append(vector.copy())
                    feature_index = feature_indices[key]
                row["detections"].append({"id": detection.track_id, "bbox": [round(x, 2) for x in detection.bbox],
                                          "conf": round(detection.confidence, 3), "anchor_y": round(anchor, 4),
                                          "anchor_valid": not detection.anchor_required or detection.anchor is not None,
                                          "anchor": detection.anchor, "anchor_required": detection.anchor_required,
                                          "feature": feature_index,
                                          "zone": track.zone if track else None,
                                          "candidate": track.candidate if track else None,
                                          "identity": track.identity_id if track else None})
                info = tracks.setdefault(detection.track_id, {"first": packet.index, "last": packet.index,
                                                             "min_y": anchor, "max_y": anchor, "samples": 0})
                info.update(last=packet.index, min_y=min(info["min_y"], anchor),
                            max_y=max(info["max_y"], anchor), samples=info["samples"] + 1)
            for event in events:
                record = dict(event.to_dict(), frame=packet.index)
                all_events.append(record)
                row["events"].append(record)
                print(packet.index, event.event_type, event.counted, event.reason, "track", event.track_id, flush=True)
            for name, value in (("read", read_seconds), ("detector", detect_seconds),
                                ("yolo", current.get("yolo", 0)), ("reid", current.get("reid", 0)), ("counter", count_seconds)):
                stages[name].append(value)
            trace.append(row)
            if video:
                image = draw_preview(packet.image, detections, cfg, engine, events,
                                     metrics={"processing_fps": 1 / max(1e-9, read_seconds + detect_seconds + count_seconds),
                                              "source_fps": reader.fps, "frame_index": packet.index,
                                              "media_seconds": (packet.timestamp - reader.start).total_seconds(),
                                              "segment_start": start})
                if writer is None:
                    writer = PreviewWriter(output.with_suffix(".mp4"), reader.fps, (image.shape[1], image.shape[0]))
                writer.write(image)
            if len(trace) % 200 == 0:
                print("processed", len(trace), "IN", engine.total_in, "OUT", engine.total_out, flush=True)
    finally:
        reader.close()
        if writer:
            writer.close()
    elapsed = time.perf_counter() - wall
    report = {"model": cfg.MODEL_PATH, "image_size": cfg.IMAGE_SIZE, "anchor_y": cfg.ANCHOR_Y,
              "reid": cfg.REID_MODEL_PATH, "tracking_mode": cfg.TRACKING_MODE, "tracker": cfg.TRACKER_PATH,
              "confidence": cfg.CONFIDENCE, "cpu_threads": cfg.CPU_THREADS, "pose_anchor": cfg.POSE_ANCHOR,
              "association_similarity": cfg.TRACKER_IDENTITY_MIN_SIMILARITY,
              "start_frame": start, "end_frame": end, "frames": len(trace), "elapsed_sec": elapsed,
              "processing_fps": len(trace) / max(1e-9, elapsed), "in": engine.total_in, "out": engine.total_out,
              "suppressed": engine.total_suppressed, "identities": len(engine.identities),
              "mean_ms": {name: statistics.mean(values) * 1000 for name, values in stages.items()},
              "tracks": tracks, "events": all_events, "trace": trace}
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(output.with_suffix(".features.npz"),
                        features=np.stack(feature_bank) if feature_bank else np.empty((0, detector.reid.dimension), np.float32))
    print(json.dumps({k: v for k, v in report.items() if k not in {"trace", "tracks", "events"}}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/test.mp4")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--anchor-y", type=float)
    parser.add_argument("--pose-anchor", choices=["head", "bbox"])
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--association-similarity", type=float)
    parser.add_argument("--tracking-mode", choices=["iou", "appearance"])
    parser.add_argument("--tracker")
    parser.add_argument("--reid")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    if args.start < 0 or args.end < args.start:
        parser.error("Expected 0 <= start <= end")
    overrides = {"VIDEO_SOURCE": args.source}
    for field, value in (("MODEL_PATH", args.model), ("IMAGE_SIZE", args.image_size), ("ANCHOR_Y", args.anchor_y),
                         ("POSE_ANCHOR", args.pose_anchor), ("CONFIDENCE", args.confidence),
                         ("TRACKER_IDENTITY_MIN_SIMILARITY", args.association_similarity),
                         ("TRACKING_MODE", args.tracking_mode), ("TRACKER_PATH", args.tracker),
                         ("REID_MODEL_PATH", args.reid)):
        if value is not None:
            overrides[field] = value
    evaluate(replace(Settings.from_env(), **overrides), args.start, args.end, args.output, video=args.video)
