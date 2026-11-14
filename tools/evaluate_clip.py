"""Reproducible, database-free diagnosis of a source frame interval."""
import argparse
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import cv2
import numpy as np

from config.settings import Settings
from config.profiles import PROFILE_NAMES, apply_profile
from core.counter_engine import PassengerCounterEngine
from core.detector import PersonDetector
from core.video_source import VideoSource, source_kind, parse_start
from core.visualization import draw_preview, PreviewWriter


def evaluate(cfg, start, end, output, *, video=False, compact=False):
    cfg.validate()
    if source_kind(cfg.VIDEO_SOURCE, cfg.SOURCE_KIND) != "file" or start < 0 or end < start:
        raise ValueError("Evaluation requires a file and 0 <= start <= end")
    output = Path(output).resolve()
    if output.suffix.lower() != ".json":
        raise ValueError("Evaluation output must be a .json report")
    if video and output.with_suffix(".mp4") == cfg.path(cfg.VIDEO_SOURCE).resolve():
        raise ValueError("Annotated output must differ from the input video")
    artifacts = [output, output.with_suffix(".features.npz")]
    if video:
        artifacts.append(output.with_suffix(".mp4"))
    if any(p.exists() for p in artifacts):
        raise ValueError("Evaluation output already exists; choose a new report name")
    output.parent.mkdir(parents=True, exist_ok=True)
    detector = PersonDetector(cfg)
    engine = PassengerCounterEngine(cfg)
    reader = VideoSource(cfg, parse_start(cfg.VIDEO_START) if cfg.VIDEO_START else
                         datetime(2026, 11, 1, tzinfo=timezone.utc), resume_frame=start)
    stages = defaultdict(float)
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
    processed = 0
    last_frame = None
    pending_frames = {}
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
            events = engine.update(packet.image, detections, packet.timestamp,
                                   child_observations=detector.child_observations)
            count_seconds = time.perf_counter() - tick
            row = {"frame": packet.index, "detections": [], "events": []}
            for detection in detections:
                x1, y1, x2, y2 = detection.bbox
                anchor = (detection.anchor[1] if detection.anchor is not None else
                          y1 + (y2 - y1) * cfg.ANCHOR_Y) / packet.image.shape[0]
                track = engine.tracks.get(detection.track_id)
                feature_index = None
                if detection.embedding is not None and not compact:
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
                                          "separated_from": detection.separated_from,
                                          "feature": feature_index,
                                          "zone": track.zone if track else None,
                                          "candidate": track.candidate if track else None,
                                          "identity": track.identity_id if track else None})
                row['detections'][-1]['child'] = dict(confirmed=track.child.confirmed, samples=track.child.samples,
                                                      adult_seen=track.child.adult_seen) if track else None
                row["detections"][-1].update(staff_probability=detection.staff_probability,
                                             staff_confirmed=track.staff.confirmed if track else False)
                info = tracks.setdefault(detection.track_id, {"first": packet.index, "last": packet.index,
                                                             "min_y": anchor, "max_y": anchor, "samples": 0})
                info.update(last=packet.index, min_y=min(info["min_y"], anchor),
                            max_y=max(info["max_y"], anchor), samples=info["samples"] + 1)
            for event in events:
                crossed = pending_frames.get(event.detected_at)
                if event.detected_at == packet.timestamp.isoformat():
                    crossed = packet.index
                record = dict(event.to_dict(), frame=packet.index, crossing_frame=crossed)
                all_events.append(record)
                row["events"].append(record)
                print(packet.index, event.event_type, event.counted, event.reason, "track", event.track_id, flush=True)
            pending_frames = {when.isoformat(): pending_frames.get(when.isoformat(), packet.index)
                              for track in engine.tracks.values() for _, when in track.pending}
            for name, value in (("read", read_seconds), ("detector", detect_seconds),
                                ("yolo", current.get("yolo", 0)), ("reid", current.get("reid", 0)), ("counter", count_seconds)):
                stages[name] += value
            processed += 1
            last_frame = packet.index
            if not compact:
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
            if processed % 500 == 0:
                progress = {"processed": processed, "last_frame": last_frame,
                            "in": engine.total_in, "out": engine.total_out,
                            "fps": processed / max(1e-9, time.perf_counter() - wall)}
                print(json.dumps(progress), flush=True)
    finally:
        reader.close()
        if writer:
            writer.close()
        engine.observe_children(detector.finish_children())
        detector.close()
    for event in engine.finish():
        record = dict(event.to_dict(), frame=last_frame, crossing_frame=pending_frames.get(event.detected_at), emitted_at_eof=True)
        all_events.append(record)
        if trace:
            trace[-1]["events"].append(record)
    elapsed = time.perf_counter() - wall
    report = {"model": cfg.MODEL_PATH, "image_size": cfg.IMAGE_SIZE, "anchor_y": cfg.ANCHOR_Y,
              "reid": cfg.REID_MODEL_PATH, "tracking_mode": cfg.TRACKING_MODE, "tracker": cfg.TRACKER_PATH,
              "confidence": cfg.CONFIDENCE, "cpu_threads": cfg.CPU_THREADS, "pose_anchor": cfg.POSE_ANCHOR,
              "association_similarity": cfg.TRACKER_IDENTITY_MIN_SIMILARITY,
              "drift_threshold": cfg.TRACKER_DRIFT_THRESHOLD,
              "source": cfg.VIDEO_SOURCE, "video_start": cfg.VIDEO_START,
              "timestamp_basis": "file presentation time; burned-in clock edits are not decoded",
              "staff_mode": cfg.STAFF_MODE, "vest_model": cfg.VEST_MODEL_PATH,
              "child_mode": cfg.CHILD_MODE, "child_model": cfg.CHILD_MODEL_PATH if cfg.CHILD_MODE == 'model' else None,
              "child_analysis": dict(detector.child_observer.stats) if detector.child_observer is not None else None,
              "vest_threshold": cfg.VEST_THRESHOLD, "staff_decision_delay": cfg.STAFF_DECISION_DELAY_SEC,
              "compact": compact, "last_frame": last_frame,
              "frame_semantics": "frame=decision emission; crossing_frame=physical transition timestamp",
              "start_frame": start, "end_frame": end, "frames": processed, "elapsed_sec": elapsed,
              "processing_fps": processed / max(1e-9, elapsed), "in": engine.total_in, "out": engine.total_out,
              "suppressed": engine.total_suppressed, "identities": len(engine.identities),
              "mean_ms": {name: value * 1000 / max(1, processed) for name, value in stages.items()},
              "tracks": tracks, "events": all_events}
    if not compact:
        report["trace"] = trace
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not compact:
        np.savez_compressed(output.with_suffix(".features.npz"),
                            features=np.stack(feature_bank) if feature_bank else np.empty((0, detector.reid.dimension), np.float32))
    print(json.dumps({k: v for k, v in report.items() if k not in {"trace", "tracks", "events"}}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/test.mp4")
    parser.add_argument("--profile", choices=PROFILE_NAMES)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--anchor-y", type=float)
    parser.add_argument("--pose-anchor", choices=["head", "bbox"])
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--association-similarity", type=float)
    parser.add_argument("--drift-threshold", type=float)
    parser.add_argument("--tracking-mode", choices=["iou", "appearance"])
    parser.add_argument("--tracker")
    parser.add_argument("--reid")
    parser.add_argument("--staff-mode", choices=["disabled", "model"])
    parser.add_argument("--child-mode", choices=["disabled", "model"])
    parser.add_argument("--child-model")
    parser.add_argument("--vest-model")
    parser.add_argument("--vest-threshold", type=float)
    parser.add_argument("--staff-decision-delay", type=float)
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--video-start")
    parser.add_argument("--compact", action="store_true", help="Save events and totals without per-frame trace/features")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    if args.start < 0 or args.end < args.start:
        parser.error("Expected 0 <= start <= end")
    overrides = {"VIDEO_SOURCE": args.source}
    for field, value in (("MODEL_PATH", args.model), ("IMAGE_SIZE", args.image_size), ("ANCHOR_Y", args.anchor_y),
                         ("POSE_ANCHOR", args.pose_anchor), ("CONFIDENCE", args.confidence),
                         ("TRACKER_IDENTITY_MIN_SIMILARITY", args.association_similarity),
                         ("TRACKER_DRIFT_THRESHOLD", args.drift_threshold),
                         ("TRACKING_MODE", args.tracking_mode), ("TRACKER_PATH", args.tracker),
                         ("REID_MODEL_PATH", args.reid), ("STAFF_MODE", args.staff_mode),
                         ("CHILD_MODE", args.child_mode), ("CHILD_MODEL_PATH", args.child_model),
                         ("VEST_MODEL_PATH", args.vest_model), ("VEST_THRESHOLD", args.vest_threshold),
                         ("STAFF_DECISION_DELAY_SEC", args.staff_decision_delay),
                         ("CPU_THREADS", args.cpu_threads), ("VIDEO_START", args.video_start)):
        if value is not None:
            overrides[field] = value
    evaluate(replace(apply_profile(Settings.from_env(), args.profile), **overrides), args.start, args.end, args.output,
             video=args.video, compact=args.compact)
