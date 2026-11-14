"""Application loop, recovery, graceful shutdown and outbox synchronisation."""
from dataclasses import asdict
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import logging
import signal
from threading import Event
import time

import cv2
import psycopg2

from core.counter_engine import PassengerCounterEngine
from core.detector import PersonDetector
from core.preview_player import PreviewPlayer
from core.video_source import VideoSource, parse_start, source_kind
from core.visualization import PreviewWriter, draw_preview
from database.db_client import DatabaseClient
from database.local_store import LocalStore

log = logging.getLogger(__name__)
TRANSIENT_DB_ERRORS = (psycopg2.OperationalError, psycopg2.InterfaceError)


def configuration_key(cfg):
    # Fingerprint models as well as thresholds so incompatible state is never reused.
    values = asdict(cfg)
    for key in list(values):
        if key.startswith("DB_") or key in {"STATE_PATH", "SYNC_INTERVAL_SEC", "CHECKPOINT_INTERVAL_SEC", "DEVICE", "CPU_THREADS"}:
            del values[key]
    if cfg.TRACKING_MODE == "iou":
        values.pop("TRACKING_MODE", None)
        values.pop("TRACKER_IDENTITY_MIN_SIMILARITY", None)
    if cfg.TRACKING_MODE == "iou" or cfg.TRACKER_DRIFT_THRESHOLD == 0:
        # Disabled guard preserves fingerprints of existing installations.
        values.pop("TRACKER_DRIFT_THRESHOLD", None)
    if cfg.CHILD_MODE == "disabled":
        for key in list(values):
            if key.startswith('CHILD_'):
                values.pop(key)
    if cfg.POSE_ANCHOR == "bbox":
        values.pop("POSE_ANCHOR", None)
        values.pop("POSE_ANCHOR_CONFIDENCE", None)
    model_names = ["MODEL_PATH", "REID_MODEL_PATH", "TRACKER_PATH"]
    if cfg.CHILD_MODE == "model":
        model_names.append('CHILD_MODEL_PATH')
    if cfg.STAFF_MODE == "model":
        model_names.append("VEST_MODEL_PATH")
    else:
        values.pop("VEST_MODEL_PATH", None)
        values.pop("STAFF_DECISION_DELAY_SEC", None)
    for name in model_names:
        path = cfg.path(getattr(cfg, name))
        with path.open("rb") as stream:
            values[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    if source_kind(cfg.VIDEO_SOURCE, cfg.SOURCE_KIND) == "file":
        path = cfg.path(cfg.VIDEO_SOURCE).resolve()
        stat = path.stat()
        values["file"] = [str(path), stat.st_size, stat.st_mtime_ns]
    # Source URLs can contain credentials: persist only the digest.
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def synchronize(store, db):
    while True:
        events = store.pending()
        # A remote checkpoint must never get ahead of its delivered events.
        finished = store.pending_count() <= len(events)
        checkpoint = store.load() if finished else None
        if not events and checkpoint is None:
            return
        db.deliver(events, checkpoint)
        store.acknowledge(events)
        if finished:
            return


def run(cfg, *, preview=False, detection_only=False, output=None, show=False,
        max_frames=None, start_frame=0):
    cfg.validate()
    if detection_only and not preview:
        raise ValueError("Detection-only mode is available only in the preview module")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max-frames must be positive")
    if start_frame < 0 or (start_frame and not preview):
        raise ValueError("start-frame is available only in preview and must be nonnegative")
    kind = source_kind(cfg.VIDEO_SOURCE, cfg.SOURCE_KIND)
    if not preview and kind == "file" and not cfg.VIDEO_START:
        raise ValueError("Set VIDEO_START / --video-start to the recording start time including timezone")
    if output and kind == "file" and cfg.path(cfg.VIDEO_SOURCE).resolve() == cfg.path(output).resolve():
        raise ValueError("Preview output must differ from the input video")
    detector = PersonDetector(cfg, detection_only=detection_only)
    if not detection_only and cfg.STAFF_MODE == "disabled":
        log.warning("STAFF_MODE=disabled: staff are counted as passengers until the vest model is configured")
    if not detection_only and cfg.CHILD_MODE == "model":
        log.warning("Experimental child filter enabled: visual age is approximate; unknown passengers count")
    engine = None if detection_only else PassengerCounterEngine(cfg)
    store = db = reader = writer = None
    stop = Event()
    previous_handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[sig] = signal.signal(sig, lambda *_: stop.set())
    processed = 0
    completed = False
    next_frame = start_frame
    source_key = None
    ready_to_sync = False
    player = None
    preview_packet = None
    segment_start = start_frame
    frame_times = deque(maxlen=30)
    try:
        start = parse_start(cfg.VIDEO_START) if cfg.VIDEO_START else datetime.now(timezone.utc)
        if not preview:
            source_key = configuration_key(cfg)
            store = LocalStore(cfg.path(cfg.STATE_PATH))
            db = DatabaseClient(cfg.DB_HOST, cfg.DB_PORT, cfg.DB_NAME, cfg.DB_USER, cfg.DB_PASS, cfg.DB_SSLMODE)
            saved = store.load()
            if saved and saved["source_key"] != source_key:
                raise ValueError("Configuration/source changed; use a separate STATE_PATH or restore the matching configuration")
            # Without local state, check the remote checkpoint before counting anything.
            try:
                remote = db.load_checkpoint(source_key)
            except TRANSIENT_DB_ERRORS:
                if saved is None:
                    raise
                remote = None
                log.warning("PostgreSQL unavailable at startup; using durable local state")
            if remote and (saved is None or remote["revision"] > saved["revision"]):
                if store.pending_count():
                    raise RuntimeError("Remote checkpoint is newer than local state with pending events; restore a consistent backup")
                store.save(source_key, remote["payload"], revision=remote["revision"])
                saved = store.load()
            ready_to_sync = True
            if saved:
                engine = PassengerCounterEngine(cfg, saved["payload"]["engine"])
                next_frame = saved["payload"]["next_frame"] if kind == "file" else 0
                if saved["payload"].get("completed"):
                    synchronize(store, db)
                    log.info("Recording already completed; pending events delivered")
                    return {"frames": 0, "in": engine.total_in, "out": engine.total_out}
                log.info("Restored checkpoint revision %s; queued events %s", saved["revision"], store.pending_count())
        resume_at = max(0, next_frame - cfg.REPLAY_FRAMES) if kind == "file" and not preview else next_frame
        reader = VideoSource(cfg, start=start, resume_frame=resume_at, stop=stop)
        last_sync = -float("inf")
        last_checkpoint = -float("inf")
        retrying = False
        while not stop.is_set():
            if player:
                while player.paused and player.seek_request is None and not stop.is_set():
                    if player.poll():
                        stop.set()
                if stop.is_set():
                    break
                target = player.take_seek()
                if target is not None:
                    segment_start = reader.seek(target)
                    detector.reset()
                    if engine is not None:
                        engine = PassengerCounterEngine(cfg)
                    frame_times.clear()
                    log.info("Preview seek to frame %s; preview counts and tracking reset", segment_start)
            frame_started = time.perf_counter()
            if store is not None and time.monotonic() - last_sync >= cfg.SYNC_INTERVAL_SEC:
                try:
                    synchronize(store, db)
                except TRANSIENT_DB_ERRORS:
                    log.warning("PostgreSQL unavailable; durable queued events: %s", store.pending_count())
                last_sync = time.monotonic()
            packet = reader.read()
            if packet is None:
                if reader.eof:
                    if preview and engine:
                        engine.observe_children(getattr(detector, 'finish_children', lambda: [])())
                        final_events = engine.finish()
                        for event in final_events:
                            log.info("%s track=%s counted=%s reason=%s (preview EOF)",
                                     event.event_type, event.track_id, event.counted, event.reason)
                        if player and preview_packet is not None:
                            image = draw_preview(preview_packet.image, detections, cfg, engine,
                                                 final_events, metrics=metrics)
                            player.present(image, preview_packet.index)
                    if player and kind == "file":
                        player.mark_end()
                        continue
                    completed = True
                    break
                if not retrying:
                    log.warning("Live source unavailable; reconnecting")
                    retrying = True
                continue
            if retrying:
                log.info("Live source reconnected")
                retrying = False
            reset_events = []
            if packet.discontinuity:
                detector.reset()
                if engine and (kind != "file" or preview):
                    reset_events = engine.reset_tracking()
            detections = detector.detect(packet.image)
            warmup = kind == "file" and not preview and packet.index < next_frame
            if warmup:
                continue
            events = reset_events + (engine.update(packet.image, detections, packet.timestamp,
                                     child_observations=getattr(detector, 'child_observations', ())) if engine else [])
            next_frame = packet.index + 1
            if store and (events or packet.timestamp.timestamp() - last_checkpoint >= cfg.CHECKPOINT_INTERVAL_SEC):
                store.save(source_key, {"bus_id": cfg.BUS_ID, "camera_id": cfg.CAMERA_ID,
                                        "engine": engine.snapshot(), "next_frame": next_frame,
                                        "completed": False}, events)
                last_checkpoint = packet.timestamp.timestamp()
            for event in events:
                log.info("%s track=%s counted=%s reason=%s", event.event_type, event.track_id,
                         event.counted, event.reason)
            if preview:
                preview_packet = packet
                elapsed = sum(frame_times)
                metrics = {"processing_fps": len(frame_times) / elapsed if elapsed else None,
                           "source_fps": reader.fps, "frame_index": packet.index,
                           "media_seconds": (packet.timestamp - start).total_seconds() if kind == "file" else None,
                           "segment_start": segment_start}
                image = draw_preview(packet.image, detections, cfg, engine, events, metrics=metrics)
                if output:
                    if writer is None:
                        height, width = image.shape[:2]
                        writer = PreviewWriter(cfg.path(output), reader.fps, (width, height))
                    writer.write(image)
                if show:
                    if player is None:
                        player = PreviewPlayer(reader.fps, reader.total_frames, seekable=kind == "file")
                    player.present(image, packet.index)
            frame_times.append(max(1e-9, time.perf_counter() - frame_started))
            if player and player.poll():
                stop.set()
            processed += 1
            if processed % 1000 == 0:
                log.info("Processed %s frames; IN=%s OUT=%s", processed,
                         engine.total_in if engine else "disabled", engine.total_out if engine else "disabled")
            if max_frames is not None and processed >= max_frames:
                break
        if engine:
            engine.observe_children(getattr(detector, 'finish_children', lambda: [])())
        if store:
            unresolved = engine.finish() if completed else []
            store.save(source_key, {"bus_id": cfg.BUS_ID, "camera_id": cfg.CAMERA_ID,
                                    "engine": engine.snapshot(), "next_frame": next_frame,
                                    "completed": completed}, unresolved)
        elif preview and engine:
            # A preview stopped by max-frames/Q has no persistent checkpoint.
            # Resolve ready delayed crossings before returning its final totals.
            for event in engine.finish():
                log.info("%s track=%s counted=%s reason=%s (preview end)",
                         event.event_type, event.track_id, event.counted, event.reason)
        return {"frames": processed, "in": engine.total_in if engine else None,
                "out": engine.total_out if engine else None}
    finally:
        # Never persist mutated in-memory state here: an earlier failed local commit
        # must leave the last consistent checkpoint + events intact.
        try:
            getattr(detector, 'close', lambda: None)()
            if reader:
                reader.close()
            if writer:
                writer.close()
            if show:
                cv2.destroyAllWindows()
            if store and db and ready_to_sync:
                try:
                    synchronize(store, db)
                except TRANSIENT_DB_ERRORS:
                    log.warning("Shutdown with %s durable queued events; replay on next start", store.pending_count())
        finally:
            if db:
                db.close()
            if store:
                store.close()
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
