"""Application loop, recovery, graceful shutdown and outbox synchronisation."""
from dataclasses import asdict
from collections import deque
import hashlib
import json
import logging
import math
import signal
from threading import Event
import time

import cv2
import psycopg2

from core.camera_clock import ClockMap
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
    if cfg.FRAME_STRIDE == 1:
        values.pop("FRAME_STRIDE", None)
    if cfg.TIME_SOURCE == "pts":
        for key in list(values):
            if key == "TIME_SOURCE" or key.startswith("CLOCK_"):
                values.pop(key)
    else:
        values.pop("VIDEO_START", None)
        if not cfg.CLOCK_MAP_PATH:
            raise ValueError("Camera time requires CLOCK_MAP_PATH / --clock-map")
        clock_payload = json.loads(cfg.path(cfg.CLOCK_MAP_PATH).read_text(encoding='utf-8'))
        semantic_clock = {name: clock_payload.get(name) for name in
                          ('source_sha256', 'model_sha256', 'timezone', 'roi', 'end_frame')}
        semantic_clock['segments'] = [{name: segment.get(name) for name in
                                      ('frame', 'pts', 'offset', 'excluded_transition_start', 'excluded_start_pts', 'reviewed')}
                                     for segment in clock_payload['segments']]
        values['CLOCK_MAP_PATH'] = hashlib.sha256(json.dumps(semantic_clock, sort_keys=True).encode()).hexdigest()
        values['VIDEO_SOURCE'] = clock_payload['source_sha256']
        values['SOURCE_KIND'] = 'file'
    model_names = ["MODEL_PATH", "REID_MODEL_PATH", "TRACKER_PATH"]
    if cfg.TIME_SOURCE == "camera":
        model_names.append("CLOCK_MODEL_PATH")
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
        values["file"] = [clock_payload["source_sha256"]] if cfg.TIME_SOURCE == "camera" else [str(path), stat.st_size, stat.st_mtime_ns]
    # Persist only the fingerprint, not the full source configuration.
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def synchronize(store, db):
    while True:
        events = store.pending()
        # A remote checkpoint must never get ahead of its delivered events.
        finished = store.pending_count() <= len(events)
        checkpoint = None
        if finished:
            header = store.checkpoint_header()
            if header is not None and (events or not store.is_synchronized(db, header)):
                checkpoint = store.load_for_delivery()
        if not events and checkpoint is None:
            return
        db.deliver(events, checkpoint)
        store.acknowledge(events)
        if checkpoint is not None:
            store.mark_synchronized(db, checkpoint)
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
    if not cfg.path(cfg.VIDEO_SOURCE).is_file():
        raise ValueError("Local video file does not exist")
    if cfg.TIME_SOURCE == "pts" and not cfg.VIDEO_START:
        raise ValueError("Set VIDEO_START / --video-start to the recording start time including timezone")
    if output and kind == "file" and cfg.path(cfg.VIDEO_SOURCE).resolve() == cfg.path(output).resolve():
        raise ValueError("Preview output must differ from the input video")
    clock_map = None
    if cfg.TIME_SOURCE == "camera":
        if not cfg.CLOCK_MAP_PATH:
            raise ValueError("Build camera time first with tools.build_clock_map and supply --clock-map")
        clock_map = ClockMap.load(cfg.path(cfg.CLOCK_MAP_PATH), cfg)
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
    buffered_events = []
    last_media_seconds = resume_media_seconds = None
    align_resume_cursor = False
    try:
        start = parse_start(cfg.VIDEO_START) if cfg.TIME_SOURCE == "pts" else None
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
                resume_media_seconds = saved['payload'].get('last_media_seconds')
                if resume_media_seconds is None and engine.last_timestamp is not None:
                    if cfg.TIME_SOURCE == 'pts':
                        resume_media_seconds = engine.last_timestamp - start.timestamp()
                    else:
                        for i, segment in enumerate(clock_map.segments):
                            media = engine.last_timestamp - segment['offset']
                            lo = clock_map.pts_starts[i]
                            hi = clock_map.pts_starts[i + 1] if i + 1 < len(clock_map.segments) else float('inf')
                            if lo <= media < hi:
                                resume_media_seconds = media
                                break
                if resume_media_seconds is not None:
                    if not isinstance(resume_media_seconds, (int, float)) or not math.isfinite(resume_media_seconds) or resume_media_seconds < 0:
                        raise ValueError('Invalid saved media cursor; restore a consistent checkpoint')
                    last_media_seconds = resume_media_seconds
                    align_resume_cursor = True
                log.info("Restored checkpoint revision %s; queued events %s", saved["revision"], store.pending_count())
        resume_at = max(0, next_frame - cfg.REPLAY_FRAMES) if kind == "file" and not preview else next_frame
        reader = VideoSource(cfg, start=start, resume_frame=resume_at, stop=stop, clock_map=clock_map)
        last_sync = -float("inf")
        last_checkpoint = -float("inf")
        loop_started = time.monotonic()
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
                continue
            warmup = not preview and (packet.media_seconds <= resume_media_seconds + 1e-7
                                      if resume_media_seconds is not None else packet.index < next_frame)
            if align_resume_cursor and not warmup:
                # MP4 random seeks may report a frame index a few frames off.
                # Decode through the saved PTS, then restore the logical cursor.
                packet.index = next_frame + (-next_frame % cfg.FRAME_STRIDE)
                reader.index = packet.index + 1
                align_resume_cursor = False
            reset_events = []
            if packet.discontinuity or packet.time_discontinuity:
                detector.reset()
                if engine and not warmup and (preview or packet.time_discontinuity):
                    reset_events = engine.reset_tracking()
            detections = detector.detect(packet.image)
            if warmup:
                continue
            events = reset_events + (engine.update(packet.image, detections, packet.timestamp,
                                     child_observations=getattr(detector, 'child_observations', ())) if engine else [])
            next_frame = packet.index + 1
            last_media_seconds = packet.media_seconds
            if store:
                buffered_events.extend(events)
            if store and (time.monotonic() - last_checkpoint >= cfg.CHECKPOINT_INTERVAL_SEC or len(buffered_events) >= 500):
                store.save(source_key, {"bus_id": cfg.BUS_ID, "camera_id": cfg.CAMERA_ID,
                                        "engine": engine.snapshot(), "next_frame": next_frame,
                                        "completed": False, "last_media_seconds": last_media_seconds}, buffered_events)
                buffered_events.clear()
                last_checkpoint = time.monotonic()
            for event in events:
                log.info("%s track=%s counted=%s reason=%s", event.event_type, event.track_id,
                         event.counted, event.reason)
            if preview:
                preview_packet = packet
                elapsed = sum(frame_times)
                metrics = {"processing_fps": len(frame_times) / elapsed if elapsed else None,
                           "source_fps": reader.fps, "frame_index": packet.index,
                           "media_seconds": packet.media_seconds,
                           "segment_start": segment_start}
                image = draw_preview(packet.image, detections, cfg, engine, events, metrics=metrics)
                if output:
                    if writer is None:
                        height, width = image.shape[:2]
                        writer = PreviewWriter(cfg.path(output), reader.fps / cfg.FRAME_STRIDE, (width, height))
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
                log.info("Processed %s frames; IN=%s OUT=%s; %.2f FPS; source_frame=%s time=%s (%s)", processed,
                         engine.total_in if engine else "disabled", engine.total_out if engine else "disabled",
                         processed / max(1e-9, time.monotonic() - loop_started), packet.index,
                         packet.timestamp.isoformat(), cfg.TIME_SOURCE)
            if max_frames is not None and processed >= max_frames:
                break
        if engine:
            engine.observe_children(getattr(detector, 'finish_children', lambda: [])())
        if store:
            unresolved = engine.finish() if completed else []
            store.save(source_key, {"bus_id": cfg.BUS_ID, "camera_id": cfg.CAMERA_ID,
                                    "engine": engine.snapshot(), "next_frame": next_frame,
                                    "completed": completed, "last_media_seconds": last_media_seconds}, buffered_events + unresolved)
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
