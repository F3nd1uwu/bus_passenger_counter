"""Crossing state machine; no IO, wall-clock timers, or global cooldowns."""
from dataclasses import asdict, dataclass, field
from datetime import datetime
import logging
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np

from config.settings import Settings
from core.reid import normalise_embedding
from core.staff_manager import StaffEvidence
from core.child_filter import ChildEvidence
from core.types import CrossingEvent

log = logging.getLogger(__name__)


@dataclass
class Identity:
    identity_id: str
    embedding: list[float]
    last_seen: float
    last_in: float | None = None
    last_out: float | None = None
    staff: bool = False
    gallery: list[list[float]] = field(default_factory=list)


@dataclass
class Track:
    last_seen: float
    identity_id: str | None = None
    zone: str | None = None
    candidate: str | None = None
    candidate_frames: int = 0
    frames: int = 0
    embedding: np.ndarray | None = None
    staff: StaffEvidence = field(default_factory=StaffEvidence)
    child: ChildEvidence = field(default_factory=ChildEvidence)
    pending: list = field(default_factory=list)
    restored: bool = False


class PassengerCounterEngine:
    def __init__(self, cfg: Settings | None = None, state=None):
        self.cfg = (cfg or Settings.from_env()).validate()
        self.tracks: dict[int, Track] = {}
        self.identities: dict[str, Identity] = {}
        self.total_in = 0
        self.total_out = 0
        self.total_suppressed = 0
        self.day = None
        self.last_timestamp = None
        self.continuity = {}
        self.orphaned_pending = []
        self.deferred = []
        if state:
            if state.get("version") != 1:
                raise ValueError("Unsupported counter checkpoint version")
            self.identities = {key: Identity(**value) for key, value in state["identities"].items()}
            for identity in self.identities.values():
                normalise_embedding(identity.embedding)
                for appearance in identity.gallery:
                    normalise_embedding(appearance)
            self.day = state["day"]
            self.total_in, self.total_out = state["total_in"], state["total_out"]
            self.total_suppressed = state["total_suppressed"]
            self.last_timestamp = state["last_timestamp"]
            self.continuity = state.get("continuity", {})
            self.orphaned_pending = state.get("orphaned_pending", [])
            self.deferred = state.get("deferred", [])

    def snapshot(self):
        continuity = dict(self.continuity)
        deferred = [saved for saved in self.deferred if saved.get('pending')]
        for tid, track in sorted(self.tracks.items(), key=lambda item: item[1].last_seen):
            if track.identity_id is not None:
                previous = continuity.get(track.identity_id)
                if previous and previous.get('pending'):
                    deferred.append(dict(previous, identity_id=track.identity_id))
                continuity[track.identity_id] = {
                    "track_id": tid, "last_seen": track.last_seen, "zone": track.zone,
                    "candidate": track.candidate, "candidate_frames": track.candidate_frames,
                    "frames": track.frames, "staff": asdict(track.staff), "child": asdict(track.child),
                    "pending": [(direction, when.isoformat()) for direction, when in track.pending]}
        orphaned_pending = list(self.orphaned_pending)
        for tid, track in self.tracks.items():
            if track.identity_id is None and track.pending:
                orphaned_pending.append({"track_id": tid,
                                         "pending": [(d, t.isoformat()) for d, t in track.pending]})
        return {"version": 1, "identities": {key: asdict(value) for key, value in self.identities.items()},
                "day": self.day, "total_in": self.total_in, "total_out": self.total_out,
                "total_suppressed": self.total_suppressed, "last_timestamp": self.last_timestamp,
                "continuity": continuity, "orphaned_pending": orphaned_pending, "deferred": deferred}

    def reset_tracking(self):
        # Tracker IDs can be reused after reconnect; person identities survive.
        unresolved = self.finish()
        self.tracks.clear()
        self.continuity.clear()
        return unresolved

    def _restore_track(self, track, identity, now):
        saved = self.continuity.get(identity.identity_id)
        if saved and 0 <= now - saved['last_seen'] <= self.cfg.TRACK_TTL_SEC:
            self.continuity.pop(identity.identity_id)
        else:
            saved = None
        # Also repair a ByteTrack ID switch while the process is running.
        if saved is None:
            for tid, old in list(self.tracks.items()):
                if (old is not track and old.identity_id == identity.identity_id
                        and 0 <= now - old.last_seen <= self.cfg.TRACK_TTL_SEC):
                    saved = {"last_seen": old.last_seen, "zone": old.zone,
                             "candidate": old.candidate, "candidate_frames": old.candidate_frames,
                             "frames": old.frames, "staff": asdict(old.staff), "child": asdict(old.child),
                             "pending": [(d, t.isoformat()) for d, t in old.pending]}
                    del self.tracks[tid]
                    break
        if saved and 0 <= now - saved["last_seen"] <= self.cfg.TRACK_TTL_SEC:
            track.zone, track.candidate = saved["zone"], saved["candidate"]
            track.candidate_frames = saved["candidate_frames"]
            track.frames = saved["frames"]
            # The current frame's evidence is observed after restoring the prior state.
            track.staff = StaffEvidence(**saved["staff"])
            track.child = ChildEvidence(**saved.get("child", {}))
            track.pending = [(d, datetime.fromisoformat(t)) for d, t in saved["pending"]]

    def _identity(self, track, now, claimed, *, restore=True, excluded=()):
        if track.identity_id in self.identities:
            identity = self.identities[track.identity_id]
            if restore and not track.restored:
                self._restore_track(track, identity, now)
                track.restored = True
            return identity
        if track.embedding is None:
            return None
        candidates = []
        for identity in self.identities.values():
            if identity.identity_id in claimed or identity.identity_id in excluded:
                continue
            vectors = np.asarray([identity.embedding, *identity.gallery], dtype=np.float32)
            if vectors.shape[1:] != track.embedding.shape:
                raise ValueError("ReID dimension differs from checkpoint; use the same encoder")
            score = float(np.max(vectors @ track.embedding))
            candidates.append((score, identity.identity_id))
        candidates.sort(reverse=True)
        if (candidates and candidates[0][0] >= self.cfg.REID_THRESHOLD
                and (len(candidates) == 1 or candidates[0][0] - candidates[1][0] >= self.cfg.REID_MARGIN)):
            identity = self.identities[candidates[0][1]]
        else:
            if len(self.identities) >= self.cfg.MAX_IDENTITIES:
                raise RuntimeError("MAX_IDENTITIES exceeded; refusing to discard hourly deduplication")
            identity = Identity(str(uuid4()), track.embedding.tolist(), now)
            self.identities[identity.identity_id] = identity
        track.identity_id = identity.identity_id
        if restore:
            self._restore_track(track, identity, now)
            track.restored = True
        return identity

    def _remember_appearance(self, identity, embedding):
        """Keep the first appearance and diverse views; an average alone forgets
        the boarding pose after a person turns or their crop changes size.
        """
        gallery = identity.gallery
        if not gallery:
            gallery.append(embedding.tolist())
            return
        vectors = np.asarray(gallery, dtype=np.float32)
        if float(np.max(vectors @ embedding)) >= self.cfg.REID_GALLERY_SIMILARITY:
            return
        if len(gallery) < self.cfg.REID_GALLERY_SIZE:
            gallery.append(embedding.tolist())
        elif len(gallery) > 1:
            # Preserve the first view for the whole identity lifetime. Replace
            # the most redundant remaining view only with a more diverse view.
            similarities = vectors @ vectors.T
            np.fill_diagonal(similarities, -1)
            redundancy = similarities.max(axis=1)
            index = 1 + int(np.argmax(redundancy[1:]))
            if float(np.max(vectors @ embedding)) < redundancy[index]:
                gallery[index] = embedding.tolist()

    def _unresolved(self, tid, track):
        events = [CrossingEvent.create(self.cfg, when, direction, tid,
                                      track.identity_id or "unresolved", False,
                                      "INSUFFICIENT_MODEL_EVIDENCE")
                  for direction, when in track.pending]
        track.pending.clear()
        return events

    def finish(self):
        """Retain uncertain completed crossings in the audit log at file EOF."""
        events = self._finish_deferred()
        for tid, track in self.tracks.items():
            events.extend(self._decide_pending(tid, track, self.last_timestamp, final=True))
            events.extend(self._unresolved(tid, track))
        for identity_id, saved in self.continuity.items():
            events.extend(self._finish_saved(identity_id, saved))
        events.extend(self._flush_orphaned())
        return events

    def _finish_deferred(self):
        # Multiple trajectories can share an hourly identity while an old age
        # decision is pending. Preserve the older event across checkpoints.
        events = []
        for saved in self.deferred:
            events.extend(self._finish_saved(saved['identity_id'], saved))
        self.deferred.clear()
        return events

    def _decide_pending(self, tid, track, timestamp, *, final=False):
        identity = self.identities.get(track.identity_id)
        model = self.cfg.STAFF_MODE == "model"
        child_confirmed = self.cfg.CHILD_MODE == 'model' and track.child.confirmed
        if identity is None or (model and not track.staff.ready(self.cfg.VEST_CONFIRM_FRAMES) and not child_confirmed):
            return []
        events, pending = [], []
        for direction, crossed_at in track.pending:
            if (self.cfg.CHILD_MODE == "model" and not final and not track.child.adult_seen
                    and timestamp - crossed_at.timestamp() < self.cfg.CHILD_DECISION_DELAY_SEC):
                pending.append((direction, crossed_at))
                continue
            if (model and not final and not track.staff.confirmed
                    and timestamp - crossed_at.timestamp() < self.cfg.STAFF_DECISION_DELAY_SEC):
                pending.append((direction, crossed_at))
                continue
            field_name = "last_in" if direction == "IN" else "last_out"
            last = getattr(identity, field_name)
            # A long-term ReID match alone must not turn a similar passenger
            # into staff. Short track handovers explicitly restore evidence.
            staff = model and track.staff.confirmed
            child = self.cfg.CHILD_MODE == "model" and track.child.confirmed
            counted = not staff and not child and (last is None or crossed_at.timestamp() - last >= self.cfg.DUPLICATE_INTERVAL_SEC)
            reason = "STAFF" if staff else "CHILD" if child else "COUNTED" if counted else "REPEAT_WITHIN_HOUR"
            same_day = crossed_at.astimezone(ZoneInfo(self.cfg.TIMEZONE)).date().isoformat() == self.day
            if counted:
                setattr(identity, field_name, crossed_at.timestamp())
                if same_day:
                    if direction == "IN":
                        self.total_in += 1
                    else:
                        self.total_out += 1
            elif same_day:
                self.total_suppressed += 1
            events.append(CrossingEvent.create(self.cfg, crossed_at, direction, tid,
                                              identity.identity_id, counted, reason))
        track.pending = pending
        return events

    def _flush_orphaned(self):
        events = [event for saved in self.orphaned_pending
                  for event in self._unresolved_saved("unresolved", saved)]
        self.orphaned_pending.clear()
        return events

    def _unresolved_saved(self, identity_id, saved):
        events = [CrossingEvent.create(self.cfg, datetime.fromisoformat(when), direction,
                                      saved.get("track_id", 0), identity_id, False,
                                      "INSUFFICIENT_MODEL_EVIDENCE")
                  for direction, when in saved["pending"]]
        saved["pending"] = []
        return events

    def _finish_saved(self, identity_id, saved):
        """Resolve a ready delayed event even when EOF follows a restart."""
        track = Track(saved["last_seen"], identity_id=identity_id,
                      staff=StaffEvidence(**saved["staff"]),
                      child=ChildEvidence(**saved.get("child", {})),
                      pending=[(d, datetime.fromisoformat(t)) for d, t in saved["pending"]])
        tid = saved.get("track_id", 0)
        events = self._decide_pending(tid, track, self.last_timestamp, final=True)
        events.extend(self._unresolved(tid, track))
        saved["pending"] = []
        return events

    def observe_children(self, observations):
        if self.cfg.CHILD_MODE != "model":
            return
        for observation in observations:
            track = self.tracks.get(observation.track_id)
            if track is None:
                continue
            source = normalise_embedding(observation.embedding)
            compatible = (source is not None and track.embedding is not None
                          and source.shape == track.embedding.shape and float(source @ track.embedding) >= .85)
            track.child.observe(observation.verdict if compatible else 'unknown', observation.sample_id,
                                self.cfg.CHILD_CONFIRM_SAMPLES)

    def _pending_ttl(self, pending, child):
        if self.cfg.CHILD_MODE == "model" and pending and not child.adult_seen:
            return max(self.cfg.TRACK_TTL_SEC, self.cfg.CHILD_DECISION_DELAY_SEC)
        return self.cfg.TRACK_TTL_SEC

    def update(self, frame, detections, now: datetime, *, warmup=False, child_observations=()):
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("Frame time must be timezone-aware")
        timestamp = now.timestamp()
        events = [] if warmup else self._flush_orphaned()
        if not warmup and self.last_timestamp is not None and timestamp < self.last_timestamp:
            raise ValueError("Frame timestamps must be monotonic")
        if not warmup:
            self.last_timestamp = timestamp
            self.observe_children(child_observations)
            events.extend(self._finish_deferred())
            day = now.astimezone(ZoneInfo(self.cfg.TIMEZONE)).date().isoformat()
            if self.day != day:
                self.day = day
                self.total_in = self.total_out = self.total_suppressed = 0
            stale = [key for key, identity in self.identities.items()
                     if timestamp - max(identity.last_seen, identity.last_in or 0,
                                        identity.last_out or 0) >= self.cfg.DUPLICATE_INTERVAL_SEC]
            for key in stale:
                del self.identities[key]
                saved = self.continuity.pop(key, None)
                if saved:
                    events.extend(self._unresolved_saved(key, saved))
            for key in list(self.continuity):
                saved = self.continuity[key]
                ttl = self._pending_ttl(saved['pending'], ChildEvidence(**saved.get('child', {})))
                if timestamp - saved["last_seen"] > ttl:
                    events.extend(self._finish_saved(key, self.continuity.pop(key)))
        for tid in list(self.tracks):
            track = self.tracks[tid]
            if timestamp - track.last_seen > self._pending_ttl(track.pending, track.child):
                if not warmup:
                    events.extend(self._decide_pending(tid, self.tracks[tid], timestamp, final=True))
                if self.tracks[tid].pending:
                    log.warning("Unresolved crossing on track %s: insufficient model evidence", tid)
                    events.extend(self._unresolved(tid, self.tracks[tid]))
                del self.tracks[tid]
        detections = list(detections)
        active = {d.track_id for d in detections}
        if len(active) != len(detections):
            raise ValueError("Duplicate tracker ID within a frame")
        claimed = {track.identity_id for tid, track in self.tracks.items()
                   if tid in active and track.identity_id is not None}
        height, width = frame.shape[:2]
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            if not np.isfinite(detection.bbox).all() or x2 <= x1 or y2 <= y1:
                continue
            if not np.isfinite(detection.confidence) or not 0 <= detection.confidence <= 1:
                continue
            cx = (x1 + x2) / (2 * width)
            if not self.cfg.DOOR_X_MIN <= cx <= self.cfg.DOOR_X_MAX:
                old = self.tracks.pop(detection.track_id, None)
                if old:
                    if not warmup:
                        events.extend(self._decide_pending(detection.track_id, old, timestamp, final=True))
                    events.extend(self._unresolved(detection.track_id, old))
                continue
            track = self.tracks.setdefault(detection.track_id, Track(timestamp))
            track.last_seen = timestamp
            reliable_appearance = (detection.embedding is not None and
                                   detection.confidence >= self.cfg.CONFIDENCE)
            if reliable_appearance:
                track.embedding = normalise_embedding(detection.embedding)
            # A newly split body must not immediately regain the rejected
            # person's identity through its recently contaminated ReID gallery.
            excluded = {self.tracks[tid].identity_id for tid in detection.separated_from
                        if tid in self.tracks and tid != detection.track_id}
            excluded.update(pid for pid, saved in self.continuity.items()
                            if saved.get("track_id") in detection.separated_from)
            identity = self._identity(track, timestamp, claimed, restore=not warmup, excluded=excluded)
            track.frames += 1
            if self.cfg.STAFF_MODE == "model" and detection.confidence >= self.cfg.CONFIDENCE:
                track.staff.observe(detection.staff_probability, self.cfg.VEST_THRESHOLD,
                                    self.cfg.VEST_CONFIRM_FRAMES)
            if identity is not None:
                claimed.add(identity.identity_id)
                if not warmup:
                    identity.last_seen = timestamp
                    if reliable_appearance:
                        self._remember_appearance(identity, track.embedding)
                        identity.embedding = normalise_embedding(
                            0.9 * np.asarray(identity.embedding) + 0.1 * track.embedding).tolist()
                    identity.staff = identity.staff or track.staff.confirmed
            if detection.confidence < self.cfg.CONFIDENCE:
                # Weak body/fragment boxes bridge tracking gaps, but a bag or
                # clipped limb must not establish a new side of the doorway.
                track.candidate, track.candidate_frames = None, 0
                if not warmup:
                    events.extend(self._decide_pending(detection.track_id, track, timestamp))
                continue
            if detection.anchor_required and detection.anchor is None:
                track.candidate, track.candidate_frames = None, 0
                continue
            if detection.anchor is not None:
                if not np.isfinite(detection.anchor).all():
                    continue
                y = detection.anchor[1] / height
            else:
                y = (y1 + (y2 - y1) * self.cfg.ANCHOR_Y) / height
            zone = "street" if y <= self.cfg.Y_STREET else "cabin" if y >= self.cfg.Y_CABIN else None
            if zone is None:
                track.candidate, track.candidate_frames = None, 0
            else:
                track.candidate_frames = track.candidate_frames + 1 if track.candidate == zone else 1
                track.candidate = zone
                if track.candidate_frames >= self.cfg.ZONE_CONFIRM_FRAMES:
                    if track.zone is not None and track.zone != zone and track.frames >= self.cfg.MIN_TRACK_FRAMES:
                        direction = "IN" if track.zone == self.cfg.ENTER_FROM else "OUT"
                        if not warmup:
                            track.pending.append((direction, now))
                    track.zone = zone
            if warmup:
                track.pending.clear()
                continue
            events.extend(self._decide_pending(detection.track_id, track, timestamp))
        return events
