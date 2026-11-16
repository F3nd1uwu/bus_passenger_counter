"""Local archive reader, presentation timestamps and verified camera clock."""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
from urllib.parse import urlsplit

import cv2


def source_kind(source, requested="auto"):
    scheme = urlsplit(str(source)).scheme.lower()
    if requested not in {"auto", "file"} or (scheme and not (len(scheme) == 1 and str(source)[1:3] in {":/", ":\\"})) or str(source).startswith(("//", "\\\\")):
        raise ValueError("Only local video files are supported; camera and network sources are disabled")
    return "file"


def parse_start(value):
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("VIDEO_START must be ISO 8601, including timezone") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("VIDEO_START must include a timezone, e.g. 2026-11-01T05:40:00+04:00")
    return result.astimezone(timezone.utc)


@dataclass
class VideoFrame:
    image: object
    timestamp: datetime
    index: int
    discontinuity: bool
    media_seconds: float = 0.0
    time_discontinuity: bool = False


class VideoSource:
    def __init__(self, cfg, start=None, resume_frame=0, stop=None, clock_map=None):
        self.cfg = cfg
        self.kind = source_kind(cfg.VIDEO_SOURCE, cfg.SOURCE_KIND)
        self.source = cfg.VIDEO_SOURCE
        if self.kind == "file":
            self.source = str(cfg.path(self.source))
            if not Path(self.source).is_file():
                raise ValueError("Video file does not exist")
        self.start = start
        self.clock_map = clock_map
        if self.clock_map is None and cfg.TIME_SOURCE == "camera":
            from core.camera_clock import ClockMap
            if not cfg.CLOCK_MAP_PATH:
                raise ValueError("Build camera time first: python -m tools.build_clock_map --source VIDEO --output data/clock.json; then set CLOCK_MAP_PATH")
            self.clock_map = ClockMap.load(cfg.path(cfg.CLOCK_MAP_PATH), cfg)
        if self.clock_map is None and self.start is None:
            raise ValueError("PTS time requires an explicit recording start")
        self.stop = stop
        self.cap = None
        self.index = resume_frame
        self.resume_frame = resume_frame
        self.fps = 25.0
        self.total_frames = 0
        self.eof = False
        self.discontinuity = True
        self.last_clock_pts = None
        self.pending_time_cut = False
        self.last_read_index = None
        self.last_video_seconds = -1.0

    def _open(self):
        self.cap = cv2.VideoCapture()
        opened = self.cap.open(self.source)
        if not opened:
            self.close()
            raise ValueError("Cannot open video file")
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if math.isfinite(fps) and fps > 0:
            self.fps = fps
        count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.total_frames = int(count) if math.isfinite(count) and count > 0 else 0
        if self.clock_map:
            self.total_frames = min(self.total_frames, self.clock_map.end_frame)
        if self.resume_frame:
            if not self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.resume_frame):
                raise RuntimeError("Video backend cannot seek to the saved frame")
        self.discontinuity = True
        return True

    def seek(self, frame_index):
        if self.kind != "file":
            raise ValueError("Seeking is supported only for video files")
        if self.cap is None:
            self._open()
        target = max(0, int(frame_index))
        if self.total_frames:
            target = min(target, self.total_frames - 1)
        if not self.cap.set(cv2.CAP_PROP_POS_FRAMES, target):
            raise RuntimeError("Video backend cannot seek to the requested frame")
        self.index = self.resume_frame = target
        self.last_clock_pts = None
        self.pending_time_cut = False
        self.last_read_index = None
        self.last_video_seconds = -1.0
        self.eof = False
        self.discontinuity = True
        return target

    def read(self):
        while True:
            packet = self._read()
            if packet is None:
                return None
            if self.clock_map and self.clock_map.is_excluded(packet.index, packet.media_seconds):
                self.pending_time_cut = self.pending_time_cut or packet.time_discontinuity
                self.discontinuity = self.discontinuity or packet.discontinuity
                continue
            packet.time_discontinuity = packet.time_discontinuity or self.pending_time_cut
            self.pending_time_cut = False
            return packet

    def _read(self):
        if self.eof:
            return None
        if self.cap is None:
            self._open()
        while self.index % self.cfg.FRAME_STRIDE and (not self.total_frames or self.index < self.total_frames):
            if not self.cap.grab():
                if self.total_frames and self.index < self.total_frames - 2:
                    raise RuntimeError("Video decoding failed while skipping frames")
                self.eof = True
                return None
            self.index += 1
        if self.clock_map and self.index >= self.clock_map.end_frame:
            self.eof = True
            return None
        ok, image = self.cap.read()
        if not ok:
            expected = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.close()
            if math.isfinite(expected) and expected > 0 and self.index < expected - 2:
                raise RuntimeError("Video decoding failed before EOF; checkpoint retained for retry")
            self.eof = True
            return None
        # Prefer presentation timestamps for variable-frame-rate recordings.
        pts = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if not math.isfinite(pts) or pts < 0 or (self.index > 0 and (pts <= 0 or pts <= self.last_video_seconds)):
            pts = max(self.index / self.fps, self.last_video_seconds + 1 / self.fps)
        self.last_video_seconds = pts
        if self.clock_map:
            timestamp, time_cut = self.clock_map.at(self.index, pts)
            if self.last_read_index is not None:
                time_cut = self.clock_map.crossed_cut(self.last_read_index, self.index, previous_pts=self.last_clock_pts, pts=pts)
        else:
            timestamp, time_cut = self.start + timedelta(seconds=pts), False
        result = VideoFrame(image, timestamp, self.index, self.discontinuity, pts, time_cut)
        self.last_clock_pts = pts
        self.last_read_index = self.index
        self.index += 1
        self.discontinuity = False
        return result

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
