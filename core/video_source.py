"""File EOF terminates; live sources reconnect with bounded network reads."""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
import time
from urllib.parse import urlsplit

import cv2


def source_kind(source, requested="auto"):
    if requested != "auto":
        return requested
    if str(source).isdigit():
        return "camera"
    if urlsplit(str(source)).scheme.lower() in {"rtsp", "rtsps", "http", "https", "rtmp", "udp", "tcp"}:
        return "stream"
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


class VideoSource:
    def __init__(self, cfg, start=None, resume_frame=0, stop=None):
        self.cfg = cfg
        self.kind = source_kind(cfg.VIDEO_SOURCE, cfg.SOURCE_KIND)
        self.source = cfg.VIDEO_SOURCE
        if self.kind == "file":
            self.source = str(cfg.path(self.source))
            if not Path(self.source).is_file():
                raise ValueError("Video file does not exist")
        if self.kind == "camera":
            try:
                self.source = int(self.source)
            except ValueError as exc:
                raise ValueError("Camera source must be a nonnegative integer") from exc
            if self.source < 0:
                raise ValueError("Camera index must be nonnegative")
        self.start = start or datetime.now(timezone.utc)
        self.clock_wall, self.clock_mono = datetime.now(timezone.utc), time.monotonic()
        self.stop = stop
        self.cap = None
        self.index = resume_frame
        self.resume_frame = resume_frame
        self.fps = 25.0
        self.total_frames = 0
        self.eof = False
        self.discontinuity = True
        self.last_video_seconds = -1.0

    def _open(self):
        self.cap = cv2.VideoCapture()
        if self.kind == "stream":
            opened = self.cap.open(self.source, cv2.CAP_FFMPEG, [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.cfg.OPEN_TIMEOUT_MS,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.cfg.READ_TIMEOUT_MS])
        else:
            opened = self.cap.open(self.source)
        if not opened:
            self.close()
            if self.kind == "file":
                raise ValueError("Cannot open video file")
            return False
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if math.isfinite(fps) and fps > 0:
            self.fps = fps
        count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.total_frames = int(count) if math.isfinite(count) and count > 0 else 0
        if self.kind == "file" and self.resume_frame:
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
        self.last_video_seconds = -1.0
        self.eof = False
        self.discontinuity = True
        return target

    def _retry_delay(self):
        if self.stop is not None:
            self.stop.wait(self.cfg.RECONNECT_DELAY_SEC)
        else:
            time.sleep(self.cfg.RECONNECT_DELAY_SEC)

    def read(self):
        if self.eof:
            return None
        if self.cap is None and not self._open():
            self._retry_delay()
            return None
        ok, image = self.cap.read()
        if not ok:
            expected = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.close()
            if self.kind == "file":
                if math.isfinite(expected) and expected > 0 and self.index < expected - 2:
                    raise RuntimeError("Video decoding failed before EOF; checkpoint retained for retry")
                self.eof = True
            else:
                self._retry_delay()
            return None
        if self.kind == "file":
            # Prefer presentation timestamps for variable-frame-rate recordings.
            pts = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if not math.isfinite(pts) or pts < 0 or (self.index > 0 and (pts <= 0 or pts <= self.last_video_seconds)):
                pts = max(self.index / self.fps, self.last_video_seconds + 1 / self.fps)
            self.last_video_seconds = pts
            timestamp = self.start + timedelta(seconds=pts)
        else:
            timestamp = self.clock_wall + timedelta(seconds=time.monotonic() - self.clock_mono)
        result = VideoFrame(image, timestamp, self.index, self.discontinuity)
        self.index += 1
        self.discontinuity = False
        return result

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
