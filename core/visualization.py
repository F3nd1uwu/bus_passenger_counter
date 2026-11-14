from pathlib import Path

import cv2


def draw_preview(frame, detections, cfg, engine=None, events=(), *, metrics=None):
    result = frame.copy()
    height, width = frame.shape[:2]
    left, right = int(width * cfg.DOOR_X_MIN), int(width * cfg.DOOR_X_MAX)
    cv2.rectangle(result, (left, 0), (right, height - 1), (140, 140, 140), 1)
    for value, label in ((cfg.Y_STREET, "STREET"), (cfg.Y_LINE, "THRESHOLD"), (cfg.Y_CABIN, "CABIN")):
        y = int(value * height)
        cv2.line(result, (left, y), (right, y), (0, 210, 255), 1)
        cv2.putText(result, label, (left + 4, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 210, 255), 1)
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection.bbox)
        status, color, person = "PERSON", (0, 220, 0), ""
        if engine:
            track = engine.tracks.get(detection.track_id)
            identity = engine.identities.get(track.identity_id) if track else None
            if identity:
                person = f" P:{identity.identity_id[:6]}"
            if track and cfg.STAFF_MODE == "model" and track.staff.confirmed:
                status, color = "STAFF", (0, 140, 255)
            elif track:
                status = "PENDING" if track.pending else (track.zone or "TRANSIT").upper()
            if track and cfg.CHILD_MODE == 'model' and not (cfg.STAFF_MODE == 'model' and track.staff.confirmed):
                if track.child.confirmed:
                    status, color = 'CHILD' + ('/PENDING' if track.pending else ''), (255, 150, 0)
                elif not track.child.adult_seen:
                    status += '/AGE?'
        if detection.confidence < cfg.CONFIDENCE:
            status += "/WEAK"
            color = (150, 150, 150)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        anchor = detection.anchor
        if anchor is None and not detection.anchor_required:
            anchor = ((x1 + x2) / 2, y1 + (y2 - y1) * cfg.ANCHOR_Y)
        if anchor is not None:
            cv2.circle(result, tuple(map(int, anchor)), 3, color, -1)
        cv2.putText(result, f"ID {detection.track_id}{person} {detection.confidence:.2f} {status}",
                    (max(0, x1), max(55, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    text = (f"IN {engine.total_in}  OUT {engine.total_out}  SUPPRESSED {engine.total_suppressed}"
            if engine else "DETECTION ONLY - counting disabled")
    if engine and cfg.STAFF_MODE == "disabled":
        text += " | STAFF FILTER OFF"
    cv2.rectangle(result, (0, 0), (width, 30), (0, 0, 0), -1)
    cv2.putText(result, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if events:
        event = events[-1]
        cv2.putText(result, f"{event.event_type} {event.reason}", (8, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    if metrics is not None:
        processing = metrics["processing_fps"]
        fps_label = f"{processing:.1f}" if processing is not None else "warming up"
        first = f"PROCESSING {fps_label} FPS | SOURCE {metrics['source_fps']:.1f} FPS"
        second = f"FRAME {metrics['frame_index']}"
        if metrics["media_seconds"] is not None:
            seconds = max(0, metrics["media_seconds"])
            hours, remaining = divmod(seconds, 3600)
            minutes, seconds = divmod(remaining, 60)
            second += f" | FILE TIME {int(hours):02}:{int(minutes):02}:{seconds:05.2f}"
        second += f" | COUNT FROM {metrics['segment_start']}"
        cv2.rectangle(result, (0, max(0, height - 49)), (width, height), (0, 0, 0), -1)
        for offset, label in ((33, first), (12, second)):
            cv2.putText(result, label, (8, height - offset), cv2.FONT_HERSHEY_SIMPLEX,
                        0.43, (255, 255, 255), 1)
    return result


class PreviewWriter:
    def __init__(self, path, fps, size):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not self.writer.isOpened():
            raise RuntimeError("Cannot create preview video")

    def write(self, frame):
        self.writer.write(frame)

    def close(self):
        self.writer.release()
