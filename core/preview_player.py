"""HighGUI controls. Programmatic slider updates never request a seek."""
import cv2


class PreviewPlayer:
    WINDOW = "Bus passenger counter"
    TIMELINE = "Frame"
    LEFT = {2424832, 65361, 63234}
    RIGHT = {2555904, 65363, 63235}

    def __init__(self, fps, total_frames=0, *, seekable=False):
        self.fps = fps
        self.total_frames = total_frames
        self.seekable = seekable
        self.frame_index = 0
        self.seek_request = None
        self.paused = False
        self.at_end = False
        self._updating = True
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        self.has_timeline = seekable and total_frames > 1
        if self.has_timeline:
            cv2.createTrackbar(self.TIMELINE, self.WINDOW, 0, total_frames - 1, self._timeline_changed)
        self._updating = False

    def _timeline_changed(self, value):
        if not self._updating and value != self.frame_index:
            self.request_seek(value)

    def request_seek(self, frame_index):
        if not self.seekable:
            return
        target = max(0, int(frame_index))
        if self.total_frames:
            target = min(target, self.total_frames - 1)
        if target != self.frame_index or self.at_end:
            self.seek_request = target

    def take_seek(self):
        target = self.seek_request
        self.seek_request = None
        if target is not None:
            self.at_end = False
        return target

    def present(self, image, frame_index):
        self.frame_index = frame_index
        cv2.imshow(self.WINDOW, image)
        if self.has_timeline and self.seek_request is None:
            self._updating = True
            try:
                cv2.setTrackbarPos(self.TIMELINE, self.WINDOW, frame_index)
            finally:
                self._updating = False

    def handle_key(self, key):
        if key in self.LEFT or key in self.RIGHT:
            base = self.seek_request if self.seek_request is not None else self.frame_index
            offset = max(1, round(5 * self.fps))
            self.request_seek(base + (offset if key in self.RIGHT else -offset))
        elif key >= 0:
            char = key & 0xFF
            if char in {ord("q"), ord("Q"), 27}:
                return True
            if char == ord(" "):
                if self.at_end:
                    self.seek_request = 0
                    self.at_end = False
                    self.paused = False
                else:
                    self.paused = not self.paused
        return False

    def poll(self):
        if self.handle_key(cv2.waitKeyEx(50 if self.paused else 1)):
            return True
        try:
            return cv2.getWindowProperty(self.WINDOW, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True

    def mark_end(self):
        self.at_end = True
        self.paused = True
