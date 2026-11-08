"""Recognise the actual work vest by a trained crop classifier, not HSV colour."""
from pathlib import Path


def person_crop(frame, bbox):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = min(width, max(0, int(x1))), min(height, max(0, int(y1)))
    x2, y2 = max(0, min(width, int(x2))), max(0, min(height, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.shape[0] < 32 or crop.shape[1] < 16:
        return None
    return crop


class VestClassifier:
    def __init__(self, model_path, device="cpu"):
        from ultralytics import YOLO
        if not Path(model_path).is_file():
            raise ValueError("VEST_MODEL_PATH must point to a trained classification model")
        self.model = YOLO(str(model_path), task="classify")
        names = self.model.names
        if self.model.task != "classify" or set(names.values()) != {"staff_vest", "passenger"}:
            raise ValueError("Vest model must classify exactly staff_vest and passenger")
        self.staff_class = next(i for i, name in names.items() if name == "staff_vest")
        self.device = device

    def predict(self, frame, bbox):
        crop = person_crop(frame, bbox)
        if crop is None:
            return None
        result = self.model.predict(crop, device=self.device, verbose=False)[0]
        if result.probs is None:
            raise RuntimeError("Vest model produced no classification probabilities")
        return float(result.probs.data[self.staff_class].cpu().item())
