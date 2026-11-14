"""Recognise the actual work vest by a trained crop classifier, not HSV colour."""
from pathlib import Path
import hashlib
import json

import cv2
import numpy as np


def spatial_consensus(first, second, minimum_iou):
    """Joint confidence for the same garment, not two unrelated bright regions.

    Rows are xyxy + confidence in the shared letterbox coordinates. Examine
    scores in descending order and stop once no remaining pair can win.
    """
    best = 0.0
    if not len(first) or not len(second):
        return best
    first = first[np.argsort(-first[:, 4])]
    second = second[np.argsort(-second[:, 4])]
    for box in first:
        if box[4] <= best or second[0, 4] <= best:
            break
        candidates = second[second[:, 4] > best]
        intersection = np.maximum(0, np.minimum(box[2:4], candidates[:, 2:4])
                                  - np.maximum(box[:2], candidates[:, :2])).prod(axis=1)
        union = (np.prod(box[2:4] - box[:2])
                 + np.prod(candidates[:, 2:4] - candidates[:, :2], axis=1) - intersection)
        agreeing = intersection >= minimum_iou * union
        if agreeing.any():
            best = max(best, float(min(box[4], candidates[agreeing, 4].max())))
    return best


def garment_tensor(crop, size):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    scale = size / max(height, width)
    w, h = max(1, round(width * scale)), max(1, round(height * scale))
    x, y = (size - w) // 2, (size - h) // 2
    canvas = np.full((size, size), 114, np.uint8)
    canvas[y:y+h, x:x+w] = cv2.resize(gray, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.repeat(canvas[None], 3, axis=0).astype(np.float32) / 255, (x, y, x+w, y+h)


def vest_tensor(crop, size=160):
    """Preserve the whole garment; discard hue before learning/inference."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    scale = size / max(height, width)
    w, h = max(1, round(width * scale)), max(1, round(height * scale))
    canvas = np.full((size, size), 114, np.uint8)
    x, y = (size - w) // 2, (size - h) // 2
    canvas[y:y+h, x:x+w] = cv2.resize(gray, (w, h), interpolation=cv2.INTER_LINEAR)
    rgb = np.repeat(canvas[None], 3, axis=0).astype(np.float32) / 255
    return (rgb - np.asarray([.485, .456, .406], np.float32)[:, None, None]) / np.asarray(
        [.229, .224, .225], np.float32)[:, None, None]


def person_crop(frame, bbox):
    if len(bbox) != 4 or not np.isfinite(bbox).all():
        return None
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
    def __init__(self, model_path, device="cpu", cpu_threads=4):
        from ultralytics import YOLO
        if not Path(model_path).is_file():
            raise ValueError("VEST_MODEL_PATH must point to a trained classification model")
        self.session = None
        self.members = None
        self.garment_detector = False
        if Path(model_path).suffix.lower() == ".json":
            path = Path(model_path).resolve()
            bundle = json.loads(path.read_text(encoding="utf-8"))
            legacy = bundle.get("format") == "grayscale_garment_ensemble_v1" and bundle.get("fusion") == "min"
            spatial = (bundle.get("format") == "grayscale_garment_ensemble_v2"
                       and bundle.get("fusion") == "spatial_min")
            if not (legacy or spatial) or len(bundle.get("members", [])) != 2:
                raise ValueError("Unsupported garment ensemble manifest")
            self.minimum_iou = bundle.get("minimum_iou") if spatial else None
            if spatial and (type(self.minimum_iou) not in (int, float)
                            or not np.isfinite(self.minimum_iou) or not 0 < self.minimum_iou <= 1):
                raise ValueError("Garment consensus requires minimum_iou in (0,1]")
            members = []
            for item in bundle["members"]:
                member = (path.parent / item["path"]).resolve()
                if member.parent != path.parent or member.suffix != ".onnx":
                    raise ValueError("Garment ensemble members must be adjacent ONNX files")
                with member.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != item["sha256"]:
                        raise ValueError("Garment ensemble model checksum mismatch")
                classifier = VestClassifier(member, device, cpu_threads)
                if not classifier.garment_detector:
                    raise ValueError("Ensemble members must detect the garment itself")
                members.append(classifier)
            if members[0].size != members[1].size:
                raise ValueError("Garment ensemble input sizes differ")
            self.members, self.garment_detector = members, True
            self.size = members[0].size
            return
        if Path(model_path).suffix.lower() == ".onnx":
            import onnxruntime as ort
            from core.onnx_runtime import cpu_options
            providers = ["CPUExecutionProvider"]
            if device != "cpu":
                if "CUDAExecutionProvider" not in ort.get_available_providers():
                    raise ValueError("Vest ONNX GPU inference requires CUDAExecutionProvider")
                providers.insert(0, "CUDAExecutionProvider")
            session = ort.InferenceSession(str(model_path), sess_options=cpu_options(cpu_threads), providers=providers)
            metadata = session.get_modelmeta().custom_metadata_map
            if metadata.get("bpc_vest_format") in {"grayscale_letterbox_v1", "grayscale_garment_detection_v1"}:
                self.garment_detector = metadata["bpc_vest_format"] == "grayscale_garment_detection_v1"
                expected = "staff_vest" if self.garment_detector else "passenger,staff_vest"
                if metadata.get("classes") != expected:
                    raise ValueError("Invalid vest ONNX class order")
                self.size = int(metadata["image_size"])
                if self.size < 32 or session.get_inputs()[0].shape[1:] != [3, self.size, self.size]:
                    raise ValueError("Invalid vest ONNX input shape")
                self.session = session
                return
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
        if self.members is not None:
            if getattr(self, "minimum_iou", None) is not None:
                return spatial_consensus(*(member._garment_detections(crop) for member in self.members),
                                         self.minimum_iou)
            scores = [member.predict(frame, bbox) for member in self.members]
            return min(scores) if all(score is not None for score in scores) else None
        if self.session is not None:
            if self.garment_detector:
                detections = self._garment_detections(crop)
                return float(detections[:, 4].max()) if len(detections) else 0.0
            logits = np.asarray(self.session.run(None, {self.session.get_inputs()[0].name:
                                vest_tensor(crop, self.size)[None]})[0], np.float32)
            if logits.shape != (1, 2) or not np.isfinite(logits).all():
                raise RuntimeError("Invalid vest classification logits")
            probabilities = np.exp(logits[0] - logits[0].max())
            return float(probabilities[1] / probabilities.sum())
        result = self.model.predict(crop, device=self.device, verbose=False)[0]
        if result.probs is None:
            raise RuntimeError("Vest model produced no classification probabilities")
        return float(result.probs.data[self.staff_class].cpu().item())

    def _garment_detections(self, crop):
        tensor, content = garment_tensor(crop, self.size)
        output = np.asarray(self.session.run(None, {self.session.get_inputs()[0].name: tensor[None]})[0])
        if output.ndim != 3 or output.shape[0] != 1 or not np.isfinite(output).all():
            raise RuntimeError("Invalid garment detection output")
        if output.shape[1] == 5:
            boxes = output[0, :4].T.copy()
            scores = output[0, 4]
            boxes[:, :2] -= boxes[:, 2:] / 2
            boxes[:, 2:] += boxes[:, :2]
        elif output.shape[2] == 6:
            boxes, scores = output[0, :, :4], output[0, :, 4]
            if np.any(output[0, :, 5] != 0):
                raise RuntimeError("Unexpected garment class index")
        else:
            raise RuntimeError("Unsupported garment detector output shape")
        if np.any((scores < 0) | (scores > 1)):
            raise RuntimeError("Invalid garment confidence")
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        valid = ((centers[:, 0] >= content[0]) & (centers[:, 0] <= content[2])
                 & (centers[:, 1] >= content[1]) & (centers[:, 1] <= content[3])
                 & np.all(boxes[:, 2:] - boxes[:, :2] >= 4, axis=1))
        boxes = boxes[valid].copy()
        boxes[:, :2] = np.maximum(boxes[:, :2], content[:2])
        boxes[:, 2:] = np.minimum(boxes[:, 2:], content[2:])
        return np.column_stack((boxes, scores[valid]))
