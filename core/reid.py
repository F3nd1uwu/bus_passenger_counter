"""ONNX/TorchScript person ReID encoder. The model must output [N, D] embeddings.

Contract: RGB, 256x128, float32, ImageNet mean/std; output is L2-normalised
here. Supply an encoder trained for person re-identification, not class logits.
"""
from pathlib import Path

import cv2
import numpy as np

from core.vest_classifier import person_crop


def normalise_embedding(value):
    if value is None:
        return None
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 1 or not value.size or not np.isfinite(value).all():
        raise ValueError("ReID embedding must be a finite nonempty vector")
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("ReID encoder returned a zero vector")
    return value / norm


class ReIDEncoder:
    def __init__(self, model_path, device="cpu", threads=4):
        if not Path(model_path).is_file():
            raise ValueError("REID_MODEL_PATH must point to a trained TorchScript or ONNX ReID encoder")
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.session = None
        if Path(model_path).suffix.lower() == ".onnx":
            import onnxruntime as ort
            from core.onnx_runtime import cpu_options
            if device != "cpu":
                raise ValueError("This ReID ONNX profile runs on CPU; use TorchScript weights for CUDA")
            self.session = ort.InferenceSession(str(model_path), sess_options=cpu_options(threads),
                                                providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            sample = self.session.run(None, {self.input_name: np.zeros((1, 3, 256, 128), np.float32)})[0]
        else:
            import torch
            self.torch = torch
            self.device = torch.device(f"cuda:{device}" if str(device).isdigit() else device)
            self.model = torch.jit.load(str(model_path), map_location=self.device).eval()
            with torch.inference_mode():
                sample = self.model(torch.zeros(1, 3, 256, 128, device=self.device))
            if not isinstance(sample, torch.Tensor):
                raise ValueError("ReID encoder must return a Tensor [N, D]")
        if sample.ndim != 2 or sample.shape[0] != 1:
            raise ValueError("ReID encoder must return features [N, D]")
        self.dimension = sample.shape[1]

    def encode(self, frame, bbox):
        return self.encode_many(frame, [bbox])[0]

    def encode_many(self, frame, bboxes):
        outputs = [None] * len(bboxes)
        images, indices = [], []
        for index, bbox in enumerate(bboxes):
            crop = person_crop(frame, bbox)
            if crop is not None:
                rgb = cv2.cvtColor(cv2.resize(crop, (128, 256)), cv2.COLOR_BGR2RGB)
                rgb = (rgb.astype(np.float32) / 255.0 - self.mean) / self.std
                images.append(rgb.transpose(2, 0, 1))
                indices.append(index)
        if not images:
            return outputs
        batch = np.ascontiguousarray(np.stack(images), dtype=np.float32)
        if self.session is not None:
            result = self.session.run(None, {self.input_name: batch})[0]
        else:
            tensor = self.torch.from_numpy(batch).to(self.device)
            with self.torch.inference_mode():
                result = self.model(tensor).cpu().numpy()
        if result.shape != (len(images), self.dimension):
            raise ValueError("ReID output shape changed")
        for index, feature in zip(indices, result):
            outputs[index] = normalise_embedding(feature)
        return outputs
