"""Single Ultralytics adapter shared by production and preview."""
from ultralytics import YOLO
import numpy as np
from types import SimpleNamespace

from core.reid import ReIDEncoder
from core.types import Detection
from core.vest_classifier import VestClassifier
from core.person_detections import distinct_person_indices


class PersonDetector:
    def __init__(self, cfg, detection_only=False):
        self.cfg = cfg
        path = cfg.path(cfg.MODEL_PATH)
        if not path.is_file():
            raise ValueError("MODEL_PATH must point to a local full-person detection model")
        self.model = YOLO(str(path))
        names = self.model.names
        label = names.get(cfg.PERSON_CLASS_ID, "").lower()
        if self.model.task not in {"detect", "pose"} or label not in {"person", "pedestrian"}:
            raise ValueError("PERSON_CLASS_ID must identify person/pedestrian; helmet/head models are unsuitable")
        if not cfg.path(cfg.TRACKER_PATH).is_file():
            raise ValueError("Tracker configuration does not exist")
        self.reid = self.vest = None
        if not detection_only:
            if not cfg.REID_MODEL_PATH:
                raise ValueError("Counting requires REID_MODEL_PATH; run python -m tools.prepare_reid")
            self.reid = ReIDEncoder(cfg.path(cfg.REID_MODEL_PATH), cfg.DEVICE, cfg.CPU_THREADS)
            if cfg.STAFF_MODE == "model":
                if not cfg.VEST_MODEL_PATH:
                    raise ValueError("STAFF_MODE=model requires VEST_MODEL_PATH")
                self.vest = VestClassifier(cfg.path(cfg.VEST_MODEL_PATH), cfg.DEVICE)
        self.ages = {}
        self.features = {}
        self._configured_backend = None
        if cfg.DEVICE == "cpu" and path.suffix == ".onnx":
            self.model.add_callback("on_predict_start", self._configure_onnx_cpu)
        self.appearance_tracker = None
        if cfg.TRACKING_MODE == "appearance":
            from ultralytics.utils import YAML
            from core.appearance_tracker import AppearanceTracker
            args = YAML.load(str(cfg.path(cfg.TRACKER_PATH)))
            if args.get("tracker_type") != "botsort" or args.get("model") != "auto":
                raise ValueError("Appearance mode requires botsort with model=auto; OSNet features are reused")
            args["with_reid"] = self.reid is not None
            self.appearance_tracker = AppearanceTracker(SimpleNamespace(**args), cfg.TRACKER_IDENTITY_MIN_SIMILARITY)

    def _configure_onnx_cpu(self, predictor):
        backend = predictor.model.backend
        if backend is self._configured_backend:
            return
        if getattr(backend, "use_io_binding", False):
            raise RuntimeError("CPU ONNX profile unexpectedly enabled GPU IO bindings")
        import onnxruntime as ort
        from core.onnx_runtime import cpu_options
        backend.session = ort.InferenceSession(str(self.cfg.path(self.cfg.MODEL_PATH)),
                                                sess_options=cpu_options(self.cfg.CPU_THREADS),
                                                providers=["CPUExecutionProvider"])
        self._configured_backend = backend

    def reset(self):
        predictor = self.model.predictor
        for tracker in getattr(predictor, "trackers", []):
            tracker.reset()
        self.ages.clear()
        self.features.clear()
        if self.appearance_tracker is not None:
            self.appearance_tracker.reset()

    def detect(self, frame):
        if self.appearance_tracker is not None:
            return self._detect_appearance(frame)
        results = self.model.track(frame, persist=True, tracker=str(self.cfg.path(self.cfg.TRACKER_PATH)),
                                   classes=[self.cfg.PERSON_CLASS_ID], conf=self.cfg.CONFIDENCE,
                                   imgsz=self.cfg.IMAGE_SIZE, device=self.cfg.DEVICE, verbose=False)
        detections = []
        boxes = results[0].boxes if results else None
        if boxes is not None and boxes.id is not None:
            keypoints = results[0].keypoints
            head_xy = keypoints.xy.cpu().numpy()[:, :5] if keypoints is not None else None
            head_conf = keypoints.conf.cpu().numpy()[:, :5] if keypoints is not None and keypoints.conf is not None else None
            for index, (tid, bbox, confidence) in enumerate(zip(boxes.id.cpu().tolist(), boxes.xyxy.cpu().tolist(),
                                              boxes.conf.cpu().tolist())):
                tid = int(tid)
                age = self.ages.get(tid, 0)
                probability = embedding = None
                if age < self.cfg.VEST_CONFIRM_FRAMES or age % self.cfg.FEATURE_INTERVAL_FRAMES == 0:
                    if self.reid is not None:
                        embedding = self.reid.encode(frame, bbox)
                        probability = self.vest.predict(frame, bbox) if self.vest is not None else None
                        if embedding is not None:
                            self.features[tid] = embedding
                self.ages[tid] = age + 1
                anchor = None
                required = self.model.task == "pose" and self.cfg.POSE_ANCHOR == "head"
                if required and head_xy is not None and head_conf is not None:
                    points, confidences = head_xy[index], head_conf[index]
                    valid = ((confidences >= self.cfg.POSE_ANCHOR_CONFIDENCE) & np.isfinite(points).all(axis=1)
                             & (points[:, 0] > 0) & (points[:, 1] > 0))
                    if valid.any():
                        anchor = tuple(np.median(points[valid], axis=0).tolist())
                detections.append(Detection(tid, tuple(bbox), confidence, self.features.get(tid), probability,
                                            anchor, required))
        active = {d.track_id for d in detections}
        self.ages = {tid: age for tid, age in self.ages.items() if tid in active}
        self.features = {tid: feature for tid, feature in self.features.items() if tid in active}
        return detections

    def _detect_appearance(self, frame):
        result = self.model.predict(frame, classes=[self.cfg.PERSON_CLASS_ID], conf=self.cfg.CONFIDENCE,
                                    imgsz=self.cfg.IMAGE_SIZE, device=self.cfg.DEVICE, verbose=False)[0]
        boxes = result.boxes.cpu().numpy()
        all_embeddings = self.reid.encode_many(frame, boxes.xyxy) if self.reid is not None else [None] * len(boxes)
        embeddings = []
        indices = []
        for index, bbox in enumerate(boxes.xyxy):
            embedding = all_embeddings[index]
            if self.reid is None or embedding is not None:
                indices.append(index)
                embeddings.append(embedding)
        selected = boxes[np.asarray(indices, dtype=int)]
        if self.reid is not None:
            keep = distinct_person_indices(selected.data, embeddings)
            indices = [indices[i] for i in keep]
            embeddings = [embeddings[i] for i in keep]
            selected = boxes[np.asarray(indices, dtype=int)]
        features = np.asarray(embeddings, dtype=np.float32) if self.reid is not None and embeddings else None
        tracks = self.appearance_tracker.update(selected, img=frame, feats=features)
        detections = []
        for row in tracks:
            tid, index = int(row[4]), int(row[-1])
            original_index = indices[index]
            bbox = tuple(boxes.xyxy[original_index].tolist())
            age = self.ages.get(tid, 0)
            probability = None
            if self.vest is not None and (age < self.cfg.VEST_CONFIRM_FRAMES or age % self.cfg.FEATURE_INTERVAL_FRAMES == 0):
                probability = self.vest.predict(frame, bbox)
            self.ages[tid] = age + 1
            anchor, required = None, self.model.task == "pose" and self.cfg.POSE_ANCHOR == "head"
            if required and result.keypoints is not None:
                points = result.keypoints.xy[original_index, :5].cpu().numpy()
                confidences = result.keypoints.conf[original_index, :5].cpu().numpy()
                valid = ((confidences >= self.cfg.POSE_ANCHOR_CONFIDENCE) & np.isfinite(points).all(axis=1)
                         & (points[:, 0] > 0) & (points[:, 1] > 0))
                if valid.any():
                    anchor = tuple(np.median(points[valid], axis=0).tolist())
            detections.append(Detection(tid, bbox, float(boxes.conf[original_index]),
                                        embeddings[index], probability, anchor, required))
        active = {d.track_id for d in detections}
        self.ages = {tid: age for tid, age in self.ages.items() if tid in active}
        return detections
