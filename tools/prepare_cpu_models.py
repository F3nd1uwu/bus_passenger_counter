"""Export validated CPU models. Requires requirements-export.txt; no runtime downloads."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO

from config.settings import ROOT
from core.onnx_runtime import cpu_options


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(detector_pt, reid_torchscript):
    detector_pt, reid_torchscript = Path(detector_pt).resolve(), Path(reid_torchscript).resolve()
    if not detector_pt.is_file() or not reid_torchscript.is_file():
        raise ValueError("Prepare local detector .pt and person-ReID TorchScript weights first")
    manifest_path = detector_pt.parent / "cpu_models.json"
    target_detector = detector_pt.with_suffix(".onnx")
    target_reid = reid_torchscript.with_suffix(".onnx")
    if manifest_path.is_file() and target_detector.is_file() and target_reid.is_file():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (saved.get("detector_source_sha256") == digest(detector_pt)
                and saved.get("reid_source_sha256") == digest(reid_torchscript)
                and saved.get("detector_onnx_sha256") == digest(target_detector)
                and saved.get("reid_onnx_sha256") == digest(target_reid)):
            print("Existing CPU model exports verified; files preserved")
            return
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix="cpu-export-", dir=detector_pt.parent) as directory:
        work = Path(directory)
        source = work / detector_pt.name
        shutil.copy2(detector_pt, source)
        detector_export = Path(YOLO(str(source)).export(format="onnx", imgsz=640, dynamic=True,
                                                        simplify=True, opset=17, device="cpu"))
        session = ort.InferenceSession(str(detector_export), sess_options=cpu_options(2), providers=["CPUExecutionProvider"])
        session.run(None, {session.get_inputs()[0].name: np.zeros((1, 3, 480, 640), np.float32)})
        del session
        reid = torch.jit.load(str(reid_torchscript), map_location="cpu").eval()
        reid_export = work / "person_reid.onnx"
        torch.onnx.export(reid, torch.zeros(1, 3, 256, 128), str(reid_export), input_names=["images"],
                          output_names=["features"], dynamic_axes={"images": {0: "batch"}, "features": {0: "batch"}},
                          opset_version=17, dynamo=False)
        session = ort.InferenceSession(str(reid_export), sess_options=cpu_options(2), providers=["CPUExecutionProvider"])
        rng = np.random.default_rng(19)
        for count in (1, 2, 3):
            sample = rng.normal(size=(count, 3, 256, 128)).astype(np.float32)
            with torch.inference_mode():
                reference = reid(torch.from_numpy(sample)).numpy()
            result = session.run(None, {"images": sample})[0]
            np.testing.assert_allclose(reference, result, rtol=1e-3, atol=1e-4)
            first = reference / np.linalg.norm(reference, axis=1, keepdims=True)
            second = result / np.linalg.norm(result, axis=1, keepdims=True)
            if not np.all(np.sum(first * second, axis=1) > 0.999999):
                raise RuntimeError("ReID ONNX export changed the representation")
        del session
        detector_export.replace(target_detector)
        reid_export.replace(target_reid)
    manifest = {"detector_source": str(detector_pt.name), "detector_source_sha256": digest(detector_pt),
                "detector_onnx_sha256": digest(target_detector), "reid_source_sha256": digest(reid_torchscript),
                "reid_onnx_sha256": digest(target_reid), "reid_tested_batches": [1, 2, 3],
                "reid_minimum_cosine_similarity": 0.999999, "precision": "FP32"}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {target_detector} and {target_reid}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", type=Path, default=ROOT / "models/yolo26n.pt")
    parser.add_argument("--reid", type=Path, default=ROOT / "models/person_reid.torchscript")
    args = parser.parse_args()
    prepare(args.detector, args.reid)
