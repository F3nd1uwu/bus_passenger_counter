"""Export a trained detector to a separate FP32 CPU candidate, keeping ReID intact."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export(weights, output):
    weights, output = Path(weights).resolve(), Path(output).resolve()
    if not weights.is_file() or weights.suffix != ".pt" or output.suffix != ".onnx":
        raise ValueError("Use existing local .pt weights and a separate .onnx output")
    pt = output.with_suffix(".pt")
    manifest = output.with_suffix(".json")
    if manifest.is_file() and output.is_file() and pt.is_file():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if (saved["source_sha256"] == digest(weights) == digest(pt)
                and saved["onnx_sha256"] == digest(output)):
            print(f"Verified existing candidate {output}")
            return
    if any(p.exists() for p in (pt, output, manifest)):
        raise ValueError("Candidate files already exist with different/missing provenance; choose another output name")
    output.parent.mkdir(parents=True, exist_ok=True)
    import onnxruntime as ort
    import torch
    from ultralytics import YOLO
    from core.onnx_runtime import cpu_options
    torch.set_num_threads(4)
    with tempfile.TemporaryDirectory(prefix="candidate-export-", dir=output.parent) as folder:
        local = Path(folder) / pt.name
        shutil.copy2(weights, local)
        model = YOLO(str(local))
        if model.task != "detect" or model.names.get(0) != "person":
            raise ValueError("Candidate must be a person detector with class 0")
        exported = Path(model.export(format="onnx", imgsz=640, dynamic=True, simplify=True,
                                     opset=17, device="cpu"))
        session = ort.InferenceSession(str(exported), sess_options=cpu_options(4),
                                        providers=["CPUExecutionProvider"])
        shapes = []
        for size in ((480, 640), (640, 640)):
            result = session.run(None, {session.get_inputs()[0].name: np.zeros((1, 3, *size), np.float32)})[0]
            # Fine-tuned checkpoints can expose the ordinary detection head;
            # Ultralytics then performs NMS. Both supported layouts are valid.
            valid_shape = (result.ndim == 3 and result.shape[0] == 1 and
                           ((result.shape[1] == 4 + len(model.names) and result.shape[2] > 0)
                            or (result.shape[1] > 0 and result.shape[2] == 6)))
            if not valid_shape or not np.isfinite(result).all():
                raise ValueError("Unexpected exported person detector output")
            shapes.append(list(result.shape))
        del session
        shutil.copy2(weights, pt)
        exported.replace(output)
    manifest.write_text(json.dumps({"source_sha256": digest(weights), "onnx_sha256": digest(output),
                                    "precision": "FP32", "image_size": 640, "classes": {"0": "person"},
                                    "tested_shapes": [[1, 3, 480, 640], [1, 3, 640, 640]],
                                    "output_shapes": shapes,
                                    "reid_changed": False, "activated": False}, indent=2), encoding="utf-8")
    print(f"Prepared candidate {output}; working model settings were not changed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/candidates/person_manual_v1.onnx"))
    args = parser.parse_args()
    export(args.weights, args.output)
