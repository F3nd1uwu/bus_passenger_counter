"""Export a separately trained grayscale garment detector with its input contract."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export(weights, output, size=320):
    weights, output = Path(weights).resolve(), Path(output).resolve()
    if not weights.is_file() or weights.suffix != ".pt" or output.suffix != ".onnx" or size < 32 or size % 32:
        raise ValueError("Use local .pt weights, a new .onnx output, and a size divisible by 32")
    if output.exists() or output.with_suffix(".json").exists():
        raise ValueError("Vest export already exists; choose a new name")
    import onnx
    import numpy as np
    import torch
    from ultralytics import YOLO
    from core.vest_classifier import VestClassifier
    torch.set_num_threads(2)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vest-export-", dir=output.parent) as folder:
        local = Path(folder) / "vest.pt"
        shutil.copy2(weights, local)
        model = YOLO(str(local))
        if model.task != "detect" or model.names != {0: "staff_vest"}:
            raise ValueError("Expected a one-class staff_vest detector trained on grayscale crops")
        exported = Path(model.export(format="onnx", imgsz=size, dynamic=False,
                                     simplify=True, opset=17, device="cpu"))
        graph = onnx.load(str(exported))
        metadata = {item.key: item.value for item in graph.metadata_props}
        metadata.update(bpc_vest_format="grayscale_garment_detection_v1", classes="staff_vest", image_size=str(size))
        onnx.helper.set_model_props(graph, metadata)
        onnx.save(graph, str(exported))
        verifier = VestClassifier(exported, cpu_threads=2)
        score = verifier.predict(np.zeros((160, 80, 3), np.uint8), (0, 0, 80, 160))
        if score is None or not 0 <= score <= 1:
            raise ValueError("Export failed inference contract check")
        del verifier
        exported.replace(output)
    output.with_suffix(".json").write_text(json.dumps({
        "format": "grayscale_garment_detection_v1", "image_size": size,
        "source_sha256": digest(weights), "onnx_sha256": digest(output),
        "input": "grayscale, square letterbox 114, three equal channels, float32 /255",
        "classes": {"0": "staff_vest"}, "activated": False,
    }, indent=2), encoding="utf-8")
    print(f"Exported {output}; evaluate passenger false exclusions before activation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=320)
    args = parser.parse_args()
    export(args.weights, args.output, args.image_size)
