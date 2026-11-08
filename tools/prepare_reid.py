"""Download official person-ReID weights, verify SHA256, export a local encoder.

Run once: python -m tools.prepare_reid
Runtime performs no downloads. OSNet architecture is vendored with its MIT license.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.request import urlopen

import torch

from config.settings import ROOT
from tools.vendor.osnet import osnet_x0_25

HF_REVISION = "a5c5cc037c24235cda3b21085b93ad77c9616224"
WEIGHTS_NAME = "osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
WEIGHTS_SHA256 = "cf55163d78fc44c62c82f85ab62d39f10438679b5abe8c698ae08cfa84aa6e18"
WEIGHTS_URL = f"https://huggingface.co/kaiyangzhou/osnet/resolve/{HF_REVISION}/{WEIGHTS_NAME}"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(output):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    # Temporary files live beside the output, including when the system disk is small.
    with tempfile.TemporaryDirectory(prefix="reid-export-", dir=output.parent) as folder:
        weights = Path(folder) / "weights.pth"
        with urlopen(WEIGHTS_URL, timeout=60) as response, weights.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        if sha256(weights) != WEIGHTS_SHA256:
            raise RuntimeError("Downloaded ReID weights failed SHA256 verification")
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
        state = checkpoint.get("state_dict", checkpoint)
        state = {key.removeprefix("module."): value for key, value in state.items()}
        model = osnet_x0_25(num_classes=1, pretrained=False).eval()
        # Training-only identity classification is deliberately not exported.
        state = {key: value for key, value in state.items() if not key.startswith("classifier.")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if set(missing) != {"classifier.weight", "classifier.bias"} or unexpected:
            raise RuntimeError(f"ReID architecture/weights mismatch: {missing}, {unexpected}")
        sample = torch.rand(1, 3, 256, 128)
        with torch.inference_mode():
            reference = model(sample)
            exported = torch.jit.trace(model, sample)
            exported = torch.jit.freeze(exported)
            # Frozen Conv/BN fusion changes the last FP32 digits on some CPUs.
            actual = exported(sample)
            torch.testing.assert_close(reference, actual, rtol=1e-3, atol=1e-4)
            similarity = torch.nn.functional.cosine_similarity(reference, actual).item()
            if similarity < 0.999999:
                raise RuntimeError("TorchScript export changed the ReID representation")
            # A different image must yield a different finite feature vector.
            other = exported(1 - sample)
            if reference.shape != (1, 512) or not torch.isfinite(other).all() or torch.allclose(reference, other):
                raise RuntimeError("Invalid ReID feature output")
        temporary_output = Path(folder) / output.name
        exported.save(str(temporary_output))
        temporary_output.replace(output)
    manifest = {"model": "OSNet x0.25, MSMT17 person-ReID", "weights_url": WEIGHTS_URL,
                "weights_sha256": WEIGHTS_SHA256, "export_sha256": sha256(output),
                "input": "RGB float32 NCHW 256x128, ImageNet mean/std",
                "output": "512-dimensional features, L2-normalised by core/reid.py",
                "torch_version": torch.__version__, "camera_calibration": "not yet evaluated on labelled bus video"}
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {output} (512-dimensional person-ReID encoder)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "models/person_reid.torchscript")
    prepare(parser.parse_args().output)
