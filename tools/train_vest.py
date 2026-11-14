"""Train a grayscale garment classifier from reviewed, temporally split crops.

The manifest is a JSON object with samples: [{path, split, label, sha256,
reviewed, episode}]. Labels are passenger=0, staff_vest=1. Entire episodes must
belong to one split. Test samples are never loaded during fitting/selection.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

from core.vest_classifier import vest_tensor


def validate_manifest(path):
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    episodes, hashes = {}, {}
    for row in data["samples"]:
        if row.get("reviewed") is not True or row["split"] not in {"train", "val", "test"} or row["label"] not in {0, 1}:
            raise ValueError("Unreviewed crop or invalid split/class")
        image = (path.parent / row["path"]).resolve()
        if not image.is_relative_to(path.parent.resolve()):
            raise ValueError("Crop path must stay inside the dataset")
        if hashlib.sha256(image.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Changed crop: {image}")
        for mapping, key in ((episodes, row["episode"]), (hashes, row["sha256"])):
            if mapping.setdefault(key, row["split"]) != row["split"]:
                raise ValueError("Episode or identical crop leaks across splits")
    for split in ("train", "val", "test"):
        if {r["label"] for r in data["samples"] if r["split"] == split} != {0, 1}:
            raise ValueError(f"Both classes required in {split}")
    return data


class Crops(Dataset):
    def __init__(self, root, rows, size, augment=False):
        self.rows, self.size, self.augment = rows, size, augment
        self.images = [cv2.imread(str(root / r["path"])) for r in rows]
        if any(im is None for im in self.images):
            raise ValueError("Unreadable crop")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        im = self.images[i].copy()
        if self.augment:
            if random.random() < .5:
                im = cv2.flip(im, 1)
            im = np.clip(im.astype(np.float32) * random.uniform(.65, 1.35) + random.uniform(-20, 20), 0, 255).astype(np.uint8)
            # Small garment occlusions discourage memorising a logo or head.
            if random.random() < .4:
                h, w = im.shape[:2]
                x, y = random.randrange(w), random.randrange(h)
                im[y:min(h, y+max(1, h//5)), x:min(w, x+max(1, w//4))] = 114
        return torch.from_numpy(vest_tensor(im, self.size)), self.rows[i]["label"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--size", type=int, default=160)
    parser.add_argument("--freeze-blocks", type=int, default=9)
    parser.add_argument("--learning-rate", type=float, default=.0003)
    args = parser.parse_args()
    if min(args.epochs, args.threads) < 1 or args.size < 32 or not 0 <= args.freeze_blocks <= 12 or not 0 < args.learning_rate < 1:
        parser.error("Invalid training dimensions")
    data = validate_manifest(args.manifest)
    args.output.mkdir(parents=True, exist_ok=False)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    torch.set_num_threads(args.threads)
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 2)
    for layer in model.features[:args.freeze_blocks]:
        for parameter in layer.parameters():
            parameter.requires_grad = False
    train_rows = [r for r in data["samples"] if r["split"] == "train"]
    val_rows = [r for r in data["samples"] if r["split"] == "val"]
    frequencies = np.bincount([r["label"] for r in train_rows], minlength=2)
    sampler = WeightedRandomSampler([1/frequencies[r["label"]] for r in train_rows], len(train_rows), replacement=True)
    training = DataLoader(Crops(args.manifest.parent, train_rows, args.size, True), batch_size=16, sampler=sampler)
    validation = DataLoader(Crops(args.manifest.parent, val_rows, args.size), batch_size=32)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=.01)
    criterion = torch.nn.CrossEntropyLoss()
    best = (float("inf"),)*3
    progress = []
    for epoch in range(args.epochs):
        model.train()
        for layer in model.features[:args.freeze_blocks]:
            layer.eval()  # Frozen backbone batch-normalization statistics must stay frozen too.
        for images, labels in training:
            optimizer.zero_grad(); loss = criterion(model(images), labels); loss.backward(); optimizer.step()
        model.eval(); probabilities=[]; targets=[]; losses=[]
        with torch.inference_mode():
            for images, labels in validation:
                output = model(images); probabilities.extend(output.softmax(1)[:, 1].tolist()); targets.extend(labels.tolist())
                losses.append(float(criterion(output, labels)))
        pred = np.asarray(probabilities) >= .90; truth = np.asarray(targets) == 1
        fp, fn = int((pred & ~truth).sum()), int((~pred & truth).sum())
        score = (fp, fn, float(np.mean(losses)))
        record = dict(epoch=epoch+1, passenger_false_exclusions=fp, vest_misses=fn, val_loss=score[2])
        progress.append(record); print(json.dumps(record), flush=True)
        if score < best:
            best=score; torch.save(model.state_dict(), args.output/'best.pt')
            (args.output/'selection.json').write_text(json.dumps(record, indent=2))
        (args.output/'progress.json').write_text(json.dumps(progress, indent=2))
    model.load_state_dict(torch.load(args.output/'best.pt', map_location='cpu', weights_only=True));model.eval()
    onnx_path=args.output/'vest.onnx'
    torch.onnx.export(model, torch.zeros(1,3,args.size,args.size), str(onnx_path), input_names=['images'], output_names=['logits'], opset_version=17, dynamo=False)
    import onnx
    graph=onnx.load(str(onnx_path))
    onnx.helper.set_model_props(graph, {'bpc_vest_format':'grayscale_letterbox_v1','classes':'passenger,staff_vest','image_size':str(args.size)})
    onnx.save(graph,str(onnx_path))
    manifest_hash=hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    (args.output/'provenance.json').write_text(json.dumps({'manifest_sha256':manifest_hash,'model_sha256':hashlib.sha256(onnx_path.read_bytes()).hexdigest(),'architecture':'torchvision mobilenet_v3_small','pretrained':'IMAGENET1K_V1','preprocessing':'grayscale_letterbox_v1','seed':42,'selection':'lexicographic val FP, FN at .90, cross entropy; test never used','arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}},indent=2))


if __name__ == '__main__':
    main()
