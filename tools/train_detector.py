"""Fine-tune pretrained nano weights after checking manual annotations."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def validate_labels(dataset):
    counts = {}
    for split in ("train", "val"):
        images = [p for p in (dataset / "images" / split).glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if not images:
            raise ValueError(f"No {split} images; split whole boarding episodes before training")
        positive = 0
        for image in images:
            label = dataset / "labels" / split / (image.stem + ".txt")
            if not label.is_file():
                raise ValueError(f"Missing manual annotation: {label}. Unlabelled images must not become negatives")
            rows = label.read_text(encoding="utf-8").splitlines()
            for line in rows:
                if not line.strip():
                    continue
                values = line.split()
                if len(values) != 5 or values[0] != "0":
                    raise ValueError(f"Expected class 0 and four YOLO coordinates: {label}")
                x, y, w, h = map(float, values[1:])
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    raise ValueError(f"Invalid normalized box in {label}")
                positive += 1
        if not positive:
            raise ValueError(f"No annotated people in {split}")
        counts[split] = {"images": len(images), "boxes": positive}
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, help="YOLO training YAML; overrides training flags except --threads")
    parser.add_argument("--validate-only", action="store_true", help="Check the local weights path and dataset without training")
    parser.add_argument("--dataset", type=Path, default=Path("datasets/bus_people"))
    parser.add_argument("--weights", default="models/yolo26n.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--freeze", type=int, default=10)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--name", default="finetune")
    args = parser.parse_args()
    recipe = None
    if args.recipe:
        from ultralytics.utils import YAML
        recipe = YAML.load(str(args.recipe))
        if recipe.get("task", "detect") != "detect" or recipe.get("mode", "train") != "train" or recipe.get("resume"):
            parser.error("Recipe must start a separate detection training run")
        data_file = Path(recipe["data"]).resolve()
        data_config = YAML.load(str(data_file))
        args.dataset = Path(data_config.get("path", data_file.parent))
        if not args.dataset.is_absolute():
            args.dataset = data_file.parent / args.dataset
        if (data_config.get("train") != "images/train" or data_config.get("val") != "images/val"
                or data_config.get("names") not in ({0: "person"}, ["person"])):
            parser.error("Recipe requires a reviewed one-class images/train, images/val dataset")
        args.weights = recipe["model"]
        if recipe.get("exist_ok"):
            parser.error("Recipe must preserve earlier runs with exist_ok=false")
    if (min(args.epochs, args.image_size, args.batch, args.threads, args.patience) <= 0 or args.freeze < 0
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        parser.error("Epochs, image size, batch, threads, patience and learning rate must be positive")
    if not Path(args.weights).is_file() or Path(args.weights).suffix.lower() != ".pt":
        parser.error("Weights must be an existing local pretrained .pt file")
    dataset = args.dataset.resolve()
    if (dataset / "baseline_protection.json").is_file():
        from tools.prepare_countpeople import verify
        verify(dataset)
    print("Verified manual labels:", validate_labels(dataset))
    if args.validate_only:
        raise SystemExit(0)
    # Rebase an exported dataset after copying Windows files to a Linux server.
    runtime_yaml = dataset / "data.runtime.yaml"
    runtime_yaml.write_text(f"path: {dataset.as_posix()}\ntrain: images/train\nval: images/val\n"
                            "test: images/test\nnames:\n  0: person\n", encoding="utf-8")
    import torch
    from ultralytics import YOLO
    model = YOLO(args.weights)
    # Device setup adjusts PyTorch pools; apply this after trainer construction.
    def start_run(trainer):
        torch.set_num_threads(args.threads)
        snapshot = {"dataset": str(dataset), "files_sha256": {}}
        snapshot["initial_weights_sha256"] = hashlib.sha256(Path(args.weights).read_bytes()).hexdigest()
        snapshot["cpu_threads"] = args.threads
        if recipe:
            snapshot["recipe"] = recipe
            snapshot["recipe_sha256"] = hashlib.sha256(args.recipe.read_bytes()).hexdigest()
        for category in ("images", "labels"):
            for path in sorted((dataset / category).glob("*/*")):
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".txt"}:
                    snapshot["files_sha256"][path.relative_to(dataset).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        audit = dataset / "annotations.json"
        if audit.is_file():
            snapshot["annotations_sha256"] = hashlib.sha256(audit.read_bytes()).hexdigest()
        (trainer.save_dir / "dataset_snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    model.add_callback("on_pretrain_routine_start", start_run)
    def record_epoch(trainer):
        record = {"epoch": trainer.epoch + 1,
                  "phase": "train_epoch" if trainer.validator.training else "best_validation",
                  "metrics": {key: float(value) for key, value in trainer.metrics.items()}}
        with (trainer.save_dir / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
    model.add_callback("on_fit_epoch_end", record_epoch)
    if recipe is not None:
        model.train(**{**recipe, "data": str(runtime_yaml), "exist_ok": False})
    else:
        model.train(data=str(runtime_yaml), epochs=args.epochs, batch=args.batch,
                    imgsz=args.image_size, device=args.device, patience=args.patience,
                    freeze=args.freeze, optimizer="AdamW", lr0=args.learning_rate,
                    close_mosaic=5, mosaic=0.5, degrees=10.0, seed=42,
                    project=str(Path("runs/bus_detector").resolve()), name=args.name, workers=0, plots=False)
