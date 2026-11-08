import argparse
from dataclasses import replace
import logging

from config.settings import Settings


def cli(preview=False):
    parser = argparse.ArgumentParser(description="Bus passenger counter" + (" preview (no database writes)" if preview else ""))
    parser.add_argument("--source")
    parser.add_argument("--source-kind", choices=["auto", "file", "stream", "camera"])
    parser.add_argument("--video-start", help="ISO 8601 recording start, including timezone")
    parser.add_argument("--model", help="Full-person YOLO weights")
    parser.add_argument("--confidence", type=float, help="Person detection confidence threshold")
    parser.add_argument("--association-similarity", type=float, help="Minimum appearance similarity for tracker association")
    parser.add_argument("--device", help="cpu, cuda:0 or Ultralytics GPU index 0")
    parser.add_argument("--max-frames", type=int)
    if preview:
        parser.add_argument("--detection-only", action="store_true", help="Draw boxes without ReID, vest classification or counting")
        parser.add_argument("--output", help="Annotated MP4 file (works on a server without a display)")
        parser.add_argument("--show", action="store_true", help="GUI: timeline, Left/Right seek 5s, Space pause, Q/Esc exit")
        parser.add_argument("--start-frame", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        cfg = Settings.from_env()
        overrides = {field: value for field, value in (
            ("VIDEO_SOURCE", args.source), ("SOURCE_KIND", args.source_kind),
            ("VIDEO_START", args.video_start), ("MODEL_PATH", args.model), ("DEVICE", args.device),
            ("CONFIDENCE", args.confidence),
            ("TRACKER_IDENTITY_MIN_SIMILARITY", args.association_similarity)) if value is not None}
        cfg = replace(cfg, **overrides)
        if preview and not (args.output or args.show):
            parser.error("Preview requires --output or --show")
        from core.service import run
        result = run(cfg, preview=preview, max_frames=args.max_frames,
                     **({"detection_only": args.detection_only, "output": args.output,
                         "show": args.show, "start_frame": args.start_frame} if preview else {}))
        logging.getLogger(__name__).info("Finished: %s", result)
    except (ValueError, RuntimeError) as exc:
        parser.exit(2, f"Configuration/runtime error: {exc}\n")
