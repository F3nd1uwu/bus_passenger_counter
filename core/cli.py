import argparse
from dataclasses import replace
import logging

from config.settings import Settings
from config.profiles import PROFILE_NAMES, apply_profile


def cli(preview=False):
    parser = argparse.ArgumentParser(description="Bus passenger counter" + (" preview (no database writes)" if preview else ""))
    parser.add_argument("--source")
    parser.add_argument("--profile", choices=PROFILE_NAMES, help="Explicit camera-specific vision preset; CLI flags override it")
    parser.add_argument("--source-kind", choices=["auto", "file", "stream", "camera"])
    parser.add_argument("--video-start", help="ISO 8601 recording start, including timezone")
    parser.add_argument("--model", help="Full-person YOLO weights")
    parser.add_argument("--confidence", type=float, help="Person detection confidence threshold")
    parser.add_argument("--association-similarity", type=float, help="Minimum appearance similarity for tracker association")
    parser.add_argument("--drift-threshold", type=float, help="Merged-crop identity guard; 0 disables it")
    parser.add_argument("--device", help="cpu, cuda:0 or Ultralytics GPU index 0")
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--staff-mode", choices=["disabled", "model"])
    parser.add_argument("--child-mode", choices=["disabled", "model"], help="Approximate small-child filter; unknown passengers are counted")
    parser.add_argument("--child-model", help="Local child ensemble manifest")
    parser.add_argument("--vest-model", help="Trained garment model; does not filter by colour")
    parser.add_argument("--vest-threshold", type=float)
    parser.add_argument("--staff-decision-delay", type=float, help="Seconds of video to wait for garment evidence")
    parser.add_argument("--max-frames", type=int)
    if preview:
        parser.add_argument("--detection-only", action="store_true", help="Draw boxes without ReID, vest classification or counting")
        parser.add_argument("--output", help="Annotated MP4 file (works on a server without a display)")
        parser.add_argument("--show", action="store_true", help="GUI: timeline, Left/Right seek 5s, Space pause, Q/Esc exit")
        parser.add_argument("--start-frame", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        cfg = apply_profile(Settings.from_env(), args.profile)
        overrides = {field: value for field, value in (
            ("VIDEO_SOURCE", args.source), ("SOURCE_KIND", args.source_kind),
            ("VIDEO_START", args.video_start), ("MODEL_PATH", args.model), ("DEVICE", args.device),
            ("CONFIDENCE", args.confidence),
            ("CPU_THREADS", args.cpu_threads), ("STAFF_MODE", args.staff_mode),
            ("CHILD_MODE", args.child_mode), ("CHILD_MODEL_PATH", args.child_model),
            ("VEST_MODEL_PATH", args.vest_model), ("VEST_THRESHOLD", args.vest_threshold),
            ("STAFF_DECISION_DELAY_SEC", args.staff_decision_delay),
            ("TRACKER_IDENTITY_MIN_SIMILARITY", args.association_similarity),
            ("TRACKER_DRIFT_THRESHOLD", args.drift_threshold)) if value is not None}
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
