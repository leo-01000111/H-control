"""Run a live camera preview with hcontrol annotations."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from hcontrol import GestureConfig, GestureEngine
from hcontrol.errors import InferenceEngineError


def _default_model_path() -> str | None:
    candidate = Path("models/gesture_recognizer.task")
    if candidate.exists():
        return str(candidate)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="hcontrol camera preview")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON/YAML config")
    parser.add_argument(
        "--gesture-model-path",
        type=str,
        default=None,
        help="Path to gesture_recognizer.task (required on Python runtimes without mp.solutions)",
    )
    args = parser.parse_args()

    model_path = args.gesture_model_path or _default_model_path()

    config = GestureConfig.from_file(args.config) if args.config else GestureConfig()
    if model_path is not None:
        config = config.model_copy(update={"gesture_model_path": model_path})

    try:
        engine = GestureEngine(config=config)
        engine.start()
    except InferenceEngineError as exc:
        print(f"Failed to start inference: {exc}")
        print(
            "Tip: download model to models/gesture_recognizer.task and rerun, "
            "or pass --gesture-model-path <path>."
        )
        raise SystemExit(1) from exc

    try:
        while engine.is_running:
            result = engine.read(timeout=0.25)
            if result is None:
                continue

            frame = result.frame_annotated if result.frame_annotated is not None else result.frame_raw
            cv2.imshow("hcontrol preview", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        engine.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
