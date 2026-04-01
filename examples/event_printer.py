"""Print gesture events from live camera input."""

from __future__ import annotations

import argparse
from pathlib import Path

from hcontrol import GestureConfig, GestureEngine
from hcontrol.errors import InferenceEngineError


def _default_model_path() -> str | None:
    candidate = Path("models/gesture_recognizer.task")
    if candidate.exists():
        return str(candidate)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="hcontrol event printer")
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
    config = config.model_copy(
        update={
            "draw_annotations": False,
            "threaded": True,
            "gesture_model_path": model_path,
        }
    )

    try:
        engine = GestureEngine(config)
        engine.start()
    except InferenceEngineError as exc:
        print(f"Failed to start inference: {exc}")
        print(
            "Tip: download model to models/gesture_recognizer.task and rerun, "
            "or pass --gesture-model-path <path>."
        )
        raise SystemExit(1) from exc

    engine.on_event(lambda event: print(event))

    try:
        while engine.is_running:
            # Drain frames to keep the loop alive while event callbacks print transitions.
            engine.read(timeout=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
