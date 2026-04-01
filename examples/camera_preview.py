"""Run a live camera preview with hcontrol annotations."""

from __future__ import annotations

import argparse

import cv2

from hcontrol import GestureConfig, GestureEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="hcontrol camera preview")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON/YAML config")
    args = parser.parse_args()

    engine = GestureEngine(config=GestureConfig(), config_file=args.config)
    engine.start()

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
