"""Download MediaPipe task models needed for local hcontrol demos."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

GESTURE_RECOGNIZER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MediaPipe models for hcontrol examples")
    parser.add_argument("--out-dir", type=str, default="models", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    destination = out_dir / "gesture_recognizer.task"

    print(f"Downloading Gesture Recognizer model to: {destination}")
    with urllib.request.urlopen(GESTURE_RECOGNIZER_URL, timeout=60) as response:
        destination.write_bytes(response.read())

    print("Done.")


if __name__ == "__main__":
    main()
