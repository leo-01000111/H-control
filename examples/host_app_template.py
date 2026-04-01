"""Minimal host app template for building custom gesture controls."""

from __future__ import annotations

from hcontrol import GestureConfig, GestureEngine
from hcontrol.types import GestureEvent


def handle_event(event: GestureEvent) -> None:
    # Map gestures to your own app logic here.
    if event.type == "GESTURE_START" and event.gesture == "Thumb_Up":
        print("[ACTION] Thumb_Up detected -> trigger custom action")
    else:
        print(event)


def main() -> None:
    config = GestureConfig(
        camera_index=0,
        draw_annotations=False,
        min_gesture_confidence=0.6,
        gesture_model_path="models/gesture_recognizer.task",
    )

    with GestureEngine(config) as engine:
        engine.on_event(handle_event)

        while engine.is_running:
            # Keep the pipeline active while callbacks handle events.
            engine.read(timeout=0.25)


if __name__ == "__main__":
    main()
