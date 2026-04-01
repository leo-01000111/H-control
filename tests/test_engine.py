from __future__ import annotations

import numpy as np

from hcontrol import GestureConfig, GestureEngine
from hcontrol.inference import InferenceOutput, RawGesturePrediction, RawHandObservation


class FakeCamera:
    def __init__(self, frames) -> None:  # type: ignore[no-untyped-def]
        self._frames = list(frames)
        self._started = False
        self._stopped = False

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._stopped = True

    def read(self):  # type: ignore[no-untyped-def]
        if not self._started or not self._frames:
            return None
        return self._frames.pop(0)


class FakeInference:
    def __init__(self) -> None:
        self.closed = False

    def process(self, frame_bgr, timestamp_ms: int) -> InferenceOutput:  # type: ignore[no-untyped-def]
        _ = frame_bgr
        _ = timestamp_ms
        hand = RawHandObservation(
            handedness="Right",
            handedness_score=0.9,
            landmarks_norm=[
                (0.1, 0.1, 0.0),
                (0.4, 0.2, 0.0),
                (0.35, 0.5, 0.0),
                (0.15, 0.45, 0.0),
            ],
            landmarks_world=[],
        )
        gesture = RawGesturePrediction(hand_index=0, gesture="Open_Palm", confidence=0.95)
        return InferenceOutput(hands=[hand], gestures=[gesture])

    def close(self) -> None:
        self.closed = True


def test_engine_sync_mode_with_stubs() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    camera = FakeCamera([frame])
    inference = FakeInference()

    config = GestureConfig(
        threaded=False,
        draw_annotations=False,
        debounce_ms=0,
        min_gesture_confidence=0.5,
    )

    engine = GestureEngine(
        config,
        camera_source=camera,  # type: ignore[arg-type]
        inference_engine=inference,  # type: ignore[arg-type]
    )

    events = []
    engine.on_event(events.append)

    engine.start()
    result = engine.read()
    engine.stop()

    assert result is not None
    assert len(result.hands) == 1
    assert result.hands[0].hand_id == 1
    assert len(result.gestures) == 1
    assert result.gestures[0].gesture == "Open_Palm"

    assert len(events) == 1
    assert events[0].type == "GESTURE_START"

    assert inference.closed is True
