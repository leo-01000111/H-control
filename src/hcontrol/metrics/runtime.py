"""Runtime metrics collection for diagnostics and observability."""

from __future__ import annotations

import threading
from collections import deque


class RuntimeMetrics:
    """Tracks rolling runtime metrics exposed in `FrameResult.metrics`."""

    def __init__(self, window_size: int = 120) -> None:
        self._capture_timestamps_ms: deque[int] = deque(maxlen=window_size)
        self._inference_timestamps_ms: deque[int] = deque(maxlen=window_size)
        self._inference_durations_ms: deque[float] = deque(maxlen=window_size)

        self._dropped_frames = 0
        self._events_emitted = 0

        self._lock = threading.Lock()

    def record_input_frame(self, timestamp_ms: int) -> None:
        with self._lock:
            self._capture_timestamps_ms.append(timestamp_ms)

    def record_inference(self, duration_ms: float, timestamp_ms: int) -> None:
        with self._lock:
            self._inference_durations_ms.append(duration_ms)
            self._inference_timestamps_ms.append(timestamp_ms)

    def record_dropped_frame(self, count: int = 1) -> None:
        with self._lock:
            self._dropped_frames += count

    def record_events(self, count: int) -> None:
        with self._lock:
            self._events_emitted += count

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return {
                "input_fps": _fps_from_timestamps(self._capture_timestamps_ms),
                "inference_fps": _fps_from_timestamps(self._inference_timestamps_ms),
                "dropped_frames": float(self._dropped_frames),
                "average_inference_time_ms": _average(self._inference_durations_ms),
                "events_emitted": float(self._events_emitted),
            }


def _fps_from_timestamps(timestamps_ms: deque[int]) -> float:
    if len(timestamps_ms) < 2:
        return 0.0

    elapsed_ms = timestamps_ms[-1] - timestamps_ms[0]
    if elapsed_ms <= 0:
        return 0.0

    return (len(timestamps_ms) - 1) / (elapsed_ms / 1000.0)


def _average(values: deque[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
