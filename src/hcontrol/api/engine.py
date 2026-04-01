"""Public GestureEngine runtime orchestrator."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

from ..adapters import ActionAdapter
from ..camera import CameraSource
from ..config import GestureConfig, load_config
from ..events import GestureEventEngine
from ..geometry import bbox_from_landmarks_norm
from ..inference import InferenceEngine, MediaPipeInference
from ..metrics import RuntimeMetrics
from ..recognition import build_hand_observations, filter_gesture_predictions
from ..render import FrameRenderer
from ..tracking import HandTracker
from ..types import FrameResult, GestureEvent

FrameCallback = Callable[[FrameResult], None]
EventCallback = Callable[[GestureEvent], None]


class GestureEngine:
    """Real-time camera to gesture-event pipeline."""

    def __init__(
        self,
        config: GestureConfig | Mapping[str, Any] | None = None,
        *,
        config_file: str | None = None,
        camera_source: CameraSource | None = None,
        inference_engine: InferenceEngine | None = None,
        renderer: FrameRenderer | None = None,
        event_engine: GestureEventEngine | None = None,
        tracker: HandTracker | None = None,
    ) -> None:
        self._config = load_config(config=config, config_file=config_file)

        self._logger = logging.getLogger("hcontrol.engine")
        self._logger.setLevel(getattr(logging, self._config.log_level.upper(), logging.INFO))

        self._camera = camera_source or CameraSource(
            camera_index=self._config.camera_index,
            frame_width=self._config.frame_width,
            frame_height=self._config.frame_height,
            frame_fps=self._config.frame_fps,
            reconnect_attempts=self._config.reconnect_attempts,
            reconnect_interval_ms=self._config.reconnect_interval_ms,
            logger=logging.getLogger("hcontrol.camera"),
        )
        self._inference: InferenceEngine = inference_engine or MediaPipeInference(
            self._config,
            logger=logging.getLogger("hcontrol.inference"),
        )
        self._renderer = renderer or FrameRenderer(
            draw_landmarks=self._config.draw_landmarks,
            label_position=self._config.label_position,
        )
        self._event_engine = event_engine or GestureEventEngine(
            debounce_ms=self._config.debounce_ms,
            hold_interval_ms=self._config.hold_interval_ms,
            cooldown_ms=self._config.cooldown_ms,
        )
        self._tracker = tracker or HandTracker(
            max_distance=self._config.tracker_max_distance,
            max_missing_ms=self._config.tracker_max_missing_ms,
        )
        self._metrics = RuntimeMetrics()

        self._frame_callbacks: list[FrameCallback] = []
        self._event_callbacks: list[EventCallback] = []
        self._adapters: list[ActionAdapter] = []

        self._result_queue: queue.Queue[FrameResult] = queue.Queue(maxsize=self._config.queue_size)
        self._latest_result: FrameResult | None = None
        self._latest_lock = threading.Lock()

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._running = False
        self._next_metrics_log_monotonic = 0.0

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def config(self) -> GestureConfig:
        return self._config

    def start(self) -> None:
        if self._running:
            return

        self._camera.start()
        self._event_engine.reset()
        self._tracker.reset()
        self._stop_event.clear()

        for adapter in self._adapters:
            self._safe_adapter_start(adapter)

        self._running = True
        self._next_metrics_log_monotonic = time.monotonic() + self._config.metrics_log_interval_sec

        if self._config.threaded:
            self._thread = threading.Thread(target=self._run_loop, name="hcontrol-engine", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        for adapter in self._adapters:
            self._safe_adapter_stop(adapter)

        self._camera.stop()
        self._inference.close()

    def read(self, timeout: float | None = None) -> FrameResult | None:
        """Read next result from queue (threaded) or process one frame (sync)."""

        if not self._running:
            return None

        if self._config.threaded:
            try:
                if timeout is None:
                    return self._result_queue.get(block=True)
                if timeout <= 0:
                    return self._result_queue.get_nowait()
                return self._result_queue.get(block=True, timeout=timeout)
            except queue.Empty:
                return None

        result = self._process_once()
        if result is not None:
            self._set_latest_result(result)
        return result

    def get_latest_result(self) -> FrameResult | None:
        with self._latest_lock:
            return self._latest_result

    def on_frame(self, callback: FrameCallback) -> None:
        self._frame_callbacks.append(callback)

    def on_event(self, callback: EventCallback) -> None:
        self._event_callbacks.append(callback)

    def register_adapter(self, adapter: ActionAdapter) -> None:
        self._adapters.append(adapter)

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            result = self._process_once()
            if result is not None:
                self._set_latest_result(result)
                self._enqueue_result(result)

            self._log_metrics_if_due()

            sleep_ms = self._config.worker_sleep_ms
            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000.0)

    def _process_once(self) -> FrameResult | None:
        frame = self._camera.read()
        if frame is None:
            self._metrics.record_dropped_frame()
            return None

        timestamp_ms = _now_ms()
        self._metrics.record_input_frame(timestamp_ms)

        inference_started = time.perf_counter()
        inference_output = self._inference.process(frame, timestamp_ms)
        inference_elapsed_ms = (time.perf_counter() - inference_started) * 1000.0
        self._metrics.record_inference(inference_elapsed_ms, timestamp_ms)

        frame_height = int(frame.shape[0])
        frame_width = int(frame.shape[1])

        bboxes_norm = [
            bbox_from_landmarks_norm(raw_hand.landmarks_norm)
            for raw_hand in inference_output.hands
        ]
        hand_ids = self._tracker.update(bboxes_norm, timestamp_ms)

        hands = build_hand_observations(
            raw_hands=inference_output.hands,
            hand_ids=hand_ids,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        gestures = filter_gesture_predictions(
            raw_gestures=inference_output.gestures,
            hand_ids=hand_ids,
            min_confidence=self._config.min_gesture_confidence,
            timestamp_ms=timestamp_ms,
        )

        events = self._event_engine.process(gestures, timestamp_ms)
        self._metrics.record_events(len(events))

        frame_annotated = None
        if self._config.draw_annotations:
            frame_annotated = self._renderer.render(frame, hands, gestures)

        result = FrameResult(
            timestamp_ms=timestamp_ms,
            frame_raw=frame,
            frame_annotated=frame_annotated,
            hands=hands,
            gestures=gestures,
            metrics=self._metrics.snapshot(),
        )

        self._dispatch_frame_callbacks(result)
        self._dispatch_events(events)

        return result

    def _dispatch_frame_callbacks(self, result: FrameResult) -> None:
        for callback in self._frame_callbacks:
            try:
                callback(result)
            except Exception:
                self._logger.exception("engine.frame_callback_error")

    def _dispatch_events(self, events: list[GestureEvent]) -> None:
        for event in events:
            for callback in self._event_callbacks:
                try:
                    callback(event)
                except Exception:
                    self._logger.exception("engine.event_callback_error")

            for adapter in self._adapters:
                try:
                    adapter.handle_event(event)
                except Exception:
                    self._logger.exception("engine.adapter_event_error")

    def _set_latest_result(self, result: FrameResult) -> None:
        with self._latest_lock:
            self._latest_result = result

    def _enqueue_result(self, result: FrameResult) -> None:
        if self._config.drop_frames_if_busy and self._result_queue.full():
            try:
                self._result_queue.get_nowait()
                self._metrics.record_dropped_frame()
            except queue.Empty:
                pass

        try:
            self._result_queue.put_nowait(result)
        except queue.Full:
            self._metrics.record_dropped_frame()

    def _log_metrics_if_due(self) -> None:
        now = time.monotonic()
        if now < self._next_metrics_log_monotonic:
            return

        self._next_metrics_log_monotonic = now + self._config.metrics_log_interval_sec
        self._logger.info("engine.metrics", extra={"metrics": self._metrics.snapshot()})

    def _safe_adapter_start(self, adapter: ActionAdapter) -> None:
        try:
            adapter.on_start()
        except Exception:
            self._logger.exception("engine.adapter_start_error")

    def _safe_adapter_stop(self, adapter: ActionAdapter) -> None:
        try:
            adapter.on_stop()
        except Exception:
            self._logger.exception("engine.adapter_stop_error")


def _now_ms() -> int:
    return int(time.time() * 1000)
