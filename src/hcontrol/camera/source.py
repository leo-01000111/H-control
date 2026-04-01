"""Camera ingestion module based on OpenCV VideoCapture."""

from __future__ import annotations

import logging
import time
from typing import Any

from numpy.typing import NDArray

from ..errors import CameraConnectionError, DependencyNotAvailableError

try:
    import cv2
except ImportError:  # pragma: no cover - optional during docs/tests without cv2
    cv2 = None  # type: ignore[assignment]

FrameArray = NDArray[Any]


class CameraSource:
    """OpenCV camera wrapper with lightweight reconnect handling."""

    def __init__(
        self,
        *,
        camera_index: int,
        frame_width: int,
        frame_height: int,
        frame_fps: int,
        reconnect_attempts: int,
        reconnect_interval_ms: int,
        logger: logging.Logger | None = None,
    ) -> None:
        self._camera_index = camera_index
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._frame_fps = frame_fps
        self._reconnect_attempts = reconnect_attempts
        self._reconnect_interval_s = reconnect_interval_ms / 1000.0
        self._logger = logger or logging.getLogger("hcontrol.camera")

        self._capture: Any | None = None
        self._failed_reads = 0
        self._running = False
        self._last_reconnect_monotonic = 0.0

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def resolution(self) -> tuple[int, int]:
        return self._frame_width, self._frame_height

    def start(self) -> None:
        if cv2 is None:
            raise DependencyNotAvailableError(
                "opencv-python is required for camera capture. Install opencv-python>=4.10."
            )
        if self._running:
            return
        self._open_capture()
        self._running = True

    def stop(self) -> None:
        self._running = False
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def read(self) -> FrameArray | None:
        """Read a frame; returns None on temporary failure."""

        if not self._running:
            return None

        if self._capture is None or not self._capture.isOpened():
            self._attempt_reconnect(force=True)
            if self._capture is None:
                return None

        ok, frame = self._capture.read()
        if ok and frame is not None:
            self._failed_reads = 0
            return frame

        self._failed_reads += 1
        self._logger.warning(
            "camera.read_failed",
            extra={"failed_reads": self._failed_reads},
        )

        if self._failed_reads >= self._reconnect_attempts:
            self._attempt_reconnect(force=False)

        return None

    def _attempt_reconnect(self, *, force: bool) -> None:
        now = time.monotonic()
        if not force and (now - self._last_reconnect_monotonic) < self._reconnect_interval_s:
            return

        self._last_reconnect_monotonic = now
        self._logger.info("camera.reconnect")

        if self._capture is not None:
            self._capture.release()
            self._capture = None

        try:
            self._open_capture()
            self._failed_reads = 0
        except CameraConnectionError:
            self._logger.exception("camera.reconnect_failed")

    def _open_capture(self) -> None:
        assert cv2 is not None

        capture = cv2.VideoCapture(self._camera_index)
        if not capture.isOpened():
            capture.release()
            raise CameraConnectionError(
                f"Could not open camera index {self._camera_index}."
            )

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._frame_width))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._frame_height))
        capture.set(cv2.CAP_PROP_FPS, float(self._frame_fps))

        self._capture = capture
