"""Frame annotation rendering utilities."""

from __future__ import annotations

from typing import Any

from ..errors import DependencyNotAvailableError
from ..types import GestureObservation, HandObservation, LabelPosition

try:
    import cv2
except ImportError:  # pragma: no cover - optional during tests
    cv2 = None  # type: ignore[assignment]


class FrameRenderer:
    """Draw hand boxes, labels, and optional landmarks onto frames."""

    def __init__(self, *, draw_landmarks: bool, label_position: LabelPosition) -> None:
        self._draw_landmarks = draw_landmarks
        self._label_position = label_position

    def render(
        self,
        frame_bgr: Any,
        hands: list[HandObservation],
        gestures: list[GestureObservation],
    ) -> Any:
        if cv2 is None:
            raise DependencyNotAvailableError("opencv-python is required for frame rendering")

        annotated = frame_bgr.copy()
        gesture_by_hand = _best_gesture_by_hand(gestures)

        for hand in hands:
            x1, y1, x2, y2 = hand.bbox_px
            color = (0, 200, 0) if hand.handedness == "Left" else (220, 120, 0)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            gesture = gesture_by_hand.get(hand.hand_id)
            gesture_text = "-"
            if gesture is not None:
                gesture_text = f"{gesture.gesture} {gesture.confidence:.2f}"

            id_text = "?" if hand.hand_id is None else str(hand.hand_id)
            label = f"{hand.handedness}#{id_text} {gesture_text}"

            label_y = y1 - 8 if self._label_position == "top" else y2 + 18
            cv2.putText(
                annotated,
                label,
                (max(0, x1), max(14, label_y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

            if self._draw_landmarks:
                frame_h, frame_w = annotated.shape[0], annotated.shape[1]
                for lx, ly, _lz in hand.landmarks_norm:
                    px = int(lx * frame_w)
                    py = int(ly * frame_h)
                    cv2.circle(annotated, (px, py), 2, color, -1)

        return annotated


def _best_gesture_by_hand(
    gestures: list[GestureObservation],
) -> dict[int | None, GestureObservation]:
    best: dict[int | None, GestureObservation] = {}
    for gesture in gestures:
        current = best.get(gesture.hand_id)
        if current is None or gesture.confidence > current.confidence:
            best[gesture.hand_id] = gesture
    return best
