"""Post-processing helpers for inference outputs."""

from __future__ import annotations

from ..geometry import bbox_from_landmarks_norm, bbox_norm_to_pixels
from ..inference import RawGesturePrediction, RawHandObservation
from ..types import GestureObservation, HandObservation


def build_hand_observations(
    *,
    raw_hands: list[RawHandObservation],
    hand_ids: list[int],
    frame_width: int,
    frame_height: int,
) -> list[HandObservation]:
    """Convert raw hand detections into typed hand observations."""

    observations: list[HandObservation] = []
    for idx, raw_hand in enumerate(raw_hands):
        bbox_norm = bbox_from_landmarks_norm(raw_hand.landmarks_norm)
        bbox_px = bbox_norm_to_pixels(bbox_norm, frame_width, frame_height)
        hand_id = hand_ids[idx] if idx < len(hand_ids) else None

        observations.append(
            HandObservation(
                hand_id=hand_id,
                handedness=raw_hand.handedness,
                handedness_score=raw_hand.handedness_score,
                bbox_px=bbox_px,
                bbox_norm=bbox_norm,
                landmarks_norm=raw_hand.landmarks_norm,
                landmarks_world=raw_hand.landmarks_world,
            )
        )

    return observations


def filter_gesture_predictions(
    *,
    raw_gestures: list[RawGesturePrediction],
    hand_ids: list[int],
    min_confidence: float,
    timestamp_ms: int,
) -> list[GestureObservation]:
    """Filter gestures by confidence and map hand index to tracked hand id."""

    accepted: list[GestureObservation] = []

    for prediction in raw_gestures:
        if prediction.confidence < min_confidence:
            continue
        if not prediction.gesture:
            continue

        hand_id = hand_ids[prediction.hand_index] if prediction.hand_index < len(hand_ids) else None

        accepted.append(
            GestureObservation(
                hand_id=hand_id,
                gesture=prediction.gesture,
                confidence=prediction.confidence,
                timestamp_ms=timestamp_ms,
            )
        )

    return accepted
