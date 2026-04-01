"""Public typed data contracts for hcontrol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

EventType = Literal["GESTURE_START", "GESTURE_HOLD", "GESTURE_END"]
Handedness = Literal["Left", "Right"]
LabelPosition = Literal["top", "bottom"]

FrameArray = NDArray[np.uint8]
Landmark3D = tuple[float, float, float]
BBoxNorm = tuple[float, float, float, float]
BBoxPx = tuple[int, int, int, int]


@dataclass(slots=True)
class HandObservation:
    """Detected hand and landmark data for a single frame."""

    hand_id: int | None
    handedness: Handedness
    handedness_score: float
    bbox_px: BBoxPx
    bbox_norm: BBoxNorm
    landmarks_norm: list[Landmark3D]
    landmarks_world: list[Landmark3D] = field(default_factory=list)


@dataclass(slots=True)
class GestureObservation:
    """Gesture classification output tied to a hand in a specific frame."""

    hand_id: int | None
    gesture: str
    confidence: float
    timestamp_ms: int


@dataclass(slots=True)
class FrameResult:
    """One processed frame and all associated observations."""

    timestamp_ms: int
    frame_raw: FrameArray
    frame_annotated: FrameArray | None
    hands: list[HandObservation]
    gestures: list[GestureObservation]
    metrics: dict[str, float]


@dataclass(slots=True)
class GestureEvent:
    """State transition event generated from gesture observations."""

    type: EventType
    gesture: str
    hand_id: int | None
    confidence: float
    timestamp_ms: int
    duration_ms: int | None
