"""Inference interfaces and shared internal contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..types import Handedness, Landmark3D


@dataclass(slots=True)
class RawHandObservation:
    """Raw hand result before bbox conversion and tracking assignment."""

    handedness: Handedness
    handedness_score: float
    landmarks_norm: list[Landmark3D]
    landmarks_world: list[Landmark3D] = field(default_factory=list)


@dataclass(slots=True)
class RawGesturePrediction:
    """Raw gesture category mapped to an inferred hand index."""

    hand_index: int
    gesture: str
    confidence: float


@dataclass(slots=True)
class InferenceOutput:
    """Inference output for one frame."""

    hands: list[RawHandObservation]
    gestures: list[RawGesturePrediction]


class InferenceEngine(Protocol):
    """Protocol for inference implementations."""

    def process(self, frame_bgr: object, timestamp_ms: int) -> InferenceOutput:
        ...

    def close(self) -> None:
        ...
