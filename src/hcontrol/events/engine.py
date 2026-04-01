"""Gesture event state machine with debounce, hold, and cooldown."""

from __future__ import annotations

from dataclasses import dataclass

from ..types import GestureEvent, GestureObservation

GestureKey = tuple[int | None, str]


@dataclass(slots=True)
class _CandidateState:
    first_seen_ms: int
    last_seen_ms: int
    confidence: float


@dataclass(slots=True)
class _ActiveState:
    start_ms: int
    last_hold_ms: int
    last_seen_ms: int
    confidence: float


class GestureEventEngine:
    """Turns per-frame gesture observations into stable stateful events."""

    def __init__(self, *, debounce_ms: int, hold_interval_ms: int, cooldown_ms: int) -> None:
        self._debounce_ms = debounce_ms
        self._hold_interval_ms = hold_interval_ms
        self._cooldown_ms = cooldown_ms

        self._candidate: dict[GestureKey, _CandidateState] = {}
        self._active: dict[GestureKey, _ActiveState] = {}
        self._cooldown_until: dict[GestureKey, int] = {}

    def reset(self) -> None:
        self._candidate.clear()
        self._active.clear()
        self._cooldown_until.clear()

    def process(self, gestures: list[GestureObservation], timestamp_ms: int) -> list[GestureEvent]:
        """Process frame-level gesture observations into start/hold/end events."""

        events: list[GestureEvent] = []
        observed = _dedupe_gestures(gestures)
        observed_keys = set(observed.keys())

        for key, active_state in list(self._active.items()):
            observed_gesture = observed.get(key)
            if observed_gesture is None:
                events.append(
                    GestureEvent(
                        type="GESTURE_END",
                        gesture=key[1],
                        hand_id=key[0],
                        confidence=active_state.confidence,
                        timestamp_ms=timestamp_ms,
                        duration_ms=timestamp_ms - active_state.start_ms,
                    )
                )
                self._cooldown_until[key] = timestamp_ms + self._cooldown_ms
                del self._active[key]
                continue

            active_state.last_seen_ms = timestamp_ms
            active_state.confidence = observed_gesture.confidence

            if (timestamp_ms - active_state.last_hold_ms) >= self._hold_interval_ms:
                events.append(
                    GestureEvent(
                        type="GESTURE_HOLD",
                        gesture=key[1],
                        hand_id=key[0],
                        confidence=observed_gesture.confidence,
                        timestamp_ms=timestamp_ms,
                        duration_ms=timestamp_ms - active_state.start_ms,
                    )
                )
                active_state.last_hold_ms = timestamp_ms

        for key in list(self._candidate.keys()):
            if key not in observed_keys:
                del self._candidate[key]

        for key, gesture in observed.items():
            if key in self._active:
                continue

            cooldown_until = self._cooldown_until.get(key)
            if cooldown_until is not None and timestamp_ms < cooldown_until:
                continue

            candidate = self._candidate.get(key)
            if candidate is None:
                self._candidate[key] = _CandidateState(
                    first_seen_ms=timestamp_ms,
                    last_seen_ms=timestamp_ms,
                    confidence=gesture.confidence,
                )
                if self._debounce_ms == 0:
                    events.append(
                        self._start_event(key, gesture.confidence, timestamp_ms)
                    )
                continue

            candidate.last_seen_ms = timestamp_ms
            candidate.confidence = gesture.confidence

            if (timestamp_ms - candidate.first_seen_ms) >= self._debounce_ms:
                events.append(self._start_event(key, gesture.confidence, timestamp_ms))

        return events

    def _start_event(self, key: GestureKey, confidence: float, timestamp_ms: int) -> GestureEvent:
        self._active[key] = _ActiveState(
            start_ms=timestamp_ms,
            last_hold_ms=timestamp_ms,
            last_seen_ms=timestamp_ms,
            confidence=confidence,
        )
        self._candidate.pop(key, None)

        return GestureEvent(
            type="GESTURE_START",
            gesture=key[1],
            hand_id=key[0],
            confidence=confidence,
            timestamp_ms=timestamp_ms,
            duration_ms=0,
        )


def _dedupe_gestures(gestures: list[GestureObservation]) -> dict[GestureKey, GestureObservation]:
    best_by_key: dict[GestureKey, GestureObservation] = {}
    for gesture in gestures:
        key = (gesture.hand_id, gesture.gesture)
        current = best_by_key.get(key)
        if current is None or gesture.confidence > current.confidence:
            best_by_key[key] = gesture
    return best_by_key
