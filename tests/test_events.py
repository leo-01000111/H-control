from __future__ import annotations

from hcontrol.events import GestureEventEngine
from hcontrol.types import GestureObservation


def _obs(ts: int, conf: float = 0.9) -> GestureObservation:
    return GestureObservation(hand_id=1, gesture="Open_Palm", confidence=conf, timestamp_ms=ts)


def test_event_engine_start_hold_end_cooldown() -> None:
    engine = GestureEventEngine(debounce_ms=100, hold_interval_ms=200, cooldown_ms=300)

    assert engine.process([_obs(0)], 0) == []

    events = engine.process([_obs(100)], 100)
    assert [event.type for event in events] == ["GESTURE_START"]

    assert engine.process([_obs(250)], 250) == []

    events = engine.process([_obs(310)], 310)
    assert [event.type for event in events] == ["GESTURE_HOLD"]

    events = engine.process([], 360)
    assert [event.type for event in events] == ["GESTURE_END"]
    assert events[0].duration_ms == 260

    assert engine.process([_obs(500)], 500) == []

    assert engine.process([_obs(660)], 660) == []
    events = engine.process([_obs(760)], 760)
    assert [event.type for event in events] == ["GESTURE_START"]
