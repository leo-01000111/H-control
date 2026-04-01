"""Desktop key adapter (optional extra dependency)."""

from __future__ import annotations

from typing import Any

from ..errors import DependencyNotAvailableError
from ..types import EventType, GestureEvent
from .base import ActionAdapter

try:
    from pynput.keyboard import Controller, Key
except ImportError:  # pragma: no cover - optional dependency
    Controller = None  # type: ignore[assignment]
    Key = None  # type: ignore[assignment]


class DesktopKeyAdapter(ActionAdapter):
    """Maps gesture events to keyboard key taps via pynput."""

    def __init__(
        self,
        *,
        gesture_to_key: dict[str, str],
        trigger_event_types: set[EventType] | None = None,
    ) -> None:
        self._gesture_to_key = gesture_to_key
        self._trigger_event_types = trigger_event_types or {"GESTURE_START"}
        self._controller: Any | None = None

    def on_start(self) -> None:
        if Controller is None:
            raise DependencyNotAvailableError(
                "Desktop adapter requires `pynput`. Install with `pip install hcontrol[desktop]`."
            )
        self._controller = Controller()

    def on_stop(self) -> None:
        self._controller = None

    def handle_event(self, event: GestureEvent) -> None:
        if self._controller is None:
            return
        if event.type not in self._trigger_event_types:
            return

        key_name = self._gesture_to_key.get(event.gesture)
        if key_name is None:
            return

        key_obj = _resolve_key(key_name)
        self._controller.press(key_obj)
        self._controller.release(key_obj)


def _resolve_key(key_name: str) -> Any:
    if len(key_name) == 1 or Key is None:
        return key_name

    special = getattr(Key, key_name, None)
    if special is None:
        return key_name
    return special
