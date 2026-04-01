"""MIDI adapter (optional extra dependency)."""

from __future__ import annotations

from typing import Any

from ..errors import DependencyNotAvailableError
from ..types import GestureEvent
from .base import ActionAdapter

try:
    import mido
except ImportError:  # pragma: no cover - optional dependency
    mido = None  # type: ignore[assignment]


class MidiNoteAdapter(ActionAdapter):
    """Maps gesture lifecycle events to MIDI note on/off messages."""

    def __init__(
        self,
        *,
        gesture_to_note: dict[str, int],
        output_name: str | None = None,
        velocity: int = 100,
        channel: int = 0,
    ) -> None:
        self._gesture_to_note = gesture_to_note
        self._output_name = output_name
        self._velocity = velocity
        self._channel = channel

        self._output: Any | None = None
        self._active_keys: set[tuple[int | None, str]] = set()

    def on_start(self) -> None:
        if mido is None:
            raise DependencyNotAvailableError(
                "MIDI adapter requires `mido` + backend. Install with `pip install hcontrol[midi]`."
            )

        self._output = mido.open_output(self._output_name) if self._output_name else mido.open_output()

    def on_stop(self) -> None:
        if self._output is not None:
            self._output.close()
            self._output = None
        self._active_keys.clear()

    def handle_event(self, event: GestureEvent) -> None:
        if self._output is None:
            return

        note = self._gesture_to_note.get(event.gesture)
        if note is None:
            return

        key = (event.hand_id, event.gesture)
        if event.type == "GESTURE_START":
            self._output.send(
                mido.Message(
                    "note_on",
                    note=note,
                    velocity=self._velocity,
                    channel=self._channel,
                )
            )
            self._active_keys.add(key)
            return

        if event.type == "GESTURE_END" and key in self._active_keys:
            self._output.send(
                mido.Message(
                    "note_off",
                    note=note,
                    velocity=0,
                    channel=self._channel,
                )
            )
            self._active_keys.remove(key)
