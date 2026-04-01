"""Optional action adapter contracts."""

from __future__ import annotations

from typing import Protocol

from ..types import GestureEvent


class ActionAdapter(Protocol):
    """Protocol for optional output/action backends."""

    def on_start(self) -> None:
        ...

    def on_stop(self) -> None:
        ...

    def handle_event(self, event: GestureEvent) -> None:
        ...


class NoopAdapter:
    """No-op adapter useful as a safe default in host applications."""

    def on_start(self) -> None:
        return None

    def on_stop(self) -> None:
        return None

    def handle_event(self, event: GestureEvent) -> None:
        _ = event
        return None
