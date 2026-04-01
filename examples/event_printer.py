"""Print gesture events from live camera input."""

from __future__ import annotations

from hcontrol import GestureConfig, GestureEngine


def main() -> None:
    engine = GestureEngine(
        GestureConfig(
            draw_annotations=False,
            threaded=True,
        )
    )

    engine.on_event(lambda event: print(event))
    engine.start()

    try:
        while engine.is_running:
            # Drain frames to keep the loop alive while event callbacks print transitions.
            engine.read(timeout=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
