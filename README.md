# hcontrol

`hcontrol` is a reusable Python library for real-time hand gesture detection from webcam streams.

## Features

- OpenCV camera ingestion with reconnect logic.
- MediaPipe hand landmark + gesture inference wrappers.
- Hand bounding boxes in normalized and pixel space.
- Gesture event state machine with debounce, hold, and cooldown.
- Pull and callback APIs.
- Optional desktop and MIDI adapters.

## Quickstart

```python
from hcontrol import GestureConfig, GestureEngine

config = GestureConfig(camera_index=0, max_hands=2, draw_annotations=True)
engine = GestureEngine(config)

engine.on_event(lambda event: print(event))
engine.start()

try:
    while engine.is_running:
        result = engine.read(timeout=0.2)
        if result is None:
            continue
        # Use result.frame_annotated in your UI loop
finally:
    engine.stop()
```

## Install

```bash
pip install hcontrol
```

Optional extras:

```bash
pip install "hcontrol[desktop]"
pip install "hcontrol[midi]"
pip install "hcontrol[dev]"
```
