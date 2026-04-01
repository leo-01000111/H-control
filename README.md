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

config = GestureConfig(
    camera_index=0,
    max_hands=2,
    draw_annotations=True,
    gesture_model_path="models/gesture_recognizer.task",
) 
with GestureEngine(config) as engine:
    engine.on_event(lambda event: print(event))
    while engine.is_running:
        result = engine.read(timeout=0.2)
        if result is None:
            continue
        # Use result.frame_annotated in your UI loop
```

## Install

From local clone (recommended for now):

```bash
pip install -e .
```

From GitHub:

```bash
pip install "git+https://github.com/leo-01000111/H-control.git@main"
```

Optional extras (local clone):

```bash
pip install -e ".[desktop]"
pip install -e ".[midi]"
pip install -e ".[dev]"
```

## Local Demo Setup

On Python runtimes where MediaPipe does not expose `mp.solutions` (for example many Python 3.12 builds),
provide a Gesture Recognizer task model file:

```bash
python examples/download_mediapipe_models.py
python examples/camera_preview.py --gesture-model-path models/gesture_recognizer.task
python examples/event_printer.py --gesture-model-path models/gesture_recognizer.task
```

## Build Your App Around It

- Start from `examples/host_app_template.py`
- Read integration notes in `docs/BUILDING_AROUND_HCONTROL.md`
