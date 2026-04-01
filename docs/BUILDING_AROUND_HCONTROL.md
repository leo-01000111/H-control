# Building Around hcontrol

This project is ready to be used as a gesture runtime library inside your own app.

## Recommended App Pattern

1. Keep `GestureEngine` as a dedicated runtime component.
2. Register `on_event` callbacks that forward to your app's command bus.
3. Keep gesture-to-action mapping in your host app, not in `hcontrol`.
4. Use `GESTURE_START` for one-shot triggers and `GESTURE_HOLD` for repeated controls.
5. Use `read(timeout=...)` in your app loop to keep processing active.

## Runtime Setup

- Install and use the venv python executable.
- Provide a MediaPipe model file at `models/gesture_recognizer.task`.
- For your machine, camera index `0` is the default.

## Suggested Structure in Your App

```text
my_app/
  gesture_runtime.py      # wraps GestureEngine lifecycle
  gesture_mapping.py      # gesture -> command mapping
  command_handlers.py     # app actions
  ui_or_loop.py           # main event loop
```

## Quick Integration Snippet

```python
from hcontrol import GestureConfig, GestureEngine

config = GestureConfig(gesture_model_path="models/gesture_recognizer.task")

with GestureEngine(config) as engine:
    engine.on_event(your_event_handler)
    while engine.is_running:
        engine.read(timeout=0.2)
```

## Useful APIs

- `GestureEngine.on_event(callback)`
- `GestureEngine.on_frame(callback)`
- `GestureEngine.remove_event_callback(callback)`
- `GestureEngine.remove_frame_callback(callback)`
- `GestureEngine.clear_callbacks()`
- `GestureEngine.read(timeout=...)`
- `GestureEngine.get_latest_result()`

## Current Gesture Labels

- `Closed_Fist`
- `Open_Palm`
- `Pointing_Up`
- `Thumb_Down`
- `Thumb_Up`
- `Victory`
- `ILoveYou`

Note: only predictions above `min_gesture_confidence` are emitted.
