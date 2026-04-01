# Gesture Control Python Library - Product Requirements Document (PRD)

- Document version: `0.1`
- Date: `2026-04-01`
- Status: `Draft for implementation`
- Product codename: `hcontrol`

## 1. Product Summary

`hcontrol` is a Python library for real-time hand gesture detection and interpretation from a live camera stream.

The library focuses on reusable base capabilities:

- Camera ingestion.
- Multi-hand detection and tracking.
- Hand bounding box generation.
- Gesture recognition labels with confidence.
- Structured event output for downstream applications.
- Optional action adapters (desktop and MIDI) as extension modules.

This PRD is only for the base library, not end-user projects built on top of it.

## 2. Problem Statement

Projects such as desktop gesture shortcuts and camera-based musical control repeatedly rebuild the same hand-vision pipeline. Most implementations are demo-level and tightly coupled to one project.

We need a clean, testable, reusable library that:

- Works in real time.
- Exposes consistent typed outputs.
- Decouples detection from project-specific behavior.
- Is simple to embed in desktop Python applications.

## 3. Goals

1. Provide a stable API for real-time hand observations and gesture labels.
2. Return an annotated live frame with hand bounding boxes and gesture names.
3. Support up to 2 hands in real time on consumer hardware.
4. Offer a robust event system with debouncing and confidence thresholds.
5. Keep project-specific actions out of the core runtime.
6. Maintain strong test coverage on core logic and APIs.

## 4. Non-Goals

1. Building full applications (theremin app, shortcut manager UI, game control app).
2. Cloud inference or remote streaming as a first milestone.
3. Full body pose tracking.
4. Sign-language translation quality claims.
5. Deep model training as part of v1 core runtime.

## 5. Primary Use Cases

1. A host app subscribes to gesture events and maps them to commands.
2. A host app receives annotated video frames for debug or UI preview.
3. A host app tracks both hands and reads gesture labels + confidence.
4. A host app toggles between camera sources at runtime.
5. A host app uses raw landmarks for custom gesture logic.

## 6. Success Metrics

1. End-to-end latency: `<= 100 ms` median at `1280x720` on a modern laptop CPU.
2. Processing FPS: `>= 24 FPS` sustained for 1-2 hands at 720p.
3. False trigger reduction: debounce system lowers accidental repeats by at least `50%` in internal test scenarios.
4. API stability: no breaking changes across patch releases in `1.x`.
5. Test coverage: `>= 80%` for core non-visual modules.

## 7. Functional Requirements

### FR-001 Camera Input

- Library SHALL support webcam capture via OpenCV.
- Library SHALL allow camera index selection.
- Library SHALL expose frame resolution and FPS settings.
- Library SHALL handle camera disconnect errors gracefully.

### FR-002 Hand Detection and Tracking

- Library SHALL detect up to `N` hands (`N` default `2`, configurable).
- Library SHALL return handedness (`Left` or `Right`) with score.
- Library SHALL return 21 landmarks per detected hand.
- Library SHALL track hands over consecutive frames.

### FR-003 Bounding Boxes

- Library SHALL provide normalized and pixel-space bounding boxes per hand.
- Library SHALL include stable hand IDs when tracking is available.

### FR-004 Gesture Recognition

- Library SHALL return gesture category and confidence per hand.
- Library SHALL support built-in gesture classes from MediaPipe model.
- Library SHALL support configurable minimum confidence threshold.

### FR-005 Frame Annotation

- Library SHALL provide an annotated frame output with:
- Hand bounding boxes.
- Gesture label text.
- Confidence value.
- Handedness.
- Library SHALL let host apps disable annotation drawing for performance.

### FR-006 Event Engine

- Library SHALL emit typed gesture events containing timestamp, hand, gesture, and confidence.
- Library SHALL support debouncing and cooldown per gesture.
- Library SHALL support stateful event types:
- `GESTURE_START`
- `GESTURE_HOLD`
- `GESTURE_END`

### FR-007 Stream Interfaces

- Library SHALL support pull-based interface (`read` / `get_latest`).
- Library SHALL support callback-based interface (`on_event`, `on_frame`).
- Library SHALL support synchronous and threaded runtime modes.

### FR-008 Configuration

- Library SHALL accept configuration from:
- Python object config.
- JSON or YAML file.
- Environment variable overrides (optional in v1.1).
- Library SHALL validate config schema at startup.

### FR-009 Extensibility

- Library SHALL define plugin hooks for action backends.
- Core library SHALL ship without hardcoded project logic.
- Optional adapters SHALL be packaged as extras.

### FR-010 Diagnostics

- Library SHALL expose internal metrics:
- input FPS
- inference FPS
- dropped frames
- average inference time
- Library SHALL support configurable logging level.

## 8. Non-Functional Requirements

### NFR-001 Performance

- Median inference + post-process time under `40 ms` at 720p on target machine class.
- CPU-only operation required.
- GPU acceleration support can be optional and non-blocking.

### NFR-002 Reliability

- Runtime should recover from temporary camera read failures.
- Invalid frames should be skipped, not crash the process.

### NFR-003 Portability

- Required support:
- Windows 11
- Linux (Ubuntu 22.04+)
- macOS 13+ (Apple Silicon priority)

### NFR-004 Developer Experience

- Typed APIs.
- Clear docstrings.
- Ready-to-run minimal examples.
- Semantic versioning.

### NFR-005 Privacy and Safety

- No network calls in core inference loop.
- Frames are processed in memory only by default.
- Recording must be explicit opt-in.

## 9. Proposed Technical Stack

## 9.1 Runtime Language

- Python `3.11` and `3.12` target support.

## 9.2 Core Dependencies

- `mediapipe>=0.10.33,<0.11` for hand landmarks and gesture recognition.
- `opencv-python>=4.10,<5` for camera I/O and drawing.
- `numpy>=1.26,<3` for frame and geometry operations.
- `pydantic>=2.8,<3` for config and data validation.

## 9.3 Optional Extras

- Desktop action backend:
- `pynput>=1.8,<2`
- `PyAutoGUI>=0.9.54,<1` as optional alternate backend.
- MIDI backend:
- `mido>=1.3,<2`
- `python-rtmidi>=1.5,<2`

## 9.4 Dev Tooling

- `pytest`, `pytest-cov`, `pytest-benchmark`.
- `mypy` for static typing.
- `ruff` for lint + format.
- `hatchling` for packaging.
- GitHub Actions for CI matrix.

## 10. High-Level Architecture

```
CameraSource -> FramePreprocessor -> MediaPipeEngine -> PostProcessor
           -> EventEngine -> OutputRouter
                             |- FrameStream (raw/annotated)
                             |- EventStream (typed gesture events)
                             |- Optional Adapters (desktop, MIDI)
```

### Module Responsibilities

- `camera`: webcam capture, frame timing, reconnection handling.
- `inference`: MediaPipe task wrappers.
- `geometry`: bbox construction, normalization, utility transforms.
- `recognition`: gesture filtering, confidence thresholding.
- `events`: debouncing, state transitions, cooldown logic.
- `render`: annotation overlays on frames.
- `api`: public classes and entrypoints.
- `adapters`: optional action outputs.

## 11. Data Contracts

### 11.1 `HandObservation`

- `hand_id: int | None`
- `handedness: Literal["Left","Right"]`
- `handedness_score: float`
- `bbox_px: tuple[int, int, int, int]`
- `bbox_norm: tuple[float, float, float, float]`
- `landmarks_norm: list[tuple[float, float, float]]`
- `landmarks_world: list[tuple[float, float, float]]`

### 11.2 `GestureObservation`

- `hand_id: int | None`
- `gesture: str`
- `confidence: float`
- `timestamp_ms: int`

### 11.3 `FrameResult`

- `timestamp_ms: int`
- `frame_raw: np.ndarray`
- `frame_annotated: np.ndarray | None`
- `hands: list[HandObservation]`
- `gestures: list[GestureObservation]`
- `metrics: dict[str, float]`

### 11.4 `GestureEvent`

- `type: Literal["GESTURE_START","GESTURE_HOLD","GESTURE_END"]`
- `gesture: str`
- `hand_id: int | None`
- `confidence: float`
- `timestamp_ms: int`
- `duration_ms: int | None`

## 12. Public API Proposal

```python
from hcontrol import GestureEngine, GestureConfig

config = GestureConfig(
    camera_index=0,
    max_hands=2,
    draw_annotations=True,
    min_gesture_confidence=0.6,
)

engine = GestureEngine(config)

engine.on_event(lambda event: print(event))
engine.start()

while engine.is_running:
    result = engine.get_latest_result()
    if result and result.frame_annotated is not None:
        # host app renders result.frame_annotated
        pass

engine.stop()
```

### Required API Surface

- `GestureEngine.start()`
- `GestureEngine.stop()`
- `GestureEngine.get_latest_result() -> FrameResult | None`
- `GestureEngine.read(timeout: float | None = None) -> FrameResult | None`
- `GestureEngine.on_event(callback)`
- `GestureEngine.on_frame(callback)`
- `GestureEngine.is_running`

## 13. Configuration Schema (v1)

### Core

- `camera_index: int = 0`
- `frame_width: int = 1280`
- `frame_height: int = 720`
- `max_hands: int = 2`
- `min_hand_detection_confidence: float = 0.5`
- `min_hand_presence_confidence: float = 0.5`
- `min_tracking_confidence: float = 0.5`
- `min_gesture_confidence: float = 0.6`

### Rendering

- `draw_annotations: bool = True`
- `draw_landmarks: bool = False`
- `label_position: Literal["top","bottom"] = "top"`

### Event Tuning

- `debounce_ms: int = 120`
- `hold_interval_ms: int = 250`
- `cooldown_ms: int = 300`

### Runtime

- `threaded: bool = True`
- `queue_size: int = 2`
- `drop_frames_if_busy: bool = True`

## 14. Packaging and Project Layout

```
hcontrol/
  pyproject.toml
  src/hcontrol/
    __init__.py
    api/
    camera/
    inference/
    geometry/
    recognition/
    events/
    render/
    adapters/
    config.py
    types.py
  tests/
  examples/
  docs/
```

### Distribution Strategy

- Main package: `hcontrol`.
- Extras:
- `hcontrol[desktop]`
- `hcontrol[midi]`
- `hcontrol[dev]`

## 15. Error Handling Strategy

- Raise typed startup exceptions for invalid config or missing model.
- Use non-fatal warnings for dropped frames.
- Emit runtime health status via metrics.
- Avoid silent failures in callback exceptions:
- catch and log
- keep engine loop alive

## 16. Observability and Logging

- Structured logs with component tag (`camera`, `inference`, `events`).
- Periodic metrics snapshot every `N` seconds.
- Debug mode can include per-frame timings.

## 17. Testing Strategy

### Unit Tests

- Config validation.
- Geometry and bbox conversion.
- Event state machine.
- Debounce/cooldown behavior.

### Integration Tests

- End-to-end pipeline using prerecorded frames.
- Gesture output contract shape and types.
- Camera failure and reconnect behavior.

### Performance Tests

- Benchmark average frame time for synthetic and recorded sessions.
- Track FPS regression budget in CI (non-blocking at first, blocking in v1.1).

### Manual Validation

- Webcam smoke test on each supported OS.
- Visual validation of annotations for 1 and 2 hands.

## 18. Risks and Mitigations

1. Risk: lighting and background variability causes unstable detection.  
Mitigation: confidence thresholds, smoothing, and calibration helpers.

2. Risk: action adapters can create unsafe repetitive triggers.  
Mitigation: strict cooldown and explicit gesture-to-action opt-in.

3. Risk: platform-specific camera driver issues.  
Mitigation: pluggable camera backend abstraction and fallback settings.

4. Risk: inference throughput drops on low-end hardware.  
Mitigation: frame resize options, frame skipping, annotation toggle.

## 19. Milestones and Implementation TODO Plan

## Milestone 0 - Foundation (Week 1)

- [ ] Initialize repository structure and packaging (`src` layout).
- [ ] Add lint/type/test tooling (`ruff`, `mypy`, `pytest`).
- [ ] Define typed models (`HandObservation`, `GestureEvent`, `FrameResult`).
- [ ] Create base configuration system with validation.

Exit criteria:

- Package installs locally.
- CI runs lint + unit tests.

## Milestone 1 - Vision Core (Week 1-2)

- [ ] Implement OpenCV camera source module.
- [ ] Implement MediaPipe hand + gesture inference wrapper.
- [ ] Convert landmarks to bounding boxes and normalized geometry.
- [ ] Produce raw `FrameResult` objects.

Exit criteria:

- Live webcam processing works.
- Results include hands, gestures, confidence, timestamps.

## Milestone 2 - Rendering and Events (Week 2)

- [ ] Implement annotation renderer for boxes + labels.
- [ ] Implement event engine with debounce, hold, cooldown.
- [ ] Add callback registration and queue-backed read API.

Exit criteria:

- Annotated frame feed available.
- Gesture events emitted predictably under repeated motion.

## Milestone 3 - Stability and Extras (Week 3)

- [ ] Add optional desktop adapter module.
- [ ] Add optional MIDI adapter module.
- [ ] Add runtime metrics and structured logs.
- [ ] Add robust error handling and reconnect flow.

Exit criteria:

- Core loop survives transient camera failures.
- Optional extras install and run via extras dependencies.

## Milestone 4 - Quality and Release Candidate (Week 3-4)

- [ ] Expand unit + integration + benchmark tests.
- [ ] Write API reference and quickstart docs.
- [ ] Add example scripts (camera preview, event printer).
- [ ] Prepare v0.1.0 release notes.

Exit criteria:

- Coverage target met.
- Public API documented and frozen for v0.1.x.

## 20. Definition of Done for v0.1.0

1. `GestureEngine` API is implemented and documented.
2. Live annotated feed works with bounding boxes + gesture labels.
3. Event system emits stable start/hold/end transitions.
4. Core tests pass on CI matrix.
5. Package can be installed with extras and imported cleanly.

## 21. Future Extensions (Post v0.1)

1. Custom gesture training and model swap workflow.
2. Multi-camera support.
3. Headless mode for server-side event relay.
4. Optional WebSocket output stream.
5. Temporal gesture classification improvements.

## 22. Open Decisions

1. Whether to include a tiny CLI in v0.1 (`hcontrol preview`) or defer to v0.2.
2. Which default action backend to document first (`pynput` vs `PyAutoGUI`).
3. Whether to enforce benchmark thresholds as blocking in initial CI.

