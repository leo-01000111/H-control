"""MediaPipe-based inference implementations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..config import GestureConfig
from ..errors import DependencyNotAvailableError, InferenceEngineError
from ..types import Handedness, Landmark3D
from .base import InferenceOutput, RawGesturePrediction, RawHandObservation

try:
    import cv2
except ImportError:  # pragma: no cover - optional during tests
    cv2 = None  # type: ignore[assignment]

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - optional during tests
    mp = None  # type: ignore[assignment]


class MediaPipeInference:
    """Inference wrapper using MediaPipe GestureRecognizer or Hands solution."""

    def __init__(self, config: GestureConfig, logger: logging.Logger | None = None) -> None:
        if cv2 is None:
            raise DependencyNotAvailableError("opencv-python is required for inference")
        if mp is None:
            raise DependencyNotAvailableError("mediapipe is required for inference")

        self._config = config
        self._logger = logger or logging.getLogger("hcontrol.inference")

        self._gesture_recognizer: Any | None = None
        self._hands_solution: Any | None = None

        if config.gesture_model_path:
            self._gesture_recognizer = self._create_gesture_recognizer(config)
            self._logger.info(
                "inference.mode",
                extra={"mode": "gesture_recognizer", "model": config.gesture_model_path},
            )
        elif _has_legacy_hands_solution():
            # Fallback keeps hand tracking available even when no gesture model path is provided.
            self._hands_solution = self._create_hands_solution(config)
            self._logger.info("inference.mode", extra={"mode": "hands_only"})
        else:
            raise InferenceEngineError(
                "MediaPipe legacy `mp.solutions.hands` is unavailable in this runtime. "
                "Set `gesture_model_path` to a valid Gesture Recognizer `.task` model file "
                "(for example: models/gesture_recognizer.task)."
            )

    def process(self, frame_bgr: object, timestamp_ms: int) -> InferenceOutput:
        if cv2 is None or mp is None:  # pragma: no cover - guarded by __init__
            raise DependencyNotAvailableError("opencv-python and mediapipe are required")

        try:
            rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        except Exception as exc:
            raise InferenceEngineError(f"Invalid input frame for inference: {exc}") from exc

        try:
            if self._gesture_recognizer is not None:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = self._gesture_recognizer.recognize(mp_image)
                return self._parse_task_result(result)

            if self._hands_solution is not None:
                result = self._hands_solution.process(rgb_frame)
                return self._parse_hands_result(result)

            raise InferenceEngineError("Inference engine is not initialized")
        except InferenceEngineError:
            raise
        except Exception as exc:
            raise InferenceEngineError(f"MediaPipe processing failed: {exc}") from exc

    def close(self) -> None:
        if self._gesture_recognizer is not None:
            close_method = getattr(self._gesture_recognizer, "close", None)
            if callable(close_method):
                close_method()
            self._gesture_recognizer = None

        if self._hands_solution is not None:
            self._hands_solution.close()
            self._hands_solution = None

    def _create_gesture_recognizer(self, config: GestureConfig) -> Any:
        assert mp is not None

        model_path = Path(config.gesture_model_path or "")
        if not model_path.exists():
            raise InferenceEngineError(
                "Gesture model file does not exist. Set `gesture_model_path` to a valid .task file."
            )

        try:
            base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
            options = mp.tasks.vision.GestureRecognizerOptions(
                base_options=base_options,
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_hands=config.max_hands,
                min_hand_detection_confidence=config.min_hand_detection_confidence,
                min_hand_presence_confidence=config.min_hand_presence_confidence,
                min_tracking_confidence=config.min_tracking_confidence,
            )
            return mp.tasks.vision.GestureRecognizer.create_from_options(options)
        except Exception as exc:
            raise InferenceEngineError(f"Could not initialize MediaPipe GestureRecognizer: {exc}") from exc

    def _create_hands_solution(self, config: GestureConfig) -> Any:
        assert mp is not None
        try:
            return mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=config.max_hands,
                min_detection_confidence=config.min_hand_detection_confidence,
                min_tracking_confidence=config.min_tracking_confidence,
            )
        except Exception as exc:
            raise InferenceEngineError(f"Could not initialize MediaPipe Hands solution: {exc}") from exc

    def _parse_task_result(self, result: Any) -> InferenceOutput:
        hands: list[RawHandObservation] = []
        gestures: list[RawGesturePrediction] = []

        hand_landmarks = result.hand_landmarks if result.hand_landmarks else []
        hand_world_landmarks = result.hand_world_landmarks if result.hand_world_landmarks else []
        handedness = result.handedness if result.handedness else []
        gesture_lists = result.gestures if result.gestures else []

        for idx, landmarks in enumerate(hand_landmarks):
            hand_label = "Right"
            hand_score = 0.0
            if idx < len(handedness) and handedness[idx]:
                category = handedness[idx][0]
                label = getattr(category, "category_name", None) or getattr(category, "display_name", None)
                if label in {"Left", "Right"}:
                    hand_label = label
                hand_score = float(getattr(category, "score", 0.0))

            world_landmarks = hand_world_landmarks[idx] if idx < len(hand_world_landmarks) else []
            hands.append(
                RawHandObservation(
                    handedness=hand_label,  # type: ignore[arg-type]
                    handedness_score=hand_score,
                    landmarks_norm=_to_landmark_tuples(landmarks),
                    landmarks_world=_to_landmark_tuples(world_landmarks),
                )
            )

            if idx < len(gesture_lists) and gesture_lists[idx]:
                category = gesture_lists[idx][0]
                gesture_name = getattr(category, "category_name", "UNKNOWN") or "UNKNOWN"
                confidence = float(getattr(category, "score", 0.0))
                gestures.append(
                    RawGesturePrediction(
                        hand_index=idx,
                        gesture=gesture_name,
                        confidence=confidence,
                    )
                )

        return InferenceOutput(hands=hands, gestures=gestures)

    def _parse_hands_result(self, result: Any) -> InferenceOutput:
        hands: list[RawHandObservation] = []

        landmark_list = result.multi_hand_landmarks if result.multi_hand_landmarks else []
        world_list = result.multi_hand_world_landmarks if result.multi_hand_world_landmarks else []
        handedness_list = result.multi_handedness if result.multi_handedness else []

        for idx, landmarks in enumerate(landmark_list):
            hand_label: Handedness = "Right"
            hand_score = 0.0

            if idx < len(handedness_list) and handedness_list[idx].classification:
                classification = handedness_list[idx].classification[0]
                if classification.label in {"Left", "Right"}:
                    hand_label = classification.label
                hand_score = float(classification.score)

            world_landmarks = world_list[idx] if idx < len(world_list) else []
            hands.append(
                RawHandObservation(
                    handedness=hand_label,
                    handedness_score=hand_score,
                    landmarks_norm=_to_landmark_tuples(landmarks.landmark),
                    landmarks_world=_to_landmark_tuples(getattr(world_landmarks, "landmark", [])),
                )
            )

        # Hands-only mode does not produce gesture classes.
        return InferenceOutput(hands=hands, gestures=[])


def _to_landmark_tuples(landmarks: Any) -> list[Landmark3D]:
    return [
        (float(landmark.x), float(landmark.y), float(landmark.z))
        for landmark in landmarks
    ]


def _has_legacy_hands_solution() -> bool:
    if mp is None:
        return False
    solutions = getattr(mp, "solutions", None)
    if solutions is None:
        return False
    return getattr(solutions, "hands", None) is not None
