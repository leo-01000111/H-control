"""Configuration models and loading helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .errors import ConfigError
from .types import LabelPosition


class GestureConfig(BaseModel):
    """Validated runtime configuration for ``GestureEngine``."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # Camera
    camera_index: int = Field(default=0, ge=0)
    frame_width: int = Field(default=1280, ge=1)
    frame_height: int = Field(default=720, ge=1)
    frame_fps: int = Field(default=30, ge=1)

    # Inference
    max_hands: int = Field(default=2, ge=1)
    min_hand_detection_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    min_hand_presence_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    min_tracking_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    min_gesture_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    gesture_model_path: str | None = None

    # Rendering
    draw_annotations: bool = True
    draw_landmarks: bool = False
    label_position: LabelPosition = "top"

    # Event tuning
    debounce_ms: int = Field(default=120, ge=0)
    hold_interval_ms: int = Field(default=250, ge=1)
    cooldown_ms: int = Field(default=300, ge=0)

    # Runtime
    threaded: bool = True
    queue_size: int = Field(default=2, ge=1)
    drop_frames_if_busy: bool = True
    worker_sleep_ms: int = Field(default=5, ge=0)

    # Camera reliability
    reconnect_attempts: int = Field(default=3, ge=0)
    reconnect_interval_ms: int = Field(default=500, ge=0)

    # Tracking
    tracker_max_distance: float = Field(default=0.2, ge=0.0, le=1.0)
    tracker_max_missing_ms: int = Field(default=500, ge=0)

    # Logging / metrics
    log_level: str = "INFO"
    metrics_log_interval_sec: int = Field(default=5, ge=1)

    @classmethod
    def from_file(cls, file_path: str | Path) -> "GestureConfig":
        """Load config from JSON or YAML file."""

        path = Path(file_path)
        if not path.exists():
            raise ConfigError(f"Config file does not exist: {path}")

        suffix = path.suffix.lower()
        raw_text = path.read_text(encoding="utf-8")

        try:
            if suffix in {".json"}:
                payload = json.loads(raw_text)
            elif suffix in {".yaml", ".yml"}:
                try:
                    import yaml
                except ImportError as exc:  # pragma: no cover - dependency check path
                    raise ConfigError(
                        "YAML config requested but PyYAML is not installed. "
                        "Install with `pip install pyyaml`."
                    ) from exc
                payload = yaml.safe_load(raw_text)
            else:
                raise ConfigError(
                    f"Unsupported config extension '{suffix}'. Use .json, .yaml or .yml"
                )
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ConfigError(f"Invalid config file '{path}': {exc}") from exc

        if not isinstance(payload, Mapping):
            raise ConfigError(f"Config file '{path}' must contain an object/map at top level")

        return cls.model_validate(dict(payload))

    def with_env_overrides(self, env_prefix: str = "HCONTROL_") -> "GestureConfig":
        """Return a new config with env var overrides applied."""

        overrides: dict[str, Any] = {}

        for field_name, field in self.__class__.model_fields.items():
            env_key = f"{env_prefix}{field_name}".upper()
            if env_key not in os.environ:
                continue
            overrides[field_name] = _parse_env_value(os.environ[env_key], field.annotation)

        if not overrides:
            return self

        return self.model_copy(update=overrides)


def load_config(
    *,
    config: GestureConfig | Mapping[str, Any] | None = None,
    config_file: str | Path | None = None,
    env_prefix: str | None = "HCONTROL_",
) -> GestureConfig:
    """Load and merge configuration from object/dict/file/env sources."""

    if config_file is not None:
        merged = GestureConfig.from_file(config_file)
    elif isinstance(config, GestureConfig):
        merged = config
    elif isinstance(config, Mapping):
        merged = GestureConfig.model_validate(dict(config))
    elif config is None:
        merged = GestureConfig()
    else:
        raise ConfigError(
            "config must be GestureConfig, mapping, or None"
        )

    if env_prefix:
        merged = merged.with_env_overrides(env_prefix=env_prefix)

    return merged


def _parse_env_value(value: str, annotation: Any) -> Any:
    """Best-effort parser for environment value overrides."""

    normalized = value.strip()

    if annotation is bool:
        return normalized.lower() in {"1", "true", "yes", "on"}

    if annotation is int:
        return int(normalized)

    if annotation is float:
        return float(normalized)

    if annotation is str:
        return normalized

    if annotation is str | None:  # py311 union syntax
        return normalized or None

    # Fallback for literal and union-like fields.
    try:
        return json.loads(normalized)
    except json.JSONDecodeError:
        return normalized
