from __future__ import annotations

import json
import os

import pytest
from pydantic import ValidationError

from hcontrol.config import GestureConfig, load_config


def test_default_config_values() -> None:
    config = GestureConfig()
    assert config.camera_index == 0
    assert config.max_hands == 2
    assert config.min_gesture_confidence == 0.6


def test_invalid_confidence_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        GestureConfig(min_gesture_confidence=1.5)


def test_load_config_from_json_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    payload = {
        "camera_index": 1,
        "frame_width": 640,
        "frame_height": 480,
        "threaded": False,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    config = load_config(config_file=config_path)
    assert config.camera_index == 1
    assert config.frame_width == 640
    assert config.threaded is False


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HCONTROL_MAX_HANDS", "1")
    monkeypatch.setenv("HCONTROL_DRAW_ANNOTATIONS", "false")

    config = GestureConfig().with_env_overrides()

    assert config.max_hands == 1
    assert config.draw_annotations is False


def test_no_env_leakage_after_monkeypatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HCONTROL_MAX_HANDS", raising=False)
    monkeypatch.delenv("HCONTROL_DRAW_ANNOTATIONS", raising=False)

    config = GestureConfig().with_env_overrides()

    assert "HCONTROL_MAX_HANDS" not in os.environ
    assert config.max_hands == 2
