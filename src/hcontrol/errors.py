"""Typed exceptions used across the hcontrol runtime."""

from __future__ import annotations


class HControlError(Exception):
    """Base exception for all library-specific failures."""


class ConfigError(HControlError):
    """Raised when configuration cannot be loaded or validated."""


class CameraConnectionError(HControlError):
    """Raised when the camera cannot be opened."""


class DependencyNotAvailableError(HControlError):
    """Raised when an optional runtime dependency is missing."""


class InferenceEngineError(HControlError):
    """Raised for inference setup or runtime failures."""
