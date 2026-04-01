from __future__ import annotations

from hcontrol.geometry import bbox_from_landmarks_norm, bbox_iou, bbox_norm_to_pixels


def test_bbox_from_landmarks_clips_to_normalized_range() -> None:
    landmarks = [
        (-0.2, 0.2, 0.0),
        (0.4, 1.2, 0.0),
        (0.8, 0.6, 0.0),
    ]

    bbox = bbox_from_landmarks_norm(landmarks)

    assert bbox == (0.0, 0.2, 0.8, 1.0)


def test_bbox_norm_to_pixels() -> None:
    bbox_px = bbox_norm_to_pixels((0.1, 0.2, 0.9, 0.8), width=1000, height=500)
    assert bbox_px == (100, 100, 900, 400)


def test_bbox_iou() -> None:
    iou = bbox_iou((0.0, 0.0, 0.5, 0.5), (0.25, 0.25, 0.75, 0.75))
    assert round(iou, 3) == 0.143
