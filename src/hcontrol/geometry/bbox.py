"""Geometry helpers for hand bounding boxes and matching metrics."""

from __future__ import annotations

from ..types import BBoxNorm, BBoxPx, Landmark3D


def bbox_from_landmarks_norm(landmarks: list[Landmark3D]) -> BBoxNorm:
    """Build a clipped normalized bbox from normalized landmarks."""

    if not landmarks:
        return (0.0, 0.0, 0.0, 0.0)

    xs = [point[0] for point in landmarks]
    ys = [point[1] for point in landmarks]

    x_min = _clamp(min(xs), 0.0, 1.0)
    y_min = _clamp(min(ys), 0.0, 1.0)
    x_max = _clamp(max(xs), 0.0, 1.0)
    y_max = _clamp(max(ys), 0.0, 1.0)

    return (x_min, y_min, x_max, y_max)


def bbox_norm_to_pixels(bbox_norm: BBoxNorm, width: int, height: int) -> BBoxPx:
    """Convert normalized bbox to pixel coordinates."""

    x_min, y_min, x_max, y_max = bbox_norm
    return (
        int(x_min * width),
        int(y_min * height),
        int(x_max * width),
        int(y_max * height),
    )


def bbox_centroid(bbox_norm: BBoxNorm) -> tuple[float, float]:
    x_min, y_min, x_max, y_max = bbox_norm
    return ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)


def centroid_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return (dx * dx + dy * dy) ** 0.5


def bbox_iou(a: BBoxNorm, b: BBoxNorm) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0

    return inter_area / union_area


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
