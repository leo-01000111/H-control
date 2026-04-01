"""Simple centroid-based tracker for stable hand ids across frames."""

from __future__ import annotations

from dataclasses import dataclass

from ..geometry import bbox_centroid, centroid_distance
from ..types import BBoxNorm


@dataclass(slots=True)
class _Track:
    track_id: int
    centroid: tuple[float, float]
    bbox_norm: BBoxNorm
    last_seen_ms: int


class HandTracker:
    """Assign stable integer IDs to hands based on bbox centroid matching."""

    def __init__(self, *, max_distance: float, max_missing_ms: int) -> None:
        self._max_distance = max_distance
        self._max_missing_ms = max_missing_ms
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def update(self, bboxes_norm: list[BBoxNorm], timestamp_ms: int) -> list[int]:
        self._drop_stale_tracks(timestamp_ms)

        if not bboxes_norm:
            return []

        detection_centroids = [bbox_centroid(bbox) for bbox in bboxes_norm]
        detection_to_id: dict[int, int] = {}
        assigned_track_ids: set[int] = set()

        candidates: list[tuple[float, int, int]] = []
        for detection_idx, centroid in enumerate(detection_centroids):
            for track_id, track in self._tracks.items():
                distance = centroid_distance(centroid, track.centroid)
                if distance <= self._max_distance:
                    candidates.append((distance, detection_idx, track_id))

        candidates.sort(key=lambda item: item[0])

        for _, detection_idx, track_id in candidates:
            if detection_idx in detection_to_id:
                continue
            if track_id in assigned_track_ids:
                continue
            detection_to_id[detection_idx] = track_id
            assigned_track_ids.add(track_id)

        for detection_idx, bbox in enumerate(bboxes_norm):
            assigned_id = detection_to_id.get(detection_idx)
            centroid = detection_centroids[detection_idx]

            if assigned_id is None:
                assigned_id = self._next_id
                self._next_id += 1
                self._tracks[assigned_id] = _Track(
                    track_id=assigned_id,
                    centroid=centroid,
                    bbox_norm=bbox,
                    last_seen_ms=timestamp_ms,
                )
                detection_to_id[detection_idx] = assigned_id
                continue

            track = self._tracks[assigned_id]
            track.centroid = centroid
            track.bbox_norm = bbox
            track.last_seen_ms = timestamp_ms

        return [detection_to_id[idx] for idx in range(len(bboxes_norm))]

    def _drop_stale_tracks(self, timestamp_ms: int) -> None:
        stale = [
            track_id
            for track_id, track in self._tracks.items()
            if (timestamp_ms - track.last_seen_ms) > self._max_missing_ms
        ]
        for track_id in stale:
            del self._tracks[track_id]
