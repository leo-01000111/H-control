from __future__ import annotations

from hcontrol.tracking import HandTracker


def test_tracker_reuses_id_for_close_detection() -> None:
    tracker = HandTracker(max_distance=0.2, max_missing_ms=500)

    ids1 = tracker.update([(0.1, 0.1, 0.3, 0.3)], 1000)
    ids2 = tracker.update([(0.12, 0.1, 0.32, 0.3)], 1030)

    assert ids1 == [1]
    assert ids2 == [1]


def test_tracker_assigns_new_id_for_far_detection() -> None:
    tracker = HandTracker(max_distance=0.1, max_missing_ms=500)

    ids1 = tracker.update([(0.1, 0.1, 0.3, 0.3)], 1000)
    ids2 = tracker.update([(0.7, 0.7, 0.9, 0.9)], 1030)

    assert ids1 == [1]
    assert ids2 == [2]
