"""Regression coverage for PTZ autotracker object reacquisition.

The tracker shell and object doubles keep these tests independent of Frigate's
object pipeline, ONVIF, and an event loop. They exercise the decision at the
reacquisition boundary directly.
"""

from __future__ import annotations

import threading
import unittest
from collections import deque
from types import SimpleNamespace
from unittest import mock

from frigate.ptz.autotrack import PtzAutoTracker

CAMERA = "front"
TRACKED_TYPES = ["person", "car", "dog", "horse"]


def camera_config():
    return SimpleNamespace(
        name=CAMERA,
        detect=SimpleNamespace(fps=5),
        frame_shape=(1080, 1920),
        onvif=SimpleNamespace(
            autotracking=SimpleNamespace(
                enabled=True,
                track=TRACKED_TYPES,
                required_zones=["center"],
            )
        ),
    )


def lost_object_data(*, label="person"):
    return {
        "id": "lost-object",
        "label": label,
        "frame_time": 10.0,
        "box": (0, 0, 10, 10),
        "region": (0, 0, 10, 10),
    }


def candidate(*, label):
    return SimpleNamespace(
        camera_config=SimpleNamespace(name=CAMERA),
        obj_data={
            "id": "candidate",
            "label": label,
            "frame_time": 11.0,
            # No overlap with the lost object's region, satisfying the
            # existing reacquisition proximity condition.
            "box": (20, 20, 30, 30),
            "region": (20, 20, 30, 30),
        },
        previous={"false_positive": False},
        false_positive=False,
        active=True,
        entered_zones=["center"],
    )


def tracker_shell(history):
    tracker = PtzAutoTracker.__new__(PtzAutoTracker)
    tracker.config = SimpleNamespace(cameras={CAMERA: camera_config()})
    tracker.autotracker_init = {CAMERA: True}
    tracker.calibrating = {CAMERA: False}
    tracker.object_types = {CAMERA: TRACKED_TYPES}
    tracker.required_zones = {CAMERA: ["center"]}
    tracker.tracked_object = {CAMERA: None}
    tracker.tracked_object_history = {CAMERA: deque([history])}
    tracker.tracked_object_metrics = {CAMERA: {}}
    tracker.ptz_metrics = {CAMERA: SimpleNamespace(tracking_active=threading.Event())}
    tracker.dispatcher = mock.Mock()
    return tracker


class AutotrackReacquisitionTest(unittest.TestCase):
    def test_reacquisition_rejects_candidate_with_different_lost_object_label(self):
        lost = lost_object_data(label="horse")
        tracker = tracker_shell(lost)
        incoming = candidate(label="dog")
        calculate = mock.Mock()
        move = mock.Mock()
        tracker._calculate_tracked_object_metrics = calculate
        tracker._autotrack_move_ptz = move

        tracker.autotrack_object(CAMERA, incoming)

        self.assertIsNone(tracker.tracked_object[CAMERA])
        self.assertEqual(list(tracker.tracked_object_history[CAMERA]), [lost])
        calculate.assert_not_called()
        move.assert_not_called()

    def test_reacquisition_accepts_candidate_with_matching_lost_object_label(self):
        tracker = tracker_shell(lost_object_data(label="dog"))
        incoming = candidate(label="dog")
        calculate = mock.Mock()
        move = mock.Mock()
        tracker._calculate_tracked_object_metrics = calculate
        tracker._autotrack_move_ptz = move

        tracker.autotrack_object(CAMERA, incoming)

        self.assertIs(tracker.tracked_object[CAMERA], incoming)
        self.assertEqual(list(tracker.tracked_object_history[CAMERA]), [incoming.obj_data])
        self.assertIsNot(tracker.tracked_object_history[CAMERA][0], incoming.obj_data)
        calculate.assert_called_once_with(CAMERA, incoming)
        move.assert_called_once_with(CAMERA, incoming)


if __name__ == "__main__":
    unittest.main()
