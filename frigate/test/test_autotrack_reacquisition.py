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


def candidate(*, label, active=True):
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
        active=active,
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
    tracker.zoom_factor = {CAMERA: 0.3}
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

    def test_reacquisition_rejects_stationary_matching_candidate(self):
        lost = lost_object_data(label="car")
        tracker = tracker_shell(lost)
        incoming = candidate(label="car", active=False)
        calculate = mock.Mock()
        move = mock.Mock()
        tracker._calculate_tracked_object_metrics = calculate
        tracker._autotrack_move_ptz = move

        tracker.autotrack_object(CAMERA, incoming)

        self.assertIsNone(tracker.tracked_object[CAMERA])
        self.assertEqual(list(tracker.tracked_object_history[CAMERA]), [lost])
        calculate.assert_not_called()
        move.assert_not_called()

    def test_current_target_is_released_when_it_becomes_stationary(self):
        previous = lost_object_data(label="car")
        tracker = tracker_shell(previous)
        incoming = candidate(label="car", active=False)
        tracker.tracked_object[CAMERA] = incoming
        tracker.ptz_metrics[CAMERA].tracking_active.set()
        calculate = mock.Mock()
        move = mock.Mock()
        tracker._calculate_tracked_object_metrics = calculate
        tracker._autotrack_move_ptz = move

        tracker.autotrack_object(CAMERA, incoming)

        self.assertIsNone(tracker.tracked_object[CAMERA])
        self.assertEqual(
            list(tracker.tracked_object_history[CAMERA]),
            [previous, incoming.obj_data],
        )
        self.assertIsNot(tracker.tracked_object_history[CAMERA][-1], incoming.obj_data)
        self.assertGreater(tracker.tracked_object_metrics[CAMERA]["max_target_box"], 0)
        # Releasing the target starts the existing maintenance return path;
        # its active state remains ON until that path returns to the preset.
        self.assertTrue(tracker.ptz_metrics[CAMERA].tracking_active.is_set())
        tracker.dispatcher.publish.assert_not_called()
        calculate.assert_not_called()
        move.assert_not_called()

    def test_released_stationary_target_reacquires_only_after_moving(self):
        tracker = tracker_shell(lost_object_data(label="car"))
        stationary = candidate(label="car", active=False)
        tracker.tracked_object[CAMERA] = stationary
        tracker._calculate_tracked_object_metrics = mock.Mock()
        tracker._autotrack_move_ptz = mock.Mock()

        tracker.autotrack_object(CAMERA, stationary)
        tracker.autotrack_object(CAMERA, candidate(label="car", active=False))

        self.assertIsNone(tracker.tracked_object[CAMERA])
        tracker._calculate_tracked_object_metrics.assert_not_called()
        tracker._autotrack_move_ptz.assert_not_called()

        moving = candidate(label="car", active=True)
        moving.obj_data["box"] = (40, 40, 50, 50)
        moving.obj_data["region"] = (40, 40, 50, 50)
        moving.obj_data["frame_time"] = 12.0
        tracker.autotrack_object(CAMERA, moving)

        self.assertIs(tracker.tracked_object[CAMERA], moving)
        tracker._calculate_tracked_object_metrics.assert_called_once_with(
            CAMERA, moving
        )
        tracker._autotrack_move_ptz.assert_called_once_with(CAMERA, moving)


if __name__ == "__main__":
    unittest.main()
