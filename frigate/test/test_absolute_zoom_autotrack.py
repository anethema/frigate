"""Regression tests for absolute autotracking zoom targets.

The fakes exercise only the autotracker queue and calibration call contracts;
they do not open an ONVIF connection.
"""

from __future__ import annotations

import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

import frigate.ptz.autotrack as autotrack_module
from frigate.config import ZoomingModeEnum
from frigate.ptz.autotrack import PtzAutoTracker

CAMERA = "front"


class Cell:
    def __init__(self, value=0.0):
        self.value = value


class ImmediateLoop:
    def call_soon_threadsafe(self, callback, *args):
        callback(*args)


def camera_config(
    zooming,
    *,
    movement_status="position",
    zoom_out_hysteresis=1.1,
    position_zoom_in_hysteresis=0.95,
    position_zoom_center_threshold=0.05,
    position_zoom_max_velocity=0.005,
):
    return SimpleNamespace(
        name=CAMERA,
        detect=SimpleNamespace(fps=5),
        frame_shape=(1080, 1920),
        onvif=SimpleNamespace(
            autotracking=SimpleNamespace(
                enabled=True,
                movement_status=movement_status,
                zooming=zooming,
                calibrate_on_startup=False,
                movement_weights=[],
                return_preset="overview",
                zoom_out_hysteresis=zoom_out_hysteresis,
                position_zoom_in_hysteresis=position_zoom_in_hysteresis,
                position_zoom_center_threshold=position_zoom_center_threshold,
                position_zoom_max_velocity=position_zoom_max_velocity,
            )
        ),
    )


def metrics():
    return SimpleNamespace(
        motor_stopped=threading.Event(),
        reset=threading.Event(),
        frame_time=Cell(100.0),
        start_time=Cell(0.0),
        stop_time=Cell(0.0),
        min_zoom=Cell(0.0),
        max_zoom=Cell(1.0),
        zoom_level=Cell(0.5),
    )


class FakeOnvif:
    def __init__(self, tracker, *, stop_after_move=False):
        self.tracker = tracker
        self.stop_after_move = stop_after_move
        self.loop = ImmediateLoop()
        self.relative_calls = []
        self.absolute_calls = []
        self.preset_calls = 0
        self.cams = {
            CAMERA: {
                # Deliberately non-normalized camera coordinates.  The public
                # _zoom_absolute argument remains normalized [0, 1].
                "absolute_zoom_range": {"XRange": {"Min": 10.0, "Max": 30.0}}
            }
        }

    async def _move_relative(self, _camera, pan, tilt, zoom, speed):
        self.relative_calls.append((pan, tilt, zoom, speed))
        self.tracker.ptz_metrics[CAMERA].motor_stopped.set()
        if self.stop_after_move:
            self.tracker.stop_event.set()

    async def _zoom_absolute(self, _camera, zoom, speed):
        self.absolute_calls.append((zoom, speed))
        self.tracker.ptz_metrics[CAMERA].motor_stopped.set()
        if self.stop_after_move:
            self.tracker.stop_event.set()

    async def _move_to_preset(self, _camera, _preset):
        self.preset_calls += 1
        self.tracker.ptz_metrics[CAMERA].motor_stopped.set()

    async def get_camera_status(self, _camera):
        return None


def tracker_shell(
    zooming,
    *,
    movement_status="position",
    stop_after_move=False,
    zoom_out_hysteresis=1.1,
    position_zoom_in_hysteresis=0.95,
    position_zoom_center_threshold=0.05,
    position_zoom_max_velocity=0.005,
):
    config = SimpleNamespace(
        cameras={
            CAMERA: camera_config(
                zooming,
                movement_status=movement_status,
                zoom_out_hysteresis=zoom_out_hysteresis,
                position_zoom_in_hysteresis=position_zoom_in_hysteresis,
                position_zoom_center_threshold=position_zoom_center_threshold,
                position_zoom_max_velocity=position_zoom_max_velocity,
            )
        }
    )
    tracker = PtzAutoTracker.__new__(PtzAutoTracker)
    tracker.config = config
    tracker.ptz_metrics = {CAMERA: metrics()}
    tracker.stop_event = threading.Event()
    tracker.move_queues = {CAMERA: asyncio.Queue()}
    tracker.move_queue_locks = {CAMERA: asyncio.Lock()}
    tracker.move_metrics = {CAMERA: []}
    tracker.move_coefficients = {CAMERA: []}
    tracker.intercept = {CAMERA: None}
    tracker.zoom_time = {CAMERA: 0.0}
    tracker.calibrating = {CAMERA: False}
    onvif = FakeOnvif(tracker, stop_after_move=stop_after_move)
    tracker.onvif = onvif
    return tracker, onvif


def zoom_policy_tracker(*, zoom_out_hysteresis=1.1, position_zoom_in_hysteresis=0.95):
    """Build the state consumed by _should_zoom_in without an ONVIF connection."""
    tracker, _ = tracker_shell(
        ZoomingModeEnum.absolute,
        zoom_out_hysteresis=zoom_out_hysteresis,
        position_zoom_in_hysteresis=position_zoom_in_hysteresis,
    )
    tracker.zoom_factor = {CAMERA: 0.3}
    tracker.tracked_object_metrics = {
        CAMERA: {
            "velocity": np.zeros(4),
            "valid_velocity": True,
            "below_distance_threshold": True,
            "target_box": 0.1,
            "original_target_box": 0.1,
            "max_target_box": 0.2,
        }
    }
    return tracker


class AbsoluteZoomAutotrackTest(unittest.IsolatedAsyncioTestCase):
    def test_position_absolute_uses_configured_zoom_in_hysteresis(self):
        tracker = zoom_policy_tracker(position_zoom_in_hysteresis=1.2)
        # This target is above the stock .95 threshold but below 1.2 times
        # the limit. Position-based absolute zoom should still move in.
        tracker.tracked_object_metrics[CAMERA]["target_box"] = 0.22

        self.assertTrue(
            tracker._should_zoom_in(CAMERA, object(), (910, 490, 1010, 590), 0)
        )

    def test_soft_size_zoom_out_requires_centering_distance(self):
        tracker = zoom_policy_tracker()
        tracker.tracked_object_metrics[CAMERA]["target_box"] = 0.3
        tracker.tracked_object_metrics[CAMERA]["below_distance_threshold"] = False

        self.assertIsNone(
            tracker._should_zoom_in(CAMERA, object(), (1300, 490, 1400, 590), 0)
        )

    def test_soft_size_zoom_out_uses_observed_box_not_prediction(self):
        tracker = zoom_policy_tracker()
        tracker.ptz_metrics[CAMERA].zoom_level.value = 1.0
        tracker.tracked_object_metrics[CAMERA]["target_box"] = 0.1
        tracker._predict_area_after_time = mock.Mock(return_value=1_000_000)

        self.assertIsNone(
            tracker._should_zoom_in(CAMERA, object(), (910, 490, 1010, 590), 1)
        )

    def test_edge_escape_still_zooms_out_when_off_center(self):
        tracker = zoom_policy_tracker()
        tracker.tracked_object_metrics[CAMERA]["below_distance_threshold"] = False

        self.assertFalse(
            tracker._should_zoom_in(CAMERA, object(), (0, 490, 100, 590), 0)
        )

    def test_high_velocity_escape_still_zooms_out_when_off_center(self):
        tracker = zoom_policy_tracker()
        tracker.tracked_object_metrics[CAMERA]["below_distance_threshold"] = False
        tracker.tracked_object_metrics[CAMERA]["velocity"] = np.array(
            [50.0, 0.0, 50.0, 0.0]
        )

        self.assertFalse(
            tracker._should_zoom_in(CAMERA, object(), (1300, 490, 1400, 590), 0)
        )

    def test_larger_hysteresis_suppresses_only_soft_zoom_out(self):
        default_tracker = zoom_policy_tracker(zoom_out_hysteresis=1.1)
        relaxed_tracker = zoom_policy_tracker(zoom_out_hysteresis=1.8)
        for tracker in (default_tracker, relaxed_tracker):
            tracker.tracked_object_metrics[CAMERA]["target_box"] = 0.3

        centered_box = (910, 490, 1010, 590)
        self.assertFalse(
            default_tracker._should_zoom_in(CAMERA, object(), centered_box, 0)
        )
        self.assertIsNone(
            relaxed_tracker._should_zoom_in(CAMERA, object(), centered_box, 0)
        )

    def test_edge_escape_bypasses_relaxed_hysteresis(self):
        tracker = zoom_policy_tracker(zoom_out_hysteresis=1.8)
        tracker.tracked_object_metrics[CAMERA]["target_box"] = 0.1

        self.assertFalse(
            tracker._should_zoom_in(CAMERA, object(), (0, 490, 100, 590), 0)
        )

    def test_position_zoom_in_requires_tight_centering(self):
        tracker = zoom_policy_tracker()

        # The broad no-pan threshold can still say True, but an object 5.2%
        # of frame width from center must not start a serialized lens move.
        self.assertIsNone(
            tracker._should_zoom_in(CAMERA, object(), (1010, 490, 1110, 590), 0)
        )

    def test_position_zoom_in_requires_low_normalized_velocity(self):
        tracker = zoom_policy_tracker()
        # 10px/frame is above .005 of a 1920px frame, while remaining below
        # the legacy broad .02/frame velocity threshold.
        tracker.tracked_object_metrics[CAMERA]["velocity"] = np.array(
            [10.0, 0.0, 10.0, 0.0]
        )

        self.assertIsNone(
            tracker._should_zoom_in(CAMERA, object(), (910, 490, 1010, 590), 0)
        )

    def test_position_zoom_in_allows_centered_still_object(self):
        tracker = zoom_policy_tracker()

        self.assertTrue(
            tracker._should_zoom_in(CAMERA, object(), (910, 490, 1010, 590), 0)
        )

    async def test_absolute_zero_target_is_enqueued_and_dispatched(self):
        tracker, onvif = tracker_shell(ZoomingModeEnum.absolute, stop_after_move=True)

        tracker._enqueue_move(CAMERA, 1.0, 0.0, 0.0, 0.0)
        await tracker._process_move_queue_impl(CAMERA)

        self.assertEqual(onvif.absolute_calls, [(0.0, 1)])
        self.assertEqual(onvif.relative_calls, [])

    def test_absolute_none_is_not_enqueued(self):
        tracker, _ = tracker_shell(ZoomingModeEnum.absolute)

        tracker._enqueue_move(CAMERA, 1.0, 0.0, 0.0, None)

        self.assertTrue(tracker.move_queues[CAMERA].empty())

    def test_native_absolute_zero_retains_its_existing_no_target_meaning(self):
        tracker, _ = tracker_shell(
            ZoomingModeEnum.absolute,
            movement_status="onvif",
        )

        tracker._enqueue_move(CAMERA, 1.0, 0.0, 0.0, 0.0)

        self.assertTrue(tracker.move_queues[CAMERA].empty())

    async def test_relative_pan_move_retains_its_zero_zoom_delta(self):
        tracker, onvif = tracker_shell(
            ZoomingModeEnum.relative,
            movement_status="onvif",
            stop_after_move=True,
        )

        tracker._enqueue_move(CAMERA, 1.0, 0.2, 0.0, 0.0)
        await tracker._process_move_queue_impl(CAMERA)

        self.assertEqual(onvif.relative_calls, [(0.2, 0.0, 0.0, 1)])
        self.assertEqual(onvif.absolute_calls, [])

    async def test_calibration_passes_normalized_absolute_zoom_endpoints(self):
        tracker, onvif = tracker_shell(ZoomingModeEnum.absolute)
        tracker.ptz_metrics[CAMERA].motor_stopped.set()
        clock = SimpleNamespace(value=0.0)

        def monotonic():
            return clock.value

        async def measured_relative(_camera, pan, tilt, zoom, speed):
            onvif.relative_calls.append((pan, tilt, zoom, speed))
            clock.value += 0.20 + 0.10 * (abs(pan) + abs(tilt))
            tracker.ptz_metrics[CAMERA].motor_stopped.set()

        onvif._move_relative = measured_relative
        tracker._write_config = mock.Mock()
        fake_time = SimpleNamespace(monotonic=monotonic, time=monotonic)

        with mock.patch.object(autotrack_module, "time", fake_time):
            await tracker._calibrate_camera(CAMERA)

        self.assertEqual(onvif.absolute_calls, [(0.0, 1), (1.0, 1)] * 2)
        self.assertEqual(len(onvif.relative_calls), 30)


if __name__ == "__main__":
    unittest.main()
