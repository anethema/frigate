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


def camera_config(zooming, *, movement_status="position"):
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


def tracker_shell(zooming, *, movement_status="position", stop_after_move=False):
    config = SimpleNamespace(
        cameras={CAMERA: camera_config(zooming, movement_status=movement_status)}
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


class AbsoluteZoomAutotrackTest(unittest.IsolatedAsyncioTestCase):
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
