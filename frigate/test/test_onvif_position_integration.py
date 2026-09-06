"""Integration-level tests for position-based PTZ completion.

These tests intentionally use the Frigate package supplied by the pinned image.
They never open a camera connection: ``OnvifController.__new__`` is supplied a
small ONVIF fake which returns fresh ``GetStatus`` samples.
"""

from __future__ import annotations

import asyncio
import threading
import unittest
from collections import deque
from types import SimpleNamespace
from unittest import mock

from pydantic import ValidationError

import frigate.ptz.autotrack as autotrack_module
from frigate.config.camera.onvif import PtzAutotrackConfig
from frigate.ptz.autotrack import PtzAutoTracker
from frigate.ptz.onvif import OnvifCommandEnum, OnvifController
from frigate.ptz.position_status import PositionStatus, PositionStatusError

CAMERA = "front"
OVERVIEW = (-0.4, 0.1, 0.0)
VERIFIED_OVERVIEW = (-0.6, 0.2, 0.0)


class Cell:
    def __init__(self, value=0.0):
        self.value = value


def metrics():
    return SimpleNamespace(
        motor_stopped=threading.Event(),
        tracking_active=threading.Event(),
        reset=threading.Event(),
        autotracker_enabled=Cell(True),
        frame_time=Cell(100.0),
        start_time=Cell(0.0),
        stop_time=Cell(0.0),
        min_zoom=Cell(0.0),
        max_zoom=Cell(1.0),
        zoom_level=Cell(0.0),
    )


def status_at(pose, *, include_zoom=True):
    position = SimpleNamespace(
        PanTilt=SimpleNamespace(x=pose[0], y=pose[1]),
    )
    if include_zoom:
        position.Zoom = SimpleNamespace(x=pose[2])
    return SimpleNamespace(Position=position)


def camera_config(*, enabled=True, movement_status="position", weights=None):
    autotracking = SimpleNamespace(
        enabled=enabled,
        enabled_in_config=True,
        movement_status=movement_status,
        zooming="disabled",
        zoom_factor=0.3,
        track=["person"],
        required_zones=["center"],
        calibrate_on_startup=False,
        movement_weights=[] if weights is None else weights,
        return_preset="overview",
        return_preset_position=None,
        preset_movement="preset",
        timeout=10,
    )
    return SimpleNamespace(
        name=CAMERA,
        detect=SimpleNamespace(fps=5),
        onvif=SimpleNamespace(autotracking=autotracking),
    )


class FakePtz:
    def __init__(self, pose=OVERVIEW):
        self.pose = pose
        self.status_responses = deque()
        self.relative_calls = 0
        self.preset_calls = 0
        self.preset_requests = []
        self.absolute_calls = 0
        self.absolute_requests = []
        self.stop_calls = 0
        self.command_error = None
        self.goto_pose = None
        self.on_goto = None

    async def GetStatus(self, _request):
        if self.status_responses:
            response = self.status_responses.popleft()
            if isinstance(response, BaseException):
                raise response
            return response
        return status_at(self.pose)

    async def RelativeMove(self, _request):
        self.relative_calls += 1
        if self.command_error:
            raise self.command_error

    async def GotoPreset(self, request):
        self.preset_calls += 1
        self.preset_requests.append(request)
        if self.command_error:
            raise self.command_error
        if self.goto_pose is not None:
            self.pose = self.goto_pose
        if self.on_goto is not None:
            self.on_goto()

    async def AbsoluteMove(self, request):
        self.absolute_calls += 1
        self.absolute_requests.append(request)
        if self.command_error:
            raise self.command_error
        pan_tilt = request["Position"].get(
            "PanTilt", {"x": self.pose[0], "y": self.pose[1]}
        )
        self.pose = (
            pan_tilt["x"],
            pan_tilt["y"],
            request["Position"]["Zoom"]["x"],
        )

    async def Stop(self, _request):
        self.stop_calls += 1


def controller_with_fake(ptz, config=None, *, state=None):
    config = config or camera_config()
    controller = OnvifController.__new__(OnvifController)
    controller.config = SimpleNamespace(cameras={CAMERA: config})
    controller.ptz_metrics = {CAMERA: metrics()}
    controller.cams = {
        CAMERA: {
            "ptz": ptz,
            "status_request": object(),
            "move_request": SimpleNamespace(ProfileToken="profile"),
            "active": False,
        }
    }
    controller.status_locks = {CAMERA: asyncio.Lock()}
    controller.position_move_locks = {CAMERA: asyncio.Lock()}
    controller.position_states = {
        CAMERA: state
        or PositionStatus(
            startup_grace=0.0,
            stable_time=0.0,
            command_timeout=0.25,
            missing_timeout=0.05,
        )
    }
    controller.position_faults = {}
    controller.position_poll_interval = 0.001
    controller.position_rpc_timeout = 0.1
    return controller


class PositionControllerTest(unittest.IsolatedAsyncioTestCase):
    def test_native_status_is_default_and_position_accepts_only_absolute_zoom(self):
        defaults = PtzAutotrackConfig()
        self.assertEqual(defaults.movement_status, "onvif")
        self.assertEqual(defaults.preset_movement, "preset")
        self.assertEqual(defaults.zoom_out_hysteresis, 1.1)
        self.assertEqual(defaults.position_zoom_center_threshold, 0.05)
        self.assertEqual(defaults.position_zoom_max_velocity, 0.005)
        self.assertEqual(
            PtzAutotrackConfig(zoom_out_hysteresis=1.8).zoom_out_hysteresis,
            1.8,
        )
        for hysteresis in (0.99, 3.01, float("nan")):
            with self.assertRaises(ValidationError):
                PtzAutotrackConfig(zoom_out_hysteresis=hysteresis)
        self.assertEqual(
            PtzAutotrackConfig(movement_status="position", zooming="absolute").zooming,
            "absolute",
        )
        with self.assertRaises(Exception):
            PtzAutotrackConfig(movement_status="position", zooming="relative")
        with self.assertRaises(Exception):
            PtzAutotrackConfig(preset_movement="absolute")
        self.assertEqual(
            PtzAutotrackConfig(
                movement_status="position", preset_movement="absolute"
            ).preset_movement,
            "absolute",
        )
        with self.assertRaises(Exception):
            PtzAutotrackConfig(return_preset_position=list(VERIFIED_OVERVIEW))
        self.assertEqual(
            PtzAutotrackConfig(
                movement_status="position",
                return_preset_position=list(VERIFIED_OVERVIEW),
            ).return_preset_position,
            list(VERIFIED_OVERVIEW),
        )

    async def test_position_tuple_allows_a_camera_that_omits_zoom(self):
        controller = controller_with_fake(FakePtz())
        self.assertEqual(
            controller._position_tuple(
                status_at(OVERVIEW, include_zoom=False).Position
            ),
            (OVERVIEW[0], OVERVIEW[1], 0.0),
        )

    async def test_absolute_zoom_updates_metric_and_can_return_to_zero(self):
        ptz = FakePtz(OVERVIEW)
        config = camera_config()
        config.onvif.autotracking.zooming = "absolute"
        controller = controller_with_fake(ptz, config)
        controller.cams[CAMERA].update(
            features=["zoom-a"],
            absolute_zoom_range={
                "URI": "http://www.onvif.org/ver10/tptz/ZoomSpaces/PositionGenericSpace",
                "XRange": {"Min": 0.0, "Max": 1.0},
            },
        )
        for target in (0.2, 0.0, 0.0):
            await controller._zoom_absolute(CAMERA, target, 1)
            self.assertEqual(ptz.pose, (OVERVIEW[0], OVERVIEW[1], target))
            self.assertEqual(controller.ptz_metrics[CAMERA].zoom_level.value, target)
            self.assertTrue(controller.ptz_metrics[CAMERA].motor_stopped.is_set())
            self.assertNotIn("PanTilt", ptz.absolute_requests[-1]["Position"])
        self.assertEqual(controller.position_states[CAMERA].command_timeout, 0.25)

    async def test_absolute_zoom_calibration_respects_verified_optical_limit(self):
        ptz = FakePtz(OVERVIEW)
        config = camera_config()
        config.onvif.autotracking.zooming = "absolute"
        config.onvif.autotracking.position_zoom_limits = [0.0, 0.06087]
        controller = controller_with_fake(ptz, config)
        controller.cams[CAMERA].update(
            features=["zoom-a"],
            absolute_zoom_range={
                "URI": "http://www.onvif.org/ver10/tptz/ZoomSpaces/PositionGenericSpace",
                "XRange": {"Min": 0.0, "Max": 1.0},
            },
        )
        await controller._zoom_absolute(CAMERA, 1.0, 1)
        self.assertEqual(ptz.pose[2], 0.06087)
        self.assertEqual(controller.ptz_metrics[CAMERA].zoom_level.value, 0.06087)

    def test_position_zoom_limits_are_validated(self):
        self.assertEqual(
            PtzAutotrackConfig(
                movement_status="position", position_zoom_limits=[0, 0.06087]
            ).position_zoom_limits,
            [0, 0.06087],
        )
        for limits in ([1, 0], [0, 2], [0, float("nan")]):
            with self.assertRaises(Exception):
                PtzAutotrackConfig(
                    movement_status="position", position_zoom_limits=limits
                )

    async def test_absolute_zoom_requires_real_zoom_feedback(self):
        config = camera_config()
        config.onvif.autotracking.zooming = "absolute"
        ptz = FakePtz()
        ptz.status_responses.append(status_at(OVERVIEW, include_zoom=False))
        controller = controller_with_fake(ptz, config)
        with self.assertRaises(PositionStatusError):
            await controller._read_position(CAMERA)

    async def test_wrong_target_faults_before_successful_idle_is_published(self):
        ptz = FakePtz()
        controller = controller_with_fake(ptz)

        async def bad_move(request):
            ptz.pose = (OVERVIEW[0], OVERVIEW[1], 0.1)

        with self.assertRaises(PositionStatusError):
            await controller._position_move(CAMERA, bad_move, {}, zoom_target=0.2)
        self.assertIn(CAMERA, controller.position_faults)
        self.assertEqual(ptz.stop_calls, 1)

    async def test_zoom_encoder_pause_does_not_finish_move_early(self):
        ptz = FakePtz()
        controller = controller_with_fake(ptz, state=PositionStatus(command_timeout=3))

        async def lens_move(request):
            ptz.pose = (OVERVIEW[0], OVERVIEW[1], 0.1)

            async def delayed_encoder():
                await asyncio.sleep(0.35)
                ptz.pose = (OVERVIEW[0], OVERVIEW[1], 0.2)

            asyncio.create_task(delayed_encoder())

        await controller._position_move(
            CAMERA, lens_move, {}, zoom_target=0.2, settle_time=1
        )
        self.assertEqual(ptz.pose[2], 0.2)
        self.assertNotIn(CAMERA, controller.position_faults)

    async def test_zero_relative_and_known_at_target_preset_settle(self):
        ptz = FakePtz(OVERVIEW)
        controller = controller_with_fake(ptz)

        await controller._position_move(
            CAMERA,
            ptz.RelativeMove,
            {"zero": True},
            allow_no_motion=True,
        )
        self.assertEqual(ptz.relative_calls, 1)
        self.assertTrue(controller.ptz_metrics[CAMERA].motor_stopped.is_set())

        await controller._position_move(
            CAMERA,
            ptz.GotoPreset,
            {"PresetToken": "overview"},
            target=OVERVIEW,
        )
        self.assertEqual(ptz.preset_calls, 1)
        self.assertTrue(controller.ptz_metrics[CAMERA].motor_stopped.is_set())

    async def test_absolute_preset_targets_advertised_coordinates_and_settles(self):
        ptz = FakePtz((-0.90, -0.30, 0.0))
        config = camera_config()
        config.onvif.autotracking.preset_movement = "absolute"
        controller = controller_with_fake(
            ptz,
            config,
            state=PositionStatus(
                startup_grace=0.0,
                stable_time=0.0,
                command_timeout=3.0,
                missing_timeout=0.05,
            ),
        )
        controller.cams[CAMERA].update(
            {
                "presets": {"overview": "token1"},
                "preset_positions": {"overview": OVERVIEW},
            }
        )

        await controller._move_to_preset(CAMERA, "Overview")

        self.assertEqual(ptz.preset_calls, 0)
        self.assertEqual(ptz.absolute_calls, 1)
        request = ptz.absolute_requests[0]
        self.assertEqual(request["ProfileToken"], "profile")
        self.assertEqual(
            (request["Position"]["PanTilt"]["x"], request["Position"]["PanTilt"]["y"]),
            OVERVIEW[:2],
        )
        self.assertEqual(request["Position"]["Zoom"]["x"], OVERVIEW[2])
        self.assertEqual(ptz.pose, OVERVIEW)
        self.assertTrue(controller.ptz_metrics[CAMERA].motor_stopped.is_set())
        self.assertEqual(controller.position_states[CAMERA].stable_time, 0.0)

    async def test_verified_return_position_overrides_bad_preset_metadata(self):
        ptz = FakePtz((-0.90, -0.30, 0.0))
        ptz.goto_pose = VERIFIED_OVERVIEW
        config = camera_config()
        config.onvif.autotracking.return_preset_position = list(VERIFIED_OVERVIEW)
        controller = controller_with_fake(
            ptz,
            config,
            state=PositionStatus(
                startup_grace=0.0,
                stable_time=0.0,
                command_timeout=3.0,
                missing_timeout=0.05,
            ),
        )
        controller.cams[CAMERA].update(
            {
                "presets": {"overview": "token1"},
                # This is the camera's bad GetPresets coordinate.  The
                # verified return position must be used for completion.
                "preset_positions": {"overview": OVERVIEW},
            }
        )
        observed_settle_times = []
        ptz.on_goto = lambda: observed_settle_times.append(
            controller.position_states[CAMERA].stable_time
        )

        await controller._move_to_preset(CAMERA, "overview")

        self.assertEqual(ptz.preset_calls, 1)
        self.assertEqual(ptz.absolute_calls, 0)
        self.assertEqual(ptz.preset_requests[0]["PresetToken"], "token1")
        self.assertEqual(ptz.pose, VERIFIED_OVERVIEW)
        self.assertNotIn(CAMERA, controller.position_faults)
        self.assertEqual(observed_settle_times, [1.0])
        self.assertEqual(controller.position_states[CAMERA].stable_time, 0.0)

    async def test_manual_preset_bypasses_latched_autotracker_fault(self):
        ptz = FakePtz((-0.90, -0.30, 0.0))
        ptz.goto_pose = OVERVIEW
        config = camera_config()
        config.onvif.autotracking.preset_movement = "absolute"
        controller = controller_with_fake(ptz, config)
        controller.cams[CAMERA].update(
            {
                "init": True,
                "presets": {"overview": "token1"},
                "preset_positions": {"overview": OVERVIEW},
            }
        )
        controller.position_faults[CAMERA] = "preset target mismatch"

        await controller.handle_command_async(
            CAMERA, OnvifCommandEnum.preset, "overview"
        )

        self.assertEqual(ptz.preset_calls, 1)
        self.assertEqual(ptz.absolute_calls, 0)
        self.assertEqual(ptz.pose, OVERVIEW)

        with self.assertRaises(PositionStatusError):
            await controller._move_to_preset(CAMERA, "overview")
        self.assertEqual(ptz.preset_calls, 1)
        self.assertEqual(ptz.absolute_calls, 0)

    async def test_pose_changed_before_first_poll_settles(self):
        ptz = FakePtz(OVERVIEW)
        controller = controller_with_fake(ptz)

        async def move(_request):
            ptz.relative_calls += 1
            ptz.pose = (-0.60, OVERVIEW[1], 0.0)

        await controller._position_move(CAMERA, move, {"move": True})
        self.assertEqual(ptz.relative_calls, 1)
        self.assertNotIn(CAMERA, controller.position_faults)

    async def test_frozen_normal_move_stops_and_latches_fault(self):
        ptz = FakePtz(OVERVIEW)
        controller = controller_with_fake(
            ptz,
            state=PositionStatus(
                startup_grace=0.0,
                stable_time=0.0,
                command_timeout=0.01,
                missing_timeout=0.01,
            ),
        )

        with self.assertRaises(PositionStatusError):
            await controller._position_move(CAMERA, ptz.RelativeMove, {"move": True})

        self.assertEqual(ptz.stop_calls, 1)
        self.assertIn(CAMERA, controller.position_faults)
        self.assertFalse(controller.config.cameras[CAMERA].onvif.autotracking.enabled)
        self.assertFalse(controller.ptz_metrics[CAMERA].autotracker_enabled.value)
        self.assertTrue(controller.ptz_metrics[CAMERA].motor_stopped.is_set())

    async def test_status_and_command_errors_stop_and_disable(self):
        ptz = FakePtz(OVERVIEW)
        ptz.status_responses.extend([status_at(OVERVIEW), RuntimeError("offline")])
        controller = controller_with_fake(ptz)
        with self.assertRaises(PositionStatusError):
            await controller._position_move(CAMERA, ptz.RelativeMove, {"move": True})
        self.assertEqual(ptz.stop_calls, 1)
        self.assertFalse(controller.config.cameras[CAMERA].onvif.autotracking.enabled)

        command_error = FakePtz(OVERVIEW)
        command_error.command_error = RuntimeError("rejected")
        controller = controller_with_fake(command_error)
        with self.assertRaises(PositionStatusError):
            await controller._position_move(
                CAMERA, command_error.RelativeMove, {"move": True}
            )
        self.assertEqual(command_error.stop_calls, 1)
        self.assertFalse(controller.config.cameras[CAMERA].onvif.autotracking.enabled)


class FakeDispatcher:
    def __init__(self):
        self.messages = []

    def publish(self, *args, **kwargs):
        self.messages.append((args, kwargs))


class FakePositionOnvif:
    def __init__(self, config, *, capability=True, raises_on_move=False):
        self.config = config
        self.cams = {CAMERA: {"init": True, "features": ["pt-r-fov"]}}
        self.capability = capability
        self.raises_on_move = raises_on_move
        self.move_calls = 0
        self.abort_calls = 0
        self.loop = object()

    async def get_service_capabilities(self, _camera):
        return self.capability

    async def get_camera_status(self, _camera):
        return None

    async def _position_failure(self, camera, _reason):
        self.abort_calls += 1
        self.config.cameras[camera].onvif.autotracking.enabled = False

    async def _move_relative(self, _camera, *_args):
        self.move_calls += 1
        if self.raises_on_move:
            raise PositionStatusError("frozen pose")


def tracker_shell(config, onvif):
    tracker = PtzAutoTracker.__new__(PtzAutoTracker)
    tracker.config = config
    tracker.onvif = onvif
    tracker.ptz_metrics = {CAMERA: metrics()}
    tracker.dispatcher = FakeDispatcher()
    tracker.stop_event = threading.Event()
    tracker.calibrating = {CAMERA: False}
    tracker.autotracker_init = {CAMERA: False}
    tracker.object_types = {}
    tracker.required_zones = {}
    tracker.zoom_factor = {}
    tracker.tracked_object = {CAMERA: None}
    tracker.tracked_object_history = {CAMERA: deque()}
    tracker.tracked_object_metrics = {}
    tracker.move_queues = {CAMERA: asyncio.Queue()}
    tracker.move_queue_locks = {CAMERA: asyncio.Lock()}
    tracker.intercept = {CAMERA: None}
    tracker.move_coefficients = {CAMERA: []}
    tracker.zoom_time = {CAMERA: 0.0}
    tracker.move_metrics = {CAMERA: []}
    return tracker


class PositionAutotrackerIntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def test_cached_initialized_camera_still_validates_position_capability(self):
        config = SimpleNamespace(cameras={CAMERA: camera_config()})
        onvif = FakePositionOnvif(config, capability=False)
        tracker = tracker_shell(config, onvif)

        await tracker._autotracker_setup(config.cameras[CAMERA], CAMERA)

        self.assertEqual(onvif.abort_calls, 1)
        self.assertFalse(config.cameras[CAMERA].onvif.autotracking.enabled)
        self.assertFalse(tracker.autotracker_init[CAMERA])

    async def test_queue_fault_drops_later_commands(self):
        config = SimpleNamespace(cameras={CAMERA: camera_config()})
        onvif = FakePositionOnvif(config, raises_on_move=True)
        tracker = tracker_shell(config, onvif)
        await tracker.move_queues[CAMERA].put((1.0, 0.2, 0.2, 0.0))
        await tracker.move_queues[CAMERA].put((2.0, 0.3, 0.3, 0.0))

        await tracker._process_move_queue(CAMERA)

        self.assertEqual(onvif.move_calls, 1)
        self.assertEqual(onvif.abort_calls, 1)
        self.assertFalse(config.cameras[CAMERA].onvif.autotracking.enabled)
        self.assertTrue(tracker.move_queues[CAMERA].empty())

    async def test_position_queue_never_refits_with_frame_timestamps(self):
        config = SimpleNamespace(cameras={CAMERA: camera_config()})
        config.cameras[CAMERA].onvif.autotracking.calibrate_on_startup = True
        onvif = FakePositionOnvif(config)
        tracker = tracker_shell(config, onvif)
        tracker.intercept[CAMERA] = 0.2
        tracker.move_metrics[CAMERA] = [
            {
                "pan": 0.1,
                "tilt": 0.1,
                "start_timestamp": 1.0,
                "end_timestamp": 1.2,
            }
        ]
        calculate = mock.Mock()
        tracker._calculate_move_coefficients = calculate

        async def successful_position_move(_camera, *_args):
            onvif.move_calls += 1
            tracker.ptz_metrics[CAMERA].motor_stopped.set()
            tracker.stop_event.set()

        onvif._move_relative = successful_position_move
        await tracker.move_queues[CAMERA].put((1.0, 0.2, 0.2, 0.0))

        await tracker._process_move_queue(CAMERA)

        self.assertEqual(onvif.move_calls, 1)
        self.assertEqual(len(tracker.move_metrics[CAMERA]), 1)
        calculate.assert_not_called()

    async def test_real_thirty_step_calibration_persists_then_reloads_weights(self):
        config = SimpleNamespace(cameras={CAMERA: camera_config()})
        clock = SimpleNamespace(value=0.0)

        def monotonic():
            return clock.value

        class CalibrationOnvif:
            def __init__(self):
                self.cams = {CAMERA: {}}
                self.relative_calls = 0
                self.preset_calls = 0

            async def _move_to_preset(self, _camera, _preset):
                self.preset_calls += 1

            async def _move_relative(self, _camera, pan, tilt, _zoom, _speed):
                self.relative_calls += 1
                # y = .20 + .10 * (abs(pan) + abs(tilt)); both regression
                # values meet the production validity limits.
                clock.value += 0.20 + 0.10 * (abs(pan) + abs(tilt))
                tracker.ptz_metrics[CAMERA].motor_stopped.set()

        onvif = CalibrationOnvif()
        tracker = tracker_shell(config, onvif)
        tracker.zoom_time = {CAMERA: 0.0}
        tracker.move_coefficients = {CAMERA: []}
        tracker.intercept = {CAMERA: None}
        writes = []
        tracker._write_config = lambda camera: writes.append(
            tracker.config.cameras[camera].onvif.autotracking.movement_weights
        )
        tracker.ptz_metrics[CAMERA].motor_stopped.set()

        fake_time = SimpleNamespace(monotonic=monotonic, time=monotonic)
        with mock.patch.object(autotrack_module, "time", fake_time):
            await tracker._calibrate_camera(CAMERA)

        self.assertFalse(tracker.calibrating[CAMERA])
        self.assertEqual(onvif.relative_calls, 30)
        self.assertEqual(len(tracker.move_metrics[CAMERA]), 30)
        self.assertEqual(len(writes), 1)
        saved = writes[0]
        self.assertEqual(len(saved.split(",")), 6)

        # A subsequent startup receives the serialized value from the config.
        reload_config = SimpleNamespace(
            cameras={CAMERA: camera_config(weights=saved.split(","))}
        )
        reload_onvif = FakePositionOnvif(reload_config, capability=True)
        reloaded = tracker_shell(reload_config, reload_onvif)

        class CompletedFuture:
            def result(self):
                return None

        def discard_background_coroutine(coro, _loop):
            coro.close()
            return CompletedFuture()

        with mock.patch.object(
            autotrack_module.asyncio,
            "run_coroutine_threadsafe",
            side_effect=discard_background_coroutine,
        ):
            await reloaded._autotracker_setup_impl(
                reload_config.cameras[CAMERA], CAMERA
            )

        self.assertTrue(reloaded.autotracker_init[CAMERA])
        self.assertEqual(reloaded.intercept[CAMERA], float(saved.split(",")[2]))
        self.assertEqual(
            list(reloaded.move_coefficients[CAMERA]),
            [float(value) for value in saved.split(",")[3:5]],
        )
        self.assertEqual(reloaded.zoom_time[CAMERA], float(saved.split(",")[5]))


if __name__ == "__main__":
    unittest.main()
