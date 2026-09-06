"""Deterministic tests for the pure position-status state machine."""

from __future__ import annotations

import math
import unittest

from frigate.ptz.position_status import (
    IDLE,
    MOVING,
    PositionStatus,
    PositionStatusTimeout,
)

BASELINE = (-0.4, 0.1, 0.0)


class PositionStatusTest(unittest.TestCase):
    def make_status(self, **overrides: object) -> PositionStatus:
        options: dict[str, object] = {
            "minimum_delta": 1e-5,
            "startup_grace": 0.25,
            "stable_time": 0.15,
            "command_timeout": 10.0,
            "missing_timeout": 0.30,
        }
        options.update(overrides)
        return PositionStatus(**options)  # type: ignore[arg-type]

    def test_passive_valid_sample_is_idle(self) -> None:
        status = self.make_status()
        self.assertEqual(IDLE, status.observe(BASELINE, 1.0))
        self.assertEqual(IDLE, status.state)

    def test_observed_trace_completes_after_stability(self) -> None:
        status = self.make_status()
        self.assertEqual(MOVING, status.begin_move(BASELINE, 0.0))
        # Position changes are sufficient; no native MoveStatus is required.
        self.assertEqual(MOVING, status.observe((-0.38, 0.1, 0.0), 0.05))
        self.assertEqual(MOVING, status.observe((-0.36, 0.1, 0.0), 0.10))
        self.assertEqual(MOVING, status.observe((-0.35, 0.1, 0.0), 0.15))
        self.assertEqual(MOVING, status.observe((-0.35, 0.1, 0.0), 0.20))
        self.assertEqual(MOVING, status.observe((-0.35, 0.1, 0.0), 0.25))
        self.assertEqual(IDLE, status.observe((-0.35, 0.1, 0.0), 0.30))

    def test_delayed_start_does_not_complete_before_motion(self) -> None:
        status = self.make_status(command_timeout=1.0)
        status.begin_move(BASELINE, 0.0)
        for now in (0.10, 0.25, 0.50):
            self.assertEqual(MOVING, status.observe(BASELINE, now))
        self.assertEqual(MOVING, status.observe((-0.35, 0.1, 0.0), 0.55))

    def test_move_completed_between_command_and_first_poll(self) -> None:
        status = self.make_status()
        status.begin_move(BASELINE, 0.0)
        self.assertEqual(MOVING, status.observe((-0.85, 0.1, 0.0), 0.30))
        self.assertEqual(MOVING, status.observe((-0.85, 0.1, 0.0), 0.35))
        self.assertEqual(IDLE, status.observe((-0.85, 0.1, 0.0), 0.45))

    def test_normal_command_with_no_displacement_times_out(self) -> None:
        status = self.make_status(command_timeout=0.50)
        status.begin_move(BASELINE, 0.0)
        self.assertEqual(MOVING, status.observe(BASELINE, 0.25))
        with self.assertRaises(PositionStatusTimeout):
            status.observe(BASELINE, 0.50)

    def test_allow_no_motion_supports_zero_calibration_and_preset_target(self) -> None:
        status = self.make_status()
        status.begin_move((0.0, 0.0, 0.0), 0.0, allow_no_motion=True)
        self.assertEqual(MOVING, status.observe((0.0, 0.0, 0.0), 0.05))
        self.assertEqual(MOVING, status.observe((0.0, 0.0, 0.0), 0.15))
        self.assertEqual(IDLE, status.observe((0.0, 0.0, 0.0), 0.25))

        preset = self.make_status()
        preset_target = (-0.42, 0.31, 0.0)
        preset.begin_move(preset_target, 0.0, allow_no_motion=True)
        self.assertEqual(MOVING, preset.observe(preset_target, 0.05))
        self.assertEqual(MOVING, preset.observe(preset_target, 0.15))
        self.assertEqual(IDLE, preset.observe(preset_target, 0.25))

    def test_allowed_no_motion_recovers_after_missing_status(self) -> None:
        status = self.make_status(missing_timeout=0.50)
        position = (0.0, 0.0, 0.0)
        status.begin_move(position, 0.0, allow_no_motion=True)
        self.assertEqual(MOVING, status.observe(position, 0.05))
        self.assertEqual(MOVING, status.observe(None, 0.10))
        # Recovery starts a new stable interval; the earlier valid reading
        # cannot be reused after the unavailable response.
        self.assertEqual(MOVING, status.observe(position, 0.15))
        self.assertEqual(MOVING, status.observe(position, 0.25))
        self.assertEqual(IDLE, status.observe(position, 0.35))

    def test_missing_and_invalid_reset_stability_then_timeout_and_latch(self) -> None:
        status = self.make_status()
        status.begin_move(BASELINE, 0.0)
        target = (-0.85, 0.1, 0.0)
        self.assertEqual(MOVING, status.observe(target, 0.05))
        self.assertEqual(MOVING, status.observe(target, 0.10))
        self.assertEqual(MOVING, status.observe(None, 0.15))
        self.assertEqual(MOVING, status.observe((math.nan, 0.1, 0.0), 0.20))
        with self.assertRaises(PositionStatusTimeout) as raised:
            status.observe(None, 0.45)
        with self.assertRaises(PositionStatusTimeout) as latched:
            status.observe(target, 0.46)
        self.assertIs(raised.exception, latched.exception)
        status.reseed(target, 0.50)
        self.assertEqual(IDLE, status.observe(target, 0.55))

    def test_frozen_timestamp_never_counts_as_fresh_stability_reading(self) -> None:
        status = self.make_status()
        status.begin_move(BASELINE, 0.0, allow_no_motion=True)
        for _ in range(5):
            self.assertEqual(MOVING, status.observe(BASELINE, 0.25))
        self.assertEqual(MOVING, status.observe(BASELINE, 0.30))
        self.assertEqual(MOVING, status.observe(BASELINE, 0.35))
        self.assertEqual(IDLE, status.observe(BASELINE, 0.40))

    def test_pan_wrap_crossing_is_small_motion_and_encoder_jitter_is_stable(
        self,
    ) -> None:
        status = self.make_status(minimum_delta=0.01)
        status.begin_move((0.99, 0.0, 0.0), 0.0)
        self.assertEqual(MOVING, status.observe((-0.99, 0.0, 0.0), 0.05))
        self.assertEqual(MOVING, status.observe((-0.95, 0.0, 0.0), 0.10))
        self.assertEqual(MOVING, status.observe((-0.948, 0.0, 0.0), 0.15))
        self.assertEqual(MOVING, status.observe((-0.952, 0.0, 0.0), 0.20))
        self.assertEqual(IDLE, status.observe((-0.949, 0.0, 0.0), 0.25))

    def test_pan_wrap_does_not_create_a_false_large_displacement(self) -> None:
        status = self.make_status(minimum_delta=0.01, command_timeout=0.30)
        status.begin_move((0.999, 0.0, 0.0), 0.0)
        # Across a [-1, 1] seam this is a 0.002 change, not 1.998.
        self.assertEqual(MOVING, status.observe((-0.999, 0.0, 0.0), 0.05))
        self.assertEqual(MOVING, status.observe((-0.999, 0.0, 0.0), 0.20))
        with self.assertRaises(PositionStatusTimeout):
            status.observe((-0.999, 0.0, 0.0), 0.30)

    def test_cumulative_slow_motion_cannot_settle(self) -> None:
        status = self.make_status(
            minimum_delta=0.01, startup_grace=0.0, stable_time=0.10
        )
        status.begin_move((0.0, 0.0, 0.0), 0.0)
        self.assertEqual(MOVING, status.observe((0.012, 0.0, 0.0), 0.05))
        # Each step is below the threshold, but they add up from the anchor.
        for now, pan in ((0.10, 0.016), (0.15, 0.021), (0.20, 0.026)):
            self.assertEqual(MOVING, status.observe((pan, 0.0, 0.0), now))
        self.assertEqual(MOVING, status.observe((0.030, 0.0, 0.0), 0.25))
        self.assertEqual(MOVING, status.observe((0.030, 0.0, 0.0), 0.30))
        self.assertEqual(IDLE, status.observe((0.030, 0.0, 0.0), 0.35))


if __name__ == "__main__":
    unittest.main()
