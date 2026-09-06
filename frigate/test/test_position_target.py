"""Target-aware completion tests for the position-status state machine."""

from __future__ import annotations

import unittest

from frigate.ptz.position_status import (
    IDLE,
    MOVING,
    PositionStatus,
    PositionStatusTimeout,
)


class PositionTargetTest(unittest.TestCase):
    def make_status(self, **overrides: object) -> PositionStatus:
        options: dict[str, object] = {
            "minimum_delta": 1e-5,
            "startup_grace": 0.0,
            "stable_time": 0.10,
            "minimum_stable_readings": 3,
            "command_timeout": 1.0,
            "missing_timeout": 0.5,
        }
        options.update(overrides)
        return PositionStatus(**options)  # type: ignore[arg-type]

    def test_intermediate_lens_plateau_remains_moving_until_target_settles(self):
        status = self.make_status()
        status.begin_move((0.0, 0.0, 0.0), 0.0, target=(0.0, 0.0, 0.2))

        for now in (0.05, 0.10, 0.15, 0.30):
            self.assertEqual(MOVING, status.observe((0.0, 0.0, 0.1), now))

        self.assertEqual(MOVING, status.observe((0.0, 0.0, 0.2), 0.35))
        self.assertEqual(MOVING, status.observe((0.0, 0.0, 0.2), 0.40))
        self.assertEqual(IDLE, status.observe((0.0, 0.0, 0.2), 0.45))

    def test_wrong_frozen_target_times_out(self):
        status = self.make_status(command_timeout=0.40)
        status.begin_move((0.0, 0.0, 0.0), 0.0, target=(0.0, 0.0, 0.5))

        self.assertEqual(MOVING, status.observe((0.0, 0.0, 0.2), 0.05))
        self.assertEqual(MOVING, status.observe((0.0, 0.0, 0.2), 0.20))
        with self.assertRaises(PositionStatusTimeout):
            status.observe((0.0, 0.0, 0.2), 0.40)

    def test_zero_zoom_target_completes_after_arrival(self):
        status = self.make_status()
        status.begin_move((0.0, 0.0, 0.2), 0.0, target=(0.0, 0.0, 0.0))

        self.assertEqual(MOVING, status.observe((0.0, 0.0, 0.0), 0.05))
        self.assertEqual(MOVING, status.observe((0.0, 0.0, 0.0), 0.10))
        self.assertEqual(IDLE, status.observe((0.0, 0.0, 0.0), 0.16))

    def test_target_uses_pan_wrap_and_component_tolerances(self):
        status = self.make_status()
        status.begin_move(
            (0.5, 0.0, 0.0),
            0.0,
            target=(-0.999, 0.10, 0.20),
            target_tolerance=(0.003, 0.01, 0.01),
        )

        near_target = (0.999, 0.105, 0.205)
        self.assertEqual(MOVING, status.observe(near_target, 0.05))
        self.assertEqual(MOVING, status.observe(near_target, 0.10))
        self.assertEqual(IDLE, status.observe(near_target, 0.16))


if __name__ == "__main__":
    unittest.main()
