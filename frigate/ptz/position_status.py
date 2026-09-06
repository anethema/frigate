"""Position-only completion tracking for PTZ commands.

The caller supplies monotonic timestamps and polls the camera position.  This
module deliberately has no I/O or sleeps, so it is usable by ONVIF code and
straightforward to test.
"""

from __future__ import annotations

import math
from typing import Final, TypeAlias

Position: TypeAlias = tuple[float, float, float]

MOVING: Final = "MOVING"
IDLE: Final = "IDLE"


class PositionStatusError(RuntimeError):
    """Base class for position-status failures."""


class PositionStatusTimeout(PositionStatusError, TimeoutError):
    """A command or its position status remained unavailable for too long."""


class PositionStatus:
    """Conservatively determine whether a position command has completed.

    ``begin_move`` must receive a known position immediately before issuing a
    command.  A regular command can complete only after a displacement from
    that baseline has been observed.  Pass ``allow_no_motion=True`` for an
    intentional no-op (for example, a zero calibration move or a preset that
    is already selected).

    ``observe`` returns only ``MOVING`` or ``IDLE``.  Missing and malformed
    readings are reported as ``MOVING`` so they never produce a false idle.
    Timeout failures are latched until ``reset``, ``reseed``, or ``begin_move``.
    """

    def __init__(
        self,
        *,
        minimum_delta: float = 1e-5,
        startup_grace: float = 0.25,
        stable_time: float = 0.15,
        minimum_stable_readings: int = 3,
        command_timeout: float = 10.0,
        missing_timeout: float = 2.0,
        pan_wrap_period: float | None = 2.0,
    ) -> None:
        if not math.isfinite(minimum_delta) or minimum_delta <= 0:
            raise ValueError("minimum_delta must be finite and positive")
        if not math.isfinite(startup_grace) or startup_grace < 0:
            raise ValueError("startup_grace must be finite and non-negative")
        if not math.isfinite(stable_time) or stable_time < 0:
            raise ValueError("stable_time must be finite and non-negative")
        if minimum_stable_readings < 3:
            raise ValueError("minimum_stable_readings must be at least 3")
        if not math.isfinite(command_timeout) or command_timeout <= 0:
            raise ValueError("command_timeout must be finite and positive")
        if not math.isfinite(missing_timeout) or missing_timeout <= 0:
            raise ValueError("missing_timeout must be finite and positive")
        if pan_wrap_period is not None and (
            not math.isfinite(pan_wrap_period) or pan_wrap_period <= 0
        ):
            raise ValueError("pan_wrap_period must be finite and positive or None")

        self.minimum_delta = minimum_delta
        self.startup_grace = startup_grace
        self.stable_time = stable_time
        self.minimum_stable_readings = minimum_stable_readings
        self.command_timeout = command_timeout
        self.missing_timeout = missing_timeout
        self.pan_wrap_period = pan_wrap_period
        self.reset()

    def reset(self) -> None:
        """Forget command state and clear a latched failure."""
        self._active = False
        self._fault: PositionStatusTimeout | None = None
        self._state = MOVING
        self._baseline: Position | None = None
        self._target: Position | None = None
        self._target_tolerance: Position | None = None
        self._command_at: float | None = None
        self._allow_no_motion = False
        self._last_position: Position | None = None
        self._last_sample_at: float | None = None
        self._missing_since: float | None = None
        self._motion_seen = False
        self._stable_anchor: Position | None = None
        self._stable_since: float | None = None
        self._stable_readings = 0

    def reseed(self, position: Position, now: float) -> str:
        """Reset after a reconnect and seed a passive, known-idle sample."""
        self.reset()
        valid_position = self._require_position(position)
        valid_now = self._require_time(now)
        self._baseline = valid_position
        self._last_position = valid_position
        self._last_sample_at = valid_now
        self._state = IDLE
        return self._state

    def begin_move(
        self,
        position: Position,
        now: float,
        *,
        allow_no_motion: bool = False,
        target: Position | None = None,
        target_tolerance: Position = (1e-4, 1e-4, 1e-4),
    ) -> str:
        """Seed a pre-command baseline and mark the just-issued command moving."""
        baseline = self._require_position(position)
        command_at = self._require_time(now)
        command_target = self._require_position(target) if target is not None else None
        tolerance = self._require_tolerance(target_tolerance)
        self.reset()
        self._active = True
        self._baseline = baseline
        self._target = command_target
        self._target_tolerance = tolerance
        self._command_at = command_at
        self._allow_no_motion = allow_no_motion
        self._last_position = baseline
        self._last_sample_at = command_at
        # A no-motion command can use the baseline as its stability anchor,
        # but both its interval and required readings begin after command issue.
        if allow_no_motion:
            self._stable_anchor = baseline
        self._state = MOVING
        return self._state

    @property
    def state(self) -> str:
        """Most recent conservative state; a reset begins as ``MOVING``."""
        return self._state

    def observe(self, position: Position | None, now: float) -> str:
        """Consume one poll result at ``now`` and return the conservative state."""
        self._raise_if_fault()
        observed_at = self._require_time(now)
        self._check_time_progress(observed_at)

        if self._active:
            self._check_command_timeout(observed_at)

        valid_position = self._coerce_position(position)
        if valid_position is None:
            return self._unavailable(observed_at)

        self._missing_since = None

        if not self._active:
            # The first usable passive sample establishes that the camera is
            # currently readable; it is not evidence of a command in flight.
            self._baseline = valid_position
            self._last_position = valid_position
            self._last_sample_at = observed_at
            self._state = IDLE
            return self._state

        if self._last_sample_at is not None and observed_at <= self._last_sample_at:
            # Replayed samples cannot satisfy the fresh-reading requirement.
            self._state = MOVING
            return self._state

        previous = self._last_position
        self._last_position = valid_position
        self._last_sample_at = observed_at

        if self._baseline is not None and self._moved_from(self._baseline, valid_position):
            self._motion_seen = True

        if previous is not None and self._moved_from(previous, valid_position):
            self._start_stability(valid_position, observed_at)
        elif self._stable_anchor is None:
            # Normal commands do not start the stable window before observed
            # movement.  An explicitly allowed no-op may restart its stable
            # window after a missing/invalid poll cleared prior stability.
            if self._motion_seen or self._allow_no_motion:
                self._start_stability(valid_position, observed_at)
        elif self._moved_from(self._stable_anchor, valid_position):
            # Individually tiny encoder steps may add up.  Restart stability
            # when their cumulative displacement crosses the movement threshold.
            self._start_stability(valid_position, observed_at)
        elif self._stable_since is None:
            # For an allowed no-op, the first post-command sample starts the
            # stability interval.  The baseline itself is not a fresh reading.
            self._stable_since = observed_at
            self._stable_readings = 1
        else:
            self._stable_readings += 1

        if not self._motion_seen and not self._allow_no_motion:
            self._state = MOVING
            return self._state

        if self._can_complete(valid_position, observed_at):
            self._active = False
            self._state = IDLE
            return self._state
        self._state = MOVING
        return self._state

    def failure(self, now: float) -> str:
        """Record an unavailable status response (an alias for ``observe(None)``)."""
        return self.observe(None, now)

    def _unavailable(self, now: float) -> str:
        self._reset_stability()
        self._state = MOVING
        if self._active:
            if self._missing_since is None:
                self._missing_since = now
            elif now - self._missing_since >= self.missing_timeout:
                self._latch_timeout("position status was unavailable for too long")
        return self._state

    def _check_command_timeout(self, now: float) -> None:
        assert self._command_at is not None
        if now - self._command_at >= self.command_timeout:
            self._latch_timeout("position command did not complete before its timeout")

    def _check_time_progress(self, now: float) -> None:
        reference_times = (self._command_at, self._last_sample_at, self._missing_since)
        for reference in reference_times:
            if reference is not None and now < reference:
                raise ValueError("timestamps must be monotonic")

    def _start_stability(self, position: Position, now: float) -> None:
        self._stable_anchor = position
        self._stable_since = now
        self._stable_readings = 1

    def _reset_stability(self) -> None:
        self._stable_anchor = None
        self._stable_since = None
        self._stable_readings = 0

    def _can_complete(self, position: Position, now: float) -> bool:
        if self._stable_since is None:
            return False
        if not self._target_matches(position):
            return False
        command_at = self._command_at
        assert command_at is not None
        return (
            now - command_at >= self.startup_grace
            and now - self._stable_since >= self.stable_time
            and self._stable_readings >= self.minimum_stable_readings
        )

    def _moved_from(self, left: Position, right: Position) -> bool:
        pan_delta = self._pan_distance(left[0], right[0])
        return (
            pan_delta >= self.minimum_delta
            or abs(left[1] - right[1]) >= self.minimum_delta
            or abs(left[2] - right[2]) >= self.minimum_delta
        )

    def _target_matches(self, position: Position) -> bool:
        if self._target is None:
            return True
        assert self._target_tolerance is not None
        return (
            self._pan_distance(position[0], self._target[0])
            <= self._target_tolerance[0]
            and abs(position[1] - self._target[1]) <= self._target_tolerance[1]
            and abs(position[2] - self._target[2]) <= self._target_tolerance[2]
        )

    def _pan_distance(self, left: float, right: float) -> float:
        pan_delta = abs(left - right)
        if self.pan_wrap_period is not None:
            pan_delta = pan_delta % self.pan_wrap_period
            pan_delta = min(pan_delta, self.pan_wrap_period - pan_delta)
        return pan_delta

    def _latch_timeout(self, message: str) -> None:
        self._fault = PositionStatusTimeout(message)
        self._state = MOVING
        raise self._fault

    def _raise_if_fault(self) -> None:
        if self._fault is not None:
            raise self._fault

    @staticmethod
    def _require_time(now: float) -> float:
        if not isinstance(now, (int, float)) or isinstance(now, bool) or not math.isfinite(now):
            raise ValueError("now must be a finite numeric monotonic timestamp")
        return float(now)

    @staticmethod
    def _coerce_position(position: object) -> Position | None:
        if not isinstance(position, tuple) or len(position) != 3:
            return None
        if any(
            not isinstance(component, (int, float))
            or isinstance(component, bool)
            or not math.isfinite(component)
            for component in position
        ):
            return None
        return (float(position[0]), float(position[1]), float(position[2]))

    def _require_position(self, position: object) -> Position:
        valid_position = self._coerce_position(position)
        if valid_position is None:
            raise ValueError("position must be a tuple of three finite floats")
        return valid_position

    @staticmethod
    def _require_tolerance(tolerance: object) -> Position:
        valid_tolerance = PositionStatus._coerce_position(tolerance)
        if valid_tolerance is None or any(component <= 0 for component in valid_tolerance):
            raise ValueError("target_tolerance must be a tuple of three finite positive floats")
        return valid_tolerance
