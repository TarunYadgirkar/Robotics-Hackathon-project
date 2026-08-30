"""Abort semantics shared by every backend.

The rule that matters: an interrupt FREEZES AND HOLDS. It never homes. Homing is
itself autonomous motion, and the moment someone hits the stop key is the worst
possible moment to start moving again — the usual reason for hitting it is that
a hand is in the way. Recovery therefore requires a second, explicit keypress,
and the homing move that follows runs at LOW_SPEED_FRACTION.
"""

import sys
import time

from . import model


class MotionAborted(RuntimeError):
    pass


class ArmFrozen(RuntimeError):
    """Raised when motion is requested while the arm is latched in a frozen state."""


_frozen_at: tuple[float, ...] | None = None


def is_frozen() -> bool:
    return _frozen_at is not None


def frozen_pose() -> tuple[float, ...] | None:
    return _frozen_at


def require_not_frozen() -> None:
    if is_frozen():
        raise ArmFrozen(
            "arm is frozen after an abort. Motion is refused until "
            "arm_io.recover_home() is called and the operator confirms at the keyboard."
        )


def latch_freeze(positions: tuple[float, ...]) -> None:
    global _frozen_at
    _frozen_at = tuple(positions)


def clear_freeze() -> None:
    global _frozen_at
    _frozen_at = None


def confirm_home_keypress(prompt_stream=sys.stderr) -> bool:
    """Explicit second keypress before any post-abort homing motion.

    A non-interactive stdin is NOT consent: it returns False rather than
    assuming yes.
    """
    if not sys.stdin.isatty():
        print(
            "[SAFETY] stdin is not a terminal; refusing to home without an operator keypress.",
            file=prompt_stream,
        )
        return False
    print(
        "[SAFETY] arm is frozen and holding. Press 'h' then Enter to home at "
        f"{int(model.LOW_SPEED_FRACTION * 100)}% speed, anything else to stay frozen: ",
        file=prompt_stream,
        end="",
        flush=True,
    )
    try:
        answer = sys.stdin.readline().strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer == "h"


def run_motion(setpoints, send, hold, realtime: bool = True, on_step=None) -> tuple[float, ...]:
    """Stream capped setpoints, freezing and holding on KeyboardInterrupt.

    `send(t, positions)` commands one setpoint; `hold(positions)` is the
    backend's stop-and-hold. Returns the final commanded pose.
    """
    require_not_frozen()
    last = setpoints[0][1]
    started = time.perf_counter()
    try:
        for t, positions in setpoints:
            if realtime:
                delay = (started + t) - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            send(t, positions)
            if on_step is not None:
                on_step(t, positions)
            last = positions
    except KeyboardInterrupt:
        hold(last)
        latch_freeze(last)
        pose = ", ".join(
            f"{name}={value:.1f}" for name, value in zip(model.JOINT_NAMES, last)
        )
        print(
            f"\n[SAFETY] interrupt: FROZEN AND HOLDING at {pose}. Not homing.",
            file=sys.stderr,
        )
        raise MotionAborted("motion aborted by operator; arm frozen and holding") from None
    return last
