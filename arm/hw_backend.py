"""Hardware backend for the in-repo YAM arm — STUB. Never run against a robot.

FACTS.md at build time: HARDWARE_PRESENT: no (no CANable on USB). `yam.arm`
imports `can` and opens a gs_usb bus at construction, so nothing from `yam` is
imported at module scope here — the import lives inside connect(), which is why
`import arm.arm_io` works fine on a machine with no CAN stack at all.

Everything below is written against the REAL signatures in yam/arm.py
(YamArm.enable/disable/read_state/command_positions/clear_errors/
recover_stale_motors, ArmState.positions in radians, SafetyLimits) rather than
guessed, but no call has been executed. The unimplemented bodies are the four
places a human has to look at the arm while typing.

Bringing this up at the venue:
  1. plug in the CANable, re-probe, edit coordination/FACTS.md:
       HARDWARE_PRESENT: yes
       HARDWARE_DEVICE_PATH: <the gs_usb channel, or 'gs_usb:0'>
  2. run the pre-flight in preflight_checklist() — in particular the
     self-collision check, which is NOT optional: yam/arm.py records that ~10%
     of in-limit poses self-collide and that the driver does not check.
  3. fill in connect(), current_positions(), send(), hold() below.
Nothing in arm_io.py, motion.py, safety.py or the gesture JSONs changes.
"""

import sys

from . import facts, model

_UNIMPLEMENTED = (
    "YAM hardware backend is a stub. FACTS.md said HARDWARE_PRESENT: {flag!r} when arm/ "
    "was written, so no call to yam.arm was ever executed. Implement {what} in "
    "arm/hw_backend.py before commanding a real arm."
)


class YamLimitMismatch(RuntimeError):
    pass


def verify_against_yam() -> None:
    """Fail loudly if arm/model.py has drifted from yam.arm's real config.

    model.py mirrors yam.arm.ARM_JOINTS in degrees so trajectories stay readable.
    A mirror is a liability unless something checks it, so this runs at connect
    time, before any motion.
    """
    import math

    from yam.arm import ARM_JOINTS, SafetyLimits

    if [j.name for j in ARM_JOINTS] != list(model.ARM_JOINT_NAMES):
        raise YamLimitMismatch(
            f"yam.arm joint names {[j.name for j in ARM_JOINTS]} != model.ARM_JOINT_NAMES "
            f"{list(model.ARM_JOINT_NAMES)}"
        )
    for joint in ARM_JOINTS:
        lo, hi = model.LIMITS_DEG[joint.name]
        if abs(math.degrees(joint.lower_limit) - lo) > 1e-3 or abs(math.degrees(joint.upper_limit) - hi) > 1e-3:
            raise YamLimitMismatch(
                f"{joint.name}: yam.arm limits "
                f"[{math.degrees(joint.lower_limit):.3f}, {math.degrees(joint.upper_limit):.3f}] deg "
                f"!= model.LIMITS_DEG [{lo:.3f}, {hi:.3f}] — update arm/model.py, do not widen it here"
            )
    step = SafetyLimits().max_step_per_tick
    if abs(step - model.YAM_MAX_STEP_PER_TICK_RAD) > 1e-9:
        raise YamLimitMismatch(
            f"yam.arm SafetyLimits.max_step_per_tick={step} != model.YAM_MAX_STEP_PER_TICK_RAD="
            f"{model.YAM_MAX_STEP_PER_TICK_RAD}; the velocity cap is derived from it"
        )


def verify_cap_against_slew(control_hz: float) -> None:
    """The driver clamps each command by max_step_per_tick, per CALL not per second.

    So the streaming rate is part of the safety arithmetic: a capped setpoint
    stream sent too slowly is silently rate-limited by the driver and the arm
    lags the schedule. Checked before motion rather than discovered during it.
    """
    import math

    driver_ceiling = math.degrees(model.YAM_MAX_STEP_PER_TICK_RAD) * control_hz
    worst = max(model.velocity_cap_deg_s()[j] for j in model.ARM_JOINT_NAMES)
    if worst > driver_ceiling:
        raise YamLimitMismatch(
            f"velocity cap {worst:.1f} deg/s exceeds what the driver can track at {control_hz:.0f} Hz "
            f"({driver_ceiling:.1f} deg/s). Raise the streaming rate or lower VELOCITY_CAP_FRACTION."
        )


def preflight_checklist() -> list[str]:
    """What a human must do before this backend is allowed to move anything."""
    return [
        "CANable attached and FACTS.md updated (HARDWARE_PRESENT: yes)",
        "uv pip install python-can gs_usb (yam.can_compat patches gs_usb for macOS)",
        "every pose in arm/gestures/*.json passed through yam.environment.ArmSafetyChecker "
        "(needs mujoco + the i2rt URDF, neither of which is on this machine) — ~10% of "
        "in-limit poses self-collide and yam.arm does not check",
        "joint2 stays below %.0f deg (self-collides against the base past ~%.0f)"
        % (model.JOINT2_AUTHORED_CEILING_DEG, model.JOINT2_SELF_COLLISION_DEG),
        "keep-alive stream running between motions: yam.arm measured a 0xD comms-lost latch "
        "at 20 Hz with 0.3-0.4s gaps, and the demo waits on human keypresses between beats",
        "gripper left alone: yam.arm.ARM_JOINTS excludes it (uncalibrated jaw travel), so the "
        "sim-only gripper channel of each trajectory is dropped on hardware",
        "e-stop within reach of the operator; freeze-and-hold is software, not a substitute",
    ]


class YamBackend:
    name = "yam-hardware"
    is_hardware = True

    def __init__(self) -> None:
        self.port = facts.device_path()
        self._arm = None

    def _todo(self, what: str) -> NotImplementedError:
        return NotImplementedError(_UNIMPLEMENTED.format(flag=facts.hardware_flag_raw(), what=what))

    def control_hz(self) -> float:
        return model.YAM_TICK_HZ

    def realtime(self) -> bool:
        return True

    def connect(self) -> None:
        try:
            from yam.arm import SafetyLimits, YamArm  # noqa: F401
        except ImportError as exc:
            raise self._todo(f"connect() — yam.arm import failed ({exc}); install python-can + gs_usb") from exc
        verify_against_yam()
        verify_cap_against_slew(self.control_hz())
        print("[HW] pre-flight, all of it manual:", file=sys.stderr)
        for item in preflight_checklist():
            print(f"[HW]   - {item}", file=sys.stderr)
        raise self._todo(
            "connect(): YamArm(safety=SafetyLimits(gain_scale=...)), then recover_stale_motors(), "
            "then enable(), then start the keep-alive stream"
        )

    def current_positions(self) -> tuple[float, ...]:
        """yam.arm.ArmState.positions is 6 radians; the 7th channel is the sim-only gripper."""
        raise self._todo(
            "current_positions(): math.degrees each of self._arm.read_state().positions, then "
            "append the last commanded gripper percent (the gripper is not on the bus)"
        )

    def send(self, t: float, positions: tuple[float, ...]) -> None:
        violations = model.check_limits(positions)
        if violations:
            raise RuntimeError("refusing to command out-of-limit pose: " + "; ".join(violations))
        if positions[1] > model.JOINT2_AUTHORED_CEILING_DEG:
            raise RuntimeError(
                f"refusing joint2={positions[1]:.1f} deg: above the authored ceiling of "
                f"{model.JOINT2_AUTHORED_CEILING_DEG} deg and heading toward the ~"
                f"{model.JOINT2_SELF_COLLISION_DEG} deg self-collision with the base"
            )
        raise self._todo(
            "send(): self._arm.command_positions(model.to_yam_radians(positions)) — six joints only, "
            "gripper channel dropped"
        )

    def hold(self, positions: tuple[float, ...]) -> None:
        print("[HW] FREEZE: keep streaming the last commanded pose; do NOT disable "
              "(a disabled motor drops the arm) and do NOT home", file=sys.stderr)
        raise self._todo(
            "hold(): stream model.to_yam_radians(positions) through command_positions() at "
            "YAM_TICK_HZ until the operator confirms recovery. yam.arm.YamArm.hold(targets, "
            "duration, rate_hz) does exactly this with a gain ramp, but it returns after "
            "`duration` — a freeze has to hold until told otherwise"
        )

    def begin_motion(self, label, traj, setpoints, report) -> None:
        print(f"[HW] {label}: {traj.duration:.2f}s, {len(setpoints)} setpoints @ {self.control_hz():.0f} Hz")
        for line in report:
            print(f"[HW] velocity cap applied: {line}")

    def end_motion(self, label, traj, setpoints) -> None:
        pass


def teach(name: str) -> None:
    """Record waypoints by hand-positioning the arm. Hardware-only, unimplemented.

    Procedure once the bus is up: YamArm.disable() so the joints back-drive, move
    the arm to each pose, capture read_state().positions (radians) on a keypress,
    convert to degrees and write the same JSON schema arm/gestures/*.json uses
    with source="taught-on-hardware". Note read_state() commands zero gains, so
    reading never applies torque — but the motors still need a command stream to
    avoid the 0xD latch.
    """
    raise NotImplementedError(
        f"teach({name!r}) needs a connected arm; FACTS.md says HARDWARE_PRESENT: "
        f"{facts.hardware_flag_raw()!r}. The gesture JSONs in arm/gestures/ are sim-authored "
        "and labelled as such in their 'source' field."
    )
