"""Hardware backend for the in-repo YAM arm, over Boris's yam.arm.YamArm.

FACTS.md now says HARDWARE_PRESENT: yes (CANable2 gs_usb, VID 0x1D50 PID 0x606F).
`yam.arm` imports `can` and opens a gs_usb bus at construction, so nothing from
`yam` is imported at module scope here — connect() does it, which keeps
`import arm.arm_io` working on a machine with no CAN stack.

Three things in here are not obvious and are load-bearing:

1. **The bus has exactly one owner, and it alternates.** Between motions,
   hwsupport.keepalive.Keepalive (Agent H1) holds the pose at 100 Hz with a
   single-pass 0xD recovery policy. During a motion, the keepalive is paused —
   pause() is a real barrier, it returns only once its in-flight exchange has
   landed — and this module's own streamer drives the setpoints. The two are
   never on the bus at once, which is H1's explicit caveat.

   Why two streamers rather than just the keepalive: Keepalive holds a pose it
   re-reads, it is not a setpoint driver, and driving motion by calling
   command_positions() from the motion thread makes each send a ~12ms bus
   transaction. Paced at 100 Hz that stretched an 18s gesture to 86s on the real
   arm. Decoupling — motion thread updates a target, streamer ships it — measured
   1.00x on every motion.

2. **We do not use `yam.arm.connected_arm`.** That helper disables the motors on
   SIGINT, which drops the arm limp. Our contract is freeze-and-hold: on
   interrupt the streamer keeps holding the last commanded pose and homing needs
   a second explicit keypress. Disable happens on a clean shutdown instead.

3. **Hardware mode is gripper-first.** The operator's rule at the arm is that the
   arm stays essentially still and the motion is the jaws. That is enforced here
   against the prepared setpoint stream before anything is sent, so replaying a
   sim trajectory on hardware is refused rather than swept through space.
"""

import math
import sys
import threading
import time

from . import facts, model


class YamLimitMismatch(RuntimeError):
    pass


class HardwareMotionRefused(RuntimeError):
    pass


class ArmUnavailable(RuntimeError):
    """The arm cannot be commanded and one automatic recovery has already failed.

    This is the ONE exception the demo layer needs to catch to keep the show
    running without motion. It is raised within a bounded time (see
    RECOVERY_BUDGET_S) — never after a hang — and it is terminal for the session:
    by the time it is raised, the in-place recovery H2's triage prescribes has
    been tried once and did not work, so the remaining fixes are physical.
    """


#: Kept as an alias so older call sites and status files still resolve. Any
#: "the arm is not usable" condition is one exception now.
ArmUnhealthy = ArmUnavailable

#: Hard ceiling on one recovery attempt. The demo cannot wait on a dead bus, so
#: this is deliberately shorter than any human would tolerate on stage.
RECOVERY_BUDGET_S = 5.0


def verify_against_yam() -> None:
    """Fail loudly if arm/model.py has drifted from yam.arm's real config.

    model.py mirrors yam.arm in degrees so trajectories stay readable. A mirror
    is a liability unless something checks it: this caught Boris's switch from a
    per-tick slew clamp to a rad/s speed ceiling on the first hardware run.
    """
    from yam.arm import ARM_JOINTS, GRIPPER_JOINT, SafetyLimits

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

    limits = SafetyLimits()
    if abs(limits.max_joint_speed - model.YAM_MAX_JOINT_SPEED_RAD_S) > 1e-9:
        raise YamLimitMismatch(
            f"yam.arm SafetyLimits.max_joint_speed={limits.max_joint_speed} != "
            f"model.YAM_MAX_JOINT_SPEED_RAD_S={model.YAM_MAX_JOINT_SPEED_RAD_S}; the velocity cap "
            "is derived from it"
        )
    if abs(limits.max_step_per_tick - model.YAM_MAX_STEP_PER_TICK_RAD) > 1e-9:
        raise YamLimitMismatch(
            f"yam.arm SafetyLimits.max_step_per_tick={limits.max_step_per_tick} != "
            f"model.YAM_MAX_STEP_PER_TICK_RAD={model.YAM_MAX_STEP_PER_TICK_RAD}"
        )

    # The gripper channel is a percent mapping onto Boris's measured stops.
    for percent in (0.0, 100.0):
        rad = model.gripper_percent_to_rad(percent)
        if not (GRIPPER_JOINT.lower_limit - 0.06) <= rad <= (GRIPPER_JOINT.upper_limit + 0.06):
            raise YamLimitMismatch(
                f"gripper {percent:.0f}% maps to {rad:.3f} rad, outside yam.arm's calibrated "
                f"[{GRIPPER_JOINT.lower_limit:.3f}, {GRIPPER_JOINT.upper_limit:.3f}] plus its 50mrad backoff"
            )


def verify_cap_against_driver(control_hz: float) -> None:
    """Our cap must sit under BOTH of yam.arm's ceilings at the rate we stream."""
    cap = max(model.velocity_cap_deg_s()[j] for j in model.ARM_JOINT_NAMES)
    speed_ceiling = math.degrees(model.YAM_MAX_JOINT_SPEED_RAD_S)
    if cap > speed_ceiling:
        raise YamLimitMismatch(
            f"velocity cap {cap:.1f} deg/s exceeds yam.arm's max_joint_speed {speed_ceiling:.1f} deg/s"
        )
    per_tick = math.radians(cap) / control_hz
    if per_tick > model.YAM_MAX_STEP_PER_TICK_RAD:
        raise YamLimitMismatch(
            f"at {control_hz:.0f} Hz our cap needs {per_tick:.4f} rad/tick, above yam.arm's "
            f"anti-lunge ceiling {model.YAM_MAX_STEP_PER_TICK_RAD} rad — stream faster or cap lower"
        )


def validate_hardware_motion(setpoints, label: str) -> dict:
    """Enforce the operator's rule: arm essentially still, motion is the gripper.

    Checked on the whole prepared stream before the first frame goes out.
    """
    start = setpoints[0][1]
    worst_excursion = {name: 0.0 for name in model.ARM_JOINT_NAMES}
    for _, positions in setpoints:
        for i, name in enumerate(model.ARM_JOINT_NAMES):
            worst_excursion[name] = max(worst_excursion[name], abs(positions[i] - start[i]))

    # joint1 first and on its own: this one is not a tunable budget, it is a
    # physical constraint the collision model cannot see (clamps on the sides of
    # the base). Checked explicitly so it cannot be widened by a config edit or
    # loosened by a checker PASS.
    if worst_excursion["joint1"] > model.HW_JOINT1_LOCKED_DEG:
        raise HardwareMotionRefused(
            f"{label}: refused — joint1 would rotate {worst_excursion['joint1']:.1f} deg. The base "
            f"is LOCKED to {model.HW_JOINT1_LOCKED_DEG} deg: there are clamps on the sides of the "
            "base that are not in the collision model, and the operator's instruction is that the "
            "base does not turn. Express the gesture with joint2-joint6 and the gripper instead."
        )

    offenders = {
        n: v for n, v in worst_excursion.items() if v > model.hw_excursion_limit(n)
    }
    if offenders:
        detail = ", ".join(
            f"{n} moves {v:.1f} deg (limit {model.hw_excursion_limit(n):.0f})"
            for n, v in offenders.items()
        )
        raise HardwareMotionRefused(
            f"{label}: refused on hardware — {detail}. Hardware gestures live in "
            "arm/gestures/hardware/; this looks like a sim trajectory."
        )

    peak_arm_speed = 0.0
    fastest_joint = ""
    for (t0, q0), (t1, q1) in zip(setpoints, setpoints[1:]):
        dt = t1 - t0
        if dt <= 0:
            continue
        for i, name in enumerate(model.ARM_JOINT_NAMES):
            speed = abs(q1[i] - q0[i]) / dt
            if speed > peak_arm_speed:
                peak_arm_speed, fastest_joint = speed, name
    # tolerance: the cap time-dilates a segment to exactly this ceiling, and the
    # result lands a few ULPs above it. Refusing that is a float bug, not a safety win.
    if peak_arm_speed > model.HW_SLOW_SPEED_DEG_S * (1 + 1e-6):
        raise HardwareMotionRefused(
            f"{label}: refused on hardware — {fastest_joint} would run at {peak_arm_speed:.1f} deg/s, "
            f"above the {model.HW_SLOW_SPEED_DEG_S} deg/s slow-motion ceiling for this arm"
        )

    peak_gripper = 0.0
    for (t0, q0), (t1, q1) in zip(setpoints, setpoints[1:]):
        dt = t1 - t0
        if dt > 0:
            peak_gripper = max(peak_gripper, abs(q1[-1] - q0[-1]) / dt)
    if peak_gripper > model.HW_GRIPPER_GENTLE_PCT_S * (1 + 1e-6):
        raise HardwareMotionRefused(
            f"{label}: refused on hardware — gripper would move at {peak_gripper:.1f} %/s, above the "
            f"gentle limit of {model.HW_GRIPPER_GENTLE_PCT_S} %/s"
        )

    for _, positions in setpoints:
        violations = model.check_limits(positions, base=start)
        if violations:
            raise HardwareMotionRefused(f"{label}: " + "; ".join(violations))

    self_collision_note = verify_self_collision_free(setpoints, label)

    return {
        "self_collision": self_collision_note,
        "worst_arm_excursion_deg": max(worst_excursion.values()),
        "peak_arm_speed_deg_s": peak_arm_speed,
        "per_joint_excursion_deg": worst_excursion,
        "peak_gripper_pct_s": peak_gripper,
        "gripper_range_pct": (
            min(p[-1] for _, p in setpoints),
            max(p[-1] for _, p in setpoints),
        ),
    }


_SELF_CHECKER = None
_SELF_CHECKER_TRIED = False
#: Every Nth setpoint is collision-checked. At 100 Hz that is a check every
#: 100ms of motion, and consecutive setpoints differ by far less than the
#: checker's own margin, so nothing meaningful slips between samples.
SELF_CHECK_STRIDE = 10


def _self_collision_checker():
    """MuJoCo self-collision checker over the official i2rt model, or None.

    Built once per process. None means the model files are not present, which is
    reported loudly rather than silently downgrading to "assumed safe".
    """
    global _SELF_CHECKER, _SELF_CHECKER_TRIED
    if _SELF_CHECKER_TRIED:
        return _SELF_CHECKER
    _SELF_CHECKER_TRIED = True

    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "hwresearch/i2rt/robot_models/arm/yam_pro/v1"
    urdf, xml = root / "yam_pro.urdf", root / "yam_pro.xml"
    if not (urdf.exists() and xml.exists()):
        print(f"[HW] WARNING: {root} missing — self-collision checking is UNAVAILABLE for this "
              "run; only the excursion and speed rules apply", file=sys.stderr)
        return None
    try:
        from yam.collision import World
        from yam.mujoco_collision import MujocoCollisionChecker

        _SELF_CHECKER = MujocoCollisionChecker(
            str(xml), World(obstacles=[], ground_z=None, margin=0.0), self_collision_margin=0.003
        )
    except Exception as exc:
        print(f"[HW] WARNING: self-collision checker unavailable ({exc}); only the excursion and "
              "speed rules apply", file=sys.stderr)
        _SELF_CHECKER = None
    return _SELF_CHECKER


def verify_self_collision_free(setpoints, label: str) -> str:
    """Refuse any motion that would fold the arm into itself.

    Runs on the prepared setpoint stream, so it covers the interpolation and not
    just the authored waypoints — the segments between poses are where a
    self-collision actually happens.
    """
    checker = _self_collision_checker()
    if checker is None:
        return "self-collision NOT checked (model files absent)"

    import math

    sampled = list(setpoints[::SELF_CHECK_STRIDE]) + [setpoints[-1]]
    for t, positions in sampled:
        q = [math.radians(v) for v in positions[:6]]
        if not checker.is_free(q):
            raise HardwareMotionRefused(
                f"{label}: refused — SELF-COLLISION at t={t:.2f}s. "
                + "; ".join(checker.explain(q))
            )
    return f"self-collision-free at {len(sampled)} sampled poses (official i2rt model)"


def read_passive(arm) -> list:
    """Read every joint without enabling anything and without applying torque.

    `YamArm.read_state()` sends the same zero-gain frames but routes through
    `_check_faults`, which raises on a DISABLED motor (error 0x0 is not 0x1).
    That is right for a control loop and wrong for a pre-enable probe, so this
    reads the raw feedback and reports the error word instead of raising on it.
    """
    from yam.dm_motor import encode_mit_command

    out = []
    for joint in arm.joints:
        feedback = arm._exchange(
            joint, encode_mit_command(joint.spec, position=0.0, kp=0.0, kd=0.0, torque=0.0)
        )
        out.append((joint, feedback))
    return out


def preflight_checklist() -> list[str]:
    """What must be true before this backend moves anything. Printed for the operator."""
    return [
        "CANable2 attached and FACTS.md amended (HARDWARE_PRESENT: yes) — verified",
        "python-can + gs_usb + pyusb installed in .venv — verified",
        "arm/model.py mirror matches yam.arm limits and SafetyLimits — verify_against_yam()",
        "SELF-COLLISION IS VERIFIED, not assumed: MuJoCo against the exact convex meshes of the "
        "official i2rt yam_pro model (github.com/i2rt-robotics/i2rt, fetched by Agent R1 into "
        "hwresearch/), run automatically over every prepared setpoint stream before it is sent, "
        "and over every gesture up front by arm/verify_poses.py",
        "BASE CLAMPS are checked only RELATIVELY: yam.mapping.base_clamps() describes them by "
        "hand, but the environment check uses link-sized spheres (up to 102mm) against a 30mm "
        "margin and so calls the arm's own folded resting pose a collision. Gestures are verified "
        "to come no CLOSER to the clamps than rest — a regression test, not proof of clearance",
        f"joint2 kept to {model.HW_PER_JOINT_EXCURSION_DEG['joint2']:.0f} deg against its "
        f"~{model.JOINT2_SELF_COLLISION_DEG:.0f} deg base-collision trap; joint3 allowed "
        f"{model.HW_PER_JOINT_EXCURSION_DEG['joint3']:.0f} deg because yam.arm records it as "
        "self-collision-free across its whole range from home",
        "NOT modelled: everything else on the table. The human watching is the authority on the "
        "rest of the workspace",
        "0xD comms-lost latch handled: recover_stale_motors() before enable, then a 100 Hz "
        "keep-alive stream for as long as the arm is enabled, including between demo beats",
        f"gripper is Boris's calibrated joint (max_torque 0.6 Nm ~ 41 N) and is commanded at "
        f"<= {model.HW_GRIPPER_GENTLE_PCT_S} %/s with gain_scale {model.HW_GAIN_SCALE}",
        "motion runs when its demo beat runs (operator's decision: the demo is the robot trying "
        "something, not a human driving it with keys). Ctrl-C remains the e-stop and freezes-and-holds",
        "one recovery pass only: recover_stale_motors()/clear_errors() before enable, and if a motor "
        "is still unhealthy after that, STOP and report rather than clearing into it again",
        "e-stop / power switch within the operator's reach — freeze-and-hold is software",
    ]


class YamBackend:
    name = "yam-hardware"
    is_hardware = True

    def __init__(self) -> None:
        self.port = facts.device_path()
        self._arm = None
        # Two locks on purpose. _bus_lock is held for the ~8ms of an actual CAN
        # exchange; _target_lock is held for a list assignment. Sharing one lock
        # made send() queue behind a bus transaction, and a motion scheduled for
        # 18s took 86s on the real arm because every one of its 1801 setpoints
        # waited for the streamer's tick.
        self._bus_lock = threading.Lock()
        self._target_lock = threading.Lock()
        self._target_rad: list[float] | None = None
        self._gripper_pct = 0.0
        self._streamer: threading.Thread | None = None
        self._streaming = False
        self._keepalive = None
        self._recovery_used = False
        self._bus_recovery_used = False
        self._fault: BaseException | None = None
        self.validation: dict | None = None
        self._ticks = 0
        self._motion_started = 0.0
        self._ticks_at_start = 0

    # -- lifecycle -----------------------------------------------------------
    def control_hz(self) -> float:
        return model.YAM_TICK_HZ

    def realtime(self) -> bool:
        return True

    def open_bus(self):
        """Open the CAN bus once per process, without enabling anything.

        libusb lets exactly one handle claim the interface: opening a second
        gs_usb bus in the same process — e.g. a read-only probe that closed its
        own bus, then connect() opening a fresh one — fails with `USBError
        [Errno 13] Access denied` on claim_interface. So the backend owns the
        bus for the process lifetime and every caller shares it.
        """
        from yam.arm import ARM_JOINTS, GRIPPER_JOINT, SafetyLimits, YamArm

        if self._arm is None:
            import dataclasses

            # Same calibrated stops, lower squeeze. See HW_GRIPPER_MAX_TORQUE_NM.
            gentle_gripper = dataclasses.replace(
                GRIPPER_JOINT, max_torque=model.HW_GRIPPER_MAX_TORQUE_NM
            )
            self._arm = YamArm(
                joints=list(ARM_JOINTS) + [gentle_gripper],
                safety=SafetyLimits(gain_scale=model.HW_GAIN_SCALE),
            )
        return self._arm

    def probe_passive(self) -> list:
        """Read every joint with no enable and no torque. Safe before connect()."""
        return read_passive(self.open_bus())

    def connect(self) -> None:
        verify_against_yam()
        verify_cap_against_driver(self.control_hz())
        try:
            self._connect_once()
        except Exception as exc:
            if not isinstance(exc, self._bus_error_types()):
                raise
            if self._arm is not None and self._recover_bus_once(exc):
                self._after_enable(self._arm.read_state())
                return
            raise ArmUnavailable(
                f"the arm could not be brought up: {type(exc).__name__}: {exc}. One in-place "
                "recovery was attempted and did not restore it. Check the CANable's USB cable and "
                "the arm's power, then run `.venv/bin/python arm/precheck.py`. The demo can "
                "continue without motion."
            ) from exc

    def _connect_once(self) -> None:
        self.open_bus()
        stale = self._arm.recover_stale_motors()
        if stale:
            print(f"[HW] cleared 0xD comms-lost latch on: {', '.join(stale)}", file=sys.stderr)

        state = self._arm.enable()
        print("[HW] enabled:\n" + state.describe(), file=sys.stderr)
        self._after_enable(state)

    def _after_enable(self, state) -> None:
        self._target_rad = list(state.positions)
        self._gripper_pct = model.gripper_rad_to_percent(state.positions[-1])
        # the pose the arm physically rests in is legal by definition; how far
        # outside a bound it sits varies per power-up (j3 has read -0.054 deg)
        self._rest_channels = tuple(
            math.degrees(r) for r in state.positions[:6]
        ) + (self._gripper_pct,)
        if self._keepalive is not None and self._keepalive.is_running:
            return

        from hwsupport.keepalive import Keepalive

        self._keepalive = Keepalive(
            self._arm, rate_hz=self.control_hz(), on_fault=self._note_fault, name="yam-hold"
        ).start()
        print("[HW] keepalive holding between motions (hwsupport.keepalive, single-pass 0xD recovery)",
              file=sys.stderr)

    def _note_fault(self, exc: BaseException) -> None:
        self._fault = exc

    def _recover_bus_once(self, exc: BaseException) -> bool:
        """One in-place bus recovery, bounded by RECOVERY_BUDGET_S. Never twice.

        Implements H2's triage for "CAN bus dies mid-session": rebuild the bus
        object in place and see whether a FRESH handle gets replies. If it does,
        the host's controller had gone bus-off and the arm was fine all along —
        which on macOS is the failure that masquerades as "the CANable needs a
        replug". If the fresh handle also hears nothing, the arm is genuinely
        unpowered or unplugged and no amount of software fixes it.
        """
        if self._bus_recovery_used or self._arm is None:
            return False
        self._bus_recovery_used = True
        deadline = time.monotonic() + RECOVERY_BUDGET_S
        print(f"\n[HW] {type(exc).__name__}: attempting ONE in-place bus recovery "
              f"(budget {RECOVERY_BUDGET_S:.0f}s)", file=sys.stderr)

        from yam.dm_motor import encode_mit_command

        try:
            self._arm.reconnect()
        except Exception as reconnect_error:
            print(f"[HW] recovery failed: fresh bus would not open ({reconnect_error})", file=sys.stderr)
            return False
        if time.monotonic() > deadline:
            print("[HW] recovery failed: out of budget after reopening the bus", file=sys.stderr)
            return False

        joint = self._arm.joints[0]
        try:
            self._arm._exchange(
                joint, encode_mit_command(joint.spec, position=0.0, kp=0.0, kd=0.0, torque=0.0),
                retries=2,
            )
        except Exception:
            print("[HW] recovery failed: a fresh bus handle hears nothing from the arm — this is "
                  "the arm being unpowered or unplugged, not a host-side bus-off", file=sys.stderr)
            return False
        if time.monotonic() > deadline:
            return False

        try:
            self._arm.recover_stale_motors()
            state = self._arm.enable()
        except Exception as enable_error:
            print(f"[HW] recovery failed at re-enable: {enable_error}", file=sys.stderr)
            return False

        with self._target_lock:
            self._target_rad = list(state.positions)
        print("[HW] recovered: fresh bus answered, motors re-enabled, holding again", file=sys.stderr)
        return True

    @staticmethod
    def _bus_error_types():
        import can
        import usb.core

        from yam.arm import MotorCommunicationError

        return (can.CanError, usb.core.USBError, OSError, MotorCommunicationError)

    def _start_streaming(self) -> None:
        """Take the bus from the keepalive for the duration of one motion."""
        if self._keepalive is not None:
            self._keepalive.pause()  # barrier: returns only once its tick has landed
        self._streaming = True
        self._streamer = threading.Thread(target=self._stream, name="yam-motion", daemon=True)
        self._streamer.start()

    def _stop_streaming(self) -> None:
        """Hand the bus back to the keepalive, which re-reads the pose we left."""
        self._streaming = False
        if self._streamer is not None:
            self._streamer.join(timeout=1.0)
            self._streamer = None
        if self._keepalive is not None and self._keepalive.is_running:
            self._keepalive.resume()

    def _stream(self) -> None:
        """Ships the current target while a motion runs. One recovery pass, then stop.

        Same policy as hwsupport.keepalive: only the benign 0xD comms latch buys a
        pass, and only one. The predicate imports H1's constant rather than
        restating the error text, so it cannot drift from yam.dm_motor.
        """
        from hwsupport.keepalive import COMMS_LOST_TEXT
        from yam.arm import MotorCommunicationError, MotorFaultError

        period = 1.0 / self.control_hz()
        while self._streaming:
            started = time.perf_counter()
            try:
                with self._target_lock:
                    target = list(self._target_rad)
                with self._bus_lock:
                    self._arm.command_positions(target)
                self._ticks += 1
            except (MotorCommunicationError, MotorFaultError) as exc:
                is_comms_latch = isinstance(exc, MotorCommunicationError) or COMMS_LOST_TEXT in str(exc)
                if not is_comms_latch or self._recovery_used:
                    # Before giving up: if this looks like the bus rather than a
                    # motor, spend the one in-place bus recovery. This is what
                    # keeps a mid-demo dropout from ending the demo.
                    if isinstance(exc, self._bus_error_types()) and not self._bus_recovery_used:
                        with self._bus_lock:
                            recovered = self._recover_bus_once(exc)
                        if recovered:
                            continue
                    why = "physical condition" if not is_comms_latch else "recovery pass already spent"
                    self._fault = exc
                    self._streaming = False
                    print(f"\n[HW] STOPPING mid-motion ({why}): {type(exc).__name__}: {exc}",
                          file=sys.stderr)
                    return
                self._recovery_used = True
                print(f"\n[HW] transient {type(exc).__name__}; spending the single recovery pass",
                      file=sys.stderr)
                try:
                    with self._bus_lock:
                        self._arm.recover_stale_motors()
                        self._arm.clear_errors()
                        self._arm.enable()
                except (MotorCommunicationError, MotorFaultError) as recovery_error:
                    self._fault = recovery_error
                    self._streaming = False
                    print(f"[HW] STOPPING: the recovery pass itself failed: {recovery_error}",
                          file=sys.stderr)
                    return
                print("[HW] recovered; continuing the motion", file=sys.stderr)
            delay = period - (time.perf_counter() - started)
            if delay > 0:
                time.sleep(delay)

    def _raise_if_faulted(self) -> None:
        if self._fault is None and self._keepalive is not None and self._keepalive.fault is not None:
            self._fault = self._keepalive.fault
        if self._fault is not None:
            raise ArmUnavailable(
                f"arm reported {type(self._fault).__name__}: {self._fault}. Automatic recovery is "
                "spent, so it is not being retried into. Run `.venv/bin/python arm/precheck.py`; "
                "the demo can continue without motion."
            ) from self._fault

    def shutdown(self) -> None:
        self._streaming = False
        if self._streamer is not None:
            self._streamer.join(timeout=1.0)
            self._streamer = None
        if self._keepalive is not None:
            print("[HW] keepalive: " + self._keepalive.report().replace("\n", " | "), file=sys.stderr)
            self._keepalive.stop()
            self._keepalive = None
        if self._arm is not None:
            self._arm.disable()
            self._arm.close()
            # Deliberately NOT calling usb.util.dispose_resources() here. It was
            # tried, and the adapter dropped off USB entirely later in the same
            # session. Whether it was the trigger is unproven, but it was never
            # needed: the single-bus-per-process rule in open_bus() already
            # solves the double-claim, and yam/can_compat.py patches out the
            # usb.reset() that breaks reopen on macOS. Touching libusb handles
            # by hand here buys nothing and risks the demo's only adapter.
            self._arm = None

    # -- state ---------------------------------------------------------------
    def current_positions(self) -> tuple[float, ...]:
        self._raise_if_faulted()
        with self._bus_lock:
            state = self._arm.read_state()
        return tuple(
            [math.degrees(p) for p in state.positions[:6]]
            + [model.gripper_rad_to_percent(state.positions[-1])]
        )

    # -- motion --------------------------------------------------------------
    def begin_motion(self, label, traj, setpoints, report) -> None:
        self._raise_if_faulted()
        self.validation = validate_hardware_motion(setpoints, label)
        lo, hi = self.validation["gripper_range_pct"]
        print(f"[HW] {label}: {traj.duration:.2f}s, {len(setpoints)} setpoints @ {self.control_hz():.0f} Hz")
        print(f"[HW] hardware-rule check PASSED: arm moves at most "
              f"{self.validation['worst_arm_excursion_deg']:.1f} deg at up to "
              f"{self.validation['peak_arm_speed_deg_s']:.1f} deg/s (limits "
              f"{model.HW_MAX_EXCURSION_DEG:.0f}/{model.HW_SLOW_SPEED_DEG_S:.0f}), "
              f"gripper {lo:.0f}->{hi:.0f}% at up to {self.validation['peak_gripper_pct_s']:.1f} %/s")
        print(f"[HW] {self.validation['self_collision']}")
        for line in report:
            print(f"[HW] velocity cap applied: {line}")
        self._motion_started = time.perf_counter()
        self._ticks_at_start = self._ticks
        self._start_streaming()

    def send(self, t: float, positions: tuple[float, ...]) -> None:
        self._raise_if_faulted()
        violations = model.check_limits(positions, base=getattr(self, "_rest_channels", None))
        if violations:
            raise HardwareMotionRefused("refusing out-of-limit pose: " + "; ".join(violations))
        with self._target_lock:
            self._target_rad = model.to_yam_radians(positions)
        self._gripper_pct = positions[-1]

    def hold(self, positions: tuple[float, ...]) -> None:
        """Freeze: stop advancing the target; the streamer keeps holding it.

        Deliberately NOT a disable — disabling drops the arm — and deliberately
        not a home.
        """
        with self._target_lock:
            self._target_rad = model.to_yam_radians(positions)
        # Hand the frozen pose to the keepalive: it holds it with a gain ramp and
        # keeps its single-pass recovery available, which a stopped motion
        # streamer would not.
        self._stop_streaming()
        print("[HW] FROZEN: keepalive is holding the last commanded pose, motors still enabled, "
              "not homing", file=sys.stderr)

    def end_motion(self, label, traj, setpoints) -> None:
        self._stop_streaming()
        self._raise_if_faulted()
        wall = time.perf_counter() - self._motion_started
        ticks = self._ticks - self._ticks_at_start
        print(f"[HW] {label}: scheduled {traj.duration:.1f}s, wall {wall:.1f}s "
              f"({wall / traj.duration:.2f}x), keep-alive streamed {ticks} frames "
              f"({ticks / wall:.0f} Hz), resyncs={self._arm.resyncs} failures={self._arm.failures}")


def teach(name: str) -> None:
    """Record waypoints by hand-positioning the arm.

    Procedure: YamArm.disable() so the joints back-drive, move the arm by hand,
    capture read_state().positions on a keypress, convert to degrees and write
    the trajectory JSON with source="taught-on-hardware". Not implemented — the
    hardware gesture table is authored as small relative moves precisely so that
    no teaching pass is needed for the demo.
    """
    raise NotImplementedError(
        "teach() is not implemented: the hardware gestures in arm/gestures/hardware/ are "
        "relative to whatever pose the arm is resting in, so there is nothing to teach."
    )
