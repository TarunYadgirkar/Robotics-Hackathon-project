"""Frozen contract: home(), replay(traj_json_path), gesture(name).

The hardware/simulator choice is read from coordination/FACTS.md at import via
arm.facts — there is deliberately no HARDWARE_PRESENT constant in this package
that could drift from what P0 measured.

Both backends receive the same velocity-capped setpoint stream and the same
freeze-and-hold abort semantics; only the four driver calls differ.
"""

from pathlib import Path

from . import facts, model, motion, safety
from .safety import ArmFrozen, MotionAborted  # re-exported for callers

_GESTURE_ROOT = Path(__file__).resolve().parent / "gestures"
#: "attempt" is the newest, for the beat where the robot is told how and tries
#: it. Added to the frozen set rather than smuggled in as a replay path so
#: callers keep using gesture(name) for everything expressive.
GESTURE_NAMES = ("attention", "decline", "point_screen", "attempt")

APPROACH_LEAD_S = 0.6  # nominal move-to-start time; the cap stretches it if needed
POSE_TOLERANCE_DEG = 0.5


def _make_backend():
    if facts.hardware_present():
        from .hw_backend import YamBackend

        return YamBackend()
    from .sim_backend import SimBackend

    return SimBackend()


_backend = _make_backend()
_connected = False

# Two gesture tables, chosen by the same FACTS flag that chooses the backend.
# The sim plays full-body gestures; the hardware table is gripper-first offsets
# from the arm's resting pose, per the operator's rule at the robot. Callers do
# not choose — gesture("decline") is the right decline for whatever is attached.
GESTURE_DIR = _GESTURE_ROOT / "hardware" if _backend.is_hardware else _GESTURE_ROOT
TASK_DEMO_PATH = GESTURE_DIR / "task_demo.json"


def backend_name() -> str:
    return _backend.name


def is_simulated() -> bool:
    return not _backend.is_hardware


def describe() -> dict:
    """What F and the build card must disclose about this module's reality."""
    return {
        "backend": _backend.name,
        "simulated": is_simulated(),
        "hardware_present_facts": facts.hardware_flag_raw(),
        "forced_sim": facts.force_sim(),
        "device_path_facts": facts.device_path(),
        "velocity_cap_fraction": model.VELOCITY_CAP_FRACTION,
        "velocity_cap_deg_s": model.velocity_cap_deg_s(),
        "limits_source": model.LIMITS_SOURCE,
        "control_hz": _backend.control_hz(),
        "gesture_dir": str(GESTURE_DIR),
        "gesture_sources": {
            name: motion.load_trajectory(GESTURE_DIR / f"{name}.json").source
            for name in GESTURE_NAMES
        },
        "hw_max_excursion_deg": model.HW_MAX_EXCURSION_DEG if _backend.is_hardware else None,
        "hw_gripper_gentle_pct_s": model.HW_GRIPPER_GENTLE_PCT_S if _backend.is_hardware else None,
    }


def probe_passive() -> list:
    """Hardware only: read every joint with no enable and no torque applied.

    Shares the backend's single bus, so a probe before connect() does not lock
    the adapter out of the session.
    """
    if not _backend.is_hardware:
        raise RuntimeError("probe_passive() is hardware-only; this is the simulator backend")
    return _backend.probe_passive()


def current_pose() -> tuple[float, ...]:
    """Live joint pose: degrees for joint1-joint6, percent-open for the gripper."""
    connect()
    return _backend.current_positions()


def connect() -> None:
    global _connected
    if not _connected:
        _backend.connect()
        _connected = True


def _with_approach(traj: motion.Trajectory) -> motion.Trajectory:
    """Prepend the move from wherever the arm is now to the trajectory's start."""
    current = _backend.current_positions()
    first = traj.waypoints[0].positions
    if all(abs(a - b) <= POSE_TOLERANCE_DEG for a, b in zip(current, first)):
        return traj
    shifted = [
        motion.Waypoint(t=w.t + APPROACH_LEAD_S, positions=w.positions, label=w.label)
        for w in traj.waypoints
    ]
    return motion.Trajectory(
        name=traj.name,
        source=traj.source,
        waypoints=(motion.Waypoint(t=0.0, positions=current, label="approach from current pose"),
                   *shifted),
        notes=traj.notes,
    )


def _execute(label: str, traj: motion.Trajectory, speed: float, amplitude: float = 1.0) -> tuple[float, ...]:
    connect()
    safety.require_not_frozen()
    if amplitude != 1.0:
        traj = motion.scale_amplitude(traj, amplitude)
    resolved = motion.resolve_relative(traj, _backend.current_positions())
    capped, setpoints, report = motion.prepare(_with_approach(resolved), speed, _backend.control_hz())
    _backend.begin_motion(label, capped, setpoints, report)
    final = safety.run_motion(
        setpoints,
        send=_backend.send,
        hold=_backend.hold,
        realtime=_backend.realtime(),
    )
    _backend.end_motion(label, capped, setpoints)
    return final


# -- public API --------------------------------------------------------------
def home(speed: float = 1.0) -> tuple[float, ...]:
    """Go to the home pose.

    On hardware this deliberately does NOT sweep to model.HOME_POSE_DEG. That
    pose was measured on a different day in a different spot, and driving to it
    from wherever the arm is now is exactly the large unattended sweep the
    operator's rule forbids. So on hardware, home() settles and holds the pose
    the arm is already in — the arm's home IS where it is resting — and says so.
    In the simulator it means what it always meant.
    """
    connect()
    current = _backend.current_positions()
    if _backend.is_hardware:
        print("[HW] home(): holding the current resting pose. No sweep to a stored home pose — "
              "that would be a large unattended motion, which this arm's rules forbid.")
        traj = motion.trajectory_from_poses(
            name="home",
            source="hardware: settle and hold the resting pose",
            poses=[(0.0, current, "resting pose"), (APPROACH_LEAD_S, current, "held")],
        )
    else:
        traj = motion.trajectory_from_poses(
            name="home",
            source="derived from model.HOME_POSE_DEG",
            poses=[(0.0, current, "current pose"), (APPROACH_LEAD_S, tuple(model.HOME_POSE_DEG), "home")],
        )
    return _execute("home", traj, speed)


def replay(traj_json_path: str | Path, speed: float = 1.0, amplitude: float = 1.0) -> tuple[float, ...]:
    """Replay a trajectory JSON. Soft limits and the velocity cap are enforced.

    `amplitude` (relative gestures only) shrinks the offsets without changing
    the timing — for bringing a gesture up on an arm whose surroundings are
    unknown: 0.25 first, watch it, then 1.0.
    """
    traj = motion.load_trajectory(traj_json_path)
    return _execute(traj.name, traj, speed, amplitude)


def gesture(name: str, speed: float = 1.0, amplitude: float = 1.0) -> tuple[float, ...]:
    if name not in GESTURE_NAMES:
        raise ValueError(f"unknown gesture {name!r}; expected one of {GESTURE_NAMES}")
    return replay(GESTURE_DIR / f"{name}.json", speed=speed, amplitude=amplitude)


def recover_home() -> tuple[float, ...] | None:
    """The ONLY way out of a freeze. Requires an explicit operator keypress.

    Homing is autonomous motion after an abort, so it never happens implicitly:
    if the operator does not confirm, the arm stays frozen and this returns None.
    """
    if not safety.is_frozen():
        raise RuntimeError("recover_home() is only valid after an aborted motion")
    if not safety.confirm_home_keypress():
        print("[SAFETY] not confirmed — staying frozen.")
        return None
    safety.clear_freeze()
    if _backend.is_hardware:
        print("[HW] recovery: resuming the hold at the frozen pose. Nothing sweeps. If the arm is "
              "somewhere it should not be, cut power or run arm_io.shutdown() and move it by hand.")
    return home(speed=model.LOW_SPEED_FRACTION)


def shutdown() -> None:
    """Disable motors and close the bus. Hardware only; a no-op in the simulator."""
    global _connected
    if _connected and hasattr(_backend, "shutdown"):
        _backend.shutdown()
        _connected = False
