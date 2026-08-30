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

GESTURE_DIR = Path(__file__).resolve().parent / "gestures"
GESTURE_NAMES = ("attention", "decline", "point_screen")
TASK_DEMO_PATH = GESTURE_DIR / "task_demo.json"

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
        "device_path_facts": facts.device_path(),
        "velocity_cap_fraction": model.VELOCITY_CAP_FRACTION,
        "velocity_cap_deg_s": model.velocity_cap_deg_s(),
        "limits_source": model.LIMITS_SOURCE,
        "control_hz": _backend.control_hz(),
        "gesture_sources": {
            name: motion.load_trajectory(GESTURE_DIR / f"{name}.json").source
            for name in GESTURE_NAMES
        },
    }


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


def _execute(label: str, traj: motion.Trajectory, speed: float) -> tuple[float, ...]:
    connect()
    safety.require_not_frozen()
    capped, setpoints, report = motion.prepare(_with_approach(traj), speed, _backend.control_hz())
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
    """Move to the home pose. Capped like every other motion."""
    connect()
    traj = motion.trajectory_from_poses(
        name="home",
        source="derived from model.HOME_POSE_DEG",
        poses=[
            (0.0, _backend.current_positions(), "current pose"),
            (APPROACH_LEAD_S, tuple(model.HOME_POSE_DEG), "home"),
        ],
    )
    return _execute("home", traj, speed)


def replay(traj_json_path: str | Path, speed: float = 1.0) -> tuple[float, ...]:
    """Replay a trajectory JSON. Soft limits and the velocity cap are enforced."""
    traj = motion.load_trajectory(traj_json_path)
    return _execute(traj.name, traj, speed)


def gesture(name: str, speed: float = 1.0) -> tuple[float, ...]:
    if name not in GESTURE_NAMES:
        raise ValueError(f"unknown gesture {name!r}; expected one of {GESTURE_NAMES}")
    return replay(GESTURE_DIR / f"{name}.json", speed=speed)


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
    return home(speed=model.LOW_SPEED_FRACTION)
