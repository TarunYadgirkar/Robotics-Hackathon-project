"""Trajectory loading, velocity capping and interpolation.

This module is backend-independent on purpose: the simulator and the (currently
stubbed) hardware driver consume the SAME capped setpoint stream, so the cap is
not a simulator courtesy — it is the only thing that will ever be sent to a
servo. Everything here is pure; no I/O beyond reading the trajectory file.
"""

from dataclasses import dataclass
from pathlib import Path
import json

from . import model

TRAJECTORY_SCHEMA_VERSION = 1


class TrajectoryError(ValueError):
    pass


class SoftLimitError(TrajectoryError):
    pass


@dataclass(frozen=True)
class Waypoint:
    t: float
    positions: tuple[float, ...]
    label: str = ""


@dataclass(frozen=True)
class Trajectory:
    name: str
    source: str
    waypoints: tuple[Waypoint, ...]
    notes: str = ""

    @property
    def duration(self) -> float:
        return self.waypoints[-1].t if self.waypoints else 0.0


def load_trajectory(path: str | Path) -> Trajectory:
    path = Path(path)
    if not path.exists():
        raise TrajectoryError(f"trajectory file not found: {path}")
    data = json.loads(path.read_text())

    for key in ("name", "source", "joint_names", "units", "waypoints"):
        if key not in data:
            raise TrajectoryError(f"{path}: missing required key '{key}'")
    if tuple(data["joint_names"]) != model.JOINT_NAMES:
        raise TrajectoryError(
            f"{path}: joint_names {data['joint_names']} do not match the arm model "
            f"{list(model.JOINT_NAMES)}"
        )
    if data["units"] != model.UNITS_LABEL:
        raise TrajectoryError(f"{path}: units must be {model.UNITS_LABEL!r}, got {data['units']!r}")
    if not data["waypoints"]:
        raise TrajectoryError(f"{path}: no waypoints")

    waypoints = []
    last_t = None
    for i, raw in enumerate(data["waypoints"]):
        positions = tuple(float(v) for v in raw["positions"])
        violations = model.check_limits(positions)
        if violations:
            raise SoftLimitError(f"{path}: waypoint {i}: " + "; ".join(violations))
        t = float(raw["t"])
        if last_t is not None and t <= last_t:
            raise TrajectoryError(f"{path}: waypoint {i}: t={t} not after {last_t}")
        last_t = t
        waypoints.append(Waypoint(t=t, positions=positions, label=raw.get("label", "")))

    if waypoints[0].t != 0.0:
        raise TrajectoryError(f"{path}: first waypoint must be at t=0")

    return Trajectory(
        name=data["name"],
        source=data["source"],
        waypoints=tuple(waypoints),
        notes=data.get("notes", ""),
    )


def trajectory_from_poses(
    name: str,
    source: str,
    poses: list[tuple[float, tuple[float, ...], str]],
    notes: str = "",
) -> Trajectory:
    return Trajectory(
        name=name,
        source=source,
        waypoints=tuple(Waypoint(t=t, positions=tuple(p), label=label) for t, p, label in poses),
        notes=notes,
    )


def scale_speed(traj: Trajectory, speed: float) -> Trajectory:
    """speed=0.5 -> half speed (durations doubled). Returns a new Trajectory."""
    if not 0.0 < speed <= 1.0:
        raise TrajectoryError(f"speed must be in (0, 1], got {speed}")
    factor = 1.0 / speed
    return Trajectory(
        name=traj.name,
        source=traj.source,
        waypoints=tuple(
            Waypoint(t=w.t * factor, positions=w.positions, label=w.label) for w in traj.waypoints
        ),
        notes=traj.notes,
    )


def enforce_velocity_cap(traj: Trajectory) -> tuple[Trajectory, list[str]]:
    """Stretch any segment that would exceed the per-joint cap.

    Time dilation, never position clipping: the path through space is preserved
    exactly and only the schedule slows down. Returns the corrected trajectory
    plus a report of every segment that had to be slowed.
    """
    caps = model.velocity_cap_deg_s()
    report: list[str] = []
    out = [traj.waypoints[0]]
    t_cursor = traj.waypoints[0].t

    for prev, cur in zip(traj.waypoints, traj.waypoints[1:]):
        dt = cur.t - prev.t
        worst_ratio = 0.0
        worst_joint = ""
        for name, a, b in zip(model.JOINT_NAMES, prev.positions, cur.positions):
            required = abs(b - a) / dt
            ratio = required / caps[name]
            if ratio > worst_ratio:
                worst_ratio, worst_joint = ratio, name
        dt_new = dt * worst_ratio if worst_ratio > 1.0 else dt
        if worst_ratio > 1.0:
            report.append(
                f"segment {prev.t:.2f}->{cur.t:.2f}s stretched to {dt_new:.2f}s "
                f"({worst_joint} would have run at {worst_ratio * caps[worst_joint]:.1f} deg/s, "
                f"cap {caps[worst_joint]:.1f})"
            )
        t_cursor += dt_new
        out.append(Waypoint(t=t_cursor, positions=cur.positions, label=cur.label))

    return (
        Trajectory(name=traj.name, source=traj.source, waypoints=tuple(out), notes=traj.notes),
        report,
    )


def to_setpoints(traj: Trajectory, hz: float = model.CONTROL_HZ) -> list[tuple[float, tuple[float, ...]]]:
    """Linear joint-space interpolation of a (already capped) trajectory."""
    step = 1.0 / hz
    setpoints: list[tuple[float, tuple[float, ...]]] = []
    n_steps = max(1, int(round(traj.duration / step)))
    idx = 0
    for k in range(n_steps + 1):
        t = min(k * step, traj.duration)
        while idx + 2 < len(traj.waypoints) and t > traj.waypoints[idx + 1].t:
            idx += 1
        a, b = traj.waypoints[idx], traj.waypoints[min(idx + 1, len(traj.waypoints) - 1)]
        span = b.t - a.t
        frac = 0.0 if span <= 0 else min(1.0, max(0.0, (t - a.t) / span))
        positions = tuple(pa + (pb - pa) * frac for pa, pb in zip(a.positions, b.positions))
        setpoints.append((t, positions))
    return setpoints


def peak_velocities(setpoints: list[tuple[float, tuple[float, ...]]]) -> dict[str, float]:
    peaks = {name: 0.0 for name in model.JOINT_NAMES}
    for (t0, q0), (t1, q1) in zip(setpoints, setpoints[1:]):
        dt = t1 - t0
        if dt <= 0:
            continue
        for name, a, b in zip(model.JOINT_NAMES, q0, q1):
            peaks[name] = max(peaks[name], abs(b - a) / dt)
    return peaks


def prepare(
    traj: Trajectory, speed: float, hz: float = model.CONTROL_HZ
) -> tuple[Trajectory, list[tuple[float, tuple[float, ...]]], list[str]]:
    """Scale, cap, interpolate — the one path every backend uses.

    `hz` is the backend's streaming rate: the simulator runs at model.CONTROL_HZ,
    the YAM driver at its own 100 Hz tick (its slew clamp is per command, not
    per second, so the rate is part of the safety arithmetic).
    """
    scaled = scale_speed(traj, speed)
    capped, report = enforce_velocity_cap(scaled)
    return capped, to_setpoints(capped, hz), report
