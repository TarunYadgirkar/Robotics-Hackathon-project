"""Simulator backend — used because FACTS.md says HARDWARE_PRESENT: no.

It is a simulator, not a stand-in that pretends to be an arm: every printed line
says SIM, and the rendered animation is labelled as schematic. It consumes the
identical capped setpoint stream `motion.prepare()` produces for hardware, so
what you watch here is the real schedule that would have been commanded.

The drawing is a stick figure, NOT the YAM's kinematics: `yam.kinematics`
parses the i2rt URDF, and neither that URDF nor mujoco is on this machine (see
FACTS.md). The joint-angle traces on the right are the real, capped commands;
only the cartoon on the left is invented.
"""

import os
import sys
from pathlib import Path

from . import model, motion

SIM_OUT = Path(__file__).resolve().parent / "sim_out"

# Schematic drawing lengths, arbitrary units. NOT YAM link lengths — the i2rt
# URDF that yam.kinematics reads is not on this machine, so the render is a
# readable cartoon of the joint schedule, not a kinematic claim.
_SCHEMATIC_LINKS = (1.0, 0.9, 0.35)
_SCHEMATIC_BASE_LEAN_DEG = 65.0

MAX_GIF_FRAMES = 150
GIF_DPI = 70
STILL_DPI = 120


def _flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _short(joint_name: str) -> str:
    return joint_name.replace("joint", "j").replace("gripper", "grip")


class SimBackend:
    name = "sim"
    is_hardware = False

    def __init__(self) -> None:
        self._positions = tuple(model.HOME_POSE_DEG)
        self._trace: list[tuple[float, tuple[float, ...]]] = []
        self.artifacts: list[str] = []

    # -- lifecycle -----------------------------------------------------------
    def connect(self) -> None:
        print(f"[SIM] no hardware per FACTS.md; simulating {model.N_JOINTS} channels: "
              f"{', '.join(model.JOINT_NAMES)} (gripper is sim-only, see arm/model.py)")

    def realtime(self) -> bool:
        return _flag("ARM_SIM_REALTIME")

    def control_hz(self) -> float:
        return model.CONTROL_HZ

    def current_positions(self) -> tuple[float, ...]:
        return self._positions

    # -- motion --------------------------------------------------------------
    def send(self, t: float, positions: tuple[float, ...]) -> None:
        self._positions = positions
        self._trace.append((t, positions))

    def hold(self, positions: tuple[float, ...]) -> None:
        self._positions = positions
        print("[SIM] hold: setpoint stream stopped, last commanded pose held", file=sys.stderr)

    def begin_motion(self, label: str, traj: motion.Trajectory, setpoints, report: list[str]) -> None:
        self._trace = []
        caps = model.velocity_cap_deg_s()
        print(f"\n[SIM] --- {label} ---")
        print(f"[SIM] source: {traj.source} | duration {traj.duration:.2f}s | "
              f"{len(traj.waypoints)} waypoints | {len(setpoints)} setpoints @ {model.CONTROL_HZ:.0f} Hz")
        for line in report:
            print(f"[SIM] velocity cap applied: {line}")
        print("[SIM] timeline:")
        for w in traj.waypoints:
            pose = " ".join(f"{_short(n)}={v:6.1f}" for n, v in zip(model.JOINT_NAMES, w.positions))
            print(f"[SIM]   t={w.t:6.2f}s  {pose}   {w.label}")
        peaks = motion.peak_velocities(setpoints)
        worst = max(peaks[j] / caps[j] for j in model.JOINT_NAMES)
        print("[SIM] peak commanded velocity vs cap (%d%% of max): " % int(model.VELOCITY_CAP_FRACTION * 100)
              + ", ".join(f"{j}={peaks[j]:.1f}/{caps[j]:.1f}" for j in model.JOINT_NAMES))
        print(f"[SIM] worst joint is at {worst * 100:.1f}% of its cap")

    def end_motion(self, label: str, traj: motion.Trajectory, setpoints) -> None:
        if not _flag("ARM_SIM_RENDER"):
            print(f"[SIM] render disabled (ARM_SIM_RENDER=0) for {label}")
            return
        SIM_OUT.mkdir(parents=True, exist_ok=True)
        written = _render(label, traj, setpoints)
        self.artifacts.extend(str(p) for p in written)
        for p in written:
            print(f"[SIM] wrote {p}")


# -- rendering ---------------------------------------------------------------
def _forward_schematic(positions: tuple[float, ...]) -> list[tuple[float, float]]:
    """Side-view stick figure from joint2/joint3/joint4. Schematic only.

    Signs follow yam.arm's description of the real arm so the cartoon does not
    contradict it: joint2 tips the chain DOWN from the folded pose, joint3 is the
    joint that lifts. The base lean is cosmetic — it makes the resting pose look
    like a resting pose instead of a flagpole.
    """
    import numpy as np

    j2, j3, j4 = positions[1], positions[2], positions[3]
    base = _SCHEMATIC_BASE_LEAN_DEG - j2
    angles = np.radians([base, base + j3, base + j3 + j4])
    pts = [(0.0, 0.0)]
    x = y = 0.0
    for length, angle in zip(_SCHEMATIC_LINKS, angles):
        x += length * float(np.cos(angle))
        y += length * float(np.sin(angle))
        pts.append((x, y))
    return pts


def _render(label: str, traj: motion.Trajectory, setpoints) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    # Frame budget, not a fixed fps: a 40s replay at 15 fps produced a 5 MB gif.
    # Deriving fps from the stride keeps playback real-time either way.
    stride = max(1, -(-len(setpoints) // MAX_GIF_FRAMES))
    gif_fps = max(model.CONTROL_HZ / stride, 1.0)
    frames = setpoints[::stride]
    times = [t for t, _ in setpoints]

    fig, (ax_arm, ax_joints) = plt.subplots(
        1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1, 1.3]}
    )
    fig.suptitle(
        f"SIM (no arm connected) — {label} [{traj.source}]\n"
        "left: schematic stick figure, not YAM kinematics   |   right: the real capped joint commands",
        fontsize=9,
    )

    ax_arm.set_xlim(-1.2, 2.4)
    ax_arm.set_ylim(-0.35, 2.45)
    ax_arm.set_aspect("equal")
    ax_arm.axhline(0.0, color="0.75", lw=1.5)
    ax_arm.set_xticks([])
    ax_arm.set_yticks([])
    ax_arm.set_xlabel("side view (schematic)", fontsize=7)
    (link_line,) = ax_arm.plot([], [], "-o", lw=4, color="#2b6cb0", markersize=6)
    (tip_marker,) = ax_arm.plot([], [], "o", color="#c05621", markersize=8)
    pose_text = ax_arm.text(-1.15, 2.4, "", fontsize=7.5, family="monospace", va="top")

    # joint1 is invisible in a side view and is the whole point of the decline and
    # point_screen gestures, so it gets its own top-down dial.
    ax_pan = ax_arm.inset_axes([0.64, 0.02, 0.34, 0.34])
    ax_pan.set_xlim(-1.15, 1.15)
    ax_pan.set_ylim(-0.35, 1.15)
    ax_pan.set_aspect("equal")
    ax_pan.set_xticks([])
    ax_pan.set_yticks([])
    ax_pan.set_title("joint1 (top-down)", fontsize=6)
    (pan_line,) = ax_pan.plot([], [], "-", lw=2.5, color="#2b6cb0")

    for name in model.JOINT_NAMES:
        idx = model.JOINT_NAMES.index(name)
        ax_joints.plot(times, [q[idx] for _, q in setpoints], lw=1.2, label=name)
    ax_joints.set_xlabel("t (s)")
    ax_joints.set_ylabel("joint angle (deg)")
    ax_joints.legend(fontsize=6, ncol=2, loc="upper right")
    cursor = ax_joints.axvline(0.0, color="k", lw=1)

    def update(frame):
        import math

        t, positions = frame
        pts = _forward_schematic(positions)
        link_line.set_data([p[0] for p in pts], [p[1] for p in pts])
        tip_marker.set_data([pts[-1][0]], [pts[-1][1]])
        tip_marker.set_markersize(4.0 + positions[-1] / 8.0)  # jaw opening, sim-only
        pan = math.radians(90.0 - positions[0])
        pan_line.set_data([0.0, math.cos(pan)], [0.0, math.sin(pan)])
        cursor.set_xdata([t, t])
        units = {name: "deg" for name in model.ARM_JOINT_NAMES}
        units[model.GRIPPER_NAME] = "%open"
        pose_text.set_text(
            f"t={t:5.2f}s\n"
            + "\n".join(
                f"{n:<8}{v:7.1f} {units[n]}" for n, v in zip(model.JOINT_NAMES, positions)
            )
        )
        return link_line, tip_marker, pan_line, cursor, pose_text

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / gif_fps, blit=False)
    gif_path = SIM_OUT / f"{label}.gif"
    fig.set_dpi(GIF_DPI)
    anim.save(gif_path, writer=PillowWriter(fps=gif_fps))

    # The still shows the pose furthest from home — the last frame is usually
    # home again, which makes for a still of nothing happening.
    png_path = SIM_OUT / f"{label}.png"
    update(max(setpoints, key=lambda s: sum((a - b) ** 2 for a, b in zip(s[1], model.HOME_POSE_DEG))))
    fig.savefig(png_path, dpi=STILL_DPI)
    plt.close(fig)
    return [gif_path, png_path]
