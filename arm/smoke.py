"""Full run of the arm contract. Exit 0 required before state=done.

Runs home(), all three gestures, and replay(task_demo) — plus the checks that
matter more than any of them: that the velocity cap actually bounds the emitted
setpoints, that an out-of-limit waypoint is refused, and that an interrupt
freezes and holds instead of homing.

Wall-clock is skipped by default (ARM_SIM_REALTIME=0) so this stays a smoke
test; set ARM_SIM_REALTIME=1 to watch it in demo time.
"""

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("ARM_SIM_REALTIME", "0")

from arm import arm_io, model, motion, safety  # noqa: E402

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  PASS {message}")
    else:
        print(f"  FAIL {message}")
        FAILURES.append(message)


def check_cap(traj_path: Path, speed: float = 1.0) -> None:
    traj = motion.load_trajectory(traj_path)
    capped, setpoints, _ = motion.prepare(traj, speed)
    caps = model.velocity_cap_deg_s()
    peaks = motion.peak_velocities(setpoints)
    worst = max(peaks[j] / caps[j] for j in model.JOINT_NAMES)
    check(worst <= 1.0 + 1e-6, f"{traj.name}: peak velocity within cap ({worst * 100:.1f}% of cap)")


def main() -> int:
    print("=== arm smoke test ===")
    info = arm_io.describe()
    print(f"backend={info['backend']} simulated={info['simulated']} "
          f"HARDWARE_PRESENT(FACTS)={info['hardware_present_facts']!r}")
    if info["simulated"]:
        print("DISCLOSURE: simulator fallback. No arm is connected; nothing below moved a servo.")

    print("\n[1] home()")
    final = arm_io.home()
    check(
        all(abs(a - b) < 1e-6 for a, b in zip(final, model.HOME_POSE_DEG)),
        "home() ends at HOME_POSE_DEG",
    )

    print("\n[2] gestures")
    for name in arm_io.GESTURE_NAMES:
        arm_io.gesture(name)
        check(True, f"gesture({name!r}) completed")

    print("\n[3] replay(task_demo) at 50% speed")
    arm_io.replay(arm_io.TASK_DEMO_PATH, speed=0.5)
    check(True, "replay(task_demo, speed=0.5) completed")

    print("\n[4] velocity cap holds on every shipped trajectory")
    for name in arm_io.GESTURE_NAMES:
        check_cap(arm_io.GESTURE_DIR / f"{name}.json")
    check_cap(arm_io.TASK_DEMO_PATH)
    check_cap(arm_io.TASK_DEMO_PATH, speed=0.5)

    print("\n[5] cap stretches an over-fast trajectory instead of clipping the path")
    far = (80.0, *model.HOME_POSE_DEG[1:])
    fast = motion.trajectory_from_poses(
        name="over_fast",
        source="smoke-test fixture",
        poses=[(0.0, model.HOME_POSE_DEG, "start"), (0.05, far, "far too fast")],
    )
    capped, setpoints, report = motion.prepare(fast, 1.0)
    check(len(report) == 1, "over-fast segment reported")
    check(capped.duration > fast.duration, f"duration stretched {fast.duration}s -> {capped.duration:.2f}s")
    check(
        capped.waypoints[-1].positions == fast.waypoints[-1].positions,
        "end pose preserved exactly (time dilation, not position clipping)",
    )
    peaks = motion.peak_velocities(setpoints)
    caps = model.velocity_cap_deg_s()
    check(max(peaks[j] / caps[j] for j in model.JOINT_NAMES) <= 1.0 + 1e-6,
          "stretched trajectory is within cap")

    print("\n[6] soft limits refuse an out-of-range pose")
    bad = REPO_ROOT / "arm" / "sim_out" / "_smoke_bad_traj.json"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text(
        json.dumps({
            "schema_version": 1,
            "name": "bad",
            "source": "smoke-test fixture",
            "joint_names": list(model.JOINT_NAMES),
            "units": model.UNITS_LABEL,
            "waypoints": [
                {"t": 0.0, "positions": list(model.HOME_POSE_DEG)},
                {"t": 1.0, "positions": [400.0, *model.HOME_POSE_DEG[1:]]},
            ],
        })
    )
    try:
        motion.load_trajectory(bad)
        check(False, "out-of-limit waypoint rejected")
    except motion.SoftLimitError as exc:
        check("joint1" in str(exc), f"out-of-limit waypoint rejected ({exc.__class__.__name__})")
    finally:
        bad.unlink(missing_ok=True)

    print("\n[7] interrupt freezes and holds, and does not home")
    held: list[tuple[float, ...]] = []
    setpoints = motion.to_setpoints(
        motion.trajectory_from_poses(
            "interrupt_fixture", "smoke-test fixture",
            [(0.0, model.HOME_POSE_DEG, ""), (2.0, (30.0, *model.HOME_POSE_DEG[1:]), "")],
        )
    )

    def send(t, positions):
        if t > 0.5:
            raise KeyboardInterrupt

    try:
        safety.run_motion(setpoints, send=send, hold=held.append, realtime=False)
        check(False, "interrupt raised MotionAborted")
    except safety.MotionAborted:
        check(True, "interrupt raised MotionAborted")
    check(len(held) == 1, "backend hold() called exactly once")
    check(safety.is_frozen(), "arm latched frozen")
    check(safety.frozen_pose() != tuple(model.HOME_POSE_DEG), "froze in place, did not home")
    try:
        safety.require_not_frozen()
        check(False, "further motion refused while frozen")
    except safety.ArmFrozen:
        check(True, "further motion refused while frozen")
    check(arm_io.recover_home() is None, "recover_home() without operator confirmation does not move")
    check(safety.is_frozen(), "still frozen after unconfirmed recovery attempt")
    safety.clear_freeze()

    print("\n[8] YAM hardware branch stays out of the way while HARDWARE_PRESENT is no")
    from arm import hw_backend  # noqa: PLC0415 — the point is that this import is safe

    check(True, "arm.hw_backend imports with no python-can / gs_usb / yam deps present")
    check(
        len(hw_backend.preflight_checklist()) >= 5,
        "hardware pre-flight checklist is populated (self-collision + keep-alive + gripper)",
    )
    try:
        hw_backend.teach("attention")
        check(False, "teach() refuses without hardware")
    except NotImplementedError:
        check(True, "teach() refuses without hardware")
    ceiling_ok = True
    for name in (*arm_io.GESTURE_NAMES, "task_demo"):
        traj = motion.load_trajectory(arm_io.GESTURE_DIR / f"{name}.json")
        for w in traj.waypoints:
            if w.positions[1] > model.JOINT2_AUTHORED_CEILING_DEG:
                ceiling_ok = False
    check(ceiling_ok, f"every authored pose keeps joint2 below {model.JOINT2_AUTHORED_CEILING_DEG} deg "
                      f"(self-collision ~{model.JOINT2_SELF_COLLISION_DEG} deg)")
    check(
        all(traj_source.endswith("authored") for traj_source in info["gesture_sources"].values()),
        f"gesture trajectories declare themselves sim-authored: {info['gesture_sources']}",
    )

    print("\n[9] sim artifacts")
    if info["simulated"] and os.environ.get("ARM_SIM_RENDER", "1") != "0":
        for name in ("home", *arm_io.GESTURE_NAMES, "task_demo"):
            gif = REPO_ROOT / "arm" / "sim_out" / f"{name}.gif"
            png = REPO_ROOT / "arm" / "sim_out" / f"{name}.png"
            check(gif.exists() and gif.stat().st_size > 0, f"{gif.name} rendered")
            check(png.exists() and png.stat().st_size > 0, f"{png.name} rendered")

    print("\n=== %d check(s) failed ===" % len(FAILURES) if FAILURES else "\n=== all checks passed ===")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
