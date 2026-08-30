"""Collision-verify every hardware gesture pose against the real YAM model.

This closes the caveat that stood through the whole build: "self-collision is
unverifiable here". Agent R1 fetched i2rt's official URDF/MJCF
(github.com/i2rt-robotics/i2rt) into hwresearch/ and installed mujoco, so
yam.environment.ArmSafetyChecker now constructs.

TWO CHECKS, AND THEY ARE NOT EQUALLY STRONG — read this before trusting a PASS:

1. SELF-COLLISION (authoritative). MuJoCo against the arm's exact convex meshes.
   This is the real answer to "does this gesture fold the arm into itself", and
   it is checked at every waypoint and along every interpolated segment.

2. BASE CLAMPS (relative only). yam.mapping.base_clamps() describes the two 25mm
   clamps by hand because LiDAR cannot resolve them. But the environment check
   works on conservative spheres fitted to whole links — up to 102mm radius —
   against a 30mm margin, and this arm rests FOLDED with every link between
   z=0.086 and z=0.266, i.e. sitting in and just above the clamp zone. Run
   absolutely, it reports the arm's own power-up resting pose as a collision,
   which is demonstrably false. Boris excludes the base link from environment
   checks for exactly this reason ("permanently in collision with the surface it
   is mounted on ... cannot move into anything new"). So the clamp check here is
   RELATIVE: a gesture passes if it never brings a link closer to the clamps than
   the arm already is at rest. That is a real regression test, not a proof of
   clearance.

Neither check knows anything about the rest of the table. A PASS means "does not
fold into itself, and gets no closer to the base clamps than it starts". The
human watching is still the authority on everything else in the workspace.

Usage:
  .venv/bin/python arm/verify_poses.py
  .venv/bin/python arm/verify_poses.py --rest 5.65 -0.01 -0.05 -4.40 13.61 11.84 72.3
"""

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from arm import arm_io, model, motion  # noqa: E402

URDF = REPO_ROOT / "hwresearch/i2rt/robot_models/arm/yam_pro/v1/yam_pro.urdf"
XML = REPO_ROOT / "hwresearch/i2rt/robot_models/arm/yam_pro/v1/yam_pro.xml"

SEGMENT_RESOLUTION_RAD = 0.05
#: How much closer to the clamps than the resting pose a gesture may get.
CLAMP_REGRESSION_TOLERANCE_M = 0.005


def build_checker():
    from yam.environment import ArmSafetyChecker
    from yam.kinematics import YamKinematics
    from yam.mapping import base_clamps
    from yam.voxel_map import VoxelMap

    if not URDF.exists() or not XML.exists():
        raise SystemExit(f"model files missing under {URDF.parent} — see hwresearch/ (Agent R1)")
    kin = YamKinematics(urdf_path=str(URDF))
    vmap = VoxelMap.from_bounds([-1.0, -1.0, -0.3], [1.0, 1.0, 1.2], resolution=0.02)
    clamps = base_clamps()
    for box in clamps:
        vmap.add_box(box["min"], box["max"])
    return ArmSafetyChecker(kin, vmap, str(XML)), clamps


def sample_segment(a, b, resolution=SEGMENT_RESOLUTION_RAD):
    steps = max(int(math.ceil(max(abs(x - y) for x, y in zip(a, b)) / resolution)), 1)
    return [
        [x + (y - x) * (i / steps) for x, y in zip(a, b)]
        for i in range(steps + 1)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rest", nargs=7, type=float, default=None)
    args = parser.parse_args()

    if args.rest is not None:
        rest = tuple(args.rest)
        print(f"resting pose (given): {rest}")
    else:
        readings = arm_io.probe_passive()
        rest = tuple(
            [math.degrees(fb.position) for _, fb in readings[:6]]
            + [model.gripper_rad_to_percent(readings[-1][1].position)]
        )
        print("resting pose (read live, no motors enabled): "
              + ", ".join(f"{n}={v:.2f}" for n, v in zip(model.JOINT_NAMES, rest)))

    checker, clamps = build_checker()
    rest_q = [math.radians(v) for v in rest[:6]]
    rest_clamp_clearance = checker.environment_clearance(rest_q)
    rest_self_free = checker.self_is_free(rest_q)
    print(f"model: official i2rt yam_pro URDF+MJCF | environment: {len(clamps)} base-clamp boxes")
    print(f"resting-pose baseline: self-collision-free={rest_self_free}, "
          f"clamp clearance {rest_clamp_clearance * 1000:+.0f} mm "
          f"(negative is the coarse sphere fit, not a real collision — see module docstring)\n")

    # Gestures that play from another gesture's END pose, not from rest — a
    # from-rest resolve is not just the wrong geometry, it violates soft limits
    # (can_fling ends 30 deg of joint2 below where it starts, by design).
    follows = {"can_fling": "can_pickup", "can_release": "can_pickup"}

    failures = 0
    for name in (*arm_io.GESTURE_NAMES, "task_demo"):
        base = rest
        if name in follows:
            pred = motion.load_trajectory(arm_io.GESTURE_DIR / f"{follows[name]}.json")
            end = motion.resolve_relative(pred, rest).waypoints[-1]
            base = end.positions
            print(f"      ({name} verified from {follows[name]}'s end pose, not rest)")
        traj = motion.load_trajectory(arm_io.GESTURE_DIR / f"{name}.json")
        resolved = motion.resolve_relative(traj, base) if traj.is_relative else traj
        qs = [[math.radians(v) for v in w.positions[:6]] for w in resolved.waypoints]

        self_bad = []
        worst_self = float("inf")
        worst_clamp = float("inf")
        for i, (a, b) in enumerate(zip(qs, qs[1:])):
            for q in sample_segment(a, b):
                worst_self = min(worst_self, checker.self_checker.clearance(q))
                worst_clamp = min(worst_clamp, checker.environment_clearance(q))
                if not checker.self_is_free(q):
                    self_bad.append((i, resolved.waypoints[i].label))
                    break

        clamp_regression = rest_clamp_clearance - worst_clamp
        clamp_ok = clamp_regression <= CLAMP_REGRESSION_TOLERANCE_M
        ok = not self_bad and clamp_ok
        failures += not ok

        print(f"{'PASS' if ok else 'FAIL'}  {name:13s} self-collision clearance {worst_self * 1000:+.0f} mm"
              f" | clamps {worst_clamp * 1000:+.0f} mm vs rest {rest_clamp_clearance * 1000:+.0f} mm"
              f" ({'no closer' if clamp_regression <= 0 else f'{clamp_regression * 1000:.0f} mm closer'})")
        for i, label in self_bad:
            print(f"        SELF-COLLISION entering segment {i} ({label})")
        if not clamp_ok:
            print(f"        moves {clamp_regression * 1000:.0f} mm closer to the base clamps than rest")

    print()
    if failures:
        print(f"{failures} gesture(s) FAILED — fix the poses, do not run them.")
        return 1
    print("PASS: no gesture folds the arm into itself at any waypoint or anywhere along an "
          "interpolated segment, and none moves closer to the base clamps than the arm rests.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
