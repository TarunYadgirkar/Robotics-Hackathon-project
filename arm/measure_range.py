"""Measure how far each joint can move from rest before it is no longer safe.

READ THE CAVEAT BEFORE USING THESE NUMBERS AS BUDGETS. This sweeps ONE joint at
a time from the folded resting pose, so what it measures is a LOWER bound on the
envelope, not the envelope. Measured example: joint4 shows only 6 deg of freedom
in the negative direction from rest, because at rest the arm is folded and the
forearm swings into itself — yet joint4 runs -28 deg happily in every authored
gesture, because joint3 lifts first and moves it clear. The safe range of a joint
depends on where the other joints are.

So these numbers set the coarse per-joint guards in HW_PER_JOINT_EXCURSION_DEG,
and the authoritative check stays where it has to be: MuJoCo over the actual
prepared setpoint stream, in hw_backend.verify_self_collision_free().

Sweeps one joint at a time from the resting pose against the same MuJoCo
self-collision checker the motion path uses, and against the base-clamp
regression rule, and reports the largest excursion that stays free in each
direction. Those numbers set HW_PER_JOINT_EXCURSION_DEG instead of a hand guess.

joint1 is reported for completeness but is NOT a recommendation: it is locked at
model.HW_JOINT1_LOCKED_DEG by operator instruction because the clamps beside the
base are outside the collision model.

Usage: .venv/bin/python arm/measure_range.py --rest 5.65 -0.01 -0.05 -4.40 13.61 11.84 71.0
"""

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from arm import arm_io, model  # noqa: E402
from arm.verify_poses import build_checker  # noqa: E402

STEP_DEG = 1.0
SAFETY_MARGIN_DEG = 5.0  # authored budgets stay this far inside the measured edge


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rest", nargs=7, type=float, default=None)
    args = parser.parse_args()

    if args.rest is not None:
        rest = tuple(args.rest)
    else:
        readings = arm_io.probe_passive()
        rest = tuple(
            [math.degrees(fb.position) for _, fb in readings[:6]]
            + [model.gripper_rad_to_percent(readings[-1][1].position)]
        )
    print("resting pose: " + ", ".join(f"{n}={v:.2f}" for n, v in zip(model.JOINT_NAMES, rest)))

    checker, _ = build_checker()
    rest_q = [math.radians(v) for v in rest[:6]]
    rest_clamp = checker.environment_clearance(rest_q)
    print(f"clamp clearance at rest: {rest_clamp * 1000:+.0f} mm\n")

    print(f"{'joint':8s} {'limit (deg)':>18s} {'free -':>8s} {'free +':>8s} {'budget':>8s}  stopped by")
    for i, name in enumerate(model.ARM_JOINT_NAMES):
        lo, hi = model.LIMITS_DEG[name]
        edges = {}
        stoppers = {}
        for direction in (-1, +1):
            reached = 0.0
            stopper = "joint limit"
            delta = 0.0
            while True:
                delta += STEP_DEG
                value = rest[i] + direction * delta
                if not (lo <= value <= hi):
                    stopper = "joint limit"
                    break
                q = list(rest_q)
                q[i] = math.radians(value)
                if not checker.self_is_free(q):
                    stopper = "self-collision"
                    break
                if rest_clamp - checker.environment_clearance(q) > 0.005:
                    stopper = "clamp zone"
                    break
                reached = delta
                if delta > 180:
                    break
            edges[direction] = reached
            stoppers[direction] = stopper

        smallest = min(edges[-1], edges[+1])
        budget = max(0.0, smallest - SAFETY_MARGIN_DEG)
        locked = name == "joint1"
        note = f"{stoppers[-1]} / {stoppers[+1]}"
        suffix = "   <-- LOCKED by operator (clamps), budget forced to %.0f" % model.HW_JOINT1_LOCKED_DEG if locked else ""
        print(f"{name:8s} [{lo:7.1f},{hi:7.1f}] {edges[-1]:8.0f} {edges[+1]:8.0f} {budget:8.0f}  {note}{suffix}")

    print(f"\nbudget = min(free-, free+) - {SAFETY_MARGIN_DEG:.0f} deg margin. "
          "joint1 is excluded from that rule by operator instruction.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
