"""Plan a collision-free path and, optionally, fly it under guard.

Dry run by default -- it plans, verifies and reports, and does not move the arm.
Executing requires --execute, on the principle that moving a robot should be the
thing you asked for rather than the default.

  python scripts/plan_and_run.py --map workcell_map.npz --degrees 0 60 90 0 0 0
  python scripts/plan_and_run.py --map workcell_map.npz --tip 0.35 0.10 0.30
  python scripts/plan_and_run.py --map workcell_map.npz --degrees ... --execute
"""

import argparse
import math

import numpy as np

from yam.arm import ARM_JOINTS, connected_arm
from yam.environment import ArmSafetyChecker
from yam.execution import ExecutionAborted, GuardedExecutor, GuardLimits
from yam.kinematics import YamKinematics, solve_ik_collision_free
from yam.planner import (PlannerConfig, PlanningError, RRTConnectPlanner, path_length,
                         resample, verify_under_tracking_error)
from yam.voxel_map import VoxelMap

ARM_XML = "../i2rt/i2rt/robot_models/arm/yam_pro/v1/yam_pro.xml"

#: Measured resting pose, stable with the motors off.
HOME = [0.0498, -0.0002, 0.0002, -0.0906, 0.0734, 1.1706]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map", required=True, help="voxel map from scripts/build_map.py")
    parser.add_argument("--degrees", type=float, nargs=6, default=None, help="goal joint angles")
    parser.add_argument("--tip", type=float, nargs=3, default=None, help="goal tip position x y z in metres")
    parser.add_argument("--from-home", action="store_true", help="plan from the recorded home pose, not the live arm")
    parser.add_argument("--margin", type=float, default=0.03, help="obstacle clearance in metres")
    parser.add_argument("--arm-xml", default=ARM_XML)
    parser.add_argument("--execute", action="store_true", help="actually move the arm")
    parser.add_argument("--gain-scale", type=float, default=0.5)
    parser.add_argument("--rate", type=float, default=100.0)
    parser.add_argument("--allow-unverified-sag", action="store_true",
                        help="execute even if the sag check fails (you are accepting the risk)")
    args = parser.parse_args()

    if (args.degrees is None) == (args.tip is None):
        raise SystemExit("give exactly one of --degrees or --tip")

    kinematics = YamKinematics()
    voxel_map = VoxelMap.load(args.map)
    checker = ArmSafetyChecker(kinematics, voxel_map, args.arm_xml, margin=args.margin)
    lower = [j.lower_limit for j in ARM_JOINTS]
    upper = [j.upper_limit for j in ARM_JOINTS]

    print(f"  map {voxel_map.shape} @ {voxel_map.resolution * 1000:.0f}mm, "
          f"{int(voxel_map.occupancy.sum()):,} occupied voxels")

    def plan_from(start):
        start = np.asarray(start, dtype=float)
        if not checker.is_free(start):
            raise SystemExit(f"  start pose is not clear: {'; '.join(checker.explain(start))}")

        if args.degrees is not None:
            goal = np.array([math.radians(d) for d in args.degrees])
            goal = np.clip(goal, lower, upper)
            if not checker.is_free(goal):
                raise SystemExit(f"  goal pose is not clear: {'; '.join(checker.explain(goal))}")
        else:
            print(f"  solving IK for tip {args.tip}...")
            goal = solve_ik_collision_free(kinematics, args.tip, checker, lower, upper, seed=start)
            if goal is None:
                raise SystemExit("  no collision-free IK solution for that point")
            reached = kinematics.tip_position(goal)
            print(f"    tip {np.round(reached, 4)}  (error {np.linalg.norm(reached - args.tip) * 1000:.1f} mm)")

        planner = RRTConnectPlanner(checker, lower, upper, PlannerConfig(seed=1))
        try:
            return goal, planner.plan(start, goal)
        except PlanningError as error:
            raise SystemExit(f"  {error}")

    def report(path):
        dense = resample(path, 0.02)
        clearances = [checker.clearance(q) for q in dense]
        print(f"\n  path: {len(path)} waypoints, {path_length(path):.2f} rad of joint travel")
        print(f"  verified {len(dense)} interpolated states, min slack {min(clearances) * 1000:+.1f} mm")

        limits = GuardLimits()
        sag = verify_under_tracking_error(checker, dense, limits.max_tracking_error, samples=12,
                                          lower=lower, upper=upper)
        print(f"\n  sag check at the guard's +-{sag['tracking_error_deg']:.0f}deg tracking limit: "
              f"{sag['failures']} of {len(dense)} waypoints fail")
        if not sag["ok"]:
            print("    The commanded path is clear, but the arm lags behind its command under gravity,")
            print("    and inside that lag envelope this path is not clear. Reduce the lag (higher gain,")
            print("    slower motion) or re-plan with a larger --margin.")
        return dense, sag

    if args.execute:
        with connected_arm(joints=ARM_JOINTS) as arm:
            arm.enable()
            start = HOME if args.from_home else arm.read_state().positions
            goal, path = plan_from(start)
            dense, sag = report(path)

            if not sag["ok"] and not args.allow_unverified_sag:
                raise SystemExit("\n  refusing to execute: sag check failed (--allow-unverified-sag to override)")

            print("\n  executing under guard...")
            executor = GuardedExecutor(arm, checker)
            try:
                result = executor.run(dense, rate_hz=args.rate, gain_scale=args.gain_scale)
                print(f"  completed {result.waypoints_sent} waypoints | "
                      f"peak tracking error {math.degrees(result.peak_tracking_error):.1f}deg | "
                      f"peak torque {result.peak_torque:.2f}Nm | "
                      f"peak residual {result.peak_torque_residual:.2f}Nm")
            except ExecutionAborted as abort:
                print(f"\n  ABORTED: {abort.reason}")
    else:
        goal, path = plan_from(HOME if args.from_home else HOME)
        report(path)
        print("\n  dry run only -- pass --execute to move the arm")


if __name__ == "__main__":
    main()
