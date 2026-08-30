"""Plan through the safety gateway and optionally execute an approved path.

Dry runs produce preview-only paths. Hardware execution additionally requires a
fresh provenance-bearing map, a raw-trace-backed calibration, and a trusted
scene-interlock file. Any missing or failed check returns a spoken refusal and
does not command a motor.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yam.arm import ARM_JOINTS, connected_arm
from yam.environment import ArmSafetyChecker
from yam.execution import ExecutionAborted, GuardedExecutor, GuardLimits
from yam.hardware_calibration import HardwareSafetyCalibration, file_sha256
from yam.kinematics import YamKinematics
from yam.safe_planning import (
    MotionGoal,
    PlanningPolicy,
    SceneInterlock,
    plan_on_demand,
)
from yam.voxel_map import VoxelMap


ARM_XML = "../i2rt/i2rt/robot_models/arm/yam_pro/v1/yam_pro.xml"
HOME = [0.0498, -0.0002, 0.0002, -0.0906, 0.0734, 1.1706]


def load_scene_interlock(path: str) -> SceneInterlock:
    data = json.loads(Path(path).read_text())
    return SceneInterlock(
        source=str(data["source"]),
        observed_at_unix=float(data["observed_at_unix"]),
        valid_for_seconds=float(data["valid_for_seconds"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True, help="map from scripts/build_map.py")
    parser.add_argument("--degrees", type=float, nargs=6, help="goal joint angles")
    parser.add_argument("--tip", type=float, nargs=3, help="jaw-tip goal x y z in metres")
    parser.add_argument("--from-home", action="store_true", help="preview from the recorded home pose")
    parser.add_argument("--margin", type=float, default=0.03, help="requested obstacle clearance in metres")
    parser.add_argument("--arm-xml", default=ARM_XML)
    parser.add_argument("--output-plan", help="save the dense path for visualization")
    parser.add_argument("--execute", action="store_true", help="move only if the gateway issues approval")
    parser.add_argument("--calibration", help="hardware safety calibration JSON")
    parser.add_argument("--scene-interlock", help="fresh trusted scene-observation JSON")
    parser.add_argument("--max-map-age", type=float, help="maximum scan age permitted by deployment policy")
    parser.add_argument("--max-calibration-age", type=float,
                        help="maximum calibration age permitted by deployment policy")
    parser.add_argument("--approval-seconds", type=float,
                        help="maximum time between planning and starting execution")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.degrees is None) == (args.tip is None):
        raise SystemExit("give exactly one of --degrees or --tip")
    if args.execute and args.from_home:
        raise SystemExit("refusing to execute from a stored home pose; hardware must plan from a live read")

    goal = (
        MotionGoal.joints_degrees(args.degrees, "requested joint pose")
        if args.degrees is not None
        else MotionGoal.tip(args.tip, "requested jaw-tip point")
    )
    policy = PlanningPolicy(
        requested_clearance_m=args.margin,
        max_map_age_seconds=args.max_map_age,
        max_calibration_age_seconds=args.max_calibration_age,
        approval_valid_seconds=args.approval_seconds,
    )

    if args.execute:
        try:
            scene_interlock = load_scene_interlock(args.scene_interlock) if args.scene_interlock else None
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            _print_refusal("scene_not_interlocked", f"the scene interlock cannot be read ({error})")
            return

        with connected_arm(joints=ARM_JOINTS) as arm:
            start = arm.read_state().positions

        outcome = plan_on_demand(
            start,
            [goal],
            args.map,
            args.arm_xml,
            policy,
            hardware_requested=True,
            calibration_path=args.calibration,
            scene_interlock=scene_interlock,
        )
    else:
        outcome = plan_on_demand(
            HOME,
            [goal],
            args.map,
            args.arm_xml,
            policy,
            hardware_requested=False,
        )

    print(json.dumps(outcome.to_dict(), indent=2, allow_nan=False))
    if not outcome.decision.allowed and outcome.preview_path is None:
        return

    path = outcome.approved_plan.path if outcome.approved_plan is not None else outcome.preview_path
    if args.output_plan:
        np.save(args.output_plan, path)
        print(f"saved {len(path)} poses to {args.output_plan}")

    if not args.execute:
        return

    calibration = HardwareSafetyCalibration.load(
        args.calibration,
        max_age_seconds=args.max_calibration_age,
    )
    voxel_map = VoxelMap.load(args.map)
    checker = ArmSafetyChecker(
        YamKinematics(),
        voxel_map,
        args.arm_xml,
        margin=args.margin,
    )
    limits = GuardLimits.from_calibration(calibration)

    with connected_arm(joints=ARM_JOINTS) as arm:
        arm.enable()
        executor = GuardedExecutor(
            arm,
            checker,
            limits,
            map_sha256=file_sha256(args.map),
        )
        try:
            result = executor.run(
                outcome.approved_plan,
                rate_hz=calibration.rate_hz,
                gain_scale=calibration.gain_scale,
            )
        except ExecutionAborted as error:
            print(f"ABORTED: {error.reason}")
            return
    print(
        f"completed {result.waypoints_sent} poses; "
        f"peak tracking error {math.degrees(result.peak_tracking_error):.2f}deg"
    )


def _print_refusal(code: str, reason: str) -> None:
    detail = reason.strip().rstrip(".")
    print(json.dumps({
        "allowed": False,
        "code": code,
        "reason": reason,
        "spoken_response": (
            f"I understand the task, but I can't do it safely: {detail}. "
            "I won't move the arm."
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
