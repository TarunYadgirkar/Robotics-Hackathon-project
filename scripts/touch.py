"""Touch points through the contact gateway.

Every decision lives in yam/contact_planning.py and yam/execution.py; this only
reads arguments and prints what came back.

  python scripts/touch.py --targets touch_targets.json --map workcell_map.npz
  python scripts/touch.py --targets touch_targets.json --execute \
      --calibration guard_calibration.json --scene-interlock scene_interlock.json \
      --max-map-age 28800 --max-calibration-age 7200 --approval-seconds 300
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yam.arm import ARM_JOINTS, connected_arm
from yam.contact_planning import approve_contact
from yam.environment import ArmSafetyChecker
from yam.execution import ExecutionAborted, GuardedExecutor, GuardLimits
from yam.hardware_calibration import HardwareSafetyCalibration, file_sha256
from yam.kinematics import YamKinematics
from yam.safe_planning import PlanningPolicy, SceneInterlock
from yam.voxel_map import VoxelMap

ARM_XML = "../i2rt/i2rt/robot_models/arm/yam_pro/v1/yam_pro.xml"
HOME = [0.0498, -0.0002, 0.0002, -0.0906, 0.0734, 1.1706]


def load_interlock(path):
    data = json.loads(Path(path).read_text())
    return SceneInterlock(source=str(data["source"]),
                          observed_at_unix=float(data["observed_at_unix"]),
                          valid_for_seconds=float(data["valid_for_seconds"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", required=True, help="JSON list of points to touch")
    parser.add_argument("--map", default="workcell_map.npz")
    parser.add_argument("--arm-xml", default=ARM_XML)
    parser.add_argument("--margin", type=float, default=0.015)
    parser.add_argument("--calibration")
    parser.add_argument("--scene-interlock")
    parser.add_argument("--max-map-age", type=float)
    parser.add_argument("--max-calibration-age", type=float)
    parser.add_argument("--approval-seconds", type=float)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    targets = json.loads(Path(args.targets).read_text())
    policy = PlanningPolicy(requested_clearance_m=args.margin,
                            max_map_age_seconds=args.max_map_age,
                            max_calibration_age_seconds=args.max_calibration_age,
                            approval_valid_seconds=args.approval_seconds)
    interlock = load_interlock(args.scene_interlock) if args.scene_interlock else None

    if not args.execute:
        for target in targets:
            outcome = approve_contact(HOME, target, args.map, args.arm_xml, policy,
                                      hardware_requested=False)
            print(f"\n{np.round(target, 3).tolist()}")
            print(json.dumps(outcome.to_dict(), indent=2, allow_nan=False)[:1200])
        return

    calibration = HardwareSafetyCalibration.load(args.calibration,
                                                 max_age_seconds=args.max_calibration_age)
    voxel_map = VoxelMap.load(args.map)
    checker = ArmSafetyChecker(YamKinematics(), voxel_map, args.arm_xml, margin=args.margin)
    kinematics = YamKinematics()

    with connected_arm(joints=ARM_JOINTS) as arm:
        arm.clear_errors()
        arm.enable()
        for target in targets:
            start = arm.read_state().positions
            outcome = approve_contact(start, target, args.map, args.arm_xml, policy,
                                      hardware_requested=True,
                                      calibration_path=args.calibration,
                                      scene_interlock=interlock)
            print(f"\n{np.round(target, 3).tolist()}  {outcome.decision.code.value}")
            if outcome.approved is None:
                print(f"  {outcome.decision.reason}")
                continue

            executor = GuardedExecutor(arm, checker, GuardLimits.from_calibration(calibration),
                                       map_sha256=file_sha256(args.map))
            try:
                report = executor.touch(outcome.approved, rate_hz=calibration.rate_hz,
                                        gain_scale=calibration.gain_scale)
            except ExecutionAborted as error:
                print(f"  ABORTED: {error.reason if hasattr(error, 'reason') else error}")
                continue

            if report.contacted:
                reached = kinematics.probe_position(np.asarray(report.pose))
                offset = float(np.dot(np.asarray(target) - reached, outcome.approved.approach_direction))
                print(f"  contact on {report.joint} at {np.round(reached, 3).tolist()}; "
                      f"{offset * 1000:+.1f} mm from where the map put the surface")
            else:
                print(f"  no contact within the approved {outcome.approved.max_travel_m * 1000:.0f} mm "
                      f"of travel; stopped at the bound")


if __name__ == "__main__":
    main()
