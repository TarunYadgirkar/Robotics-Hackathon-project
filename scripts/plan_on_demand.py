"""Structured on-demand planner for action/world-context callers.

The caller proposes Cartesian or joint goals in JSON. The response is always a
safety decision with fixed refusal wording; no free-form model output is used to
authorize motion.

Example request::

    {
      "start_joint_radians": [0.05, 0, 0, -0.09, 0.07, 1.17],
      "goals": [{"type": "tip", "position_m": [0.3, 0.1, 0.2], "label": "near object"}],
      "map_path": "workcell_map.npz",
      "arm_xml_path": "../i2rt/i2rt/robot_models/arm/yam_pro/v1/yam_pro.xml",
      "requested_clearance_m": 0.03
    }
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yam.safe_planning import MotionGoal, PlanningPolicy, SceneInterlock, plan_on_demand
from yam.safety_contract import SafetyCode, SafetyDecision


def parse_goal(raw: dict) -> MotionGoal:
    kind = raw.get("type")
    label = str(raw.get("label", ""))
    if kind == "tip":
        return MotionGoal.tip(raw["position_m"], label)
    if kind == "joints_radians":
        return MotionGoal.joints_radians(raw["positions"], label)
    if kind == "joints_degrees":
        return MotionGoal.joints_degrees(raw["positions"], label)
    raise ValueError(f"unknown goal type {kind!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", help="JSON planning request")
    parser.add_argument("--output-plan", help="save preview or approved path as .npy")
    args = parser.parse_args()

    try:
        request = json.loads(Path(args.request).read_text())
        goals = [parse_goal(raw) for raw in request["goals"]]
        policy_data = request.get("policy", {})
        policy = PlanningPolicy(
            requested_clearance_m=float(
                policy_data.get("requested_clearance_m", request.get("requested_clearance_m", 0.03))
            ),
            self_collision_margin_m=float(policy_data.get("self_collision_margin_m", 0.003)),
            path_step_rad=float(policy_data.get("path_step_rad", 0.02)),
            verification_step_rad=float(policy_data.get("verification_step_rad", 0.01)),
            planner_seeds=tuple(int(seed) for seed in policy_data.get("planner_seeds", (1, 7, 23))),
            max_map_age_seconds=policy_data.get("max_map_age_seconds"),
            max_calibration_age_seconds=policy_data.get("max_calibration_age_seconds"),
            approval_valid_seconds=policy_data.get("approval_valid_seconds"),
        )
        hardware_requested = request.get("mode", "preview") == "hardware"
        interlock_data = request.get("scene_interlock")
        scene_interlock = None if interlock_data is None else SceneInterlock(
            source=str(interlock_data["source"]),
            observed_at_unix=float(interlock_data["observed_at_unix"]),
            valid_for_seconds=float(interlock_data["valid_for_seconds"]),
        )
        outcome = plan_on_demand(
            request["start_joint_radians"],
            goals,
            request["map_path"],
            request["arm_xml_path"],
            policy,
            hardware_requested=hardware_requested,
            calibration_path=request.get("calibration_path"),
            scene_interlock=scene_interlock,
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        decision = SafetyDecision.refuse(SafetyCode.INVALID_REQUEST, str(error))
        print(json.dumps(decision.to_dict(), indent=2))
        return 2

    print(json.dumps(outcome.to_dict(), indent=2, allow_nan=False))
    path = outcome.approved_plan.path if outcome.approved_plan is not None else outcome.preview_path
    if path is not None and args.output_plan:
        np.save(args.output_plan, path)
    return 0 if outcome.decision.allowed or outcome.preview_path is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
