import os
import tempfile
import time
import unittest

import numpy as np

from yam.action_safety import apply_motion_safety
from yam.execution import ExecutionAborted, GuardedExecutor, GuardLimits
from yam.hardware_calibration import file_sha256
from yam.planner import PlannerConfig, RRTConnectPlanner
from yam.safe_planning import PlanningOutcome, validate_map_for_hardware, verify_tracking_envelope
from yam.safety_contract import SafetyCode, SafetyDecision
from yam.voxel_map import VoxelMap


class BoxObstacleChecker:
    def is_free(self, q):
        x, y = np.asarray(q)
        return not (-0.25 < x < 0.25 and -0.35 < y < 0.35)

    def segment_is_free(self, start, end, resolution=0.02):
        start, end = np.asarray(start), np.asarray(end)
        steps = max(int(np.ceil(np.max(np.abs(end - start)) / resolution)), 1)
        return all(self.is_free(start + (end - start) * (i / steps)) for i in range(steps + 1))


class CornerCollisionChecker:
    def is_free(self, q):
        return not np.all(np.asarray(q) > 0.05)

    def clearance(self, q):
        return 1.0 if self.is_free(q) else -1.0

    def explain(self, q):
        return ["corner collision"] if not self.is_free(q) else []


class NeverCommandArm:
    joints = []

    def read_state(self):
        raise AssertionError("raw path must be rejected before reading or commanding the arm")


class MotionSafetyTests(unittest.TestCase):
    def test_rrt_path_always_runs_start_to_goal_after_tree_swaps(self):
        start = np.array([-0.9, 0.0])
        goal = np.array([0.9, 0.0])
        for seed in range(12):
            planner = RRTConnectPlanner(
                BoxObstacleChecker(),
                [-1.0, -1.0],
                [1.0, 1.0],
                PlannerConfig(seed=seed, step_size=0.12, collision_resolution=0.02),
            )
            path = planner.plan(start, goal)
            self.assertTrue(np.allclose(path[0], start), seed)
            self.assertTrue(np.allclose(path[-1], goal), seed)

    def test_hardware_map_rejects_changed_source(self):
        with tempfile.TemporaryDirectory() as directory:
            scan_path = os.path.join(directory, "scan.ply")
            registration_path = os.path.join(directory, "registration.json")
            with open(scan_path, "wb") as handle:
                handle.write(b"measured scan")
            with open(registration_path, "wb") as handle:
                handle.write(b"measured registration")

            voxel_map = VoxelMap.from_bounds([-1, -1, -1], [1, 1, 1], 0.1)
            voxel_map.provenance = {
                "schema_version": 1,
                "scan": {"path": scan_path, "sha256": file_sha256(scan_path), "captured_at_unix": time.time()},
                "registration": {"path": registration_path, "sha256": file_sha256(registration_path)},
            }
            with open(scan_path, "ab") as handle:
                handle.write(b" changed")

            decision = validate_map_for_hardware(voxel_map, 60.0)
            self.assertFalse(decision.allowed)
            self.assertEqual(decision.code, SafetyCode.MAP_SOURCE_CHANGED)

    def test_tracking_check_includes_six_dimensional_corners(self):
        report = verify_tracking_envelope(
            CornerCollisionChecker(),
            [np.zeros(6)],
            [0.1] * 6,
            [-1.0] * 6,
            [1.0] * 6,
        )
        self.assertFalse(report["ok"])
        self.assertEqual(report["samples_per_pose"], 77)
        self.assertEqual(report["first_failure"]["offset_deg"], [5.73] * 6)

    def test_executor_rejects_raw_path_even_with_validated_limits(self):
        limits = GuardLimits(
            hardware_validated=True,
            calibration_sha256="a" * 64,
            calibrated_rate_hz=100.0,
            calibrated_gain_scale=0.5,
        )
        executor = GuardedExecutor(NeverCommandArm(), limits=limits, map_sha256="b" * 64)
        with self.assertRaisesRegex(ExecutionAborted, "no safety approval certificate"):
            executor.run([[0.0] * 6])

    def test_understood_task_becomes_explicit_safety_refusal(self):
        task = {"tier": "act", "matched_task_id": "known-task"}
        planning = PlanningOutcome(SafetyDecision.refuse(SafetyCode.PATH_UNSAFE, "path intersects table"))
        combined = apply_motion_safety(task, planning)

        self.assertEqual(combined["tier"], "act")
        self.assertFalse(combined["motion_allowed"])
        self.assertEqual(combined["response_kind"], "safety_refusal")
        self.assertIn("I won't move the arm", combined["spoken_response"])


if __name__ == "__main__":
    unittest.main()
