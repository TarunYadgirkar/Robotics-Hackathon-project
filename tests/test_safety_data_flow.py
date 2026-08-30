import os
import tempfile
import unittest

import numpy as np

from yam.enrollment import EnrollmentSession, PoseSample, touch_repeatability
from yam.execution import ExecutionAborted, GuardedExecutor
from yam.kinematics import numerical_jacobian
from yam.lidar import gravity_aligned_kabsch, kabsch, scan_timestamp_from_path
from yam.scan_registration import ARKIT_TO_ZUP, yaw_matrix
from yam.voxel_map import VoxelMap


class ProbeOnlyKinematics:
    def probe_position(self, q):
        values = np.asarray(q, dtype=float)
        return values[:3]

    def tip_position(self, q):
        raise AssertionError("gripper frame origin must not be used as the jaw tip")


class CommandRejectingArm:
    joints = []

    def command_positions(self, targets, gain_scale):
        raise AssertionError("unvalidated execution must stop before commanding the arm")


class SafetyDataFlowTests(unittest.TestCase):
    def test_scan_timestamp_selects_pose_held_at_upload(self):
        session = EnrollmentSession()
        session.pose_log = [
            PoseSample(100.0, [0.0] * 6),
            PoseSample(120.0, [1.0] * 6),
            PoseSample(160.0, [2.0] * 6),
        ]

        timestamp = scan_timestamp_from_path("/tmp/phone_scan_150.ply")
        sample, index = session.pose_at(timestamp)

        self.assertEqual(index, 1)
        self.assertEqual(sample.joint_angles, [1.0] * 6)

    def test_kabsch_rejects_collinear_correspondences(self):
        points = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0]])
        with self.assertRaisesRegex(ValueError, "collinear"):
            kabsch(points, points)

    def test_three_landmarks_cannot_claim_validation(self):
        source = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.3, 0.0]])
        registration = kabsch(source, source)
        self.assertFalse(registration.is_trustworthy)

    def test_landmark_fit_preserves_arkit_gravity(self):
        robot = np.array([
            [-0.4, -0.3, 0.0],
            [-0.4, 0.3, 0.0],
            [0.4, -0.3, -0.4],
            [0.4, 0.3, -0.4],
        ])
        robot_to_upright = yaw_matrix(0.6)
        translation = np.array([0.3, -0.2, 0.5])
        scan = (robot @ robot_to_upright.T + translation) @ ARKIT_TO_ZUP

        registration = gravity_aligned_kabsch(scan, robot)

        self.assertTrue(registration.is_trustworthy)
        self.assertTrue(np.allclose(registration.apply(scan), robot, atol=1e-9))
        self.assertTrue(np.allclose(registration.rotation[2], [0.0, 1.0, 0.0], atol=1e-9))

    def test_registration_uncertainty_survives_map_round_trip(self):
        voxel_map = VoxelMap.from_bounds([-1, -1, -1], [1, 1, 1], resolution=0.1)
        voxel_map.uncertainty = 0.037
        voxel_map.add_points([[0.2, 0.3, 0.4]])

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "map.npz")
            voxel_map.save(path)
            loaded = VoxelMap.load(path)

        self.assertAlmostEqual(loaded.uncertainty, 0.037)
        self.assertTrue(np.array_equal(loaded.occupancy, voxel_map.occupancy))

    def test_robot_carving_preserves_geometry_below_mounting_datum(self):
        voxel_map = VoxelMap.from_bounds([-0.2, -0.2, -0.2], [0.2, 0.2, 0.2], resolution=0.02)
        below = np.array([[0.0, 0.0, -0.05]])
        above = np.array([[0.0, 0.0, 0.05]])
        voxel_map.add_points(np.vstack([below, above]))

        voxel_map.carve_spheres(
            centers=[[0.0, 0.0, 0.0]],
            radii=[0.1],
            protect_below_z=0.0,
        )

        below_index = voxel_map.to_indices(below)[0]
        above_index = voxel_map.to_indices(above)[0]
        self.assertTrue(voxel_map.occupancy[tuple(below_index)])
        self.assertFalse(voxel_map.occupancy[tuple(above_index)])

    def test_cartesian_helpers_use_jaw_tip_probe(self):
        kinematics = ProbeOnlyKinematics()
        q = np.arange(6, dtype=float)

        jacobian = numerical_jacobian(kinematics, q)
        repeatability = touch_repeatability(kinematics, [q, q])

        self.assertTrue(np.allclose(jacobian[:, :3], np.eye(3), atol=1e-8))
        self.assertTrue(np.allclose(jacobian[:, 3:], 0.0, atol=1e-8))
        self.assertEqual(repeatability["max_error_mm"], 0.0)

    def test_unvalidated_contact_guard_cannot_command_arm(self):
        executor = GuardedExecutor(CommandRejectingArm())
        with self.assertRaisesRegex(ExecutionAborted, "not hardware-validated"):
            executor.run([[0.0] * 6])


if __name__ == "__main__":
    unittest.main()
