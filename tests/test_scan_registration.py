import unittest

import numpy as np

from yam.scan_registration import (
    ARKIT_TO_ZUP,
    ScanRegistration,
    refine_from_seed,
    yaw_matrix,
)


class ScanRegistrationTests(unittest.TestCase):
    def test_same_base_with_opposite_yaw_does_not_agree(self):
        first = ScanRegistration(np.eye(3), np.zeros(3), 0.01, 100, 100, 1)
        second = ScanRegistration(yaw_matrix(np.pi), np.zeros(3), 0.01, 100, 100, 1)

        self.assertFalse(first.agrees_with(second))

    def test_arm_fit_without_independent_surfaces_is_inconclusive(self):
        result = ScanRegistration(
            rotation=np.eye(3),
            translation=np.zeros(3),
            rmse=0.01,
            inliers=100,
            model_points=100,
            searched=1,
            model_p95_error=0.02,
        )

        self.assertEqual(result.verdict, "inconclusive")
        self.assertFalse(result.is_trustworthy)

    def test_seeded_fit_recovers_gravity_constrained_transform(self):
        rng = np.random.default_rng(4)
        model = np.vstack([
            rng.uniform([0.00, -0.04, 0.00], [0.08, 0.04, 0.24], size=(100, 3)),
            rng.uniform([0.04, -0.03, 0.18], [0.30, 0.03, 0.24], size=(100, 3)),
        ])
        surfaces = np.array([
            [-0.45, -0.30, 0.00],
            [-0.40, 0.35, 0.00],
            [0.38, -0.32, -0.42],
            [0.46, 0.31, -0.42],
            [0.18, 0.48, -0.42],
        ])
        true_robot_to_upright = yaw_matrix(0.72)
        true_translation = np.array([0.32, -0.24, 0.48])
        upright = np.vstack([
            model @ true_robot_to_upright.T + true_translation,
            surfaces @ true_robot_to_upright.T + true_translation,
            rng.uniform([-0.8, -0.8, -0.2], [0.8, 0.8, 1.0], size=(300, 3)),
        ])
        upright += rng.normal(scale=0.002, size=upright.shape)
        scan = upright @ ARKIT_TO_ZUP
        seed = true_translation @ ARKIT_TO_ZUP

        result = refine_from_seed(
            scan,
            model,
            seed,
            surface_points=surfaces,
            search_radius=0.12,
        )

        self.assertIsNotNone(result)
        expected_rotation = true_robot_to_upright.T @ ARKIT_TO_ZUP
        expected_translation = -true_robot_to_upright.T @ true_translation
        relative = result.rotation @ expected_rotation.T
        rotation_error = np.arccos(
            np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
        )
        self.assertLess(rotation_error, np.deg2rad(3.0))
        self.assertLess(np.linalg.norm(result.translation - expected_translation), 0.03)
        self.assertTrue(result.is_trustworthy)
        self.assertTrue(np.allclose(result.rotation[2], [0.0, 1.0, 0.0], atol=1e-8))


if __name__ == "__main__":
    unittest.main()
