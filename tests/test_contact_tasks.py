import hashlib
import time
import unittest
from dataclasses import dataclass
from typing import List

import numpy as np

from yam.contact_planning import corridor_obstructions, straight_probe, travel_bound
from yam.execution import ContactReport, ExecutionAborted, GuardedExecutor, GuardLimits
from yam.kinematics import YamKinematics
from yam.safety_contract import ApprovedContact, SafetyCode
from yam.surface import SurfaceUnknown, estimate_normal
from yam.voxel_map import VoxelMap


def plane_map(axis: int = 2, thickness: int = 1) -> VoxelMap:
    """A single flat scanned surface, the way a LiDAR sees one: a shell."""
    voxel_map = VoxelMap.from_bounds([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5], resolution=0.02)
    index = voxel_map.occupancy.shape[axis] // 2
    slicer = [slice(None)] * 3
    slicer[axis] = slice(index, index + thickness)
    voxel_map.occupancy[tuple(slicer)] = True
    return voxel_map


@dataclass
class FakeFeedback:
    temperature_mos: float = 30.0
    temperature_rotor: float = 30.0
    is_healthy: bool = True
    error_message: str = ""


@dataclass
class FakeJoint:
    name: str


class ScriptedArm:
    """An arm whose torque follows a script, so a probe can be tested offline."""

    def __init__(self, torque_by_step: List[List[float]]):
        self.joints = [FakeJoint(f"joint{i + 1}") for i in range(6)]
        self.torque_by_step = torque_by_step
        self.commands: List[np.ndarray] = []

    def _state(self):
        index = min(len(self.commands), len(self.torque_by_step) - 1)
        torque = self.torque_by_step[max(index, 0)]

        class State:
            positions = list(self.commands[-1]) if self.commands else [0.0] * 6
            torques = list(torque)
            feedback = [FakeFeedback() for _ in range(6)]

        return State()

    def read_state(self):
        return self._state()

    def command_positions(self, target, gain_scale=None):
        self.commands.append(np.asarray(target, dtype=float))
        return self._state()


def approved_contact(approach, probe, limits, map_sha="map", travel=0.05):
    def digest(array):
        return hashlib.sha256(np.asarray(array, dtype="<f8").tobytes()).hexdigest()

    return ApprovedContact(
        approach=approach, probe=probe,
        approach_sha256=digest(approach), probe_sha256=digest(probe),
        map_sha256=map_sha, calibration_sha256=limits.calibration_sha256,
        issued_at_unix=time.time(), valid_for_seconds=60.0,
        start_tolerance_rad=(0.2,) * 6, approach_direction=[0, 0, -1], max_travel_m=travel)


def probe_limits(residual=(1.0,) * 6):
    return GuardLimits(max_torque_residual=residual, absolute_torque=50.0,
                       max_tracking_error=1.0, max_temperature=90.0,
                       warmup_seconds=0.0, baseline_seconds=0.35,
                       hardware_validated=True, calibration_sha256="cal",
                       calibrated_rate_hz=100.0, calibrated_gain_scale=0.5)


class SurfaceNormals(unittest.TestCase):
    def test_normal_is_perpendicular_to_a_scanned_plane(self):
        for axis in (0, 1, 2):
            sample = estimate_normal(plane_map(axis), np.zeros(3), radius=0.10)
            expected = np.zeros(3)
            expected[axis] = 1.0
            self.assertAlmostEqual(abs(float(np.dot(sample.axis, expected))), 1.0, places=2)

    def test_both_approach_directions_are_offered(self):
        sample = estimate_normal(plane_map(), np.zeros(3), radius=0.10)
        directions = sample.approach_directions()
        self.assertEqual(len(directions), 2)
        self.assertAlmostEqual(float(np.dot(directions[0], directions[1])), -1.0, places=6)

    def test_empty_space_is_refused_rather_than_guessed(self):
        with self.assertRaises(SurfaceUnknown):
            estimate_normal(plane_map(), np.array([0.0, 0.0, 0.4]), radius=0.05)

    def test_a_corner_is_refused_as_not_planar(self):
        voxel_map = plane_map(axis=2)
        voxel_map.occupancy[:, voxel_map.occupancy.shape[1] // 2:, :] = True
        with self.assertRaises(SurfaceUnknown):
            estimate_normal(voxel_map, np.zeros(3), radius=0.10)


class TravelBounds(unittest.TestCase):
    def test_travel_covers_registration_and_patch_uncertainty(self):
        voxel_map = plane_map()
        voxel_map.uncertainty = 0.035
        bound = travel_bound(voxel_map, residual_m=0.005)
        self.assertGreater(bound, voxel_map.uncertainty)
        self.assertAlmostEqual(bound, 0.035 + 0.010 + 0.02, places=9)

    def test_a_noisier_patch_earns_more_travel(self):
        voxel_map = plane_map()
        voxel_map.uncertainty = 0.01
        self.assertGreater(travel_bound(voxel_map, 0.02), travel_bound(voxel_map, 0.002))


class StraightProbe(unittest.TestCase):
    def setUp(self):
        self.kinematics = YamKinematics()
        self.start = np.array([0.0498, 0.6, 0.4, -0.0906, 0.0734, 1.1706])
        self.lower = np.full(6, -3.0)
        self.upper = np.full(6, 3.0)

    def test_the_tip_travels_along_the_requested_direction_only(self):
        direction = np.array([0.0, 0.0, -1.0])
        poses = straight_probe(self.kinematics, self.start, direction, 0.05, 0.001,
                               self.lower, self.upper)
        tips = np.array([self.kinematics.probe_position(q) for q in poses])
        travel = tips[-1] - tips[0]
        self.assertAlmostEqual(travel[2], -0.05, places=3)
        self.assertLess(np.linalg.norm(travel[:2]), 0.002)

    def test_the_arm_keeps_its_configuration_instead_of_drifting(self):
        poses = straight_probe(self.kinematics, self.start, np.array([0.0, 0.0, -1.0]),
                               0.05, 0.001, self.lower, self.upper)
        self.assertLess(float(np.degrees(np.abs(poses[:, 0] - poses[0, 0]).max())), 2.0)


class CorridorChecks(unittest.TestCase):
    def test_a_body_link_driven_into_the_scan_is_reported(self):
        kinematics = YamKinematics()
        voxel_map = VoxelMap.from_bounds([-1, -1, -1], [1, 1, 1], resolution=0.02)
        voxel_map.occupancy[:] = True
        poses = np.array([[0.0, 0.5, 0.5, 0.0, 0.0, 0.0]])
        found = corridor_obstructions(kinematics, voxel_map, poses,
                                      body_links=("link1", "link2", "link3", "link4", "link5"))
        self.assertTrue(found)
        self.assertEqual(found[0]["probe_index"], 0)

    def test_open_space_reports_nothing(self):
        kinematics = YamKinematics()
        voxel_map = VoxelMap.from_bounds([-1, -1, -1], [1, 1, 1], resolution=0.02)
        poses = np.array([[0.0, 0.5, 0.5, 0.0, 0.0, 0.0]])
        self.assertEqual(
            corridor_obstructions(kinematics, voxel_map, poses,
                                  body_links=("link1", "link2", "link3", "link4", "link5")),
            [])


class ProbeExecution(unittest.TestCase):
    def setUp(self):
        self.limits = probe_limits()
        self.approach = np.zeros((3, 6))
        self.probe = np.linspace(0.0, 0.01, 40)[:, None] * np.ones((1, 6))

    def _executor(self, torques):
        arm = ScriptedArm(torques)
        executor = GuardedExecutor(arm, checker=None, limits=self.limits, map_sha256="map")
        executor.limits.require_free = False
        return arm, executor

    def test_a_torque_step_during_the_probe_is_contact_not_an_abort(self):
        quiet = [[0.0] * 6] * 20
        step = [[0.0, 0.0, 5.0, 0.0, 0.0, 0.0]] * 60
        arm, executor = self._executor(quiet + step)
        report = executor.touch(approved_contact(self.approach, self.probe, self.limits),
                                rate_hz=100.0, gain_scale=0.5)
        self.assertIsInstance(report, ContactReport)
        self.assertTrue(report.contacted)
        self.assertEqual(report.joint, "joint3")
        self.assertLess(len(arm.commands), len(self.approach) + len(self.probe))

    def test_a_probe_that_feels_nothing_stops_at_its_approved_travel(self):
        arm, executor = self._executor([[0.0] * 6] * 500)
        report = executor.touch(approved_contact(self.approach, self.probe, self.limits),
                                rate_hz=100.0, gain_scale=0.5)
        self.assertFalse(report.contacted)
        self.assertEqual(len(arm.commands), len(self.approach) + len(self.probe))

    def test_a_substituted_probe_is_refused(self):
        contact = approved_contact(self.approach, self.probe, self.limits)
        object.__setattr__(contact, "probe", self.probe * 2.0)
        _, executor = self._executor([[0.0] * 6] * 500)
        with self.assertRaises(ExecutionAborted):
            executor.touch(contact, rate_hz=100.0, gain_scale=0.5)

    def test_an_uncalibrated_guard_cannot_probe(self):
        limits = probe_limits()
        limits.hardware_validated = False
        _, executor = self._executor([[0.0] * 6] * 500)
        executor.limits = limits
        with self.assertRaises(ExecutionAborted):
            executor.touch(approved_contact(self.approach, self.probe, limits),
                           rate_hz=100.0, gain_scale=0.5)


if __name__ == "__main__":
    unittest.main()
