"""The safety checker the planner actually runs against.

Two independent questions, answered by the tool suited to each:

* **Does the arm hit the world?** Answered against the voxel distance field,
  using conservative spheres fitted to the arm's meshes. This covers arbitrary
  scanned geometry -- including a table the arm can reach underneath -- which no
  primitive-shape model describes.
* **Does the arm hit itself?** Answered by MuJoCo against exact convex meshes.

The base link is excluded from environment checks: it is bolted to the table and
resting against it, so it is permanently "in collision" with the surface it is
mounted on, and it cannot move into anything new.
"""

from typing import List, Optional, Sequence

import numpy as np

from yam.collision import Box, GRIPPER_LINKS, World
from yam.mujoco_collision import MujocoCollisionChecker
from yam.voxel_map import VoxelMap

MOVING_LINKS = ("link1", "link2", "link3", "link4", "link5", "gripper", "tip_left", "tip_right")


class ArmSafetyChecker:
    def __init__(
        self,
        kinematics,
        voxel_map: VoxelMap,
        arm_xml_path: str,
        margin: float = 0.03,
        self_collision_margin: float = 0.003,
        links: Sequence[str] = MOVING_LINKS,
        calibration_samples: int = 1200,
    ):
        self.kinematics = kinematics
        self.map = voxel_map
        self.requested_margin = margin
        self.registration_uncertainty = voxel_map.uncertainty
        self.measured_margin = margin + self.registration_uncertainty
        self.synthetic_margin = margin
        # Kept for reporting compatibility: the largest world-model margin.
        self.margin = margin + self.registration_uncertainty
        self.links = tuple(links)
        self.map.compute_distance_field()

        synthetic_boxes = [
            Box(
                f"base_clamp_{index}",
                np.asarray(box["min"], dtype=float),
                np.asarray(box["max"], dtype=float),
            )
            for index, box in enumerate(
                voxel_map.provenance.get("synthetic_geometry", {}).get("base_clamps", [])
            )
        ]

        # MuJoCo checks the exact arm meshes against explicit boxes and also
        # checks self-collision. The URDF spheres remain authoritative for the
        # measured LiDAR and for gripper geometry missing from the MJCF.
        self.self_checker = MujocoCollisionChecker(
            arm_xml_path,
            World(obstacles=synthetic_boxes, ground_z=None, margin=self.synthetic_margin),
            calibration_samples=calibration_samples,
            self_collision_margin=self_collision_margin,
        )

    def environment_clearances(self, q: Sequence[float]) -> tuple[float, float]:
        centers, radii = self.kinematics.collision_spheres(q, self.links)
        if len(centers) == 0:
            return float("inf"), float("inf")
        measured = float((self.map.measured_distance_at(centers) - radii).min())
        synthetic = self._synthetic_clearance(q, centers, radii)
        return measured, synthetic

    def _synthetic_clearance(
        self,
        q: Sequence[float],
        centers: np.ndarray,
        radii: np.ndarray,
    ) -> float:
        boxes = (
            self.map.provenance.get("synthetic_geometry", {}).get("base_clamps", [])
            if self.map.provenance
            else []
        )
        if not boxes:
            return float((self.map.synthetic_distance_at(centers) - radii).min())

        smallest = self.self_checker.obstacle_clearance(q) + self.synthetic_margin
        gripper_centers, gripper_radii = self.kinematics.collision_spheres(q, GRIPPER_LINKS)
        for box in boxes:
            minimum = np.asarray(box["min"], dtype=float)
            maximum = np.asarray(box["max"], dtype=float)
            outside = np.maximum(
                np.maximum(minimum - gripper_centers, gripper_centers - maximum),
                0.0,
            )
            smallest = min(
                smallest,
                float((np.linalg.norm(outside, axis=1) - gripper_radii).min()),
            )
        return smallest

    def environment_clearance(self, q: Sequence[float]) -> float:
        """Raw gap to the nearest measured or explicitly modeled obstacle."""
        return min(self.environment_clearances(q))

    def environment_slack(self, q: Sequence[float]) -> float:
        measured, synthetic = self.environment_clearances(q)
        return min(measured - self.measured_margin, synthetic - self.synthetic_margin)

    def environment_is_free(self, q: Sequence[float]) -> bool:
        return self.environment_slack(q) > 0.0

    def self_is_free(self, q: Sequence[float]) -> bool:
        return self.self_checker.self_is_free(q)

    def is_free(self, q: Sequence[float]) -> bool:
        return self.environment_is_free(q) and self.self_is_free(q)

    def clearance(self, q: Sequence[float]) -> float:
        return min(self.environment_slack(q), self.self_checker.self_clearance(q))

    def explain(self, q: Sequence[float]) -> List[str]:
        reasons = []
        measured_gap, synthetic_gap = self.environment_clearances(q)
        if measured_gap <= self.measured_margin:
            centers, radii = self.kinematics.collision_spheres(q, self.links)
            worst = int(np.argmin(self.map.measured_distance_at(centers) - radii))
            reasons.append(
                f"measured environment: arm sphere at {np.round(centers[worst], 3)} is "
                f"{measured_gap * 1000:+.1f}mm from LiDAR (needs {self.measured_margin * 1000:.0f}mm: "
                f"{self.requested_margin * 1000:.0f}mm clearance + "
                f"{self.registration_uncertainty * 1000:.0f}mm registration uncertainty)"
            )
        if synthetic_gap <= self.synthetic_margin:
            centers, radii = self.kinematics.collision_spheres(q, self.links)
            reasons.append(
                f"modeled environment: moving arm is {synthetic_gap * 1000:+.1f}mm "
                f"from explicit geometry "
                f"(needs {self.synthetic_margin * 1000:.0f}mm)"
            )
        reasons.extend(self.self_checker.explain(q))
        return reasons

    def segment_is_free(self, start: Sequence[float], end: Sequence[float], resolution: float = 0.05) -> bool:
        start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        steps = max(int(np.ceil(np.abs(end - start).max() / resolution)), 1)
        for index in range(steps + 1):
            if not self.is_free(start + (end - start) * (index / steps)):
                return False
        return True
