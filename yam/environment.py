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

from yam.collision import World
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
        self.margin = margin
        self.links = tuple(links)
        self.map.compute_distance_field()

        # An empty world: this instance answers self-collision only, so the
        # environment lives entirely in the voxel map.
        self.self_checker = MujocoCollisionChecker(
            arm_xml_path,
            World(obstacles=[], ground_z=None, margin=0.0),
            calibration_samples=calibration_samples,
            self_collision_margin=self_collision_margin,
        )

    def environment_clearance(self, q: Sequence[float]) -> float:
        centers, radii = self.kinematics.collision_spheres(q, self.links)
        if len(centers) == 0:
            return float("inf")
        return float((self.map.distance_at(centers) - radii).min())

    def environment_is_free(self, q: Sequence[float]) -> bool:
        return self.environment_clearance(q) > self.margin

    def self_is_free(self, q: Sequence[float]) -> bool:
        return self.self_checker.is_free(q)

    def is_free(self, q: Sequence[float]) -> bool:
        return self.environment_is_free(q) and self.self_is_free(q)

    def clearance(self, q: Sequence[float]) -> float:
        return min(self.environment_clearance(q) - self.margin, self.self_checker.clearance(q))

    def explain(self, q: Sequence[float]) -> List[str]:
        reasons = []
        gap = self.environment_clearance(q)
        if gap <= self.margin:
            centers, radii = self.kinematics.collision_spheres(q, self.links)
            worst = int(np.argmin(self.map.distance_at(centers) - radii))
            reasons.append(
                f"environment: arm sphere at {np.round(centers[worst], 3)} is {gap * 1000:+.1f}mm "
                f"from the map (needs {self.margin * 1000:.0f}mm)"
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
