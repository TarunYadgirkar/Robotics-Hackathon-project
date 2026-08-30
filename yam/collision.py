"""Obstacles and whole-arm collision checking.

Obstacles are axis-aligned boxes and spheres in the robot's base frame, plus an
optional ground plane. The arm is the union of the sphere set from
`YamKinematics.collision_spheres`, so a check covers every link -- the elbow and
the back of the arm included, not just the gripper.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class Box:
    """Axis-aligned box in the base frame."""

    name: str
    minimum: np.ndarray
    maximum: np.ndarray

    @property
    def center(self) -> np.ndarray:
        return (self.minimum + self.maximum) / 2.0

    @property
    def size(self) -> np.ndarray:
        return self.maximum - self.minimum

    def distance_to_points(self, points: np.ndarray) -> np.ndarray:
        """Euclidean distance from each point to the box surface; 0 inside."""
        outside = np.maximum(np.maximum(self.minimum - points, points - self.maximum), 0.0)
        return np.linalg.norm(outside, axis=1)

    def to_dict(self) -> Dict:
        return {"type": "box", "name": self.name, "min": self.minimum.tolist(), "max": self.maximum.tolist()}


@dataclass
class Sphere:
    name: str
    center: np.ndarray
    radius: float

    def distance_to_points(self, points: np.ndarray) -> np.ndarray:
        return np.maximum(np.linalg.norm(points - self.center, axis=1) - self.radius, 0.0)

    def to_dict(self) -> Dict:
        return {"type": "sphere", "name": self.name, "center": self.center.tolist(), "radius": self.radius}


@dataclass
class World:
    obstacles: List = field(default_factory=list)
    ground_z: Optional[float] = None
    margin: float = 0.03

    def add_box_from_points(self, name: str, points: Sequence[Sequence[float]], padding: float = 0.0) -> Box:
        array = np.asarray(points, dtype=float)
        box = Box(name, array.min(axis=0) - padding, array.max(axis=0) + padding)
        self.obstacles.append(box)
        return box

    def clearance(self, centers: np.ndarray, radii: np.ndarray) -> float:
        """Smallest gap between any arm sphere and any obstacle. Negative means overlap."""
        if len(centers) == 0:
            return float("inf")

        smallest = float("inf")
        for obstacle in self.obstacles:
            smallest = min(smallest, float((obstacle.distance_to_points(centers) - radii).min()))
        if self.ground_z is not None:
            smallest = min(smallest, float((centers[:, 2] - radii - self.ground_z).min()))
        return smallest

    def is_free(self, centers: np.ndarray, radii: np.ndarray) -> bool:
        return self.clearance(centers, radii) > self.margin

    def to_dict(self) -> Dict:
        return {
            "obstacles": [o.to_dict() for o in self.obstacles],
            "ground_z": self.ground_z,
            "margin": self.margin,
        }

    def save(self, path: str) -> None:
        with open(path, "w") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def from_dict(cls, data: Dict) -> "World":
        obstacles = []
        for entry in data.get("obstacles", []):
            if entry["type"] == "box":
                obstacles.append(Box(entry["name"], np.array(entry["min"]), np.array(entry["max"])))
            elif entry["type"] == "sphere":
                obstacles.append(Sphere(entry["name"], np.array(entry["center"]), float(entry["radius"])))
        return cls(obstacles=obstacles, ground_z=data.get("ground_z"), margin=data.get("margin", 0.03))

    @classmethod
    def load(cls, path: str) -> "World":
        with open(path) as handle:
            return cls.from_dict(json.load(handle))


#: The MJCF i2rt ships models base..link5 only, so these are the links MuJoCo cannot see.
GRIPPER_LINKS = ("gripper", "tip_left", "tip_right")


class CollisionChecker:
    """Sphere-based checking, optionally restricted to a subset of links.

    The sphere fit is deliberately conservative, which makes it a poor whole-arm
    model -- the base spheres are fat enough to sink through the table plane and
    reject every pose. It earns its place covering the gripper and tips, which
    the shipped MJCF has no geometry for at all.
    """

    def __init__(self, kinematics, world: World, links: Optional[Sequence[str]] = None):
        self.kinematics = kinematics
        self.world = world
        self.links = links

    def clearance(self, q: Sequence[float]) -> float:
        centers, radii = self.kinematics.collision_spheres(q, self.links)
        return self.world.clearance(centers, radii)

    def is_free(self, q: Sequence[float]) -> bool:
        centers, radii = self.kinematics.collision_spheres(q, self.links)
        return self.world.is_free(centers, radii)

    def segment_is_free(self, start: Sequence[float], end: Sequence[float], resolution: float = 0.05) -> bool:
        """Check a straight line in joint space, densely enough that nothing is stepped over."""
        start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        steps = max(int(np.ceil(np.abs(end - start).max() / resolution)), 1)
        for index in range(steps + 1):
            if not self.is_free(start + (end - start) * (index / steps)):
                return False
        return True
