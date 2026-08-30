"""Collision-free joint-space planning with RRT-Connect and shortcut smoothing.

RRT-Connect finds a feasible path quickly; on its own that path wanders, because
it is built from random samples. The shortcutting pass is what makes the result
short -- it repeatedly tries to replace a chunk of the path with a straight line
in joint space and keeps the replacement whenever it stays collision-free.

The result is locally optimal, not globally: it is a good route around the
obstacles, not a proof that no better homotopy exists.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class PlannerConfig:
    step_size: float = 0.15
    max_iterations: int = 6000
    goal_bias: float = 0.1
    collision_resolution: float = 0.05
    shortcut_iterations: int = 300
    seed: Optional[int] = None


class PlanningError(RuntimeError):
    pass


class RRTConnectPlanner:
    def __init__(self, checker, lower_limits: Sequence[float], upper_limits: Sequence[float],
                 config: Optional[PlannerConfig] = None):
        self.checker = checker
        self.lower = np.asarray(lower_limits, dtype=float)
        self.upper = np.asarray(upper_limits, dtype=float)
        self.config = config or PlannerConfig()
        self.rng = np.random.default_rng(self.config.seed)

    def _sample(self) -> np.ndarray:
        return self.lower + self.rng.random(len(self.lower)) * (self.upper - self.lower)

    def _steer(self, origin: np.ndarray, target: np.ndarray) -> np.ndarray:
        delta = target - origin
        distance = np.linalg.norm(delta)
        if distance <= self.config.step_size:
            return target
        return origin + delta * (self.config.step_size / distance)

    def _extend(self, tree: List[np.ndarray], parents: List[int], target: np.ndarray) -> Optional[int]:
        nodes = np.array(tree)
        nearest = int(np.argmin(np.linalg.norm(nodes - target, axis=1)))
        candidate = self._steer(nodes[nearest], target)

        if not self.checker.segment_is_free(nodes[nearest], candidate, self.config.collision_resolution):
            return None
        tree.append(candidate)
        parents.append(nearest)
        return len(tree) - 1

    @staticmethod
    def _path_to_root(tree: List[np.ndarray], parents: List[int], index: int) -> List[np.ndarray]:
        path = []
        while index != -1:
            path.append(tree[index])
            index = parents[index]
        return path[::-1]

    def plan(self, start: Sequence[float], goal: Sequence[float]) -> List[np.ndarray]:
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)

        if not self.checker.is_free(start):
            raise PlanningError("start configuration is already in collision")
        if not self.checker.is_free(goal):
            raise PlanningError("goal configuration is in collision")

        if self.checker.segment_is_free(start, goal, self.config.collision_resolution):
            return self.shortcut([start, goal])

        start_tree, start_parents = [start], [-1]
        goal_tree, goal_parents = [goal], [-1]

        for iteration in range(self.config.max_iterations):
            target = goal if self.rng.random() < self.config.goal_bias else self._sample()

            grown = self._extend(start_tree, start_parents, target)
            if grown is not None:
                bridge = self._extend(goal_tree, goal_parents, start_tree[grown])
                if bridge is not None and self.checker.segment_is_free(
                    start_tree[grown], goal_tree[bridge], self.config.collision_resolution
                ):
                    path = self._path_to_root(start_tree, start_parents, grown)
                    path += self._path_to_root(goal_tree, goal_parents, bridge)[::-1]
                    return self.shortcut(path)

            start_tree, start_parents, goal_tree, goal_parents = goal_tree, goal_parents, start_tree, start_parents

        raise PlanningError(
            f"no collision-free path found in {self.config.max_iterations} iterations; "
            "the goal may be walled off, or the margin may be too large"
        )

    def shortcut(self, path: List[np.ndarray]) -> List[np.ndarray]:
        path = [np.asarray(p, dtype=float) for p in path]
        for _ in range(self.config.shortcut_iterations):
            if len(path) <= 2:
                break
            i, j = sorted(self.rng.integers(0, len(path), size=2))
            if j - i < 2:
                continue
            if self.checker.segment_is_free(path[i], path[j], self.config.collision_resolution):
                path = path[: i + 1] + path[j:]
        return path


def path_length(path: Sequence[np.ndarray]) -> float:
    return float(sum(np.linalg.norm(path[i + 1] - path[i]) for i in range(len(path) - 1)))


def resample(path: Sequence[np.ndarray], step: float = 0.02) -> List[np.ndarray]:
    """Densify a waypoint path so it can be streamed to the controller."""
    dense = [np.asarray(path[0], dtype=float)]
    for start, end in zip(path[:-1], path[1:]):
        start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        steps = max(int(np.ceil(np.linalg.norm(end - start) / step)), 1)
        for index in range(1, steps + 1):
            dense.append(start + (end - start) * (index / steps))
    return dense
