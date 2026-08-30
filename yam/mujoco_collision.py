"""MuJoCo-backed collision checking: exact convex mesh geometry plus self-collision.

Obstacles are injected into the arm's MJCF as world-frame boxes, so one
`mj_forward` answers both "does the arm hit the world" and "does the arm hit
itself". Adjacent links touch by construction, so the pairs that are always in
contact are discovered by sampling and then ignored -- the same way MoveIt
builds its allowed-collision matrix, rather than hand-listing pairs.
"""

import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Sequence, Set, Tuple

import mujoco
import numpy as np

from yam.collision import Box, Sphere, World


def build_model_xml(arm_xml_path: str, world: World) -> str:
    tree = ET.parse(arm_xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")

    for obstacle in world.obstacles:
        if isinstance(obstacle, Box):
            center, size = obstacle.center, obstacle.size / 2.0
            ET.SubElement(worldbody, "geom", {
                "name": f"obstacle_{obstacle.name}",
                "type": "box",
                "pos": " ".join(f"{v:.6f}" for v in center),
                "size": " ".join(f"{max(v, 1e-4):.6f}" for v in size),
                "rgba": "0.85 0.3 0.3 0.4",
            })
        elif isinstance(obstacle, Sphere):
            ET.SubElement(worldbody, "geom", {
                "name": f"obstacle_{obstacle.name}",
                "type": "sphere",
                "pos": " ".join(f"{v:.6f}" for v in obstacle.center),
                "size": f"{obstacle.radius:.6f}",
                "rgba": "0.85 0.3 0.3 0.4",
            })

    if world.ground_z is not None:
        ET.SubElement(worldbody, "geom", {
            "name": "obstacle_ground",
            "type": "plane",
            "pos": f"0 0 {world.ground_z:.6f}",
            "size": "2 2 0.05",
            "rgba": "0.5 0.5 0.5 0.3",
        })

    return ET.tostring(root, encoding="unicode")


def load_mesh_assets(arm_xml_path: str) -> Dict[str, bytes]:
    """Mesh bytes keyed by the path the MJCF refers to.

    `from_xml_string` has no directory to resolve `assets/link2.stl` against, and
    writing a temp MJCF into the vendored model directory to use `from_xml_path`
    would mean mutating a read-only dependency.
    """
    root = ET.parse(arm_xml_path).getroot()
    model_dir = os.path.dirname(os.path.abspath(arm_xml_path))
    compiler = root.find("compiler")
    mesh_dir = compiler.get("meshdir", "") if compiler is not None else ""

    assets = {}
    for mesh in root.iter("mesh"):
        filename = mesh.get("file")
        if not filename:
            continue
        full_path = os.path.join(model_dir, mesh_dir, filename)
        if os.path.isfile(full_path):
            with open(full_path, "rb") as handle:
                assets[filename] = handle.read()
    return assets


class MujocoCollisionChecker:
    def __init__(self, arm_xml_path: str, world: World, calibration_samples: int = 400, seed: int = 0,
                 self_collision_margin: float = 0.003):
        self.world = world
        self.self_collision_margin = self_collision_margin
        self.model = mujoco.MjModel.from_xml_string(
            build_model_xml(arm_xml_path, world), load_mesh_assets(arm_xml_path)
        )
        self.data = mujoco.MjData(self.model)

        # Widen contact detection to the largest threshold we test against; the
        # per-pair threshold is then applied in `_violations`.
        self.model.geom_margin[:] = np.maximum(
            self.model.geom_margin, max(world.margin, self_collision_margin)
        )

        self.joint_count = self.model.nq
        self.ignored_pairs: Set[Tuple[int, int]] = set()
        self._calibrate_ignored_pairs(calibration_samples, seed)
        self._ignore_mounted_base()

    def _ignore_mounted_base(self) -> None:
        """The base is bolted to the table, so its resting contact with the ground is not a collision."""
        ground = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle_ground")
        if ground < 0:
            return
        base = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        for geom_id in range(self.model.ngeom):
            if self.model.geom_bodyid[geom_id] == base:
                self.ignored_pairs.add(tuple(sorted((geom_id, ground))))

    def _contact_pairs(self, q: Sequence[float]) -> List[Tuple[int, int, float]]:
        self.data.qpos[: self.joint_count] = np.asarray(q, dtype=float)[: self.joint_count]
        mujoco.mj_forward(self.model, self.data)
        pairs = []
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            g1, g2 = sorted((int(contact.geom1), int(contact.geom2)))
            pairs.append((g1, g2, float(contact.dist)))
        return pairs

    def _calibrate_ignored_pairs(self, samples: int, seed: int) -> None:
        """Ignore link pairs that touch in essentially every pose: those are adjacent by construction.

        Calibration runs at *zero* margin, and deliberately not at the checking
        margin. With a 20 mm margin, links that merely pass near each other in
        most poses also look permanently in contact -- link1/link3 does -- and
        excluding those would blind the checker to a real self-collision. At zero
        margin only genuine constant overlap qualifies, which on this model is
        the base/link1 shroud alone.
        """
        limits = self.model.jnt_range[: self.joint_count]
        rng = np.random.default_rng(seed)
        counts: Dict[Tuple[int, int], int] = {}

        margins = self.model.geom_margin.copy()
        self.model.geom_margin[:] = 0.0
        try:
            for _ in range(samples):
                q = limits[:, 0] + rng.random(self.joint_count) * (limits[:, 1] - limits[:, 0])
                # One vote per pair per pose: a flat face yields several contact
                # points, which would otherwise count as several observations.
                touching = {
                    (g1, g2)
                    for g1, g2, _dist in self._contact_pairs(q)
                    if not (self._is_obstacle(g1) or self._is_obstacle(g2))
                }
                for pair in touching:
                    counts[pair] = counts.get(pair, 0) + 1
        finally:
            self.model.geom_margin[:] = margins

        self.ignored_pairs = {pair for pair, count in counts.items() if count > 0.98 * samples}
        self.pair_frequencies = {pair: count / samples for pair, count in counts.items()}

    def _is_obstacle(self, geom_id: int) -> bool:
        name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        return bool(name and name.startswith("obstacle_"))

    def _threshold_for(self, g1: int, g2: int) -> float:
        """Clearance required of this pair.

        Obstacle pairs get the full keep-away margin. Self-collision pairs get a
        much smaller one, because links that are structurally near-adjacent sit
        closer than any useful obstacle margin -- base/link2 rides at 9 mm and
        link3/link5 at 12 mm in ordinary poses. Judging those by a 20 mm obstacle
        margin would mark almost every pose in collision.
        """
        if self._is_obstacle(g1) or self._is_obstacle(g2):
            return self.world.margin
        return self.self_collision_margin

    def _violations(self, q: Sequence[float]) -> List[Tuple[int, int, float, float]]:
        found = []
        for g1, g2, dist in self._contact_pairs(q):
            if (g1, g2) in self.ignored_pairs:
                continue
            threshold = self._threshold_for(g1, g2)
            if dist < threshold:
                found.append((g1, g2, dist, threshold))
        return found

    def is_free(self, q: Sequence[float]) -> bool:
        return not self._violations(q)

    def clearance(self, q: Sequence[float]) -> float:
        """Worst margin-relative slack: how much room is left before the tightest pair trips."""
        smallest = float("inf")
        for g1, g2, dist in self._contact_pairs(q):
            if (g1, g2) in self.ignored_pairs:
                continue
            smallest = min(smallest, dist - self._threshold_for(g1, g2))
        return smallest

    def explain(self, q: Sequence[float]) -> List[str]:
        """One line per offending pair, worst first.

        A flat contact between two links produces several contact points, so the
        violations are collapsed per pair -- otherwise a single overlap reports
        as five identical lines.
        """
        def body_name(geom_id: int) -> str:
            return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.model.geom_bodyid[geom_id])

        worst: Dict[Tuple[int, int], Tuple[float, float]] = {}
        for g1, g2, dist, threshold in self._violations(q):
            previous = worst.get((g1, g2))
            if previous is None or dist < previous[0]:
                worst[(g1, g2)] = (dist, threshold)

        lines = []
        for (g1, g2), (dist, threshold) in sorted(worst.items(), key=lambda item: item[1][0]):
            kind = "obstacle" if (self._is_obstacle(g1) or self._is_obstacle(g2)) else "self"
            lines.append(
                f"{kind}: {body_name(g1)} <-> {body_name(g2)} at {dist * 1000:+.1f}mm "
                f"(needs {threshold * 1000:.0f}mm)"
            )
        return lines


class RedundantCollisionChecker:
    """Requires two independent models to agree that a pose is safe.

    The MuJoCo model has exact convex meshes and catches self-collision, but the
    shipped MJCF carries no gripper geom. The URDF sphere model covers the
    gripper and tips and is deliberately conservative. Neither alone covers the
    arm; a pose is accepted only if both pass, so a gap in either is caught by
    the other and the failure mode is refusing to move rather than a crash.
    """

    def __init__(self, sphere_checker, mujoco_checker: MujocoCollisionChecker):
        self.sphere_checker = sphere_checker
        self.mujoco_checker = mujoco_checker

    def is_free(self, q: Sequence[float]) -> bool:
        return self.sphere_checker.is_free(q) and self.mujoco_checker.is_free(q)

    def clearance(self, q: Sequence[float]) -> float:
        return min(self.sphere_checker.clearance(q), self.mujoco_checker.clearance(q))

    def explain(self, q: Sequence[float]) -> List[str]:
        return self.sphere_checker.explain(q) + self.mujoco_checker.explain(q)

    def segment_is_free(self, start: Sequence[float], end: Sequence[float], resolution: float = 0.05) -> bool:
        start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        steps = max(int(np.ceil(np.abs(end - start).max() / resolution)), 1)
        for index in range(steps + 1):
            if not self.is_free(start + (end - start) * (index / steps)):
                return False
        return True
