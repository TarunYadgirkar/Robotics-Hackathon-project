"""Register a scan to the robot by finding the robot in it.

Clicking matching landmarks works, but it is neither scalable nor precise: it
costs three careful taps per scan and its accuracy is the accuracy of a
fingertip on a phone screen.

There is a better target already in the scan. The arm's surface is known
exactly -- URDF geometry through forward kinematics at a measured joint
configuration -- so the arm is a calibration object of known shape sitting in
the scene. Fitting that known shape to the cloud yields the scan-to-robot
transform with no human input, and its precision comes from thousands of
surface points rather than three taps.

Two facts collapse the search from six degrees of freedom to three:

* ARKit's world frame is gravity-aligned, so its Y axis and the robot's Z axis
  are the same physical direction. Only a yaw remains.
* The arm is bolted to a table, and a table is a large horizontal plane -- the
  strongest horizontal surfaces in the cloud are the few candidate heights.

What is left is yaw and two translations, coarse-searched against a distance
field and then refined by ICP.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

#: ARKit is Y-up, the robot is Z-up: a +90 degree rotation about X maps one to
#: the other, and leaves yaw as the only unknown rotation.
ARKIT_TO_ZUP = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
])


@dataclass
class ScanRegistration:
    rotation: np.ndarray
    translation: np.ndarray
    rmse: float
    inliers: int
    model_points: int
    searched: int

    @property
    def inlier_fraction(self) -> float:
        return self.inliers / max(self.model_points, 1)

    @property
    def is_trustworthy(self) -> bool:
        """A good fit needs both closeness and coverage.

        A low RMSE over a handful of points is what a wrong pose looks like when
        a few model points happen to land on a wall.
        """
        return self.rmse < 0.02 and self.inlier_fraction > 0.5

    def apply(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=float).reshape(-1, 3) @ self.rotation.T + self.translation

    def describe(self) -> str:
        return (
            f"{self.rmse * 1000:.1f} mm RMSE over {self.inliers}/{self.model_points} "
            f"model points ({self.inlier_fraction:.0%}), {self.searched} poses searched"
        )


def yaw_matrix(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def arm_surface_points(kinematics, pose: Sequence[float], gripper_opening: float = 1.0,
                       max_points: int = 4000, seed: int = 0) -> np.ndarray:
    """The arm's outer surface in robot coordinates, at one joint configuration."""
    frames = kinematics.link_transforms(pose, gripper_opening)
    chunks = []

    for name, geometry in kinematics.link_geometry.items():
        mesh_path = None
        if name not in frames:
            continue
        # Reuse the fitted sphere centres as a cheap surface sample: they follow
        # the mesh and are already in link coordinates.
        if len(geometry.centers) == 0:
            continue
        frame = frames[name]
        chunks.append(geometry.centers @ frame[:3, :3].T + frame[:3, 3])

    if not chunks:
        return np.zeros((0, 3))
    points = np.vstack(chunks)

    if len(points) > max_points:
        rng = np.random.default_rng(seed)
        points = points[rng.choice(len(points), max_points, replace=False)]
    return points


def dense_arm_surface(kinematics, pose: Sequence[float], gripper_opening: float = 1.0,
                      max_points: int = 6000, seed: int = 0) -> np.ndarray:
    """Denser surface sample, taken from the visual meshes themselves."""
    import os
    import xml.etree.ElementTree as ET

    from yam.kinematics import read_binary_stl, rpy_to_matrix

    frames = kinematics.link_transforms(pose, gripper_opening)
    root = ET.parse(kinematics.urdf_path).getroot()
    rng = np.random.default_rng(seed)
    chunks = []

    for link in root.findall("link"):
        name = link.get("name")
        visual = link.find("visual")
        if visual is None or name not in frames:
            continue
        mesh = visual.find("geometry/mesh")
        if mesh is None:
            continue
        path = os.path.join(kinematics.root_dir, mesh.get("filename"))
        if not os.path.isfile(path):
            continue

        vertices = read_binary_stl(path)
        origin = visual.find("origin")
        if origin is not None:
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            vertices = vertices @ rpy_to_matrix(*rpy).T + xyz

        take = min(len(vertices), max(max_points // 8, 200))
        vertices = vertices[rng.choice(len(vertices), take, replace=False)]
        frame = frames[name]
        chunks.append(vertices @ frame[:3, :3].T + frame[:3, 3])

    return np.vstack(chunks) if chunks else np.zeros((0, 3))


def candidate_heights(points: np.ndarray, bins: int = 240, count: int = 4) -> List[float]:
    """Heights holding unusually many points: floors, tables, worktops."""
    histogram, edges = np.histogram(points[:, 2], bins=bins)
    order = np.argsort(histogram)[::-1]

    chosen: List[float] = []
    for index in order:
        height = float((edges[index] + edges[index + 1]) / 2)
        if all(abs(height - existing) > 0.10 for existing in chosen):
            chosen.append(height)
        if len(chosen) >= count:
            break
    return chosen


def _score_against(field, origin: np.ndarray, resolution: float, shape, points: np.ndarray,
                   cutoff: float) -> np.ndarray:
    """Distance-field lookup for many points at once; outside the grid reads as cutoff."""
    indices = np.floor((points - origin) / resolution).astype(int)
    inside = np.all((indices >= 0) & (indices < np.array(shape)), axis=1)
    distances = np.full(len(points), cutoff)
    if inside.any():
        valid = indices[inside]
        distances[inside] = np.minimum(field[valid[:, 0], valid[:, 1], valid[:, 2]], cutoff)
    return distances


def register_arm_to_scan(
    scan_points: np.ndarray,
    arm_points: np.ndarray,
    resolution: float = 0.03,
    yaw_steps: int = 72,
    translation_step: float = 0.10,
    cutoff: float = 0.15,
    refine_iterations: int = 30,
    search_radius: float = 2.5,
) -> Optional[ScanRegistration]:
    """Find the rigid transform taking scan coordinates into robot coordinates."""
    from scipy import ndimage

    scan = np.asarray(scan_points, dtype=float).reshape(-1, 3)
    model = np.asarray(arm_points, dtype=float).reshape(-1, 3)
    if len(scan) < 100 or len(model) < 20:
        return None

    # Work in a gravity-aligned copy of the scan, where only yaw is unknown.
    upright = scan @ ARKIT_TO_ZUP.T

    low = upright.min(axis=0) - 0.2
    high = upright.max(axis=0) + 0.2
    shape = np.maximum(np.ceil((high - low) / resolution).astype(int), 1)
    occupancy = np.zeros(shape, dtype=bool)
    indices = np.floor((upright - low) / resolution).astype(int)
    keep = np.all((indices >= 0) & (indices < shape), axis=1)
    indices = indices[keep]
    occupancy[indices[:, 0], indices[:, 1], indices[:, 2]] = True
    field = ndimage.distance_transform_edt(~occupancy, sampling=resolution)

    centre = upright.mean(axis=0)
    span = min(search_radius, float(np.linalg.norm(high - low)) / 2 + 0.5)
    offsets = np.arange(-span, span + 1e-9, translation_step)
    heights = candidate_heights(upright)

    best = None
    searched = 0
    coarse = model if len(model) <= 900 else model[np.linspace(0, len(model) - 1, 900).astype(int)]

    for yaw in np.linspace(0, 2 * np.pi, yaw_steps, endpoint=False):
        rotated = coarse @ yaw_matrix(yaw).T
        for height in heights:
            for dx in offsets:
                for dy in offsets:
                    translation = np.array([centre[0] + dx, centre[1] + dy, height])
                    distances = _score_against(field, low, resolution, shape, rotated + translation, cutoff)
                    score = float(distances.mean())
                    searched += 1
                    if best is None or score < best[0]:
                        best = (score, yaw, translation)

    if best is None:
        return None

    # Refine against the actual points rather than the voxelised field.
    tree = cKDTree(upright)
    rotation = yaw_matrix(best[1])
    translation = best[2]

    for _ in range(refine_iterations):
        placed = model @ rotation.T + translation
        distances, neighbours = tree.query(placed, k=1)
        inliers = distances < cutoff
        if inliers.sum() < 10:
            break

        source = model[inliers]
        target = upright[neighbours[inliers]]
        source_centre = source.mean(axis=0)
        target_centre = target.mean(axis=0)
        u, _, vt = np.linalg.svd((source - source_centre).T @ (target - target_centre))
        correction = np.eye(3)
        correction[2, 2] = np.sign(np.linalg.det(vt.T @ u.T))
        rotation = vt.T @ correction @ u.T
        translation = target_centre - rotation @ source_centre

    placed = model @ rotation.T + translation
    distances, _ = tree.query(placed, k=1)
    inliers = distances < cutoff
    if inliers.sum() < 10:
        return None

    # rotation/translation map MODEL -> upright scan; we want scan -> robot,
    # which is the inverse, composed with the gravity alignment.
    scan_to_robot_rotation = rotation.T @ ARKIT_TO_ZUP
    scan_to_robot_translation = -rotation.T @ translation

    return ScanRegistration(
        rotation=scan_to_robot_rotation,
        translation=scan_to_robot_translation,
        rmse=float(np.sqrt((distances[inliers] ** 2).mean())),
        inliers=int(inliers.sum()),
        model_points=len(model),
        searched=searched,
    )
