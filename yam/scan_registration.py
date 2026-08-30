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
    def verdict(self) -> str:
        """"good", "inconclusive" or "bad" -- never a bare boolean.

        A single fit's residual cannot separate right from wrong on this
        hardware. Measured: the true pose scores ~3mm on a clean synthetic
        cloud, a wrong yaw at the same position scores 21-22mm, and a
        registration confirmed correct by four independent means on a real phone
        sweep scores 19.6mm. Wrong-yaw and confirmed-correct overlap, so any
        threshold placed between them is guessing.

        The band where they overlap is therefore reported as inconclusive rather
        than resolved in either direction. Use `agrees_with`: a wrong pose is a
        local minimum that an unrelated seed will not find, so convergence from
        separate seeds does discriminate where a residual does not.
        """
        if self.inlier_fraction <= 0.9:
            return "bad"
        if self.rmse < 0.010:
            return "good"
        if self.rmse < 0.035:
            return "inconclusive"
        return "bad"

    @property
    def is_trustworthy(self) -> bool:
        """Only a fit that is unambiguously good on its own evidence.

        Deliberately conservative: "inconclusive" answers False here, so a
        caller that ignores `verdict` fails safe rather than acting on a fit
        that could be a wrong yaw.
        """
        return self.verdict == "good"

    def agrees_with(self, other: "ScanRegistration", tolerance: float = 0.05) -> bool:
        """Whether two fits put the robot base in the same place.

        Convergence from unrelated seeds is far stronger evidence than either
        fit's own residual: a wrong pose is a local minimum that a different
        seed will not find.
        """
        here = -self.translation @ self.rotation
        there = -other.translation @ other.rotation
        return bool(np.linalg.norm(here - there) < tolerance)

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


def _build_field(upright: np.ndarray, resolution: float = 0.02):
    from scipy import ndimage

    low = upright.min(axis=0) - 0.4
    high = upright.max(axis=0) + 0.4
    shape = np.maximum(np.ceil((high - low) / resolution).astype(int), 1)
    occupancy = np.zeros(shape, dtype=bool)
    cells = np.floor((upright - low) / resolution).astype(int)
    cells = cells[np.all((cells >= 0) & (cells < shape), axis=1)]
    occupancy[cells[:, 0], cells[:, 1], cells[:, 2]] = True
    field = ndimage.distance_transform_edt(~occupancy, sampling=resolution)

    def lookup(points: np.ndarray) -> np.ndarray:
        cell = np.floor((points - low) / resolution).astype(int)
        inside = np.all((cell >= 0) & (cell < shape), axis=-1)
        out = np.full(points.shape[:-1], 5.0)
        valid = cell[inside]
        out[inside] = field[valid[..., 0], valid[..., 1], valid[..., 2]]
        return out

    return lookup


def _free_space_shell(model: np.ndarray, count: int, seed: int = 0) -> np.ndarray:
    """Points that must be EMPTY if the model is where the arm is.

    Without this, "the model lies on a surface" is satisfied just as well by a
    wall as by the arm.
    """
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(len(model) * 2, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    shell = np.repeat(model, 2, axis=0) + directions * rng.uniform(0.09, 0.16, (len(model) * 2, 1))
    shell = shell[cKDTree(model).query(shell, k=1)[0] > 0.07]
    if len(shell) > count:
        shell = shell[rng.choice(len(shell), count, replace=False)]
    return shell


def _sweep(lookup, model, shell, centre, radius, step, yaws, free_space_weight):
    offsets = np.arange(-radius, radius + 1e-9, step)
    grid = np.array([[dx, dy, dz] for dx in offsets for dy in offsets for dz in offsets])
    candidates = centre + grid

    best = None
    for yaw in yaws:
        rotation = yaw_matrix(yaw)
        placed = model @ rotation.T
        empty = shell @ rotation.T
        on_surface = np.minimum(lookup(placed[None] + candidates[:, None, :]), 0.10).mean(axis=1)
        free = np.minimum(lookup(empty[None] + candidates[:, None, :]), 0.10).mean(axis=1)
        score = on_surface - free_space_weight * free
        index = int(np.argmin(score))
        if best is None or score[index] < best[0]:
            best = (float(score[index]), float(yaw), candidates[index].copy())
    return best, len(candidates) * len(yaws)


def refine_from_seed(
    scan_points: np.ndarray,
    arm_points: np.ndarray,
    seed_point: Sequence[float],
    search_radius: float = 0.40,
    cutoff: float = 0.12,
    iterations: int = 40,
    free_space_weight: float = 0.8,
) -> Optional[ScanRegistration]:
    """Align the arm to the scan given a rough idea of where the arm is.

    A global search over a whole room is where this problem is hard: the arm is a
    small object in a large cloud, and point-to-cloud distance alone is happy to
    park it against any dense surface. One coarse indication of where the arm is
    -- a tap in the AR view, accurate to a hand's width -- removes that entirely.

    Searched coarse-to-fine rather than at one resolution. A single fine sweep
    over yaw and a 0.4m translation cube is hundreds of millions of lookups and
    took ~90 seconds, long enough that the phone gave up on the request before
    the answer existed. Two passes reach the same place in a few seconds.

    Precision does not come from the seed, or from the sweep. It comes from the
    ICP that follows, which fits the whole model.

    `seed_point` is in the scan's own (ARKit) frame -- what a raycast returns.
    """
    scan = np.asarray(scan_points, dtype=float).reshape(-1, 3)
    model = np.asarray(arm_points, dtype=float).reshape(-1, 3)
    if len(scan) < 100 or len(model) < 20:
        return None

    upright = scan @ ARKIT_TO_ZUP.T
    seed = np.asarray(seed_point, dtype=float) @ ARKIT_TO_ZUP.T
    lookup = _build_field(upright)

    def thin(points, count):
        return points if len(points) <= count else points[np.linspace(0, len(points) - 1, count).astype(int)]

    coarse_model = thin(model, 250)
    fine_model = thin(model, 700)
    coarse_shell = _free_space_shell(coarse_model, 250)
    fine_shell = _free_space_shell(fine_model, 400)

    best, searched = _sweep(lookup, coarse_model, coarse_shell, seed, search_radius, 0.10,
                            np.linspace(0, 2 * np.pi, 24, endpoint=False), free_space_weight)
    if best is None:
        return None

    yaws = best[1] + np.linspace(-np.pi / 24, np.pi / 24, 13)
    refined, more = _sweep(lookup, fine_model, fine_shell, best[2], 0.10, 0.033,
                           yaws, free_space_weight)
    searched += more
    if refined is not None:
        best = refined

    tree = cKDTree(upright)
    rotation = yaw_matrix(best[1])
    translation = best[2]

    for _ in range(iterations):
        placed = model @ rotation.T + translation
        gaps, neighbours = tree.query(placed, k=1)
        inliers = gaps < cutoff
        if inliers.sum() < 20:
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
    gaps, _ = tree.query(placed, k=1)
    inliers = gaps < cutoff
    if inliers.sum() < 20:
        return None

    return ScanRegistration(
        rotation=rotation.T @ ARKIT_TO_ZUP,
        translation=-rotation.T @ translation,
        rmse=float(np.sqrt((gaps[inliers] ** 2).mean())),
        inliers=int(inliers.sum()),
        model_points=len(model),
        searched=searched,
    )
