"""Register an ARKit scan to the robot frame using measured geometry.

The arm shape supplies a dense local fit. Touched surface points supply the
independent evidence that resolves otherwise plausible wrong yaws. Gravity is
kept fixed throughout: ARKit is Y-up and the robot is Z-up, leaving yaw and
translation as the only unknowns.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

#: ARKit is Y-up, the robot is Z-up: a +90 degree rotation about X maps one to
#: the other, and leaves yaw as the only unknown rotation.
ARKIT_TO_ZUP = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
])

MIN_MODEL_COVERAGE = 0.95
MAX_EVIDENCE_ERROR = 0.05
MIN_SURFACE_SPREAD = 0.10


@dataclass
class ScanRegistration:
    rotation: np.ndarray
    translation: np.ndarray
    rmse: float
    inliers: int
    model_points: int
    searched: int
    model_p95_error: Optional[float] = None
    surface_rmse: Optional[float] = None
    surface_max_error: Optional[float] = None
    surface_points: int = 0
    surface_spread: float = 0.0

    @property
    def inlier_fraction(self) -> float:
        return self.inliers / max(self.model_points, 1)

    @property
    def verdict(self) -> str:
        """Return ``good`` only when arm and independent surface evidence agree."""
        model_p95 = self.model_p95_error if self.model_p95_error is not None else self.rmse
        if self.inlier_fraction < MIN_MODEL_COVERAGE or model_p95 > MAX_EVIDENCE_ERROR:
            return "bad"
        if self.surface_points < 4 or self.surface_spread < MIN_SURFACE_SPREAD:
            return "inconclusive"
        if self.surface_max_error is None or self.surface_max_error > MAX_EVIDENCE_ERROR:
            return "bad"
        return "good"

    @property
    def is_trustworthy(self) -> bool:
        """Whether the fit has passed both shape and touched-surface checks."""
        return self.verdict == "good"

    @property
    def uncertainty(self) -> float:
        errors = [self.rmse]
        if self.model_p95_error is not None:
            errors.append(self.model_p95_error)
        if self.surface_max_error is not None:
            errors.append(self.surface_max_error)
        return max(errors)

    def agrees_with(
        self,
        other: "ScanRegistration",
        translation_tolerance: float = 0.05,
        rotation_tolerance: float = np.deg2rad(5.0),
    ) -> bool:
        """Whether two fits agree on both base position and orientation."""
        here = -self.translation @ self.rotation
        there = -other.translation @ other.rotation
        relative_rotation = self.rotation @ other.rotation.T
        cosine = np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
        rotation_error = float(np.arccos(cosine))
        return bool(
            np.linalg.norm(here - there) < translation_tolerance
            and rotation_error < rotation_tolerance
        )

    def apply(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=float).reshape(-1, 3) @ self.rotation.T + self.translation

    def describe(self) -> str:
        description = (
            f"{self.rmse * 1000:.1f} mm RMSE over {self.inliers}/{self.model_points} "
            f"model points ({self.inlier_fraction:.0%}), {self.searched} poses searched"
        )
        if self.surface_rmse is not None and self.surface_max_error is not None:
            description += (
                f", touched surfaces {self.surface_rmse * 1000:.1f} mm RMS / "
                f"{self.surface_max_error * 1000:.1f} mm max"
            )
        return description


def yaw_matrix(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


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


def _surface_spread(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    singular_values = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    return float(singular_values[1]) if len(singular_values) > 1 else 0.0


def _refine_with_measured_surfaces(
    tree: cKDTree,
    model: np.ndarray,
    surface_points: np.ndarray,
    initial_yaw: float,
    initial_translation: np.ndarray,
    translation_radius: float,
    iterations: int,
):
    fit_model = model if len(model) <= 1500 else model[
        np.linspace(0, len(model) - 1, 1500).astype(int)
    ]

    def residuals(parameters, points):
        rotation = yaw_matrix(parameters[0])
        return tree.query(points @ rotation.T + parameters[1:], k=1)[0]

    def objective(parameters):
        model_gaps = residuals(parameters, fit_model)
        score = float(np.mean(np.minimum(model_gaps, 0.08) ** 2))
        if len(surface_points):
            surface_gaps = residuals(parameters, surface_points)
            score += float(np.mean(np.minimum(surface_gaps, 0.10) ** 2))
        return score

    initial = np.r_[initial_yaw, initial_translation]
    radius = min(translation_radius, 0.15)
    bounds = [
        (initial_yaw - np.pi / 6, initial_yaw + np.pi / 6),
        *[(value - radius, value + radius) for value in initial_translation],
    ]
    result = minimize(
        objective,
        initial,
        method="Powell",
        bounds=bounds,
        options={"maxiter": max(iterations, 1), "xtol": 1e-5, "ftol": 1e-7},
    )
    return result.x, result.nfev


def refine_from_seed(
    scan_points: np.ndarray,
    arm_points: np.ndarray,
    seed_point: Sequence[float],
    surface_points: Optional[np.ndarray] = None,
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

    The final refinement keeps gravity fixed and balances two evidence classes:
    the scanned arm surface and physical points touched in the workcell. Arm
    shape alone can produce a plausible wrong yaw and is never marked trusted.

    `seed_point` is in the scan's own (ARKit) frame -- what a raycast returns.
    """
    scan = np.asarray(scan_points, dtype=float).reshape(-1, 3)
    model = np.asarray(arm_points, dtype=float).reshape(-1, 3)
    measured_surfaces = np.asarray(
        np.zeros((0, 3)) if surface_points is None else surface_points, dtype=float
    ).reshape(-1, 3)
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
    parameters, evaluations = _refine_with_measured_surfaces(
        tree,
        model,
        measured_surfaces,
        initial_yaw=best[1],
        initial_translation=best[2],
        translation_radius=search_radius,
        iterations=iterations,
    )
    searched += evaluations
    robot_to_upright_rotation = yaw_matrix(parameters[0])
    translation = parameters[1:]

    model_gaps = tree.query(model @ robot_to_upright_rotation.T + translation, k=1)[0]
    inliers = model_gaps < cutoff
    if inliers.sum() < 20:
        return None

    surface_gaps = tree.query(
        measured_surfaces @ robot_to_upright_rotation.T + translation, k=1
    )[0] if len(measured_surfaces) else np.zeros(0)

    return ScanRegistration(
        rotation=robot_to_upright_rotation.T @ ARKIT_TO_ZUP,
        translation=-robot_to_upright_rotation.T @ translation,
        rmse=float(np.sqrt((model_gaps[inliers] ** 2).mean())),
        inliers=int(inliers.sum()),
        model_points=len(model),
        searched=searched,
        model_p95_error=float(np.percentile(model_gaps, 95)),
        surface_rmse=(
            float(np.sqrt(np.mean(surface_gaps ** 2))) if len(surface_gaps) else None
        ),
        surface_max_error=float(surface_gaps.max()) if len(surface_gaps) else None,
        surface_points=len(measured_surfaces),
        surface_spread=_surface_spread(measured_surfaces),
    )
