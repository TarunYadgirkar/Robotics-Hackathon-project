"""Estimate which way a scanned surface faces, so contact can approach along it.

Touching is not a downward motion; it is a motion along the surface normal. A
tabletop happens to face up, but a table leg faces sideways and a shelf
underside faces down. Assuming -Z works for exactly one of those and silently
approaches the other two edge-on.

The normal comes from the scan itself: the occupied voxels around a point form
a small patch, and the direction of least variance across that patch is its
normal. Two things decide whether the answer is usable, and both are reported
rather than assumed -- how many voxels supported the fit, and how planar they
were. A normal fitted to a corner or to eight scattered returns is not a
normal, and a contact approach built on one drives in at the wrong angle.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from yam.voxel_map import VoxelMap


class SurfaceUnknown(ValueError):
    """The scan does not describe a surface well enough to approach it."""


@dataclass(frozen=True)
class SurfaceSample:
    """A local surface patch. `axis` is unsigned: see `approach_directions`."""

    point: np.ndarray
    axis: np.ndarray
    support: int
    planarity: float
    residual_m: float

    def approach_directions(self) -> list:
        """Both directions the surface could be touched from.

        The sign of a fitted normal is not decidable from this map. The LiDAR
        recorded faces, not volume, so there are no voxels behind a tabletop to
        mark its underside, and marching the occupancy answers confidently and
        wrongly -- objects resting on a table make its open side look like the
        blocked one. The caller resolves the sign by finding which side admits
        a reachable, collision-free standoff, which is the question that
        actually matters.
        """
        return [-self.axis, self.axis]

    def to_dict(self) -> dict:
        return {
            "point_m": self.point.tolist(),
            "axis": self.axis.tolist(),
            "support_voxels": int(self.support),
            "planarity": float(self.planarity),
            "patch_residual_m": float(self.residual_m),
        }


def occupied_points(voxel_map: VoxelMap) -> np.ndarray:
    indices = np.argwhere(voxel_map.occupancy)
    if voxel_map.synthetic_occupancy is not None:
        synthetic = np.argwhere(voxel_map.synthetic_occupancy)
        if len(synthetic):
            indices = np.vstack([indices, synthetic])
    return indices * voxel_map.resolution + voxel_map.origin


def estimate_normal(
    voxel_map: VoxelMap,
    point: np.ndarray,
    radius: float = 0.10,
    min_support: int = 20,
    min_planarity: float = 2.0,
    cloud: Optional[np.ndarray] = None,
) -> SurfaceSample:
    """Fit a local plane to the scanned patch around `point` and return its normal.

    `planarity` is the ratio of the patch's second-smallest spread to its
    smallest. A flat patch is much wider than it is thick and scores high; a
    corner or a noise blob scores near one, and is refused rather than
    approached at a guessed angle.
    """
    point = np.asarray(point, dtype=float).reshape(3)
    points = occupied_points(voxel_map) if cloud is None else np.asarray(cloud, dtype=float)
    patch = points[np.linalg.norm(points - point, axis=1) <= radius]

    if len(patch) < min_support:
        raise SurfaceUnknown(
            f"only {len(patch)} scanned voxels within {radius * 1000:.0f}mm of "
            f"{np.round(point, 3).tolist()}; {min_support} needed to fit a surface"
        )

    centred = patch - patch.mean(axis=0)
    # Eigenvectors of the patch covariance: the smallest is normal to the plane.
    eigenvalues, eigenvectors = np.linalg.eigh(centred.T @ centred / len(patch))
    order = np.argsort(eigenvalues)
    smallest, middle = eigenvalues[order[0]], eigenvalues[order[1]]
    normal = eigenvectors[:, order[0]]

    planarity = float(np.sqrt(middle / smallest)) if smallest > 1e-12 else float("inf")
    if planarity < min_planarity:
        raise SurfaceUnknown(
            f"the scan around {np.round(point, 3).tolist()} is not planar enough to give a "
            f"normal (planarity {planarity:.1f}, need {min_planarity:.1f}); it may be an edge or a corner"
        )

    return SurfaceSample(
        point=point,
        axis=normal / np.linalg.norm(normal),
        support=len(patch),
        planarity=planarity,
        residual_m=float(np.sqrt(max(smallest, 0.0))),
    )
