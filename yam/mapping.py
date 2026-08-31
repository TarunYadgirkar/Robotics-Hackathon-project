"""Build a planning map from registered LiDAR and explicitly specified geometry."""

from typing import Any, Mapping, Optional, Sequence

import numpy as np

from yam.lidar import crop_to_workspace, filter_robot_from_scan
from yam.voxel_map import VoxelMap

DEFAULT_BOUNDS_MIN = (-0.95, -0.95, -0.75)
DEFAULT_BOUNDS_MAX = (0.95, 0.95, 1.00)
INCH = 0.0254
BASE_SIDE_Y = 0.100

#: Shoulder pivot height in the robot frame, from the kinematic model.
SHOULDER_PIVOT_Z = 0.1135


def base_clamps(
    height: float = 5.75 * INCH,
    length: float = 3 * INCH,
    width: float = 1 * INCH,
    top_z: float = SHOULDER_PIVOT_Z + 0.75 * INCH,
) -> list:
    """The clamps holding the base to the table, from measurement.

    LiDAR does not resolve them -- they are 25mm wide beside a much larger arm
    -- so their height is given rather than sensed, and an error here is
    invisible to every check downstream.

    Height is keyed to the shoulder pivot rather than to the table, because the
    pivot is a point the kinematic model knows exactly while the table's height
    in the robot frame is inferred. Measured: the clamps stand 5.75 in above the
    table and the pivot 5.00 in above it, so their tops sit 0.75 in above the
    pivot regardless of where the table plane is judged to be.
    """
    boxes = []
    for side in (-1.0, 1.0):
        inner_y = side * BASE_SIDE_Y
        outer_y = inner_y + side * width
        boxes.append({
            "min": [-length / 2, min(inner_y, outer_y), top_z - height],
            "max": [length / 2, max(inner_y, outer_y), top_z],
        })
    return boxes


def build_map(
    scan_points: Optional[np.ndarray] = None,
    scan_is_registered: bool = False,
    registration_uncertainty: float = 0.0,
    resolution: float = 0.02,
    bounds_min: Sequence[float] = DEFAULT_BOUNDS_MIN,
    bounds_max: Sequence[float] = DEFAULT_BOUNDS_MAX,
    kinematics=None,
    scan_poses: Optional[Sequence[Sequence[float]]] = None,
    synthetic_boxes: Optional[Sequence[dict]] = None,
    self_filter_padding: float = 0.08,
    protect_below_z: Optional[float] = None,
    provenance: Optional[Mapping[str, Any]] = None,
) -> VoxelMap:
    voxel_map = VoxelMap.from_bounds(bounds_min, bounds_max, resolution)
    voxel_map.uncertainty = float(registration_uncertainty)
    voxel_map.provenance = dict(provenance or {})

    if scan_points is not None and len(scan_points):
        # Both the crop and the self-filter reason in ROBOT coordinates. A scan
        # straight off the phone is in ARKit's frame, so running them on an
        # unregistered cloud crops the wrong region and subtracts the arm from
        # coordinates the arm is not in -- silently, and the result looks fine.
        if not scan_is_registered:
            raise ValueError(
                "scan_points must be registered into the robot frame first. "
                "Cropping and self-filtering are robot-frame operations; on an "
                "unregistered scan they produce a plausible-looking wrong map."
            )
        points = crop_to_workspace(np.asarray(scan_points, dtype=float))
        if kinematics is not None and scan_poses is not None and len(scan_poses):
            before = len(points)
            points = filter_robot_from_scan(points, kinematics, scan_poses,
                                            padding=self_filter_padding,
                                            protect_below_z=protect_below_z)
            print(f"  self-filter: {before:,} -> {len(points):,} points "
                  f"({before - len(points):,} removed as robot across {len(scan_poses)} poses)")
        voxel_map.add_points(points)

    # Even after point-level filtering, stray returns land on the arm: the scan
    # pose is only known to the accuracy of FK, and LiDAR noise smears surfaces.
    # Voxels inside the padded scan-time arm model are cleared, or the arm is
    # walled in by its own LiDAR image.
    if kinematics is not None and scan_poses is not None:
        for pose in scan_poses:
            centers, radii = kinematics.collision_spheres(pose)
            if protect_below_z is not None:
                # Do not carve measured geometry below the robot mounting datum.
                keep = centers[:, 2] + radii > protect_below_z
                centers, radii = centers[keep], radii[keep]
            voxel_map.carve_spheres(
                centers,
                radii,
                padding=self_filter_padding,
                protect_below_z=protect_below_z,
            )

    for box in synthetic_boxes or []:
        voxel_map.add_box(box["min"], box["max"], synthetic=True)

    voxel_map.compute_distance_field()
    return voxel_map
