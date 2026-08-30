"""Assemble the planning map from everything we know about the workcell.

Three sources, in increasing order of richness:

* **Touched obstacles** -- boxes fitted to points the arm physically reached.
  Sparse, but expressed in the robot's own frame with no registration step.
* **A LiDAR scan** -- dense geometry for whatever the arm cannot conveniently
  touch, registered into the base frame using touched reference points.
* **A table slab** -- the fallback when there is no scan yet, described as a
  finite box rather than a half-space, because this arm overhangs its table edge.

Everything lands in one voxel grid, so the planner does not care which source a
given obstacle came from.
"""

from typing import Optional, Sequence

import numpy as np

from yam.enrollment import EnrollmentSession
from yam.lidar import crop_to_workspace, filter_robot_from_scan
from yam.voxel_map import VoxelMap

DEFAULT_BOUNDS_MIN = (-0.95, -0.95, -0.75)
DEFAULT_BOUNDS_MAX = (0.95, 0.95, 1.00)


def build_map(
    session: Optional[EnrollmentSession] = None,
    scan_points: Optional[np.ndarray] = None,
    scan_is_registered: bool = False,
    table: Optional[dict] = None,
    resolution: float = 0.02,
    bounds_min: Sequence[float] = DEFAULT_BOUNDS_MIN,
    bounds_max: Sequence[float] = DEFAULT_BOUNDS_MAX,
    kinematics=None,
    scan_poses: Optional[Sequence[Sequence[float]]] = None,
    self_filter_padding: float = 0.08,
    protect_below_z: Optional[float] = None,
) -> VoxelMap:
    voxel_map = VoxelMap.from_bounds(bounds_min, bounds_max, resolution)

    if table is not None:
        voxel_map.add_box(table["min"], table["max"])

    if session is not None:
        for box in session.to_world_boxes():
            voxel_map.add_box(box.minimum, box.maximum)

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
    # Any voxel the arm provably occupies is cleared, or the arm is walled in
    # by an image of itself.
    if kinematics is not None and scan_poses is not None:
        for pose in scan_poses:
            centers, radii = kinematics.collision_spheres(pose)
            if protect_below_z is not None:
                # Do not carve below the mounting plane: that is the table.
                keep = centers[:, 2] + radii > protect_below_z
                centers, radii = centers[keep], radii[keep]
            voxel_map.carve_spheres(centers, radii, padding=self_filter_padding)

    voxel_map.compute_distance_field()
    return voxel_map


#: The clamps holding the base to the table. LiDAR does not resolve them -- they
#: are 25mm wide next to a much larger arm -- so they are described by hand.
#: Dimensions as given: 5.5in tall, 1in wide, 4in long, long axis along the arm's
#: forward direction, centred 4in either side of the base centreline.
def base_clamps(offset_y: float = 0.1016, height: float = 0.1397, width: float = 0.0254,
                length: float = 0.1016, table_z: float = -0.02) -> list:
    boxes = []
    for sign in (+1, -1):
        centre_y = sign * offset_y
        boxes.append({
            "min": [-length / 2, centre_y - width / 2, table_z],
            "max": [+length / 2, centre_y + width / 2, table_z + height],
        })
    return boxes


def table_slab(surface_z: float, edge_x: float, extent: float = 0.95, thickness: float = 0.04) -> dict:
    """A table the arm is mounted at the edge of.

    `edge_x` is where the tabletop stops; beyond it there is nothing, and the arm
    may legitimately swing below `surface_z`.
    """
    return {
        "min": [-extent, -extent, surface_z - thickness],
        "max": [edge_x, extent, surface_z],
    }
