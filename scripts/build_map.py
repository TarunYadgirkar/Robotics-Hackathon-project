"""Turn enrollment (and optionally a LiDAR scan) into a planning map.

  python scripts/build_map.py --enrollment enrollment.json --table-z -0.02 --table-edge 0.30
  python scripts/build_map.py --scan room.ply --scan-pose-from enrollment.json

Writes a voxel map (.npz) the planner loads directly.
"""

import argparse

import numpy as np

from yam.enrollment import EnrollmentSession
from yam.kinematics import YamKinematics
from yam.lidar import kabsch, load_point_cloud
from yam.mapping import base_clamps, build_map, table_slab


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--enrollment", default=None, help="enrollment.json from scripts/enroll.py")
    parser.add_argument("--scan", default=None, help="LiDAR export (.ply/.obj/.stl)")
    parser.add_argument("--registration", default=None,
                        help="JSON holding the scan-to-robot transform (rotation/translation, as "
                             "written by pick_seed.py) or matched 'scan'/'robot' point lists")
    parser.add_argument("--table-z", type=float, default=None, help="tabletop height in base frame (m)")
    parser.add_argument("--table-edge", type=float, default=0.30,
                        help="x where the tabletop ends; beyond this the arm may go below table level")
    parser.add_argument("--obstacle-mode", choices=("box", "spheres", "planes", "none"), default="box",
                        help="box: one bounding box round all touched points. spheres: each "
                             "touched point is its own small obstacle, which is what you want "
                             "when the points are separate things rather than one object. "
                             "planes: fit horizontal surfaces to the touched points, which is "
                             "what they are when someone taps a tabletop and a floor. "
                             "none: the scan is the map; touched points add nothing.")
    parser.add_argument("--table-edge-x", type=float, default=None,
                        help="where the tabletop stops. Defaults to the most conservative value "
                             "the touched points allow: the nearest floor point, since a table "
                             "assumed too short is a table the arm plans straight through.")
    parser.add_argument("--sphere-radius", type=float, default=0.08)
    parser.add_argument("--min-base-distance", type=float, default=0.15,
                        help="ignore touched points nearer the base than this; they are the robot")
    parser.add_argument("--clamps", action="store_true",
                        help="add the two base clamps, which LiDAR is too coarse to resolve")
    parser.add_argument("--resolution", type=float, default=0.02)
    parser.add_argument("--output", default="workcell_map.npz")
    args = parser.parse_args()

    kinematics = YamKinematics()
    session = EnrollmentSession.load(args.enrollment) if args.enrollment else None
    if session is not None:
        from yam.enrollment import recompute_positions

        corrected = recompute_positions(session, kinematics)
        if corrected:
            print(f"  corrected {corrected} captured points to the jaw tips "
                  f"(sessions recorded before the probe offset was measured are 134mm out)")

    points = None
    registered = False
    if args.scan:
        points = load_point_cloud(args.scan)
        print(f"  loaded {len(points):,} points from {args.scan}")

        if args.registration:
            import json

            with open(args.registration) as handle:
                data = json.load(handle)

            if "rotation" in data:
                # A transform solved by fitting the arm's own shape to the scan.
                rotation = np.array(data["rotation"])
                translation = np.array(data["translation"])
                points = points @ rotation.T + translation
                print(f"  registered by arm fit: {data.get('rmse_mm', 0):.1f} mm RMSE, "
                      f"{data.get('inliers', '?')} inliers")
            else:
                registration = kabsch(np.array(data["scan"]), np.array(data["robot"]))
                points = registration.apply(points)
                print(f"  registered: {registration.rmse * 1000:.1f} mm RMSE over {len(data['scan'])} pairs")
            registered = True
        else:
            print("  NOTE: no --registration given. A scan is in the phone's frame until it is")
            print("        aligned, and everything downstream assumes robot coordinates, so it")
            print("        will be refused rather than silently mapped into the wrong place.")
            registered = False

    # Every pose the arm was logged in during enrollment, so a continuous sweep
    # can have the arm subtracted along its whole trajectory rather than at one
    # instant. Falls back to the touched poses for sessions recorded before
    # pose logging existed.
    scan_poses = [sample.joint_angles for sample in session.pose_log] if session else []
    if session and not scan_poses:
        scan_poses = [p.joint_angles for obstacle in session.objects for p in obstacle.points]
    if scan_poses:
        print(f"  {len(scan_poses)} arm poses available for scan subtraction")

    table = table_slab(args.table_z, args.table_edge) if args.table_z is not None else None

    voxel_map = build_map(
        session=None if args.obstacle_mode in ("spheres", "planes", "none") else session,
        scan_points=points, table=table, resolution=args.resolution,
        kinematics=kinematics, scan_poses=scan_poses, scan_is_registered=registered,
    )

    if args.obstacle_mode == "planes" and session is not None:

        points = np.vstack([o.positions() for o in session.objects])
        upper = points[points[:, 2] > -0.2]
        lower = points[points[:, 2] <= -0.2]
        extent = 0.95

        if len(lower):
            floor_z = float(lower[:, 2].mean())
            voxel_map.add_box([-extent, -extent, floor_z - 0.06], [extent, extent, floor_z])
            print(f"  floor  plane at z={floor_z:+.3f} m from {len(lower)} touched points")

        if len(upper):
            table_z = float(upper[:, 2].mean())
            # Where the table stops is not measured, only bounded: it is somewhere
            # between the furthest table point and the nearest floor point. Take the
            # far end. Over-estimating the table forbids space that is actually
            # free; under-estimating it plans the arm through a tabletop.
            edge = args.table_edge_x
            if edge is None:
                edge = float(lower[:, 0].min()) if len(lower) else extent
            voxel_map.add_box([-extent, -extent, table_z - 0.06], [edge, extent, table_z])
            print(f"  table  plane at z={table_z:+.3f} m from {len(upper)} touched points, "
                  f"edge at x={edge:+.3f} m")
            if len(lower):
                span = (lower[:, 0].min() - upper[:, 0].max()) * 1000
                print(f"         edge is bounded to x={upper[:, 0].max():+.3f}..{lower[:, 0].min():+.3f} "
                      f"({span:.0f}mm unknown); using the conservative end")
        voxel_map.compute_distance_field()

    if args.obstacle_mode == "spheres" and session is not None:
        kept = skipped = 0
        for obstacle in session.objects:
            for position in obstacle.positions():
                if np.linalg.norm(position) < args.min_base_distance:
                    skipped += 1
                    continue
                voxel_map.add_box(np.array(position) - args.sphere_radius,
                                  np.array(position) + args.sphere_radius)
                kept += 1
        if skipped:
            print(f"  ignored {skipped} touched point(s) within {args.min_base_distance * 100:.0f}cm "
                  f"of the base -- those are the robot, not an obstacle")
        print(f"  {kept} touched points as {args.sphere_radius * 100:.0f}cm obstacles")
        voxel_map.compute_distance_field()
    voxel_map.save(args.output)

    if args.clamps:
        for box in base_clamps(table_z=args.table_z if args.table_z is not None else -0.02):
            voxel_map.add_box(box["min"], box["max"])
        print(f"  added 2 base clamps (140 x 25 x 102 mm, +-102mm either side of centreline)")
        voxel_map.compute_distance_field()

    occupied = int(voxel_map.occupancy.sum())
    print(f"\n  map: {voxel_map.shape} voxels at {args.resolution * 1000:.0f}mm, {occupied:,} occupied")
    print(f"  saved {args.output}")


if __name__ == "__main__":
    main()
