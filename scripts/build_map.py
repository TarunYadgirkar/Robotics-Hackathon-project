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
                        help="JSON with 'scan' and 'robot' point lists, to align the scan")
    parser.add_argument("--table-z", type=float, default=None, help="tabletop height in base frame (m)")
    parser.add_argument("--table-edge", type=float, default=0.30,
                        help="x where the tabletop ends; beyond this the arm may go below table level")
    parser.add_argument("--obstacle-mode", choices=("box", "spheres"), default="box",
                        help="box: one bounding box round all touched points. spheres: each "
                             "touched point is its own small obstacle, which is what you want "
                             "when the points are separate things rather than one object.")
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
                pairs = json.load(handle)
            registration = kabsch(np.array(pairs["scan"]), np.array(pairs["robot"]))
            points = registration.apply(points)
            registered = True
            print(f"  registered: {registration.rmse * 1000:.1f} mm RMSE over {len(pairs['scan'])} pairs")
            if not registration.is_trustworthy:
                print("  WARNING: residual above 20mm -- re-touch the reference points before planning on this")
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
        session=None if args.obstacle_mode == "spheres" else session,
        scan_points=points, table=table, resolution=args.resolution,
        kinematics=kinematics, scan_poses=scan_poses, scan_is_registered=registered,
    )

    if args.obstacle_mode == "spheres" and session is not None:
        import numpy as np

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
