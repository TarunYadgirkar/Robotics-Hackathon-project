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
from yam.mapping import build_map, table_slab


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--enrollment", default=None, help="enrollment.json from scripts/enroll.py")
    parser.add_argument("--scan", default=None, help="LiDAR export (.ply/.obj/.stl)")
    parser.add_argument("--registration", default=None,
                        help="JSON with 'scan' and 'robot' point lists, to align the scan")
    parser.add_argument("--table-z", type=float, default=None, help="tabletop height in base frame (m)")
    parser.add_argument("--table-edge", type=float, default=0.30,
                        help="x where the tabletop ends; beyond this the arm may go below table level")
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
        session=session, scan_points=points, table=table, resolution=args.resolution,
        kinematics=kinematics, scan_poses=scan_poses, scan_is_registered=registered,
    )
    voxel_map.save(args.output)

    occupied = int(voxel_map.occupancy.sum())
    print(f"\n  map: {voxel_map.shape} voxels at {args.resolution * 1000:.0f}mm, {occupied:,} occupied")
    print(f"  saved {args.output}")


if __name__ == "__main__":
    main()
