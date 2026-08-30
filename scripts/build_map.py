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

    points = None
    scan_pose = None
    if args.scan:
        points = load_point_cloud(args.scan)
        print(f"  loaded {len(points):,} points from {args.scan}")

        if args.registration:
            import json

            with open(args.registration) as handle:
                pairs = json.load(handle)
            registration = kabsch(np.array(pairs["scan"]), np.array(pairs["robot"]))
            points = registration.apply(points)
            print(f"  registered: {registration.rmse * 1000:.1f} mm RMSE over {len(pairs['scan'])} pairs")
            if not registration.is_trustworthy:
                print("  WARNING: residual above 20mm -- re-touch the reference points before planning on this")
        else:
            print("  NOTE: no --registration given, so the scan is assumed to already be in the base frame")

    if session and session.objects and session.objects[0].points:
        scan_pose = session.objects[0].points[-1].joint_angles

    table = table_slab(args.table_z, args.table_edge) if args.table_z is not None else None

    voxel_map = build_map(
        session=session, scan_points=points, table=table, resolution=args.resolution,
        kinematics=kinematics, scan_pose=scan_pose,
    )
    voxel_map.save(args.output)

    occupied = int(voxel_map.occupancy.sum())
    print(f"\n  map: {voxel_map.shape} voxels at {args.resolution * 1000:.0f}mm, {occupied:,} occupied")
    print(f"  saved {args.output}")


if __name__ == "__main__":
    main()
