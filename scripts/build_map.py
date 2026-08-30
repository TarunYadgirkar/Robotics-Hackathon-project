"""Build a collision map in the robot frame.

The command uses measured LiDAR points and can add the two explicitly specified
base clamps. It does not synthesize table planes or fill occluded regions.

  python scripts/build_map.py \
      --scan phone_scan_123.ply \
      --registration registration.json \
      --clamps
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from yam.enrollment import EnrollmentSession
from yam.kinematics import YamKinematics
from yam.lidar import gravity_aligned_kabsch, load_point_cloud, scan_timestamp_from_path
from yam.mapping import base_clamps, build_map


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_rotation(rotation: np.ndarray) -> None:
    if rotation.shape != (3, 3):
        raise ValueError(f"registration rotation must be 3x3, got {rotation.shape}")
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5):
        raise ValueError("registration rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError("registration rotation is not a proper rotation")


def pose_for_scan(data: dict, scan_path: str, session: EnrollmentSession | None) -> list:
    saved_pose = data.get("scan_pose")
    if saved_pose is not None:
        if len(saved_pose) != 6 or not np.isfinite(saved_pose).all():
            raise ValueError("registration scan_pose must contain six finite joint angles")
        return [float(value) for value in saved_pose]

    if session is None:
        raise ValueError(
            "registration has no scan-time pose; provide --enrollment so it can be recovered"
        )
    timestamp = scan_timestamp_from_path(scan_path)
    if timestamp is None:
        raise ValueError(
            "scan filename has no timestamp and registration has no scan_pose"
        )
    sample, _ = session.pose_at(timestamp)
    return sample.joint_angles


def apply_registration(
    scan_points: np.ndarray,
    scan_path: str,
    registration_path: str,
    session: EnrollmentSession | None,
):
    with open(registration_path) as handle:
        data = json.load(handle)

    if "rotation" in data:
        if data.get("schema_version") != 2:
            raise ValueError(
                "legacy arm-fit registration is not accepted; regenerate it with pick_seed.py"
            )
        if data.get("trustworthy") is not True or data.get("verdict") != "good":
            raise ValueError("registration did not pass arm and touched-surface validation")
        if data.get("gravity_constrained") is not True:
            raise ValueError("registration was allowed to tilt away from measured gravity")
        expected_hash = data.get("scan_sha256")
        if not expected_hash or expected_hash != file_sha256(scan_path):
            raise ValueError("registration was produced for a different scan file")

        rotation = np.asarray(data["rotation"], dtype=float)
        translation = np.asarray(data["translation"], dtype=float)
        validate_rotation(rotation)
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError("registration translation must contain three finite values")

        uncertainty = float(data.get("uncertainty_mm", np.inf)) / 1000.0
        if not np.isfinite(uncertainty) or uncertainty <= 0.0:
            raise ValueError("registration has no finite positive uncertainty estimate")
        pose = pose_for_scan(data, scan_path, session)
        return scan_points @ rotation.T + translation, uncertainty, pose, data

    registration = gravity_aligned_kabsch(
        np.asarray(data["scan"]), np.asarray(data["robot"])
    )
    if not registration.is_trustworthy:
        raise ValueError(
            "landmark registration needs at least four spread-out pairs, "
            "under 20 mm RMS and 30 mm worst error"
        )
    pose = pose_for_scan(data, scan_path, session)
    registered = registration.apply(scan_points)
    details = {
        "method": "paired landmarks",
        "rmse_mm": registration.rmse * 1000.0,
        "surface_max_error_mm": registration.uncertainty * 1000.0,
    }
    return registered, registration.uncertainty, pose, details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scan", required=True, help="LiDAR export (.ply/.obj/.stl)")
    parser.add_argument("--registration", required=True,
                        help="validated scan-to-robot transform from pick_seed.py or paired landmarks")
    parser.add_argument("--enrollment", default=None,
                        help="needed only when the registration lacks its scan-time arm pose")
    parser.add_argument("--self-filter-padding", type=float, default=0.10,
                        help="clearance around the measured scan-time arm while removing it")
    parser.add_argument("--protect-below-z", type=float, default=0.0,
                        help="do not erase measured scan points below the robot mounting datum")
    parser.add_argument("--clamps", action="store_true",
                        help="add the two user-specified base clamp boxes")
    parser.add_argument("--resolution", type=float, default=0.02)
    parser.add_argument("--output", default="workcell_map.npz")
    args = parser.parse_args()

    kinematics = YamKinematics()
    session = EnrollmentSession.load(args.enrollment) if args.enrollment else None
    scan_points = load_point_cloud(args.scan)
    print(f"  loaded {len(scan_points):,} measured points from {args.scan}")

    try:
        registered_points, uncertainty, scan_pose, details = apply_registration(
            scan_points, args.scan, args.registration, session
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"  refusing to build map: {error}") from error

    voxel_diagonal = args.resolution * np.sqrt(3.0)
    minimum_self_filter_padding = uncertainty + 0.03 + voxel_diagonal
    if args.self_filter_padding < minimum_self_filter_padding:
        raise SystemExit(
            f"  refusing to build map: self-filter padding "
            f"({args.self_filter_padding * 1000:.0f} mm) must cover registration "
            f"uncertainty, the default planning margin, and one voxel diagonal "
            f"({minimum_self_filter_padding * 1000:.0f} mm total)"
        )

    print(f"  registration: {details.get('method', 'measured fit')}, "
          f"{details.get('rmse_mm', 0):.1f} mm arm RMS, "
          f"{uncertainty * 1000:.1f} mm uncertainty")
    voxel_map = build_map(
        scan_points=registered_points,
        resolution=args.resolution,
        kinematics=kinematics,
        scan_poses=[scan_pose],
        synthetic_boxes=base_clamps() if args.clamps else None,
        scan_is_registered=True,
        registration_uncertainty=uncertainty,
        self_filter_padding=args.self_filter_padding,
        protect_below_z=args.protect_below_z,
        provenance={
            "schema_version": 1,
            "map_built_at_unix": time.time(),
            "scan": {
                "path": os.path.abspath(args.scan),
                "sha256": file_sha256(args.scan),
                "captured_at_unix": scan_timestamp_from_path(args.scan),
            },
            "registration": {
                "path": os.path.abspath(args.registration),
                "sha256": file_sha256(args.registration),
                "schema_version": details.get("schema_version"),
                "verdict": details.get("verdict"),
                "trustworthy": details.get("trustworthy"),
            },
            "parameters": {
                "resolution_m": args.resolution,
                "self_filter_padding_m": args.self_filter_padding,
                "protect_below_z_m": args.protect_below_z,
                "registration_uncertainty_m": uncertainty,
            },
            "synthetic_geometry": {
                "base_clamps": base_clamps() if args.clamps else [],
            },
        },
    )
    voxel_map.save(args.output)

    occupied = int(voxel_map.occupancy.sum())
    synthetic = int(voxel_map.synthetic_occupancy.sum()) if voxel_map.synthetic_occupancy is not None else 0
    print(f"\n  map: {voxel_map.shape} voxels at {args.resolution * 1000:.0f} mm, "
          f"{occupied:,} occupied")
    print(f"  sources: {occupied - synthetic:,} measured LiDAR voxels, "
          f"{synthetic:,} synthetic clamp voxels")
    if args.clamps:
        print("  clamps: 5in tall x 3in along robot X x 1in wide; "
              "outside base sides, spanning z=-1in..+4in")
    print("  no generated planes or filled regions")
    print(f"  saved {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
