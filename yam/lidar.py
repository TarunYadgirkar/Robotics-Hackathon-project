"""Import a phone LiDAR scan and register it into the robot's base frame.

A scan on its own is useless for planning: it lives in the phone's arbitrary
frame, and the planner needs metres from the robot's base. Registration is what
connects them, and the correspondences come from the arm itself -- touch a
feature with the gripper to get its position in robot coordinates, click the
same feature in the scan, repeat. Three non-collinear pairs determine the rigid
transform; more pairs let us report how well it actually fits.

Kabsch gives the least-squares optimal rotation and translation for those pairs.
Scale is deliberately fixed at 1: ARKit-derived scans are metrically scaled, so
solving for scale would mostly absorb touch error and flatter the residual.
"""

import os
import re
import struct
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


def _load_ply(path: str) -> np.ndarray:
    with open(path, "rb") as handle:
        raw = handle.read()

    end = raw.find(b"end_header")
    if end < 0:
        raise ValueError(f"{path}: no PLY header")
    header = raw[:end].decode("ascii", errors="replace")
    body = raw[raw.find(b"\n", end) + 1:]

    fmt_match = re.search(r"format\s+(\S+)", header)
    fmt = fmt_match.group(1) if fmt_match else "ascii"
    count = int(re.search(r"element vertex\s+(\d+)", header).group(1))

    properties = re.findall(r"property\s+(\S+)\s+(\S+)", header)
    vertex_properties = properties[: len(properties)]

    if fmt == "ascii":
        values = np.array([line.split()[:3] for line in body.decode().split("\n")[:count] if line.strip()],
                          dtype=float)
        return values

    type_codes = {
        "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
        "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
        "ushort": "u2", "uint16": "u2", "short": "i2", "int16": "i2",
        "uint": "u4", "uint32": "u4", "int": "i4", "int32": "i4",
    }
    order = "<" if "little" in fmt else ">"
    dtype = np.dtype([(name, order + type_codes[kind]) for kind, name in vertex_properties if kind in type_codes])
    array = np.frombuffer(body, dtype=dtype, count=count)
    return np.stack([array["x"], array["y"], array["z"]], axis=1).astype(float)


def _load_obj(path: str) -> np.ndarray:
    points = []
    with open(path) as handle:
        for line in handle:
            if line.startswith("v "):
                points.append([float(v) for v in line.split()[1:4]])
    return np.array(points, dtype=float)


def load_point_cloud(path: str) -> np.ndarray:
    """Load points from a PLY, OBJ or STL export."""
    extension = os.path.splitext(path)[1].lower()
    if extension == ".ply":
        return _load_ply(path)
    if extension == ".obj":
        return _load_obj(path)
    if extension == ".stl":
        from yam.kinematics import read_binary_stl

        return read_binary_stl(path)
    raise ValueError(
        f"unsupported scan format {extension!r}; export the scan as PLY, OBJ or STL "
        "(USDZ is a zipped USD container and is not read directly)"
    )


@dataclass
class Registration:
    rotation: np.ndarray
    translation: np.ndarray
    rmse: float
    per_point_error: np.ndarray

    def apply(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=float).reshape(-1, 3) @ self.rotation.T + self.translation

    @property
    def is_trustworthy(self) -> bool:
        return self.rmse < 0.02


def kabsch(source: np.ndarray, target: np.ndarray) -> Registration:
    """Least-squares rigid transform mapping `source` points onto `target`."""
    source = np.asarray(source, dtype=float).reshape(-1, 3)
    target = np.asarray(target, dtype=float).reshape(-1, 3)
    if len(source) != len(target):
        raise ValueError(f"need matching counts, got {len(source)} and {len(target)}")
    if len(source) < 3:
        raise ValueError("need at least 3 correspondences to fix a rigid transform")

    source_centre = source.mean(axis=0)
    target_centre = target.mean(axis=0)
    covariance = (source - source_centre).T @ (target - target_centre)
    u, _, vt = np.linalg.svd(covariance)

    # Guard against a reflection: a naive SVD can produce det(R) = -1, which
    # fits the points beautifully and mirrors the entire scan.
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ correction @ u.T
    translation = target_centre - rotation @ source_centre

    residuals = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
    return Registration(rotation, translation, float(np.sqrt((residuals ** 2).mean())), residuals)


def filter_robot_from_scan(points: np.ndarray, kinematics, poses, padding: float = 0.03) -> np.ndarray:
    """Drop scan points that are the robot itself, across every pose it was seen in.

    A sweep of the workcell inevitably includes the arm. Left in, the arm becomes
    a permanent obstacle sitting exactly where it has to move, and the planner
    can never leave the pose it was scanned in.

    `poses` is a sequence of joint vectors, not one pose: a phone sweep takes
    tens of seconds, and anything holding the arm during it drags the arm
    through many configurations. Filtering only the final pose leaves the rest
    of the trajectory in the map as a smear of phantom obstacles.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    poses = np.atleast_2d(np.asarray(poses, dtype=float))

    keep = np.ones(len(points), dtype=bool)
    for pose in poses:
        centers, radii = kinematics.collision_spheres(pose)
        for center, radius in zip(centers, radii):
            keep &= np.linalg.norm(points - center, axis=1) > (radius + padding)
        if not keep.any():
            break
    return points[keep]


def crop_to_workspace(points: np.ndarray, radius: float = 1.0, floor: float = -1.0) -> np.ndarray:
    """Keep only what the arm could ever reach; a room scan is mostly irrelevant."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    within = (np.linalg.norm(points[:, :2], axis=1) <= radius) & (points[:, 2] >= floor)
    return points[within]
