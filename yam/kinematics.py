"""Forward kinematics and collision geometry for the YAM arm, straight from its URDF.

The i2rt SDK does this through MuJoCo; parsing the URDF ourselves keeps the
dependency list to numpy, which matters because mujoco's build chain does not
install cleanly here. Link shapes are approximated by spheres fitted to the
visual meshes, so collision checks cover the whole arm and not just the tip.
"""

import os
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_URDF = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "i2rt", "i2rt", "robot_models", "arm", "yam_pro", "v1", "yam_pro.urdf",
)


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ])


def transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


@dataclass
class Joint:
    name: str
    parent: str
    child: str
    joint_type: str
    origin: np.ndarray
    axis: np.ndarray

    def transform_for(self, angle: float) -> np.ndarray:
        if self.joint_type == "revolute":
            return self.origin @ transform(axis_angle_to_matrix(self.axis, angle), np.zeros(3))
        if self.joint_type == "prismatic":
            return self.origin @ transform(np.eye(3), self.axis * angle)
        return self.origin


@dataclass
class LinkGeometry:
    name: str
    centers: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    radii: np.ndarray = field(default_factory=lambda: np.zeros(0))


def read_binary_stl(path: str) -> np.ndarray:
    with open(path, "rb") as handle:
        handle.seek(80)
        (triangle_count,) = struct.unpack("<I", handle.read(4))
        raw = np.frombuffer(handle.read(triangle_count * 50), dtype=np.uint8)

    if raw.size < triangle_count * 50:
        raise ValueError(f"{path}: truncated STL")

    records = raw.reshape(triangle_count, 50)
    vertex_bytes = np.ascontiguousarray(records[:, 12:48])
    return vertex_bytes.view(np.float32).reshape(-1, 3).astype(np.float64)


def fit_spheres(points: np.ndarray, target_count: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """Cover a point cloud with spheres strung along its longest axis.

    Deliberately conservative: each sphere's radius is the farthest point in its
    slice, so the union always encloses the mesh. Being slightly too fat costs a
    little clearance; being too thin would let a link pass through an obstacle.
    """
    if len(points) == 0:
        return np.zeros((0, 3)), np.zeros(0)

    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, principal = np.linalg.svd(centered, full_matrices=False)
    axis = principal[0]

    projections = centered @ axis
    low, high = projections.min(), projections.max()
    span = high - low
    count = 1 if span < 1e-6 else max(1, min(target_count, int(np.ceil(span / 0.05))))

    edges = np.linspace(low, high, count + 1)
    centers, radii = [], []
    for index in range(count):
        start, end = edges[index], edges[index + 1]
        mask = (projections >= start) & (projections <= end) if index == count - 1 else (projections >= start) & (projections < end)
        if not mask.any():
            continue
        slice_points = points[mask]
        center = slice_points.mean(axis=0)
        centers.append(center)
        radii.append(float(np.linalg.norm(slice_points - center, axis=1).max()))

    return np.array(centers), np.array(radii)


class YamKinematics:
    """Forward kinematics for the 6 driven arm joints."""

    ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

    def __init__(self, urdf_path: str = DEFAULT_URDF, load_geometry: bool = True):
        self.urdf_path = os.path.normpath(urdf_path)
        self.root_dir = os.path.dirname(self.urdf_path)
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()

        self.joints: Dict[str, Joint] = {}
        self.joint_order: List[str] = []
        for element in root.findall("joint"):
            origin = element.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            axis_element = element.find("axis")
            axis = np.fromstring(axis_element.get("xyz"), sep=" ") if axis_element is not None else np.array([0.0, 0.0, 1.0])

            joint = Joint(
                name=element.get("name"),
                parent=element.find("parent").get("link"),
                child=element.find("child").get("link"),
                joint_type=element.get("type"),
                origin=transform(rpy_to_matrix(*rpy), xyz),
                axis=axis,
            )
            self.joints[joint.name] = joint
            self.joint_order.append(joint.name)

        self.link_geometry: Dict[str, LinkGeometry] = {}
        if load_geometry:
            self._load_geometry(root)

    def _load_geometry(self, root: ET.Element) -> None:
        for link in root.findall("link"):
            name = link.get("name")
            visual = link.find("visual")
            if visual is None:
                continue
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue

            path = os.path.join(self.root_dir, mesh.get("filename"))
            if not os.path.isfile(path):
                continue

            points = read_binary_stl(path)
            scale = mesh.get("scale")
            if scale:
                points = points * np.fromstring(scale, sep=" ")

            origin = visual.find("origin")
            if origin is not None:
                xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
                points = points @ rpy_to_matrix(*rpy).T + xyz

            centers, radii = fit_spheres(points)
            self.link_geometry[name] = LinkGeometry(name, centers, radii)

    def link_transforms(self, q: Sequence[float]) -> Dict[str, np.ndarray]:
        """World transform of every link, given the 6 arm joint angles."""
        if len(q) != len(self.ARM_JOINT_NAMES):
            raise ValueError(f"expected {len(self.ARM_JOINT_NAMES)} angles, got {len(q)}")

        angles = dict(zip(self.ARM_JOINT_NAMES, q))
        frames: Dict[str, np.ndarray] = {"base": np.eye(4)}

        for name in self.joint_order:
            joint = self.joints[name]
            if joint.parent not in frames:
                continue
            frames[joint.child] = frames[joint.parent] @ joint.transform_for(angles.get(name, 0.0))
        return frames

    def tip_position(self, q: Sequence[float]) -> np.ndarray:
        """World position of the gripper frame origin."""
        return self.link_transforms(q)["gripper"][:3, 3]

    def collision_spheres(self, q: Sequence[float],
                          links: Optional[Sequence[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Collision spheres in world coordinates, optionally for a subset of links."""
        frames = self.link_transforms(q)
        selected = None if links is None else set(links)
        centers, radii = [], []
        for name, geometry in self.link_geometry.items():
            if name not in frames or len(geometry.centers) == 0:
                continue
            if selected is not None and name not in selected:
                continue
            frame = frames[name]
            centers.append(geometry.centers @ frame[:3, :3].T + frame[:3, 3])
            radii.append(geometry.radii)

        if not centers:
            return np.zeros((0, 3)), np.zeros(0)
        return np.vstack(centers), np.concatenate(radii)
