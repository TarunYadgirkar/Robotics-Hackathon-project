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
    #: The two jaws, driven together off the single gripper motor.
    JAW_JOINT_NAMES = ["joint7", "joint8"]
    #: Prismatic travel of one jaw, from the URDF limit. 0 is fully open.
    JAW_TRAVEL = 0.04695

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

    def link_transforms(self, q: Sequence[float], gripper_opening: float = 1.0) -> Dict[str, np.ndarray]:
        """World transform of every link.

        `gripper_opening` runs 0 (closed) to 1 (open). It defaults to open, which
        is the conservative choice for collision -- open jaws sweep the larger
        volume -- but it must be supplied to place the jaws correctly, and
        anything reasoning about where the jaws actually are needs to pass it.
        Leaving the jaw joints at 0 models a closed gripper 84mm too wide.
        """
        if len(q) != len(self.ARM_JOINT_NAMES):
            raise ValueError(f"expected {len(self.ARM_JOINT_NAMES)} angles, got {len(q)}")

        angles = dict(zip(self.ARM_JOINT_NAMES, q))
        jaw = -(1.0 - min(max(gripper_opening, 0.0), 1.0)) * self.JAW_TRAVEL
        for name in self.JAW_JOINT_NAMES:
            angles[name] = jaw
        frames: Dict[str, np.ndarray] = {"base": np.eye(4)}

        for name in self.joint_order:
            joint = self.joints[name]
            if joint.parent not in frames:
                continue
            frames[joint.child] = frames[joint.parent] @ joint.transform_for(angles.get(name, 0.0))
        return frames

    def tip_position(self, q: Sequence[float]) -> np.ndarray:
        """World position of the gripper frame origin.

        Independent of jaw opening: this is the wrist frame the jaws hang off,
        which is what makes it a stable probe point for enrollment.
        """
        return self.link_transforms(q)["gripper"][:3, 3]

    def jaw_gap(self, q: Sequence[float], gripper_opening: float = 1.0) -> float:
        """Opening between the jaws in metres, measured along the axis they travel.

        Not the 3D distance between the two tip frames: those origins also sit
        ~48mm apart laterally, an offset the jaws never close, so a plain norm
        reports a gap that cannot go below 48mm and never reaches zero.
        """
        frames = self.link_transforms(q, gripper_opening)
        gripper = frames["gripper"]
        travel_axis = gripper[:3, 1]   # jaws slide along the gripper frame's Y
        separation = frames["tip_left"][:3, 3] - frames["tip_right"][:3, 3]
        return float(abs(separation @ travel_axis))

    def collision_spheres(self, q: Sequence[float],
                          links: Optional[Sequence[str]] = None,
                          gripper_opening: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Collision spheres in world coordinates, optionally for a subset of links."""
        frames = self.link_transforms(q, gripper_opening)
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


def numerical_jacobian(kinematics: "YamKinematics", q: np.ndarray, delta: float = 1e-6) -> np.ndarray:
    """3x6 position Jacobian by central differences.

    The URDF gives joint axes, so an analytic Jacobian is available, but central
    differences cannot disagree with `tip_position` -- and it is `tip_position`
    that every enrolled point was measured through.
    """
    jacobian = np.zeros((3, len(q)))
    for index in range(len(q)):
        forward, backward = q.copy(), q.copy()
        forward[index] += delta
        backward[index] -= delta
        jacobian[:, index] = (kinematics.tip_position(forward) - kinematics.tip_position(backward)) / (2 * delta)
    return jacobian


def solve_ik(
    kinematics: "YamKinematics",
    target: Sequence[float],
    seed: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    tolerance: float = 1e-4,
    max_iterations: int = 200,
    damping: float = 0.05,
) -> Tuple[np.ndarray, float, bool]:
    """Damped least-squares IK for tip position only, clamped to joint limits.

    Damping keeps the step finite near singularities, where an undamped pseudo-
    inverse asks for enormous joint velocities. Orientation is left free: this
    arm has six joints, and spending three of them on orientation would rule out
    many otherwise reachable points around an obstacle.
    """
    target = np.asarray(target, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    q = np.clip(np.asarray(seed, dtype=float), lower, upper)

    for _ in range(max_iterations):
        error = target - kinematics.tip_position(q)
        distance = float(np.linalg.norm(error))
        if distance < tolerance:
            return q, distance, True

        jacobian = numerical_jacobian(kinematics, q)
        step = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + (damping ** 2) * np.eye(3), error
        )
        q = np.clip(q + np.clip(step, -0.2, 0.2), lower, upper)

    return q, float(np.linalg.norm(target - kinematics.tip_position(q))), False


def solve_ik_collision_free(
    kinematics: "YamKinematics",
    target: Sequence[float],
    checker,
    lower: Sequence[float],
    upper: Sequence[float],
    seed: Optional[Sequence[float]] = None,
    attempts: int = 60,
    seed_rng: Optional[np.random.Generator] = None,
) -> Optional[np.ndarray]:
    """IK restarted from random seeds until a solution is both accurate and collision-free."""
    rng = seed_rng or np.random.default_rng(0)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    for attempt in range(attempts):
        start = np.asarray(seed, dtype=float) if (attempt == 0 and seed is not None) \
            else lower + rng.random(len(lower)) * (upper - lower)
        q, error, converged = solve_ik(kinematics, target, start, lower, upper)
        if converged and error < 5e-3 and checker.is_free(q):
            return q
    return None
