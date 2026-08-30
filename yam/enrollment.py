"""Enrolling obstacles by touching them with the arm.

The arm is its own measuring instrument: hold it limp, put the gripper on a
feature, and forward kinematics turns the joint angles into a 3D point in the
robot's own base frame. That last part is why this matters even when a LiDAR
scan is available -- the scan arrives in the phone's arbitrary frame, and these
touched points are the only thing that ties it to the robot.

Coverage is tracked by azimuth around the object being enrolled, because points
clustered on one face pin down a box badly. The viewer turns those sectors into
the ring the operator is actually filling in.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

#: Coverage is a direction on a sphere around the object, not an angle on a dial:
#: what matters is that the object was touched from several sides, and "several
#: sides" in 3D includes above and below.
AZIMUTH_SECTORS = 8
ELEVATION_BANDS = 3
PATCH_COUNT = AZIMUTH_SECTORS * ELEVATION_BANDS

#: Enough distinct directions to pin a box down; more is better but not required.
TARGET_PATCHES = 6
TARGET_POINTS = 6


@dataclass
class CapturedPoint:
    position: List[float]
    joint_angles: List[float]
    timestamp: float
    label: str = ""


@dataclass
class EnrolledObject:
    name: str
    points: List[CapturedPoint] = field(default_factory=list)
    padding: float = 0.02

    def positions(self) -> np.ndarray:
        """Obstacle points only.

        Reference points anchor the scan; they are not part of the object's
        shape, and folding them into the bounding box would inflate it out to
        whatever landmark was convenient to touch.
        """
        return np.array(
            [p.position for p in self.points if p.label != "reference"], dtype=float
        ).reshape(-1, 3)

    @property
    def centroid(self) -> np.ndarray:
        points = self.positions()
        return points.mean(axis=0) if len(points) else np.zeros(3)

    def bounds(self):
        points = self.positions()
        if len(points) == 0:
            return None
        return points.min(axis=0) - self.padding, points.max(axis=0) + self.padding

    def patch_coverage(self) -> np.ndarray:
        """Which directions around the object have been touched from.

        Each point is turned into a direction from the object's centroid and
        binned by azimuth and elevation. Points clustered on one face leave most
        patches dark, which is exactly the feedback the operator needs -- a box
        fitted from one face is a guess about the other five.
        """
        filled = np.zeros(PATCH_COUNT, dtype=bool)
        points = self.positions()
        if len(points) < 2:
            return filled

        offsets = points - points.mean(axis=0)
        lengths = np.linalg.norm(offsets, axis=1)
        # A point sitting on the centroid has no direction to report.
        offsets = offsets[lengths > 1e-6]
        lengths = lengths[lengths > 1e-6]
        if len(offsets) == 0:
            return filled

        azimuth = np.arctan2(offsets[:, 1], offsets[:, 0]) % (2 * np.pi)
        elevation = np.arcsin(np.clip(offsets[:, 2] / lengths, -1.0, 1.0))

        azimuth_index = (azimuth / (2 * np.pi) * AZIMUTH_SECTORS).astype(int) % AZIMUTH_SECTORS
        band = np.clip(((elevation + np.pi / 2) / np.pi * ELEVATION_BANDS).astype(int), 0, ELEVATION_BANDS - 1)
        for a, b in zip(azimuth_index, band):
            filled[b * AZIMUTH_SECTORS + a] = True
        return filled

    @property
    def progress(self) -> float:
        """Blend of "enough points" and "enough different directions"."""
        by_count = min(len(self.points) / TARGET_POINTS, 1.0)
        by_spread = min(int(self.patch_coverage().sum()) / TARGET_PATCHES, 1.0)
        return float(0.4 * by_count + 0.6 * by_spread)

    def to_dict(self) -> Dict:
        low, high = self.bounds() if self.points else (None, None)
        return {
            "name": self.name,
            "padding": self.padding,
            "points": [asdict(p) for p in self.points],
            "min": None if low is None else low.tolist(),
            "max": None if high is None else high.tolist(),
            "patches": self.patch_coverage().tolist(),
            "progress": self.progress,
        }


@dataclass
class PoseSample:
    timestamp: float
    joint_angles: List[float]


@dataclass
class EnrollmentSession:
    objects: List[EnrolledObject] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    #: Every distinct pose the arm was seen in. A phone LiDAR sweep is
    #: continuous, so the arm appears in the scan across a whole trajectory, not
    #: in one pose. Filtering one pose out of the cloud leaves the rest of the
    #: sweep behind as a smear of phantom obstacles along the arm's path.
    pose_log: List[PoseSample] = field(default_factory=list)

    def begin_object(self, name: str, padding: float = 0.02) -> EnrolledObject:
        obstacle = EnrolledObject(name=name, padding=padding)
        self.objects.append(obstacle)
        return obstacle

    @property
    def current(self) -> Optional[EnrolledObject]:
        return self.objects[-1] if self.objects else None

    #: Points touched purely to anchor a LiDAR scan, not to describe an obstacle.
    REFERENCE_LABEL = "reference"

    def reference_points(self) -> List[CapturedPoint]:
        return [p for o in self.objects for p in o.points if p.label == self.REFERENCE_LABEL]

    def capture(self, position: Sequence[float], joint_angles: Sequence[float], label: str = "") -> CapturedPoint:
        if self.current is None:
            self.begin_object("object_1")
        point = CapturedPoint(
            position=[float(v) for v in position],
            joint_angles=[float(v) for v in joint_angles],
            timestamp=time.time(),
            label=label,
        )
        self.current.points.append(point)
        return point

    def log_pose(self, joint_angles: Sequence[float], min_change: float = 0.03) -> bool:
        """Record a pose if it differs from the last logged one.

        Thresholded rather than sampled on a timer: holding still for a minute
        should cost one entry, and a fast sweep should not be undersampled.
        """
        angles = [float(v) for v in joint_angles]
        if self.pose_log:
            previous = np.asarray(self.pose_log[-1].joint_angles)
            if np.abs(np.asarray(angles) - previous).max() < min_change:
                return False
        self.pose_log.append(PoseSample(timestamp=time.time(), joint_angles=angles))
        return True

    def undo(self) -> bool:
        if self.current and self.current.points:
            self.current.points.pop()
            return True
        return False

    def to_dict(self) -> Dict:
        return {
            "started_at": self.started_at,
            "objects": [o.to_dict() for o in self.objects],
            "pose_log": [asdict(p) for p in self.pose_log],
        }

    def save(self, path: str) -> None:
        with open(path, "w") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def load(cls, path: str) -> "EnrollmentSession":
        with open(path) as handle:
            data = json.load(handle)
        session = cls(started_at=data.get("started_at", time.time()))
        session.pose_log = [PoseSample(**p) for p in data.get("pose_log", [])]
        for entry in data.get("objects", []):
            obstacle = EnrolledObject(name=entry["name"], padding=entry.get("padding", 0.02))
            obstacle.points = [CapturedPoint(**p) for p in entry["points"]]
            session.objects.append(obstacle)
        return session

    def to_world_boxes(self):
        from yam.collision import Box

        boxes = []
        for obstacle in self.objects:
            bounds = obstacle.bounds()
            if bounds is not None:
                boxes.append(Box(obstacle.name, bounds[0], bounds[1]))
        return boxes


def touch_repeatability(kinematics, joint_configurations: Sequence[Sequence[float]]) -> Dict:
    """Spread of FK positions for the same physical point touched from different poses.

    Touching one fixed feature from several arm configurations and comparing the
    computed positions is the one check that exercises the whole kinematic chain
    at once. If FK, the joint scaling or the zero offsets are wrong, these points
    scatter; the scatter is the error budget every enrolled obstacle inherits.
    """
    positions = np.array([kinematics.tip_position(q) for q in joint_configurations])
    centroid = positions.mean(axis=0)
    deviations = np.linalg.norm(positions - centroid, axis=1)
    return {
        "count": len(positions),
        "centroid": centroid.tolist(),
        "mean_error_mm": float(deviations.mean() * 1000),
        "max_error_mm": float(deviations.max() * 1000),
        "positions": positions.tolist(),
    }
