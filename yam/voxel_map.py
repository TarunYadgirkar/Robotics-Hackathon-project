"""Voxel occupancy map with a Euclidean distance field, for planning against scans.

A half-space ground plane cannot describe this arm's workspace: it is mounted on
the edge of a table, so on one side the arm legitimately reaches *below* the
table surface. The table therefore has to be real, finite geometry, which is
what a scan gives.

Occupied voxels come from LiDAR points or explicit boxes. `scipy`'s exact
Euclidean distance transform then turns that into a distance field, so a
collision query is an array lookup rather than a geometry test -- fast enough to
sit inside an RRT inner loop, and indifferent to how complicated the scan is.
"""

import json
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

#: Space outside the mapped volume is unknown, and unknown is not the same as
#: empty. Treating it as blocked keeps the arm inside the region actually
#: measured; the alternative lets it swing confidently into unscanned space.
UNKNOWN_IS_BLOCKED = True


@dataclass
class VoxelMap:
    origin: np.ndarray
    resolution: float
    occupancy: np.ndarray
    distance_field: Optional[np.ndarray] = None

    @classmethod
    def from_bounds(cls, minimum: Sequence[float], maximum: Sequence[float], resolution: float = 0.02) -> "VoxelMap":
        minimum = np.asarray(minimum, dtype=float)
        maximum = np.asarray(maximum, dtype=float)
        shape = np.maximum(np.ceil((maximum - minimum) / resolution).astype(int), 1)
        return cls(origin=minimum, resolution=resolution, occupancy=np.zeros(shape, dtype=bool))

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.occupancy.shape

    @property
    def maximum(self) -> np.ndarray:
        return self.origin + np.array(self.shape) * self.resolution

    def to_indices(self, points: np.ndarray) -> np.ndarray:
        return np.floor((np.asarray(points, dtype=float) - self.origin) / self.resolution).astype(int)

    def _inside(self, indices: np.ndarray) -> np.ndarray:
        return np.all((indices >= 0) & (indices < np.array(self.shape)), axis=1)

    def add_points(self, points: np.ndarray) -> int:
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        indices = self.to_indices(points)
        keep = indices[self._inside(indices)]
        if len(keep):
            self.occupancy[keep[:, 0], keep[:, 1], keep[:, 2]] = True
        self.distance_field = None
        return len(keep)

    def add_box(self, minimum: Sequence[float], maximum: Sequence[float]) -> None:
        low = np.clip(self.to_indices(np.asarray(minimum)[None, :])[0], 0, np.array(self.shape) - 1)
        high = np.clip(self.to_indices(np.asarray(maximum)[None, :])[0], 0, np.array(self.shape) - 1)
        self.occupancy[low[0]:high[0] + 1, low[1]:high[1] + 1, low[2]:high[2] + 1] = True
        self.distance_field = None

    def carve_spheres(self, centers: np.ndarray, radii: np.ndarray, padding: float = 0.02) -> int:
        """Clear voxels inside the given spheres.

        This is how the robot removes itself from its own scan. A LiDAR sweep of
        the workcell contains the arm, and without this the arm's own body
        becomes a permanent obstacle sitting exactly where it needs to move.
        """
        centers = np.asarray(centers, dtype=float).reshape(-1, 3)
        radii = np.asarray(radii, dtype=float).reshape(-1)
        cleared = 0

        for center, radius in zip(centers, radii):
            reach = radius + padding
            low = np.clip(self.to_indices((center - reach)[None, :])[0], 0, np.array(self.shape) - 1)
            high = np.clip(self.to_indices((center + reach)[None, :])[0], 0, np.array(self.shape) - 1)
            if np.any(high < low):
                continue

            grids = np.meshgrid(
                *[np.arange(low[axis], high[axis] + 1) for axis in range(3)], indexing="ij"
            )
            coordinates = self.origin + (np.stack(grids, axis=-1) + 0.5) * self.resolution
            inside = np.linalg.norm(coordinates - center, axis=-1) <= reach
            block = self.occupancy[low[0]:high[0] + 1, low[1]:high[1] + 1, low[2]:high[2] + 1]
            cleared += int((block & inside).sum())
            block[inside] = False

        self.distance_field = None
        return cleared

    def compute_distance_field(self) -> np.ndarray:
        """Metres from each voxel to the nearest occupied voxel."""
        if not self.occupancy.any():
            self.distance_field = np.full(self.shape, np.inf)
        else:
            self.distance_field = ndimage.distance_transform_edt(
                ~self.occupancy, sampling=self.resolution
            )
        return self.distance_field

    def distance_at(self, points: np.ndarray) -> np.ndarray:
        """Distance to the nearest occupied voxel for each point."""
        if self.distance_field is None:
            self.compute_distance_field()

        points = np.asarray(points, dtype=float).reshape(-1, 3)
        indices = self.to_indices(points)
        inside = self._inside(indices)

        distances = np.full(len(points), 0.0 if UNKNOWN_IS_BLOCKED else np.inf)
        if inside.any():
            valid = indices[inside]
            distances[inside] = self.distance_field[valid[:, 0], valid[:, 1], valid[:, 2]]
        return distances

    def occupied_points(self) -> np.ndarray:
        """Centre of every occupied voxel, for visualization."""
        indices = np.argwhere(self.occupancy)
        return self.origin + (indices + 0.5) * self.resolution

    def save(self, path: str) -> None:
        np.savez_compressed(
            path, origin=self.origin, resolution=self.resolution, occupancy=np.packbits(self.occupancy),
            shape=np.array(self.shape),
        )

    @classmethod
    def load(cls, path: str) -> "VoxelMap":
        data = np.load(path)
        shape = tuple(int(v) for v in data["shape"])
        count = int(np.prod(shape))
        occupancy = np.unpackbits(data["occupancy"])[:count].astype(bool).reshape(shape)
        return cls(origin=data["origin"], resolution=float(data["resolution"]), occupancy=occupancy)
