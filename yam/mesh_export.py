"""Decimate the arm's meshes into something a browser can draw.

The shipped STLs are ~1.5 MB each, which is far too much to ship to a viewer
that only needs a recognisable arm. Vertex clustering collapses every vertex in
a grid cell to one representative, which keeps the silhouette while cutting the
triangle count by a couple of orders of magnitude. A convex hull would be
smaller still, but the links are L-shaped and a hull fills in the bends.
"""

import json
from typing import Dict, List, Sequence

import numpy as np


#: The two long segments carry a white band in the middle; everything else on
#: this arm is black. Colour is baked per vertex because the band is part of a
#: link, not a whole link, and a per-link material cannot express that.
BANDED_LINKS = ("link2", "link3")
BAND_HALF_WIDTH = 0.22   # fraction of the link's length, either side of centre


def band_colors(points: np.ndarray, banded: bool) -> np.ndarray:
    """Per-vertex black/white, with a white band across the middle of a long link."""
    colors = np.zeros((len(points), 3))
    if not banded or len(points) == 0:
        return colors

    centred = points - points.mean(axis=0)
    _, _, principal = np.linalg.svd(centred, full_matrices=False)
    along = centred @ principal[0]
    span = along.max() - along.min()
    if span < 1e-6:
        return colors

    normalized = (along - along.min()) / span - 0.5
    colors[np.abs(normalized) < BAND_HALF_WIDTH] = 1.0
    return colors


def decimate(triangle_vertices: np.ndarray, cell_size: float = 0.008, banded: bool = False) -> Dict[str, List]:
    points = np.asarray(triangle_vertices, dtype=float).reshape(-1, 3)
    if len(points) == 0:
        return {"positions": [], "indices": []}

    quantized = np.round(points / cell_size).astype(np.int64)
    _, first_index, inverse = np.unique(quantized, axis=0, return_index=True, return_inverse=True)

    representatives = np.zeros((len(first_index), 3))
    np.add.at(representatives, inverse, points)
    counts = np.bincount(inverse, minlength=len(first_index))
    representatives /= counts[:, None]

    faces = inverse.reshape(-1, 3)
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    faces = faces[keep]

    return {
        "positions": np.round(representatives, 5).ravel().tolist(),
        "indices": faces.ravel().tolist(),
        "colors": band_colors(representatives, banded).ravel().tolist(),
    }


def export_arm_meshes(kinematics, cell_size: float = 0.008) -> Dict[str, Dict]:
    """Decimated mesh per link, in each link's own frame."""
    from yam.kinematics import read_binary_stl
    import os
    import xml.etree.ElementTree as ET

    root = ET.parse(kinematics.urdf_path).getroot()
    meshes: Dict[str, Dict] = {}

    for link in root.findall("link"):
        name = link.get("name")
        visual = link.find("visual")
        if visual is None:
            continue
        mesh = visual.find("geometry/mesh")
        if mesh is None:
            continue

        path = os.path.join(kinematics.root_dir, mesh.get("filename"))
        if not os.path.isfile(path):
            continue

        points = read_binary_stl(path)
        origin = visual.find("origin")
        if origin is not None:
            from yam.kinematics import rpy_to_matrix

            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            points = points @ rpy_to_matrix(*rpy).T + xyz

        meshes[name] = decimate(points, cell_size, banded=name in BANDED_LINKS)

    return meshes


def export_kinematic_chain(kinematics) -> List[Dict]:
    """Joint chain description so a viewer can pose the arm itself."""
    return [
        {
            "name": joint.name,
            "parent": joint.parent,
            "child": joint.child,
            "type": joint.joint_type,
            "origin": np.asarray(joint.origin).ravel().tolist(),
            "axis": np.asarray(joint.axis).tolist(),
        }
        for joint in (kinematics.joints[n] for n in kinematics.joint_order)
    ]
