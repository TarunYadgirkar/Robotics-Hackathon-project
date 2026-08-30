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


def decimate(triangle_vertices: np.ndarray, cell_size: float = 0.008) -> Dict[str, List]:
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

        meshes[name] = decimate(points, cell_size)

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
