"""Render a planned path, the arm, and every obstacle to a standalone HTML page.

Written to be opened before anything moves. A plan that verifies numerically can
still be the wrong plan, and the cheapest way to notice that is to look at it.
"""

import argparse
import json
import os
import sys
import webbrowser
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yam.arm import ARM_JOINTS
from yam.kinematics import YamKinematics
from yam.mesh_export import export_arm_meshes
from yam.voxel_map import VoxelMap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", default="workcell_map.npz")
    parser.add_argument("--plan", default="/tmp/plan.npy")
    parser.add_argument("--contacts", default=None,
                        help="optional JSON list of contact names and path indices")
    parser.add_argument("--output", default="plan_preview.html")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    kinematics = YamKinematics()
    path = np.load(args.plan)
    voxel_map = VoxelMap.load(args.map)

    frames = []
    for q in path:
        transforms = kinematics.link_transforms(q)
        frames.append({name: np.round(matrix.ravel(), 5).tolist() for name, matrix in transforms.items()})

    tip_track = [np.round(kinematics.probe_position(q), 4).tolist() for q in path]
    measured = voxel_map.measured_points()
    synthetic = voxel_map.synthetic_points()
    contacts = []
    if args.contacts:
        with open(args.contacts) as handle:
            for contact in json.load(handle):
                index = int(contact["index"])
                if index < 0 or index >= len(path):
                    raise ValueError(f"contact index {index} is outside the {len(path)}-pose path")
                contacts.append({
                    "name": str(contact["name"]),
                    "index": index,
                    "position": tip_track[index],
                })

    payload = {
        "meshes": export_arm_meshes(kinematics),
        "frames": frames,
        "tip": tip_track,
        "voxels": np.round(measured, 4).ravel().tolist(),
        "syntheticVoxels": np.round(synthetic, 4).ravel().tolist(),
        "voxelSize": voxel_map.resolution,
        "joints": [j.name for j in ARM_JOINTS],
        "angles": np.round(np.degrees(path), 2).tolist(),
        "contacts": contacts,
    }

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "preview_template.html")) as handle:
        template = handle.read()

    html = template.replace("__PAYLOAD__", json.dumps(payload))
    with open(args.output, "w") as handle:
        handle.write(html)

    print(f"  wrote {args.output}  ({len(html) // 1024} KB, {len(path)} poses, "
          f"{len(measured):,} measured + {len(synthetic):,} synthetic obstacle voxels)")
    if not args.no_browser:
        webbrowser.open("file://" + os.path.abspath(args.output))


if __name__ == "__main__":
    main()
