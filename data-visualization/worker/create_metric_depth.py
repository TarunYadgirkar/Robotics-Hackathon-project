#!/usr/bin/env python3
"""Create a metric-depth mesh artifact for the one-video validation sample."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import DepthProForDepthEstimation, DepthProImageProcessor


MODEL_ID = "apple/DepthPro-hf"
TARGET_WIDTH = 640
GRID_STRIDE = 5
ASSUMED_HORIZONTAL_FOV_DEGREES = 90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timestamp", type=float, default=14.681)
    parser.add_argument("--model", default=MODEL_ID)
    return parser.parse_args()


def rounded(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def read_frame(video_path: Path, timestamp: float) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not read frame at {timestamp:.3f}s")
    return frame


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame_bgr = read_frame(args.input, args.timestamp)
    source_height, source_width = frame_bgr.shape[:2]
    target_height = round(source_height * TARGET_WIDTH / source_width)
    frame_bgr = cv2.resize(frame_bgr, (TARGET_WIDTH, target_height), interpolation=cv2.INTER_AREA)
    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    device = select_device()
    print(f"Loading {args.model} on {device.type}")
    dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32
    processor = DepthProImageProcessor.from_pretrained(args.model)
    model = DepthProForDepthEstimation.from_pretrained(
        args.model,
        dtype=dtype,
        attn_implementation="sdpa",
    ).to(device).eval()
    inputs = {name: tensor.to(device) for name, tensor in processor(images=image, return_tensors="pt").items()}
    with torch.inference_mode():
        outputs = model(**inputs)
    prediction = processor.post_process_depth_estimation(
        outputs,
        target_sizes=[(target_height, TARGET_WIDTH)],
    )[0]
    depth_meters = prediction["predicted_depth"].float().cpu().numpy()
    focal_length_pixels = float(prediction["focal_length"].float().cpu())
    field_of_view_degrees = float(prediction["field_of_view"].float().cpu())
    depth_meters = np.clip(depth_meters, 0.05, 20.0)

    preview_low, preview_high = np.percentile(depth_meters, (2, 98))
    normalized = np.clip((depth_meters - preview_low) / max(1e-6, preview_high - preview_low), 0, 1)
    preview = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.imwrite(str(args.output / "metric-depth-preview.png"), preview)

    focal = focal_length_pixels
    points: list[list[float | int]] = []
    sampled_depths: list[float] = []
    for y in range(0, target_height, GRID_STRIDE):
        for x in range(0, TARGET_WIDTH, GRID_STRIDE):
            depth = float(depth_meters[y, x])
            px = (x - TARGET_WIDTH / 2) * depth / focal
            py = -(y - target_height / 2) * depth / focal
            blue, green, red = frame_bgr[y, x]
            points.append([
                rounded(px), rounded(py), rounded(-depth),
                int(red), int(green), int(blue),
            ])
            sampled_depths.append(depth)

    stats = np.asarray(sampled_depths)
    scene = {
        "schema_version": "1.0",
        "method": "Apple Depth Pro zero-shot metric monocular depth mesh",
        "model_id": args.model,
        "metric_scale": True,
        "depth_unit": "meters",
        "coordinate_system": "camera-space",
        "point_format": ["x", "y", "z", "r", "g", "b"],
        "point_count": len(points),
        "source_size": {"width": TARGET_WIDTH, "height": target_height},
        "source_frame_seconds": rounded(args.timestamp, 3),
        "estimated_focal_length_pixels": rounded(focal_length_pixels, 2),
        "estimated_horizontal_fov_degrees": rounded(field_of_view_degrees, 2),
        "depth_stats_meters": {
            "min": rounded(np.min(stats), 2),
            "p05": rounded(np.percentile(stats, 5), 2),
            "median": rounded(np.median(stats), 2),
            "p95": rounded(np.percentile(stats, 95), 2),
            "max": rounded(np.max(stats), 2),
        },
        "limitations": [
            "Depth is inferred from one RGB frame and is not LiDAR or surveyed ground truth.",
            "Camera intrinsics were unavailable, so 3D projection uses the model-estimated focal length.",
            "Occluded and unseen surfaces are not reconstructed.",
        ],
        "points": points,
    }
    output_path = args.output / "metric-scene.json"
    output_path.write_text(json.dumps(scene, separators=(",", ":")))
    print(f"Wrote {len(points):,} metric vertices to {output_path}")
    print(json.dumps(scene["depth_stats_meters"], indent=2))


if __name__ == "__main__":
    main()
