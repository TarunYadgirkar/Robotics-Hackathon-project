#!/usr/bin/env python3
"""Composite sampled metric depth over every frame of the source video."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np


NEAR_METERS = 0.5
FAR_METERS = 4.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--depths", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth-fps", type=float, default=2.0)
    parser.add_argument("--width", type=int, default=960)
    return parser.parse_args()


def colorize(depth: np.ndarray) -> np.ndarray:
    normalized = np.clip((depth - NEAR_METERS) / (FAR_METERS - NEAR_METERS), 0, 1)
    return cv2.applyColorMap(((1.0 - normalized) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def main() -> None:
    args = parse_args()
    depths = np.load(args.depths)["depths"]
    if not np.isfinite(depths).all():
        raise RuntimeError("Depth artifact contains invalid values")

    capture = cv2.VideoCapture(str(args.input_video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {args.input_video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_height = round(source_height * args.width / source_width)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = subprocess.Popen(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{args.width}x{output_height}", "-r", str(fps), "-i", "-",
            "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-movflags", "+faststart", "-pix_fmt", "yuv420p", str(args.output),
        ],
        stdin=subprocess.PIPE,
    )

    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_index / fps
            depth_position = min(timestamp * args.depth_fps, len(depths) - 1)
            before = int(np.floor(depth_position))
            after = min(before + 1, len(depths) - 1)
            mix = depth_position - before
            depth = depths[before] * (1.0 - mix) + depths[after] * mix
            if depth.shape != (output_height, args.width):
                depth = cv2.resize(depth, (args.width, output_height), interpolation=cv2.INTER_LINEAR)
            heatmap = colorize(depth)
            resized = cv2.resize(frame, (args.width, output_height), interpolation=cv2.INTER_AREA)
            composite = cv2.addWeighted(resized, 0.42, heatmap, 0.58, 0)
            if ffmpeg.stdin is None:
                raise RuntimeError("ffmpeg input pipe closed")
            ffmpeg.stdin.write(composite.tobytes())
            frame_index += 1
    finally:
        capture.release()
        if ffmpeg.stdin:
            ffmpeg.stdin.close()
        return_code = ffmpeg.wait()
    if return_code:
        raise RuntimeError(f"ffmpeg exited with status {return_code}")
    print(f"Wrote {frame_index} smooth overlay frames to {args.output}")


if __name__ == "__main__":
    main()
