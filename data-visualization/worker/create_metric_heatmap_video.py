#!/usr/bin/env python3
"""Render a synchronized Apple Depth Pro heat-map overlay for the sample clip."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import DepthProForDepthEstimation, DepthProImageProcessor


MODEL_ID = "apple/DepthPro-hf"
NEAR_METERS = 0.5
FAR_METERS = 4.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-fps", type=float, default=3.0)
    parser.add_argument("--width", type=int, default=960)
    return parser.parse_args()


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def infer_depth(
    frame_bgr: np.ndarray,
    processor: DepthProImageProcessor,
    model: DepthProForDepthEstimation,
    device: torch.device,
) -> np.ndarray:
    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    inputs = {name: tensor.to(device) for name, tensor in processor(images=image, return_tensors="pt").items()}
    with torch.inference_mode():
        outputs = model(**inputs)
    result = processor.post_process_depth_estimation(
        outputs,
        target_sizes=[frame_bgr.shape[:2]],
    )[0]
    return result["predicted_depth"].float().cpu().numpy()


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    normalized = np.clip((depth - NEAR_METERS) / (FAR_METERS - NEAR_METERS), 0, 1)
    # Near objects are warm and distant surfaces are cool.
    return cv2.applyColorMap(((1.0 - normalized) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {args.input}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_height = round(source_height * args.width / source_width)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    inference_step = max(1, round(fps / args.sample_fps))

    device = select_device()
    dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32
    print(f"Loading {MODEL_ID} on {device.type}")
    processor = DepthProImageProcessor.from_pretrained(MODEL_ID)
    model = DepthProForDepthEstimation.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        attn_implementation="sdpa",
    ).to(device).eval()

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
    heatmap: np.ndarray | None = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            resized = cv2.resize(frame, (args.width, output_height), interpolation=cv2.INTER_AREA)
            if heatmap is None or frame_index % inference_step == 0:
                depth = infer_depth(resized, processor, model, device)
                heatmap = colorize_depth(depth)
                print(f"Depth frame {frame_index + 1}/{frame_count}", flush=True)
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
    print(f"Wrote {frame_index} synchronized frames to {args.output}")


if __name__ == "__main__":
    main()
