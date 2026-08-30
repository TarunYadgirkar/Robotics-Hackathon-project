#!/usr/bin/env python3
"""Generate a one-video hand-landmark and relative-depth 3D proof."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


def rounded(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hand-model", type=Path, required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--source-start-seconds", type=float, default=0.0)
    parser.add_argument("--sample-fps", type=float, default=3.0)
    parser.add_argument("--depth-model", default="depth-anything/Depth-Anything-V2-Small-hf")
    return parser.parse_args()


def video_metadata(capture: cv2.VideoCapture) -> dict[str, float | int]:
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return {
        "fps": fps,
        "frame_count": frames,
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration_seconds": frames / fps if fps else 0.0,
    }


def serialize_hand(result: vision.HandLandmarkerResult, index: int) -> dict:
    handedness = result.handedness[index][0]
    landmarks = result.hand_landmarks[index]
    world_landmarks = result.hand_world_landmarks[index]
    return {
        "handedness": handedness.category_name,
        "confidence": rounded(handedness.score, 4),
        "landmarks": [
            {"x": rounded(item.x), "y": rounded(item.y), "z": rounded(item.z)}
            for item in landmarks
        ],
        "world_landmarks": [
            {"x": rounded(item.x), "y": rounded(item.y), "z": rounded(item.z)}
            for item in world_landmarks
        ],
    }


def draw_hands(frame: np.ndarray, hands: list[dict]) -> np.ndarray:
    output = frame.copy()
    height, width = output.shape[:2]
    for hand in hands:
        points = [
            (int(point["x"] * width), int(point["y"] * height))
            for point in hand["landmarks"]
        ]
        color = (74, 222, 128) if hand["handedness"] == "Left" else (251, 191, 36)
        for start, end in HAND_CONNECTIONS:
            cv2.line(output, points[start], points[end], color, 4, cv2.LINE_AA)
        for point in points:
            cv2.circle(output, point, 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(output, point, 5, color, 2, cv2.LINE_AA)
    return output


def extract_hands(args: argparse.Namespace) -> tuple[dict, np.ndarray, dict]:
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {args.input}")
    metadata = video_metadata(capture)
    source_fps = float(metadata["fps"])
    frame_step = max(1, round(source_fps / args.sample_fps))
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(args.hand_model),
            delegate=python.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.45,
        min_hand_presence_confidence=0.45,
        min_tracking_confidence=0.45,
    )
    frames: list[dict] = []
    best_frame: np.ndarray | None = None
    best_record: dict | None = None
    best_score = -1.0
    frame_index = 0
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_step:
                frame_index += 1
                continue
            local_seconds = frame_index / source_fps
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(image, int(local_seconds * 1000))
            hands = [serialize_hand(result, index) for index in range(len(result.hand_landmarks))]
            record = {
                "timestamp_seconds": rounded(args.source_start_seconds + local_seconds, 3),
                "clip_timestamp_seconds": rounded(local_seconds, 3),
                "hands": hands,
            }
            frames.append(record)
            score = len(hands) * 10 + sum(hand["confidence"] for hand in hands)
            if score > best_score:
                best_score = score
                best_frame = frame.copy()
                best_record = record
            frame_index += 1
    capture.release()
    if best_frame is None or best_record is None:
        raise RuntimeError("No frames were sampled")
    detected_frames = sum(bool(frame["hands"]) for frame in frames)
    payload = {
        "schema_version": "1.0",
        "source_url": args.source_url,
        "source_start_seconds": args.source_start_seconds,
        "sample_fps": args.sample_fps,
        "video": metadata,
        "joint_names": [
            "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
            "index_mcp", "index_pip", "index_dip", "index_tip",
            "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
            "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
            "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
        ],
        "connections": [list(edge) for edge in HAND_CONNECTIONS],
        "summary": {
            "sampled_frames": len(frames),
            "frames_with_hands": detected_frames,
            "detection_coverage": rounded(detected_frames / max(1, len(frames)), 4),
            "maximum_hands": max((len(frame["hands"]) for frame in frames), default=0),
        },
        "frames": frames,
    }
    return payload, best_frame, best_record


def create_depth_scene(
    frame_bgr: np.ndarray,
    model_name: str,
    output_dir: Path,
) -> dict:
    height, width = frame_bgr.shape[:2]
    target_width = 640
    target_height = round(height * target_width / width)
    frame_bgr = cv2.resize(frame_bgr, (target_width, target_height), interpolation=cv2.INTER_AREA)
    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name)
    model.eval()
    inputs = processor(images=image, return_tensors="pt")
    with torch.inference_mode():
        prediction = model(**inputs).predicted_depth
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=(target_height, target_width),
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()
    low, high = np.percentile(prediction, (2, 98))
    normalized = np.clip((prediction - low) / max(1e-6, high - low), 0, 1)
    depth_preview = (normalized * 255).astype(np.uint8)
    cv2.imwrite(str(output_dir / "depth-preview.png"), depth_preview)

    # Depth Anything V2 Small produces relative, not metric, depth. Convert it
    # into a normalized camera-space point cloud for visual feasibility testing.
    z_map = 0.65 + (1.0 - normalized) * 3.35
    focal = target_width / (2 * math.tan(math.radians(90) / 2))
    points: list[list[float | int]] = []
    stride = 5
    for y in range(0, target_height, stride):
        for x in range(0, target_width, stride):
            z = float(z_map[y, x])
            px = (x - target_width / 2) * z / focal
            py = -(y - target_height / 2) * z / focal
            blue, green, red = frame_bgr[y, x]
            points.append([
                rounded(px, 4), rounded(py, 4), rounded(-z, 4),
                int(red), int(green), int(blue),
            ])
    return {
        "schema_version": "1.0",
        "method": "Depth Anything V2 Small relative-depth point cloud",
        "metric_scale": False,
        "coordinate_system": "camera-space",
        "point_format": ["x", "y", "z", "r", "g", "b"],
        "point_count": len(points),
        "source_size": {"width": target_width, "height": target_height},
        "assumed_horizontal_fov_degrees": 90,
        "limitations": [
            "Relative depth only; distances are not measured in meters.",
            "This single-frame feasibility preview is not a multi-view Gaussian reconstruction.",
            "Occluded and unseen surfaces are not reconstructed.",
        ],
        "points": points,
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    hand_payload, best_frame, best_record = extract_hands(args)
    overlay = draw_hands(best_frame, best_record["hands"])
    cv2.imwrite(str(args.output / "hand-overlay-preview.jpg"), overlay)
    (args.output / "hands.json").write_text(json.dumps(hand_payload, separators=(",", ":")))
    scene = create_depth_scene(best_frame, args.depth_model, args.output)
    (args.output / "scene.json").write_text(json.dumps(scene, separators=(",", ":")))
    manifest = {
        "schema_version": "1.0",
        "video_id": "clip_322c7pdpympec",
        "title": "Axle shaft cutting — one-video proof",
        "category": "Axle shaft cutting",
        "source_url": args.source_url,
        "source_start_seconds": args.source_start_seconds,
        "test_clip_duration_seconds": hand_payload["video"]["duration_seconds"],
        "status": "preview_ready",
        "artifacts": {
            "hands": "hands.json",
            "scene": "scene.json",
            "hand_preview": "hand-overlay-preview.jpg",
            "depth_preview": "depth-preview.png",
        },
        "quality": {
            "hand_detection_coverage": hand_payload["summary"]["detection_coverage"],
            "maximum_hands": hand_payload["summary"]["maximum_hands"],
            "scene_point_count": scene["point_count"],
            "scene_type": "relative-depth preview",
            "gaussian_splat_ready": False,
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
