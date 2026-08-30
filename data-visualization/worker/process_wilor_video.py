#!/usr/bin/env python3
"""Run WiLoR hand-mesh inference using precomputed MediaPipe hand boxes."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "darwin")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import torch

# WiLoR's MANO files can unpickle the legacy chumpy package. Keep that package
# working on modern Python/NumPy without changing the model checkpoint.
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
for legacy_name, replacement in {
    "bool": np.bool_,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}.items():
    if legacy_name not in np.__dict__:
        setattr(np, legacy_name, replacement)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--hand-tracks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hopformer-root", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=float, default=60.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--fast", action="store_true")
    return parser.parse_args()


def rounded_array(value: np.ndarray, digits: int = 6) -> list:
    return np.round(value.astype(np.float64), digits).tolist()


def project_points(
    points: np.ndarray,
    camera_translation: np.ndarray,
    focal_length: float,
    image_size: np.ndarray,
) -> np.ndarray:
    camera_center = image_size / 2.0
    camera_points = points + camera_translation
    normalized = camera_points / np.maximum(camera_points[:, 2:3], 1e-6)
    output = np.empty((len(points), 2), dtype=np.float32)
    output[:, 0] = focal_length * normalized[:, 0] + camera_center[0]
    output[:, 1] = focal_length * normalized[:, 1] + camera_center[1]
    return output


def make_items(capture, hand_payload, model_cfg, duration, max_frames):
    from src.models.wilor.datasets.vitdet_dataset import ViTDetDataset

    items = []
    records = [
        record
        for record in hand_payload["frames"]
        if record["clip_timestamp_seconds"] < duration and record["hands"]
    ]
    if max_frames is not None:
        records = records[:max_frames]
    for frame_number, record in enumerate(records):
        timestamp = float(record["clip_timestamp_seconds"])
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ok, frame = capture.read()
        if not ok:
            continue
        height, width = frame.shape[:2]
        boxes = []
        right_flags = []
        valid_hands = []
        for hand in record["hands"]:
            coords = np.asarray(
                [[point["x"] * width, point["y"] * height] for point in hand["landmarks"]],
                dtype=np.float32,
            )
            minimum = coords.min(axis=0)
            maximum = coords.max(axis=0)
            if np.any(maximum - minimum < 8):
                continue
            boxes.append([minimum[0], minimum[1], maximum[0], maximum[1]])
            right_flags.append(1.0 if hand["handedness"] == "Right" else 0.0)
            valid_hands.append(hand)
        if not boxes:
            continue
        dataset = ViTDetDataset(
            model_cfg,
            frame,
            np.asarray(boxes),
            np.asarray(right_flags),
            rescale_factor=2.0,
        )
        for hand_index in range(len(dataset)):
            item = dataset[hand_index]
            item["timestamp_seconds"] = timestamp
            item["source_hand"] = valid_hands[hand_index]
            item["frame_number"] = frame_number
            items.append(item)
    return items


def stack_batch(items, device, use_half):
    tensor_keys = ("img", "box_center", "box_size", "img_size", "right")
    batch = {}
    for key in tensor_keys:
        values = [torch.as_tensor(item[key]) for item in items]
        batch[key] = torch.stack(values).float().to(device)
    batch["img"] = batch["img"].half() if use_half else batch["img"].float()
    return batch


def draw_preview(frame, hands, faces):
    overlay = frame.copy()
    colors = [(80, 220, 120), (255, 170, 40)]
    for hand_index, hand in enumerate(hands):
        points = np.asarray(hand["vertices_2d"], dtype=np.int32)
        color = colors[hand_index % len(colors)]
        for face in faces[::6]:
            polygon = points[np.asarray(face, dtype=np.int32)]
            cv2.polylines(overlay, [polygon], True, color, 1, cv2.LINE_AA)
        for point in np.asarray(hand["joints_2d"], dtype=np.int32):
            cv2.circle(overlay, tuple(point), 3, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(overlay, tuple(point), 3, color, 1, cv2.LINE_AA)
    return cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    tracks_path = args.hand_tracks.resolve()
    output_dir = args.output.resolve()
    hopformer_root = args.hopformer_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("PYOPENGL_PLATFORM", "darwin")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.chdir(hopformer_root)
    sys.path.insert(0, str(hopformer_root))

    from src.models.wilor import load_wilor
    from src.models.wilor.utils.renderer import cam_crop_to_full

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    started = time.perf_counter()
    model, model_cfg = load_wilor(
        checkpoint_path="./pretrained_models/wilor/wilor_final.ckpt",
        cfg_path="./pretrained_models/wilor/model_config.yaml",
    )
    if args.fast:
        torch.set_float32_matmul_precision("high")
        model = model.half()
        # MANO receives explicitly float32 pose parameters in WiLoR's forward
        # pass, so keep its fixed template tensors in float32 as well.
        model.mano = model.mano.float()
        model.backbone.skip_blocks = True
    model = model.to(device).eval()
    load_seconds = time.perf_counter() - started

    hand_payload = json.loads(tracks_path.read_text())
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {input_path}")
    items = make_items(
        capture,
        hand_payload,
        model_cfg,
        args.duration_seconds,
        args.max_frames,
    )
    capture.release()

    results = []
    preview_candidates = []
    inference_started = time.perf_counter()
    for offset in range(0, len(items), args.batch_size):
        item_batch = items[offset : offset + args.batch_size]
        batch = stack_batch(item_batch, device, args.fast)
        with torch.inference_mode():
            output = model(batch)
        multiplier = 2 * batch["right"] - 1
        pred_camera = output["pred_cam"].clone()
        pred_camera[:, 1] = multiplier * pred_camera[:, 1]
        image_size = batch["img_size"].float()
        focal = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * image_size.max()
        camera_full = cam_crop_to_full(
            pred_camera.float(),
            batch["box_center"].float(),
            batch["box_size"].float(),
            image_size,
            focal,
        ).detach().cpu().numpy()
        vertices_batch = output["pred_vertices"].detach().float().cpu().numpy()
        joints_batch = output["pred_keypoints_3d"].detach().float().cpu().numpy()

        for index, item in enumerate(item_batch):
            right = float(item["right"])
            vertices = vertices_batch[index]
            joints = joints_batch[index]
            mirror = 2 * right - 1
            vertices[:, 0] *= mirror
            joints[:, 0] *= mirror
            size = np.asarray(item["img_size"], dtype=np.float32)
            focal_value = float(focal.detach().cpu())
            vertices_2d = project_points(vertices, camera_full[index], focal_value, size)
            joints_2d = project_points(joints, camera_full[index], focal_value, size)
            record = {
                "timestamp_seconds": round(float(item["timestamp_seconds"]), 3),
                "handedness": item["source_hand"]["handedness"],
                "detection_confidence": item["source_hand"]["confidence"],
                "camera_translation": rounded_array(camera_full[index]),
                "joints": rounded_array(joints),
                "joints_2d": rounded_array(joints_2d, 3),
                "vertices": rounded_array(vertices),
            }
            results.append(record)
            preview_candidates.append(
                {
                    "timestamp_seconds": record["timestamp_seconds"],
                    "vertices_2d": rounded_array(vertices_2d, 3),
                    "joints_2d": record["joints_2d"],
                }
            )
    if device.type == "mps":
        torch.mps.synchronize()
    inference_seconds = time.perf_counter() - inference_started

    output_payload = {
        "schema_version": "1.0",
        "model": "WiLoR",
        "source": {
            "local_clip": str(input_path),
            "url": hand_payload.get("source_url"),
            "start_seconds": hand_payload.get("source_start_seconds", 0.0),
        },
        "duration_seconds": args.duration_seconds,
        "sample_fps": hand_payload["sample_fps"],
        "device": str(device),
        "fast_mode": args.fast,
        "mesh": {
            "format": "MANO",
            "vertex_count": int(model.mano.faces.max()) + 1,
            "face_count": len(model.mano.faces),
            "joint_count": 21,
            "faces": np.asarray(model.mano.faces, dtype=np.int32).tolist(),
        },
        "summary": {
            "sampled_frames": len({item["frame_number"] for item in items}),
            "reconstructed_hands": len(results),
            "model_load_seconds": round(load_seconds, 3),
            "inference_seconds": round(inference_seconds, 3),
            "hands_per_second": round(len(results) / max(inference_seconds, 1e-6), 3),
        },
        "hands": results,
    }
    (output_dir / "wilor-hands.json").write_text(
        json.dumps(output_payload, separators=(",", ":"))
    )

    if preview_candidates:
        target = min(preview_candidates, key=lambda item: abs(item["timestamp_seconds"] - 30.0))
        timestamp = target["timestamp_seconds"]
        selected = [
            item for item in preview_candidates if item["timestamp_seconds"] == timestamp
        ]
        preview_capture = cv2.VideoCapture(str(input_path))
        preview_capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ok, frame = preview_capture.read()
        preview_capture.release()
        if ok:
            preview = draw_preview(frame, selected, np.asarray(model.mano.faces))
            cv2.imwrite(str(output_dir / "wilor-mesh-preview.jpg"), preview)

    print(json.dumps(output_payload["summary"], indent=2))


if __name__ == "__main__":
    main()
