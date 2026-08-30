#!/usr/bin/env python3
"""Render WiLoR MANO meshes into a browser-playable video overlay."""

from __future__ import annotations

import argparse
import bisect
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--wilor-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=float, default=60.0)
    return parser.parse_args()


def project_vertices(
    vertices: np.ndarray,
    camera_translation: np.ndarray,
    focal_length: float,
    width: int,
    height: int,
) -> np.ndarray:
    camera_points = vertices + camera_translation
    normalized = camera_points[:, :2] / np.maximum(camera_points[:, 2:3], 1e-6)
    projected = normalized * focal_length
    projected[:, 0] += width / 2.0
    projected[:, 1] += height / 2.0
    return projected.astype(np.int32)


def interpolate_hand(track: list[dict], track_times: list[float], timestamp: float) -> dict | None:
    """Interpolate short detector gaps and the normal 3 FPS sampling interval."""
    position = bisect.bisect_left(track_times, timestamp)
    if position == 0:
        return track[0] if track_times[0] - timestamp <= 0.5 else None
    if position == len(track):
        return track[-1] if timestamp - track_times[-1] <= 0.5 else None

    before = track[position - 1]
    after = track[position]
    before_time = track_times[position - 1]
    after_time = track_times[position]
    gap = after_time - before_time
    if gap > 3.5:
        nearest = before if timestamp - before_time <= after_time - timestamp else after
        nearest_time = before_time if nearest is before else after_time
        return nearest if abs(timestamp - nearest_time) <= 0.5 else None

    mix = (timestamp - before_time) / max(gap, 1e-6)
    return {
        "handedness": before["handedness"],
        "vertices": before["vertices"] * (1.0 - mix) + after["vertices"] * mix,
        "camera_translation": before["camera_translation"] * (1.0 - mix) + after["camera_translation"] * mix,
        "joints_2d": before["joints_2d"] * (1.0 - mix) + after["joints_2d"] * mix,
    }


def main() -> None:
    args = parse_args()
    payload = json.loads(args.wilor_data.read_text())
    faces = np.asarray(payload["mesh"]["faces"], dtype=np.int32)[::5]

    tracks: dict[str, list[dict]] = {"Left": [], "Right": []}
    for hand in payload["hands"]:
        hand["vertices"] = np.asarray(hand["vertices"], dtype=np.float32)
        hand["camera_translation"] = np.asarray(hand["camera_translation"], dtype=np.float32)
        hand["joints_2d"] = np.asarray(hand["joints_2d"], dtype=np.float32)
        tracks.setdefault(hand["handedness"], []).append(hand)
    for track in tracks.values():
        track.sort(key=lambda hand: hand["timestamp_seconds"])
    track_times = {
        side: [float(hand["timestamp_seconds"]) for hand in track]
        for side, track in tracks.items()
    }

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {args.input}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = min(
        int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(round(args.duration_seconds * fps)),
    )
    focal_length = 5000.0 / 256.0 * max(width, height)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", f"{fps:.8f}", "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert encoder.stdin is not None

    colors = {"Left": (100, 255, 125), "Right": (255, 210, 55)}
    rendered = 0
    try:
        while rendered < total_frames:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = rendered / fps
            hands = [
                hand
                for side, track in tracks.items()
                if (hand := interpolate_hand(track, track_times[side], timestamp)) is not None
            ]
            overlay = frame.copy()
            for hand in hands:
                vertices = hand["vertices"]
                translation = hand["camera_translation"]
                points = project_vertices(vertices, translation, focal_length, width, height)
                polygons = [points[face] for face in faces]
                color = colors.get(hand["handedness"], (80, 220, 120))
                cv2.polylines(overlay, polygons, True, color, 1, cv2.LINE_AA)
                for joint in hand["joints_2d"].astype(np.int32):
                    cv2.circle(overlay, tuple(joint), 3, (245, 255, 248), -1, cv2.LINE_AA)
                    cv2.circle(overlay, tuple(joint), 4, color, 1, cv2.LINE_AA)
            frame = cv2.addWeighted(overlay, 0.82, frame, 0.18, 0)
            cv2.putText(
                frame,
                f"WiLoR MANO mesh  |  {timestamp:05.1f}s  |  {len(hands)} hand(s)",
                (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (225, 255, 232), 2, cv2.LINE_AA,
            )
            encoder.stdin.write(frame.tobytes())
            rendered += 1
            if rendered % 300 == 0:
                print(f"Rendered {rendered}/{total_frames} frames", flush=True)
    finally:
        capture.release()
        encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError("ffmpeg failed while encoding WiLoR overlay")
    print(f"Wrote {args.output} ({rendered} frames at {fps:.3f} fps)")


if __name__ == "__main__":
    main()
