#!/usr/bin/env python3
"""Track hand identity by motion and resample landmarks to the display frame rate."""

from __future__ import annotations

import argparse
import bisect
import itertools
import json
import math
from pathlib import Path

import numpy as np


PALM_INDICES = (0, 5, 9, 13, 17)
SIDES = ("Left", "Right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-fps", type=float, default=18.0)
    parser.add_argument("--duration-seconds", type=float, default=300.0)
    parser.add_argument("--max-interpolation-gap-seconds", type=float, default=0.35)
    return parser.parse_args()


def anchor(hand: dict) -> np.ndarray:
    points = np.asarray(
        [[hand["landmarks"][index]["x"], hand["landmarks"][index]["y"]] for index in PALM_INDICES],
        dtype=np.float32,
    )
    return points.mean(axis=0)


def blend_points(before: list[dict], after: list[dict], mix: float) -> list[dict]:
    output = []
    for first, second in zip(before, after):
        output.append({
            axis: round(float(first[axis] * (1.0 - mix) + second[axis] * mix), 6)
            for axis in ("x", "y", "z")
        })
    return output


class TrackState:
    def __init__(self, side: str) -> None:
        self.side = side
        self.position: np.ndarray | None = None
        self.velocity = np.zeros(2, dtype=np.float32)
        self.last_time: float | None = None
        self.history: list[dict] = []

    def predicted_position(self, timestamp: float) -> np.ndarray | None:
        if self.position is None or self.last_time is None:
            return None
        elapsed = min(max(timestamp - self.last_time, 0.0), 0.5)
        return self.position + self.velocity * elapsed

    def assignment_cost(self, hand: dict, timestamp: float) -> float:
        detected_position = anchor(hand)
        predicted = self.predicted_position(timestamp)
        label_mismatch = hand["handedness"] != self.side
        if predicted is None or self.last_time is None:
            return (0.0 if not label_mismatch else 0.45) + abs(detected_position[0] - 0.5) * 0.05
        stale = timestamp - self.last_time
        distance = float(np.linalg.norm(detected_position - predicted))
        if stale > 1.0:
            return distance * 0.25 + (0.0 if not label_mismatch else 0.38)
        return distance + (0.0 if not label_mismatch else 0.10)

    def update(self, hand: dict, timestamp: float) -> None:
        detected_position = anchor(hand)
        if self.position is not None and self.last_time is not None:
            elapsed = timestamp - self.last_time
            if 0.0 < elapsed <= 0.5:
                measured_velocity = (detected_position - self.position) / elapsed
                self.velocity = self.velocity * 0.65 + measured_velocity * 0.35
            else:
                self.velocity[:] = 0.0
        self.position = detected_position
        self.last_time = timestamp
        tracked = dict(hand)
        tracked["raw_handedness"] = hand["handedness"]
        tracked["handedness"] = self.side
        tracked["track_id"] = self.side.lower()
        tracked["interpolated"] = False
        tracked["clip_timestamp_seconds"] = timestamp
        self.history.append(tracked)


def assign_detections(states: dict[str, TrackState], hands: list[dict], timestamp: float) -> None:
    if not hands:
        return
    if len(hands) == 1:
        selected = min(SIDES, key=lambda side: states[side].assignment_cost(hands[0], timestamp))
        states[selected].update(hands[0], timestamp)
        return

    hands = hands[:2]
    best_assignment = min(
        itertools.permutations(SIDES, len(hands)),
        key=lambda assignment: sum(
            states[side].assignment_cost(hand, timestamp)
            for side, hand in zip(assignment, hands)
        ),
    )
    for side, hand in zip(best_assignment, hands):
        states[side].update(hand, timestamp)


def sample_track(
    history: list[dict],
    times: list[float],
    timestamp: float,
    max_gap: float,
    output_fps: float,
) -> dict | None:
    position = bisect.bisect_left(times, timestamp)
    if position < len(times) and abs(times[position] - timestamp) < 1e-4:
        return {key: value for key, value in history[position].items() if key != "clip_timestamp_seconds"}
    if position == 0 or position == len(times):
        candidate_index = 0 if position == 0 else len(times) - 1
        if abs(times[candidate_index] - timestamp) > 1.0 / output_fps:
            return None
        return {key: value for key, value in history[candidate_index].items() if key != "clip_timestamp_seconds"}

    before = history[position - 1]
    after = history[position]
    before_time = times[position - 1]
    after_time = times[position]
    gap = after_time - before_time
    if gap > max_gap:
        nearest_index = position - 1 if timestamp - before_time <= after_time - timestamp else position
        if abs(times[nearest_index] - timestamp) > 1.5 / output_fps:
            return None
        return {key: value for key, value in history[nearest_index].items() if key != "clip_timestamp_seconds"}

    mix = (timestamp - before_time) / max(gap, 1e-9)
    return {
        "handedness": before["handedness"],
        "raw_handedness": f'{before["raw_handedness"]}->{after["raw_handedness"]}',
        "track_id": before["track_id"],
        "confidence": round(float(before["confidence"] * (1.0 - mix) + after["confidence"] * mix), 4),
        "landmarks": blend_points(before["landmarks"], after["landmarks"], mix),
        "world_landmarks": blend_points(before["world_landmarks"], after["world_landmarks"], mix),
        "interpolated": True,
    }


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text())
    states = {side: TrackState(side) for side in SIDES}
    for frame in payload["frames"]:
        assign_detections(states, frame["hands"], float(frame["clip_timestamp_seconds"]))

    histories = {side: states[side].history for side in SIDES}
    history_times = {
        side: [float(hand["clip_timestamp_seconds"]) for hand in history]
        for side, history in histories.items()
    }
    frame_count = int(round(args.duration_seconds * args.output_fps))
    frames = []
    interpolated_hands = 0
    for frame_index in range(frame_count):
        timestamp = frame_index / args.output_fps
        hands = []
        for side in SIDES:
            sampled = sample_track(
                histories[side], history_times[side], timestamp,
                args.max_interpolation_gap_seconds, args.output_fps,
            )
            if sampled is not None:
                interpolated_hands += int(sampled.get("interpolated", False))
                hands.append(sampled)
        frames.append({
            "timestamp_seconds": round(timestamp, 3),
            "clip_timestamp_seconds": round(timestamp, 3),
            "hands": hands,
        })

    frames_with_hands = sum(bool(frame["hands"]) for frame in frames)
    frames_with_both = sum(len(frame["hands"]) == 2 for frame in frames)
    output = {
        "schema_version": "1.1",
        "source_url": payload["source_url"],
        "source_start_seconds": 0.0,
        "sample_fps": args.output_fps,
        "source_detection_fps": payload["sample_fps"],
        "video": {
            **payload["video"],
            "fps": args.output_fps,
            "frame_count": frame_count,
            "duration_seconds": args.duration_seconds,
        },
        "joint_names": payload["joint_names"],
        "connections": payload["connections"],
        "tracking": {
            "identity_method": "palm-position motion prediction with handedness prior",
            "max_interpolation_gap_seconds": args.max_interpolation_gap_seconds,
            "preserves_long_occlusions": True,
        },
        "summary": {
            "sampled_frames": frame_count,
            "frames_with_hands": frames_with_hands,
            "frames_with_both_hands": frames_with_both,
            "detection_coverage": round(frames_with_hands / frame_count, 4),
            "both_hands_coverage": round(frames_with_both / frame_count, 4),
            "maximum_hands": 2,
            "interpolated_hand_instances": interpolated_hands,
        },
        "frames": frames,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, separators=(",", ":")))
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
