#!/usr/bin/env python3
"""Extract MediaPipe hand tracks without running the depth model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from process_one_video import draw_hands, extract_hands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hand-model", type=Path, required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--source-start-seconds", type=float, default=0.0)
    parser.add_argument("--sample-fps", type=float, default=3.0)
    parser.add_argument("--duration-seconds", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    payload, best_frame, best_record = extract_hands(args)
    (args.output / "hands.json").write_text(
        json.dumps(payload, separators=(",", ":"))
    )
    overlay = draw_hands(best_frame, best_record["hands"])
    import cv2

    cv2.imwrite(str(args.output / "mediapipe-preview.jpg"), overlay)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
