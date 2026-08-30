"""Reuses pipeline/extract.py's MediaPipe hand-tracking exactly, for live human demos.

Imports pipeline.extract's model settings and per-frame math directly rather than
reimplementing them, so a live take is extracted identically to a corpus clip:
same detector confidence, same landmark subset, same aperture/speed definitions.
Only the frame *source* differs (webcam/synthetic vs. ffmpeg-decoded video file) -
track_clip() in extract.py is coupled to a video file path, so the per-frame loop
body is replicated here verbatim against frames already in memory.
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "pipeline"))
import extract as pipeline_extract  # noqa: E402

FPS = pipeline_extract.FPS
W = pipeline_extract.W
H = pipeline_extract.H
PALM_IDS = pipeline_extract.PALM_IDS
WRIST = pipeline_extract.WRIST
MID_MCP = pipeline_extract.MID_MCP
THUMB_TIP = pipeline_extract.THUMB_TIP
INDEX_TIP = pipeline_extract.INDEX_TIP


def process_frames(frames):
    """Same per-frame extraction as pipeline.extract.track_clip, given in-memory RGB frames."""
    hands = pipeline_extract._get_hands()
    rows = {k: [] for k in (
        "t_s", "n_hands",
        "h0_cx", "h0_cy", "h0_size", "h0_aperture", "h0_score", "h0_label", "h0_lm",
        "h1_cx", "h1_cy", "h1_size", "h1_aperture", "h1_score", "h1_label", "h1_lm",
    )}
    for i, frame in enumerate(frames):
        res = hands.process(frame)
        found = []
        if res.multi_hand_landmarks:
            handed = res.multi_handedness or []
            for j, lms in enumerate(res.multi_hand_landmarks):
                pts = np.array([[p.x, p.y] for p in lms.landmark], np.float32)
                centroid = pts[list(PALM_IDS)].mean(0)
                size = float(np.linalg.norm(pts[MID_MCP] - pts[WRIST]))
                aperture = float(np.linalg.norm(pts[THUMB_TIP] - pts[INDEX_TIP]))
                label, score = "", 0.0
                if j < len(handed) and handed[j].classification:
                    label = handed[j].classification[0].label
                    score = float(handed[j].classification[0].score)
                found.append({
                    "cx": float(centroid[0]), "cy": float(centroid[1]),
                    "size": size,
                    "aperture": aperture / size if size > 1e-6 else float("nan"),
                    "score": score, "label": label,
                    "lm": pts.reshape(-1).tolist(),
                })
        found.sort(key=lambda h: h["cx"])
        rows["t_s"].append(i / FPS)
        rows["n_hands"].append(len(found))
        for slot in (0, 1):
            h = found[slot] if slot < len(found) else None
            p = f"h{slot}_"
            rows[p + "cx"].append(h["cx"] if h else None)
            rows[p + "cy"].append(h["cy"] if h else None)
            rows[p + "size"].append(h["size"] if h else None)
            rows[p + "aperture"].append(h["aperture"] if h else None)
            rows[p + "score"].append(h["score"] if h else None)
            rows[p + "label"].append(h["label"] if h else None)
            rows[p + "lm"].append(h["lm"] if h else None)
    return rows


def add_speeds(rows):
    return pipeline_extract.add_speeds(rows)


def build_table(rows, clip_id, task_id):
    """Same column layout as pipeline.extract.process's Parquet output.

    gyro_rms / torso_lean_deg are all-null: live webcam takes carry no IMU stream,
    but the columns are kept so downstream code expecting the corpus schema does
    not need a special case for live data.
    """
    n = len(rows["t_s"])
    rows["clip_id"] = [clip_id] * n
    rows["task_id"] = [task_id] * n
    rows["gyro_rms"] = [None] * n
    rows["torso_lean_deg"] = [None] * n
    return pa.table({
        "clip_id": pa.array(rows["clip_id"], pa.string()),
        "task_id": pa.array(rows["task_id"], pa.string()),
        "t_s": pa.array(rows["t_s"], pa.float32()),
        "n_hands": pa.array(rows["n_hands"], pa.uint8()),
        "gyro_rms": pa.array(rows["gyro_rms"], pa.float32()),
        "torso_lean_deg": pa.array(rows["torso_lean_deg"], pa.float32()),
        **{
            f"h{s}_{f}": pa.array(rows[f"h{s}_{f}"], typ)
            for s in (0, 1)
            for f, typ in (
                ("cx", pa.float32()), ("cy", pa.float32()),
                ("size", pa.float32()), ("aperture", pa.float32()),
                ("speed", pa.float32()), ("score", pa.float32()),
                ("label", pa.string()), ("lm", pa.list_(pa.float32())),
            )
        },
    })
