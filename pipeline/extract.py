"""Track both hands across sampled frames of every clip; write one Parquet per clip.

Sampling is 2 fps at 640x360. The camera is torso-mounted, so normalised image
coordinates are body-relative hand position and need no calibration.
"""
import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wcdata

FPS = 2.0
W, H = 640, 360
PALM_IDS = (0, 5, 9, 13, 17)
WRIST, MID_MCP, THUMB_TIP, INDEX_TIP = 0, 9, 4, 8

_hands = None


def _get_hands():
    global _hands
    if _hands is None:
        import mediapipe as mp_lib

        _hands = mp_lib.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.3,
            model_complexity=1,
        )
    return _hands


def iter_frames(path):
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-vf", f"fps={FPS},scale={W}:{H}",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=W * H * 3 * 4)
    nbytes = W * H * 3
    try:
        while True:
            buf = proc.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            yield np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    finally:
        proc.stdout.close()
        proc.wait()


def imu_per_second(clip, n_seconds):
    """Gyro RMS and torso-lean deviation per second.

    Lean is the angle between the low-passed accelerometer vector (gravity) and
    the clip's median gravity direction. Axes are camera-native and unreoriented,
    so an absolute pitch is not recoverable; deviation magnitude is.
    """
    gyro_rms = np.full(n_seconds, np.nan, np.float32)
    lean = np.full(n_seconds, np.nan, np.float32)
    path = wcdata.imu_path(clip)
    if not path.exists():
        return gyro_rms, lean
    with open(path) as fh:
        d = json.load(fh)
    t = np.asarray(d["t"], np.float32)
    gyro = np.asarray(d["gyro"], np.float32)
    accl = np.asarray(d["accl"], np.float32)
    if t.size == 0:
        return gyro_rms, lean

    rate = max(1, int(round(d.get("rate_hz") or 200)))
    win = rate * 2  # 2 s moving average isolates gravity from motion
    kernel = np.ones(win, np.float32) / win
    lp = np.stack([np.convolve(accl[:, i], kernel, mode="same") for i in range(3)], 1)
    norm = np.linalg.norm(lp, axis=1, keepdims=True)
    unit = lp / np.where(norm == 0, 1, norm)
    ref = np.median(unit, axis=0)
    ref = ref / max(np.linalg.norm(ref), 1e-9)
    ang = np.degrees(np.arccos(np.clip(unit @ ref, -1, 1)))

    gmag2 = (gyro ** 2).sum(1)
    sec = np.clip(t.astype(np.int32), 0, n_seconds - 1)
    for s in range(n_seconds):
        m = sec == s
        if m.any():
            gyro_rms[s] = np.sqrt(gmag2[m].mean())
            lean[s] = ang[m].mean()
    return gyro_rms, lean


def track_clip(clip):
    """Returns a dict of per-frame columns. Two hand slots, ordered by x."""
    hands = _get_hands()
    rows = {k: [] for k in (
        "t_s", "n_hands",
        "h0_cx", "h0_cy", "h0_size", "h0_aperture", "h0_score", "h0_label", "h0_lm",
        "h1_cx", "h1_cy", "h1_size", "h1_aperture", "h1_score", "h1_label", "h1_lm",
    )}
    for i, frame in enumerate(iter_frames(wcdata.video_path(clip))):
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
        # Slot by horizontal position: stable frame-to-frame even when the
        # handedness classifier flips, which it does when hands cross.
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
    """Finite difference of palm centroid, normalised units/s.

    None across a detection gap: a gap is missing data, not stillness.
    """
    n = len(rows["t_s"])
    for slot in (0, 1):
        p = f"h{slot}_"
        cx, cy = rows[p + "cx"], rows[p + "cy"]
        speed = [None] * n
        for i in range(1, n):
            if cx[i] is not None and cx[i - 1] is not None:
                dt = rows["t_s"][i] - rows["t_s"][i - 1]
                if dt > 0:
                    speed[i] = float(np.hypot(cx[i] - cx[i - 1], cy[i] - cy[i - 1]) / dt)
        rows[p + "speed"] = speed
    return rows


def process(clip):
    out = wcdata.frames_path(clip)
    if out.exists():
        try:
            pq.read_metadata(out)
            return clip["clip_id"], "skip", 0.0
        except Exception:
            out.unlink(missing_ok=True)
    t0 = time.time()
    try:
        rows = add_speeds(track_clip(clip))
        n = len(rows["t_s"])
        if n == 0:
            return clip["clip_id"], "empty", time.time() - t0
        seconds = int(np.ceil(n / FPS))
        gyro_rms, lean = imu_per_second(clip, seconds)
        idx = np.clip((np.asarray(rows["t_s"]) ).astype(np.int32), 0, seconds - 1)
        rows["gyro_rms"] = [float(v) if np.isfinite(v) else None for v in gyro_rms[idx]]
        rows["torso_lean_deg"] = [float(v) if np.isfinite(v) else None for v in lean[idx]]
        rows["clip_id"] = [clip["clip_id"]] * n
        rows["task_id"] = [clip["canonical_task_id"]] * n

        table = pa.table({
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
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, out)
        return clip["clip_id"], "ok", time.time() - t0
    except Exception as exc:  # one bad clip must not kill the corpus pass
        return clip["clip_id"], f"error: {exc}", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", action="append", help="task id (repeatable)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    if args.all:
        todo = wcdata.clips()
    elif args.task:
        todo = [c for t in args.task for c in wcdata.clips_for_task(t)]
    else:
        ap.error("pass --task or --all")
    if args.limit:
        todo = todo[: args.limit]

    print(f"{len(todo)} clips, {args.workers} workers", flush=True)
    t0 = time.time()
    done = 0
    if args.workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers, maxtasksperchild=8) as pool:
            for cid, status, dt in pool.imap_unordered(process, todo):
                done += 1
                print(f"[{done}/{len(todo)}] {cid} {status} {dt:.1f}s "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
    else:
        for clip in todo:
            cid, status, dt = process(clip)
            done += 1
            print(f"[{done}/{len(todo)}] {cid} {status} {dt:.1f}s "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)
    print(f"finished in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
