"""Collapse per-frame tracking into a per-second corpus payload for the explorer.

The web app receives raw per-second observations (hand count and hand speed),
not baked-in labels, so the state threshold stays a live control in the UI.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wcdata

CLIP_SECONDS = 300
FPS = 2
HEAT_W, HEAT_H = 32, 18
APERTURE_BINS = 40
OUT = wcdata.REPO / "web" / "public" / "data"
SPEED_SCALE = 100.0  # normalised units/s -> uint8; 254 saturates, 255 = missing
MISSING = 255


def load(clip):
    p = wcdata.frames_path(clip)
    if not p.exists():
        return None
    try:
        return pq.read_table(p)
    except Exception:
        return None


def col(t, name):
    return np.array([np.nan if v is None else v for v in t[name].to_pylist()], np.float32)


def per_second(t):
    """Per-second hand count and max hand speed, reduced from 2 fps."""
    n = np.array(t["n_hands"].to_pylist(), np.uint8)
    s0, s1 = col(t, "h0_speed"), col(t, "h1_speed")
    stack = np.stack([s0, s1])
    smax = np.where(np.isnan(stack).all(0), np.nan, np.nanmax(np.where(np.isnan(stack), -np.inf, stack), 0))
    frames = len(n)
    sec_n = np.zeros(CLIP_SECONDS, np.uint8)
    sec_s = np.full(CLIP_SECONDS, MISSING, np.uint8)
    for s in range(CLIP_SECONDS):
        lo, hi = s * FPS, min((s + 1) * FPS, frames)
        if lo >= frames:
            break
        sec_n[s] = n[lo:hi].max()
        window = smax[lo:hi]
        window = window[np.isfinite(window)]
        if window.size:
            sec_s[s] = min(int(round(window.max() * SPEED_SCALE)), 254)
    return sec_n, sec_s


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = {t["canonical_task_id"]: t for t in wcdata.tasks()}
    previews = wcdata.previews()

    clip_rows, states, missing = [], [], 0
    task_acc = {}

    for clip in wcdata.clips():
        t = load(clip)
        if t is None:
            missing += 1
            continue
        sec_n, sec_s = per_second(t)
        states.append(sec_n.tobytes())
        states.append(sec_s.tobytes())

        prev = previews.get(clip["clip_id"])
        clip_rows.append({
            "id": clip["clip_id"],
            "task": clip["canonical_task_id"],
            "idx": clip["task_clip_index"],
            "cam": clip["camera_id"],
            "rep": clip["independent_repetition_id"],
            "seq": clip["sequence_id"],
            "path": clip["relative_path"],
            "thumb": clip["thumbnail_path"],
            "preview": prev["relative_path"] if prev else None,
            "preview_start": prev["source_start_s"] if prev else None,
        })

        a = task_acc.setdefault(clip["canonical_task_id"], {
            "cy": [], "cx": [], "ap": [], "asym": [], "gyro": [], "lean": [],
            "frames": 0, "det1": 0, "det2": 0, "heat": np.zeros(HEAT_H * HEAT_W, np.float64),
        })
        n = np.array(t["n_hands"].to_pylist(), np.uint8)
        a["frames"] += len(n)
        a["det1"] += int((n >= 1).sum())
        a["det2"] += int((n == 2).sum())
        for slot in (0, 1):
            cx, cy = col(t, f"h{slot}_cx"), col(t, f"h{slot}_cy")
            ap = col(t, f"h{slot}_aperture")
            m = np.isfinite(cx) & np.isfinite(cy)
            a["cx"].append(cx[m]); a["cy"].append(cy[m])
            a["ap"].append(ap[np.isfinite(ap)])
            gx = np.clip((cx[m] * HEAT_W).astype(int), 0, HEAT_W - 1)
            gy = np.clip((cy[m] * HEAT_H).astype(int), 0, HEAT_H - 1)
            np.add.at(a["heat"], gy * HEAT_W + gx, 1)
        s0, s1 = col(t, "h0_speed"), col(t, "h1_speed")
        both = np.isfinite(s0) & np.isfinite(s1)
        if both.any():
            lo = np.minimum(s0[both], s1[both])
            hi = np.maximum(s0[both], s1[both])
            a["asym"].append(np.where(hi > 1e-6, (hi - lo) / (hi + lo + 1e-9), 0.0))
        g, l = col(t, "gyro_rms"), col(t, "torso_lean_deg")
        a["gyro"].append(g[np.isfinite(g)]); a["lean"].append(l[np.isfinite(l)])

    # Global speed distribution -> the default threshold, one rule for all 50 tasks.
    all_speed = np.frombuffer(b"".join(states[1::2]), np.uint8)
    valid = all_speed[all_speed != MISSING].astype(np.float32) / SPEED_SCALE
    v_hi = float(np.percentile(valid, 75)) if valid.size else 0.3

    task_rows, heat = [], []
    for tid, a in sorted(task_acc.items()):
        cx = np.concatenate(a["cx"]) if a["cx"] else np.array([0.5])
        cy = np.concatenate(a["cy"]) if a["cy"] else np.array([0.5])
        ap = np.concatenate(a["ap"]) if a["ap"] else np.array([0.0])
        asym = np.concatenate(a["asym"]) if a["asym"] else np.array([0.0])
        gyro = np.concatenate(a["gyro"]) if a["gyro"] else np.array([np.nan])
        lean = np.concatenate(a["lean"]) if a["lean"] else np.array([np.nan])
        meta = tasks.get(tid, {})
        hist, _ = np.histogram(np.clip(ap, 0, 1.2), bins=APERTURE_BINS, range=(0, 1.2))
        h = a["heat"]
        heat.append((h / max(h.max(), 1) * 65535).astype(np.uint16).tobytes())
        task_rows.append({
            "id": tid,
            "name": meta.get("display_name", tid),
            "clips": meta.get("clip_count"),
            "minutes": meta.get("delivered_minutes"),
            "reps": meta.get("independent_repetition_count"),
            "cameras": meta.get("camera_count"),
            "warnings": meta.get("diversity_warning") or [],
            "det1": a["det1"] / max(a["frames"], 1),
            "det2": a["det2"] / max(a["frames"], 1),
            "palm": [float(cx.mean()), float(cy.mean())],
            "envelope": [float(np.percentile(cx, 5)), float(np.percentile(cy, 5)),
                         float(np.percentile(cx, 95)), float(np.percentile(cy, 95))],
            "aperture": [float(np.percentile(ap, p)) for p in (10, 25, 50, 75, 90)],
            "aperture_hist": hist.tolist(),
            "asymmetry": float(np.median(asym)),
            "gyro_rms": float(np.nanmean(gyro)),
            "lean_span": float(np.nanpercentile(lean, 95) - np.nanpercentile(lean, 5)),
        })

    (OUT / "states.bin").write_bytes(b"".join(states))
    (OUT / "heatmaps.bin").write_bytes(b"".join(heat))
    (OUT / "corpus.json").write_text(json.dumps({
        "config": {
            "clip_seconds": CLIP_SECONDS, "fps": FPS, "speed_scale": SPEED_SCALE,
            "missing": MISSING, "v_hi": round(v_hi, 3),
            "heat_w": HEAT_W, "heat_h": HEAT_H, "aperture_bins": APERTURE_BINS,
        },
        "clips": clip_rows,
        "tasks": task_rows,
    }, separators=(",", ":")))

    print(f"clips {len(clip_rows)} (missing {missing}) tasks {len(task_rows)} v_hi={v_hi:.3f}")
    for f in ("states.bin", "heatmaps.bin", "corpus.json"):
        print(f"  {f} {(OUT / f).stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
