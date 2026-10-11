"""Sample seconds at random, render contact sheets, and hold the index to account.

There are no labels in this dataset, so there is no accuracy to compute. What can
be checked is whether a human looking at a frame agrees with the state the index
assigned it. This script produces the sheets; a person fills in the verdicts.
"""
import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dsdata

STATE_NAMES = ["ABSENT", "TRANSIT", "ONE-HANDED", "TWO-HANDED"]
TILE_W, TILE_H = 320, 180
COLS, ROWS = 4, 3
OUT = dsdata.WORK / "validation"
DETECTION_FLOOR = 0.5


def states_for(clip, v_hi, cfg):
    """Mirror of the browser's classifier, including the median-of-3 smoothing."""
    t = pq.read_table(dsdata.frames_path(clip))
    n = np.array(t["n_hands"].to_pylist(), np.uint8)
    def col(name):
        return np.array([np.nan if v is None else v for v in t[name].to_pylist()], np.float32)
    s0, s1 = col("h0_speed"), col("h1_speed")
    stack = np.stack([s0, s1])
    smax = np.where(np.isnan(stack).all(0), np.nan,
                    np.nanmax(np.where(np.isnan(stack), -np.inf, stack), 0))
    S, fps = cfg["clip_seconds"], cfg["fps"]
    out = np.zeros(S, np.uint8)
    for s in range(S):
        lo, hi = s * fps, min((s + 1) * fps, len(n))
        if lo >= len(n):
            break
        cnt = n[lo:hi].max()
        w = smax[lo:hi][np.isfinite(smax[lo:hi])]
        sp = w.max() if w.size else None
        out[s] = 0 if cnt == 0 else (1 if (sp is not None and sp >= v_hi) else (3 if cnt >= 2 else 2))
    sm = out.copy()
    for i in range(1, S - 1):
        if out[i - 1] == out[i + 1]:
            sm[i] = out[i - 1]
    return sm


def _frame(clip, t, w, h):
    cmd = ["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(dsdata.video_path(clip)),
           "-frames:v", "1", "-vf", f"scale={w}:{h}", "-f", "image2pipe",
           "-vcodec", "mjpeg", "-"]
    buf = subprocess.run(cmd, capture_output=True).stdout
    if not buf:
        return None
    import io
    return Image.open(io.BytesIO(buf)).convert("RGB")


def grab(clip, second):
    """Both frames of the second, side by side.

    A single still cannot confirm or refute a TRANSIT call, which is a claim about
    motion. Two frames half a second apart can.
    """
    half = TILE_W // 2
    a = _frame(clip, second, half, TILE_H)
    b = _frame(clip, second + 0.5, half, TILE_H)
    if a is None and b is None:
        return None
    tile = Image.new("RGB", (TILE_W, TILE_H), (8, 9, 11))
    if a:
        tile.paste(a, (0, 0))
    if b:
        tile.paste(b, (half, 0))
    return tile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    cfg = json.loads((dsdata.REPO / "web/public/data/corpus.json").read_text())["config"]
    corpus_tasks = json.loads((dsdata.REPO / "web/public/data/corpus.json").read_text())["tasks"]
    ok_tasks = {t["id"] for t in corpus_tasks if t["det1"] >= DETECTION_FLOOR}

    rng = random.Random(args.seed)
    pool = [c for c in dsdata.clips()
            if dsdata.frames_path(c).exists() and c["canonical_task_id"] in ok_tasks]
    rng.shuffle(pool)

    picks, cache = [], {}
    while len(picks) < args.n and pool:
        clip = pool[len(picks) % len(pool)]
        cid = clip["clip_id"]
        if cid not in cache:
            cache[cid] = states_for(clip, cfg["v_hi"], cfg)
        second = rng.randrange(cfg["clip_seconds"])
        picks.append({"clip_id": cid, "task": clip["canonical_task_id"],
                      "second": second, "state": int(cache[cid][second]),
                      "state_name": STATE_NAMES[int(cache[cid][second])]})

    OUT.mkdir(parents=True, exist_ok=True)
    by_id = {c["clip_id"]: c for c in dsdata.clips()}
    sheets = []
    for start in range(0, len(picks), COLS * ROWS):
        batch = picks[start:start + COLS * ROWS]
        sheet = Image.new("RGB", (COLS * TILE_W, ROWS * (TILE_H + 18)), (8, 9, 11))
        draw = ImageDraw.Draw(sheet)
        for k, p in enumerate(batch):
            img = grab(by_id[p["clip_id"]], p["second"])
            x, y = (k % COLS) * TILE_W, (k // COLS) * (TILE_H + 18)
            if img:
                sheet.paste(img, (x, y))
            draw.text((x + 4, y + TILE_H + 3),
                      f'#{start + k:02d}  {p["state_name"]}  {p["task"][:22]}',
                      fill=(255, 176, 32))
        path = OUT / f"sheet_{start // (COLS * ROWS):02d}.jpg"
        sheet.save(path, quality=82)
        sheets.append(str(path))
        print("wrote", path, flush=True)

    (OUT / "key.json").write_text(json.dumps(
        {"seed": args.seed, "v_hi": cfg["v_hi"], "picks": picks}, indent=2))
    print(f"{len(picks)} samples across {len({p['task'] for p in picks})} tasks")


if __name__ == "__main__":
    main()
