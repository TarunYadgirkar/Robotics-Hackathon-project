"""Encode the index's own pick for each task into a small self-contained clip.

The montage in the explorer normally streams from the media server. These tiles let
the deployed build show the same index-selected moments with no media server and no
USB drive attached — and unlike the shipped 8-second proxies, the window is chosen by
the index rather than fixed in advance.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dsdata
from validate import states_for

WINDOW = 8
BIMANUAL = 3
OUT = dsdata.REPO / "web" / "public" / "tiles"
DETECTION_FLOOR = 0.5


def best_window(states):
    """Window with the most two-handed seconds; ties go to the earliest."""
    hits = (states == BIMANUAL).astype(np.int32)
    if len(hits) < WINDOW:
        return 0, 0
    cum = np.concatenate([[0], np.cumsum(hits)])
    scores = cum[WINDOW:] - cum[:-WINDOW]
    start = int(np.argmax(scores))
    return start, int(scores[start])


def main():
    corpus = json.loads((dsdata.REPO / "web/public/data/corpus.json").read_text())
    cfg = corpus["config"]
    ok = {t["id"] for t in corpus["tasks"] if t["det1"] >= DETECTION_FLOOR}

    best = {}
    for clip in dsdata.clips():
        tid = clip["canonical_task_id"]
        if tid not in ok or not dsdata.frames_path(clip).exists():
            continue
        start, score = best_window(states_for(clip, cfg["v_hi"], cfg))
        if tid not in best or score > best[tid]["score"]:
            best[tid] = {"clip": clip, "start": start, "score": score}

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for tid, pick in sorted(best.items()):
        clip = pick["clip"]
        name = f"{clip['clip_id']}_{pick['start']}.mp4"
        path = OUT / name
        if not path.exists():
            subprocess.run([
                "ffmpeg", "-v", "error", "-y", "-ss", str(pick["start"]),
                "-i", str(dsdata.video_path(clip)), "-t", str(WINDOW),
                "-vf", "scale=320:180", "-an", "-c:v", "libx264", "-crf", "30",
                "-preset", "veryfast", "-movflags", "+faststart", str(path),
            ], check=True)
        manifest[tid] = {
            "clip_id": clip["clip_id"], "start": pick["start"],
            "score": pick["score"], "tile": f"tiles/{name}",
        }
        print(f"{tid} {pick['score']}/{WINDOW} @ {pick['start']}s", flush=True)

    (dsdata.REPO / "web/public/data/tiles.json").write_text(json.dumps(manifest, indent=2))
    total = sum(p.stat().st_size for p in OUT.glob("*.mp4"))
    print(f"{len(manifest)} tiles, {total // 1024 // 1024} MB")


if __name__ == "__main__":
    main()
