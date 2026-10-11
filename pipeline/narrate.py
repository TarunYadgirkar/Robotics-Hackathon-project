"""Speak the index. The script is generated from the measured per-second timeline.

Nothing here describes the scene. Every sentence is a restatement of numbers this
pipeline computed, which is the point: if the narration matches the footage, the
index is reading the footage.

Voice: ElevenLabs when ELEVENLABS_API_KEY is set, macOS `say` otherwise.
"""
import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dsdata
from validate import states_for

NAMES = {0: "hands out of frame", 1: "transit", 2: "one-handed work", 3: "two-handed work"}
OUT = dsdata.REPO / "web" / "public" / "audio"
VOICE = os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
MIN_RUN = 4  # seconds; shorter runs are noise at 2 fps


def spoken_time(s):
    if s == 0:
        return "at the start"
    if s < 60:
        return f"{s} second{'s' if s != 1 else ''} in"
    m, r = s // 60, s % 60
    mins = f"{m} minute{'s' if m > 1 else ''}"
    return f"{mins} in" if r == 0 else f"{mins} {r} second{'s' if r != 1 else ''} in"


def script_for(clip, task, states, window=90):
    runs = []
    start = 0
    for i in range(1, len(states) + 1):
        if i == len(states) or states[i] != states[start]:
            runs.append((start, i, int(states[start])))
            start = i

    lines = [f"{task['display_name']}. Clip {clip['task_clip_index'] + 1} of {task['clip_count']}."]
    said = 0
    for a, b, st in runs:
        if a >= window:
            break
        if b - a < MIN_RUN:
            continue
        lines.append(f"{spoken_time(a).capitalize()}, {NAMES[st]} for {b - a} seconds.")
        said += 1
        if said >= 5:
            break
    if said == 0:
        lines.append("No run longer than four seconds in the first minute and a half.")

    tally = np.bincount(states, minlength=4) / len(states)
    lines.append(
        f"Across the full five minutes: {tally[3] * 100:.0f} percent two-handed, "
        f"{tally[2] * 100:.0f} percent one-handed, {tally[1] * 100:.0f} percent transit, "
        f"{tally[0] * 100:.0f} percent with no hands in frame."
    )
    return " ".join(lines)


def speak_elevenlabs(text, path, key):
    req = urllib.request.Request(
        f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE}",
        data=json.dumps({"text": text, "model_id": "eleven_turbo_v2_5"}).encode(),
        headers={"xi-api-key": key, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        path.write_bytes(r.read())


def speak_say(text, path):
    aiff = path.with_suffix(".aiff")
    subprocess.run(["say", "-v", "Samantha", "-o", str(aiff), text], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(aiff), "-b:a", "96k", str(path)],
                   check=True)
    aiff.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="clips to narrate, one per task")
    args = ap.parse_args()

    key = os.environ.get("ELEVENLABS_API_KEY")
    print("voice:", "elevenlabs" if key else "macos say (set ELEVENLABS_API_KEY to upgrade)")

    corpus = json.loads((dsdata.REPO / "web/public/data/corpus.json").read_text())
    cfg = corpus["config"]
    ok = {t["id"] for t in corpus["tasks"] if t["det1"] >= 0.5}
    tasks = {t["canonical_task_id"]: t for t in dsdata.tasks()}

    seen, chosen = set(), []
    for clip in dsdata.clips():
        tid = clip["canonical_task_id"]
        if tid in seen or tid not in ok or not dsdata.frames_path(clip).exists():
            continue
        seen.add(tid)
        chosen.append(clip)
        if len(chosen) >= args.n:
            break

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for clip in chosen:
        states = states_for(clip, cfg["v_hi"], cfg)
        text = script_for(clip, tasks[clip["canonical_task_id"]], states)
        path = OUT / f"{clip['clip_id']}.mp3"
        try:
            if key:
                speak_elevenlabs(text, path, key)
            else:
                speak_say(text, path)
            manifest[clip["clip_id"]] = {"text": text, "audio": f"audio/{path.name}"}
            print(f"{clip['clip_id']} {clip['canonical_task_id']}", flush=True)
        except Exception as exc:
            print(f"{clip['clip_id']} FAILED {exc}", flush=True)

    (dsdata.REPO / "web/public/data/narration.json").write_text(json.dumps(manifest, indent=2))
    print(f"{len(manifest)} narrations")


if __name__ == "__main__":
    main()
