"""Shared dataset access: metadata joins and paths."""
import json
import os
from functools import lru_cache
from pathlib import Path

DATA = Path(os.environ.get("WC_DATA", Path.home() / "TarunsCode/wc-hack"))
VIDEO_ROOT = Path(os.environ.get("WC_VIDEOS", "/Volumes/WC23/WORLD_CONTEXT_EXPLORER_V3"))
REPO = Path(__file__).resolve().parent.parent
WORK = Path(os.environ.get("WC_WORK", REPO / "work"))


def _jsonl(name):
    with open(DATA / "meta" / name) as fh:
        return [json.loads(line) for line in fh if line.strip()]


@lru_cache(maxsize=1)
def clips():
    return _jsonl("clips.jsonl")


@lru_cache(maxsize=1)
def tasks():
    return _jsonl("tasks.jsonl")


@lru_cache(maxsize=1)
def sequences():
    return _jsonl("sequences.jsonl")


@lru_cache(maxsize=1)
def previews():
    return {p["clip_id"]: p for p in _jsonl("ui_previews.jsonl")}


def clips_for_task(task_id):
    return [c for c in clips() if c["canonical_task_id"] == task_id]


def video_path(clip):
    return VIDEO_ROOT / clip["relative_path"]


def imu_path(clip):
    return DATA / clip["imu_path"]


def frames_path(clip):
    return WORK / "frames" / clip["canonical_task_id"] / f"{clip['clip_id']}.parquet"
