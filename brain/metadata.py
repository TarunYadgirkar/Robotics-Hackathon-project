"""Metadata access for the decision brain. Reuses pipeline/dsdata.py loaders."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline import dsdata  # noqa: E402


def load_tasks():
    return dsdata.tasks()


def load_clips():
    return dsdata.clips()


def corpus_stats(tasks, clips):
    total_hours = sum(c.get("duration_s", 0) or 0 for c in clips) / 3600.0
    return {
        "n_tasks": len(tasks),
        "n_clips": len(clips),
        "hours": round(total_hours, 2),
    }


def clips_for_task(clips, task_id):
    return [c for c in clips if c["canonical_task_id"] == task_id]
