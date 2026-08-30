"""Optional inputs: variance/results/variance.json (Agent A) and
feedback/live_index.json (Agent E). Both are read defensively -- absence
is a normal, expected state, not an error."""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VARIANCE_PATH = REPO_ROOT / "variance" / "results" / "variance.json"
LIVE_INDEX_PATH = REPO_ROOT / "feedback" / "live_index.json"


def load_variance(path=None):
    p = Path(path) if path else VARIANCE_PATH
    if not p.exists():
        return None
    with open(p) as fh:
        return json.load(fh)


def load_live_index(path=None):
    p = Path(path) if path else LIVE_INDEX_PATH
    if not p.exists():
        return None
    with open(p) as fh:
        return json.load(fh)
