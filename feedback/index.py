"""feedback/live_index.json read/update helpers.

Frozen schema (from PLAN_v2):
  {task_query: {n_live_demos, n_live_demonstrators, takes: [
      {file, n_frames, detection_rate, mean_speed, grip_aperture_range}
  ]}}

One additive, non-frozen field is written per task_query: `live_stat_line`, a
single sentence of measured fact for voice/speak.py's abstain-after-feedback
template (e.g. "peak hand speed 812.3 px/s, 63% of frames tracked"). It is
computed here from the same per-take numbers already in the schema, so any
consumer that ignores unknown keys still reads a schema-conformant file.
"""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def compute_take_stats(table):
    """detection_rate, mean_speed, grip_aperture_range from a written take's Parquet table."""
    n_frames = table.num_rows
    n_hands = np.asarray(table.column("n_hands").to_pylist(), dtype=np.int64)
    detection_rate = float((n_hands > 0).mean()) if n_frames else 0.0

    speeds = []
    apertures = []
    for slot in (0, 1):
        s = table.column(f"h{slot}_speed").to_pylist()
        a = table.column(f"h{slot}_aperture").to_pylist()
        speeds.extend(v for v in s if v is not None and np.isfinite(v))
        apertures.extend(v for v in a if v is not None and np.isfinite(v))

    mean_speed = float(np.mean(speeds)) if speeds else 0.0
    if apertures:
        grip_aperture_range = float(np.max(apertures) - np.min(apertures))
    else:
        grip_aperture_range = 0.0

    return {
        "n_frames": n_frames,
        "detection_rate": round(detection_rate, 4),
        "mean_speed": round(mean_speed, 4),
        "grip_aperture_range": round(grip_aperture_range, 4),
    }


def peak_speed_from_parquet(path):
    table = pq.read_table(path)
    peaks = []
    for slot in (0, 1):
        s = table.column(f"h{slot}_speed").to_pylist()
        peaks.extend(v for v in s if v is not None and np.isfinite(v))
    return max(peaks) if peaks else 0.0


def build_live_stat_line(takes_dir, takes):
    """One sentence of measured fact across all takes for a task_query. Real numbers only."""
    if not takes:
        return "no live demonstrations recorded yet"
    peak_speed = 0.0
    for t in takes:
        p = Path(t["file"])
        if p.exists():
            peak_speed = max(peak_speed, peak_speed_from_parquet(p))
    mean_detection = float(np.mean([t["detection_rate"] for t in takes]))
    return (
        f"peak hand speed {peak_speed:.1f} normalized-units/s, "
        f"{mean_detection * 100:.0f}% of frames tracked across {len(takes)} take(s)"
    )


def load_index(index_path):
    if index_path.exists():
        with open(index_path) as fh:
            return json.load(fh)
    return {}


def save_index(index_path, index):
    index_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = index_path.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        json.dump(index, fh, indent=2, sort_keys=True)
    tmp.replace(index_path)


def update_index(index_path, task_query, take_files_and_stats, n_demonstrators):
    """Merges new takes for task_query into the index and rewrites live_stat_line."""
    index = load_index(index_path)
    entry = index.get(task_query, {"n_live_demos": 0, "n_live_demonstrators": 0, "takes": []})

    existing_files = {t["file"] for t in entry["takes"]}
    for file_path, stats in take_files_and_stats:
        if file_path in existing_files:
            continue
        entry["takes"].append({"file": file_path, **stats})

    entry["n_live_demos"] = len(entry["takes"])
    entry["n_live_demonstrators"] = max(entry.get("n_live_demonstrators", 0), n_demonstrators)
    entry["live_stat_line"] = build_live_stat_line(index_path.parent, entry["takes"])

    index[task_query] = entry
    save_index(index_path, index)
    return entry


def validate_index_schema(index):
    """Raises AssertionError on any frozen-schema violation. Used by smoke tests."""
    assert isinstance(index, dict)
    for task_query, entry in index.items():
        assert isinstance(task_query, str)
        for key in ("n_live_demos", "n_live_demonstrators", "takes"):
            assert key in entry, f"{task_query} missing {key}"
        assert isinstance(entry["n_live_demos"], int)
        assert isinstance(entry["n_live_demonstrators"], int)
        assert isinstance(entry["takes"], list)
        for take in entry["takes"]:
            for key in ("file", "n_frames", "detection_rate", "mean_speed", "grip_aperture_range"):
                assert key in take, f"{task_query} take missing {key}"
