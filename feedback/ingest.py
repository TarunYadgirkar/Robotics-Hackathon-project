"""feedback/ingest.py - human feedback ingest, frozen CLI contract:

    python feedback/ingest.py --task "bottle flip" --record 3
        -> captures N webcam takes (spacebar start/stop, ~10s each, 2fps/640px),
           runs the same MediaPipe extraction as pipeline/extract.py, writes one
           Parquet per take to feedback/live_corpus/<slug>/*.parquet, and updates
           feedback/live_index.json.

    python feedback/ingest.py --selftest
        -> end-to-end smoke test through the identical extraction + parquet +
           index code. Uses a real 5s webcam grab if the camera opens, otherwise
           (per coordination/FACTS.md: WEBCAM_STATUS=blocked, TCC permission
           denial) falls back to a synthetic frame sequence. Exits 0 on success.
           Cleans up its own probe entry so the real demo index is untouched.

No variance/silhouette claim is computed from live takes: n_live_demonstrators
defaults to 1 and 3 same-person takes are not evidence of cross-worker strategy
diversity. That computation belongs to variance/ (Agent A) on the corpus only.
"""
import argparse
import re
import shutil
import sys
import time
from pathlib import Path

import cv2
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capture
import extract_shim
import index as live_index

REPO_ROOT = Path(__file__).resolve().parent.parent
LIVE_CORPUS_DIR = REPO_ROOT / "feedback" / "live_corpus"
INDEX_PATH = REPO_ROOT / "feedback" / "live_index.json"


def slugify(task_query):
    s = re.sub(r"[^a-z0-9]+", "_", task_query.strip().lower())
    return s.strip("_") or "task"


def ask_demonstrators(default=1):
    try:
        raw = input(f"number of demonstrators for this batch [default {default}]: ").strip()
        return int(raw) if raw else default
    except (EOFError, ValueError):
        return default


def run_record(task_query, n_takes, demonstrators, device):
    cap = capture.open_camera(device)
    if cap is None:
        print(
            f"webcam not available (device {device}). Per coordination/FACTS.md this is "
            "WEBCAM_STATUS=blocked - a macOS camera-permission (TCC) denial, not missing "
            "hardware. Grant Camera access to this process in System Settings > Privacy "
            "& Security > Camera, then retry.",
            file=sys.stderr,
        )
        sys.exit(1)

    if demonstrators is None:
        demonstrators = ask_demonstrators()

    slug = slugify(task_query)
    out_dir = LIVE_CORPUS_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    take_records = []
    try:
        for i in range(1, n_takes + 1):
            frames = capture.record_take_interactive(cap, max_seconds=10.0, take_label=str(i))
            if not frames:
                print(f"[ingest] take {i}/{n_takes}: no frames captured, skipping", file=sys.stderr)
                continue
            rows = extract_shim.add_speeds(extract_shim.process_frames(frames))
            clip_id = f"{slug}_live_{int(time.time())}_{i}"
            table = extract_shim.build_table(rows, clip_id=clip_id, task_id=slug)
            out_path = out_dir / f"{clip_id}.parquet"
            pq.write_table(table, out_path, compression="zstd")
            stats = live_index.compute_take_stats(table)
            rel = str(out_path.relative_to(REPO_ROOT))
            take_records.append((rel, stats))
            print(
                f"[ingest] take {i}/{n_takes} -> {rel} "
                f"frames={stats['n_frames']} detection_rate={stats['detection_rate']}",
                file=sys.stderr,
            )
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not take_records:
        print("[ingest] no successful takes recorded", file=sys.stderr)
        sys.exit(1)

    entry = live_index.update_index(INDEX_PATH, task_query, take_records, demonstrators)
    print(f"[ingest] live_stat_line: {entry['live_stat_line']}", file=sys.stderr)
    return 0


def run_selftest():
    cap = capture.open_camera(0)
    if cap is not None:
        frames = capture.record_take_auto(cap, duration_s=5.0)
        cap.release()
        source = "webcam"
        if not frames:
            source = "synthetic (webcam opened but yielded 0 frames)"
            frames = capture.synthetic_frame_sequence(duration_s=5.0)
    else:
        source = "synthetic (camera did not open - see FACTS.md WEBCAM_STATUS)"
        frames = capture.synthetic_frame_sequence(duration_s=5.0)

    rows = extract_shim.add_speeds(extract_shim.process_frames(frames))
    clip_id = "selftest_probe"
    task_slug = "_selftest_probe"
    table = extract_shim.build_table(rows, clip_id=clip_id, task_id=task_slug)

    out_dir = LIVE_CORPUS_DIR / task_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{clip_id}.parquet"
    pq.write_table(table, out_path, compression="zstd")

    reopened = pq.read_table(out_path)
    assert reopened.num_rows == table.num_rows, "parquet round-trip row count mismatch"
    assert reopened.num_rows > 0, "extraction produced zero frames"

    stats = live_index.compute_take_stats(reopened)

    pre_existing = INDEX_PATH.exists()
    pre_content = live_index.load_index(INDEX_PATH) if pre_existing else None

    rel = str(out_path.relative_to(REPO_ROOT))
    entry = live_index.update_index(INDEX_PATH, "_selftest_probe", [(rel, stats)], n_demonstrators=1)

    full_index = live_index.load_index(INDEX_PATH)
    live_index.validate_index_schema(full_index)

    print(
        f"[selftest] source={source} frames={stats['n_frames']} "
        f"detection_rate={stats['detection_rate']} mean_speed={stats['mean_speed']} "
        f"grip_aperture_range={stats['grip_aperture_range']}",
        file=sys.stderr,
    )
    print(f"[selftest] live_stat_line: {entry['live_stat_line']}", file=sys.stderr)

    shutil.rmtree(out_dir, ignore_errors=True)
    if pre_existing:
        live_index.save_index(INDEX_PATH, pre_content)
    else:
        INDEX_PATH.unlink(missing_ok=True)

    print("OK selftest passed")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", help="task query string, e.g. 'bottle flip'")
    ap.add_argument("--record", type=int, help="number of takes to capture")
    ap.add_argument("--demonstrators", type=int, default=None)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(run_selftest())

    if not args.task or not args.record:
        ap.error("pass --task \"<query>\" --record N, or --selftest")

    sys.exit(run_record(args.task, args.record, args.demonstrators, args.device))


if __name__ == "__main__":
    main()
