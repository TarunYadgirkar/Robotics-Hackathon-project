"""Strategy-variance analysis per task: DTW clustering, significance, confound checks.

Reads the already-extracted per-clip Parquets (pipeline/extract.py output) — never
re-extracts video. Feature representation per frame: (palm centroid x, palm centroid y,
grip aperture, hand speed), all already computed by extract.py. Z-scored per clip before
DTW because the camera is torso-mounted and mount position varies by camera far more than
strategy does.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score, silhouette_score
from dtaidistance import dtw, dtw_ndim

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pipeline"))
import wcdata  # noqa: E402

DETECTION_FLOOR = 0.5          # same rule as pipeline/segment.py's det1 exclusion
MIN_CLIPS_FOR_CLUSTERING = 4   # below this, k=2 clustering is not meaningful
MIN_VALID_FRAMES = 3           # per-clip floor on usable (detected) frames
CONFOUND_ARI = 0.30            # adjusted-rand-index threshold for "leaks the partition"
N_PERMUTATIONS = 1000
PERM_SEED = 42

# PLAN_v2's metadata-only heuristic for the 21 "clean" tasks (>=2 families, >=2 cameras,
# >=8 clips, no diversity_warning) — sanity-checked against this script's actual
# clustering-based clean set (excluded=False, family_confounded=False, camera_confounded=False).
PLAN_CLEAN_TASKS = {
    "belly-band-assembly", "binding-pre-fold-stitching", "fabric-cutting-machine",
    "fabric-layering", "garment-back-panel-attachment", "garment-belly-band-wrapping",
    "garment-button-attachment", "garment-carton-packing", "garment-folding-cardboard-insert",
    "garment-folding-general", "garment-inside-out", "garment-iron-press",
    "garment-loop-attachment", "garment-packing-general", "garment-pair-folding",
    "garment-quality-checking", "garment-safety-sticker", "garment-stitching-overlock",
    "loop-tape-preparation", "processing-fabric-cut", "processing-fabric-spread",
}


def task_detection_rate(clips):
    """Fraction of sampled frames with >=1 hand detected, aggregated over the task's clips."""
    total = hits = 0
    for c in clips:
        p = wcdata.frames_path(c)
        if not p.exists():
            continue
        t = pq.read_table(p, columns=["n_hands"])
        n = np.asarray(t["n_hands"].to_pylist(), dtype=np.uint8)
        total += n.size
        hits += int((n >= 1).sum())
    return (hits / total) if total else 0.0


def _col(table, name):
    return np.array([np.nan if v is None else v for v in table[name].to_pylist()], dtype=np.float64)


def clip_landmark_series(clip):
    """Per-frame (cx, cy, aperture, speed), detected frames only, z-scored within the clip."""
    p = wcdata.frames_path(clip)
    if not p.exists():
        return None
    t = pq.read_table(p, columns=["n_hands", "h0_cx", "h0_cy", "h0_aperture", "h0_speed"])
    n = np.asarray(t["n_hands"].to_pylist(), dtype=np.int64)
    mask = n >= 1
    if not mask.any():
        return None
    cx, cy = _col(t, "h0_cx")[mask], _col(t, "h0_cy")[mask]
    ap, sp = _col(t, "h0_aperture")[mask], _col(t, "h0_speed")[mask]
    sp = np.where(np.isnan(sp), 0.0, sp)  # first frame after a gap: no speed measured yet
    ap = np.where(np.isnan(ap), (np.nanmedian(ap) if np.isfinite(np.nanmedian(ap)) else 0.0), ap)
    keep = np.isfinite(cx) & np.isfinite(cy)
    cx, cy, ap, sp = cx[keep], cy[keep], ap[keep], sp[keep]
    if cx.size < MIN_VALID_FRAMES:
        return None
    feats = np.stack([cx, cy, ap, sp], axis=1)
    mu, sigma = feats.mean(0), feats.std(0)
    sigma = np.where(sigma < 1e-9, 1.0, sigma)
    return ((feats - mu) / sigma).astype(np.double)


def clip_imu_series(clip):
    """Fallback: z-scored gyro RMS time series (already computed by extract.py per frame)."""
    p = wcdata.frames_path(clip)
    if not p.exists():
        return None
    t = pq.read_table(p, columns=["gyro_rms"])
    g = _col(t, "gyro_rms")
    g = g[np.isfinite(g)]
    if g.size < MIN_VALID_FRAMES:
        return None
    mu, sigma = g.mean(), g.std()
    sigma = sigma if sigma > 1e-9 else 1.0
    return ((g - mu) / sigma).reshape(-1, 1).astype(np.double)


def pairwise_dtw(series_list, ndim):
    if ndim:
        return dtw_ndim.distance_matrix(series_list)
    return dtw.distance_matrix(series_list)


def cluster_k2(dm):
    n = dm.shape[0]
    if n < 2:
        return None
    condensed = squareform(dm, checks=False)
    z = linkage(condensed, method="average")
    labels = fcluster(z, 2, criterion="maxclust")
    return labels - 1  # -> 0/1


def permutation_p(dm, labels, real_score, n_perm=N_PERMUTATIONS, seed=PERM_SEED):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    count = 0
    tries = 0
    for _ in range(n_perm):
        perm = rng.permutation(labels)
        if len(set(perm.tolist())) < 2:
            continue
        tries += 1
        try:
            s = silhouette_score(dm, perm, metric="precomputed")
        except Exception:
            continue
        if s >= real_score:
            count += 1
    return count / tries if tries else 1.0


def confound_flag(labels, group_ids):
    """True if the k=2 split leaks the partition, OR the partition has <2 groups.

    A single family/camera means every clip IS the same worker/rig: there is no
    cross-worker (or cross-mount) variation for the split to possibly reflect, so
    per PLAN_v2's known landmine ("single-recording-family tasks CANNOT support
    cross-worker variance claims") this counts as confounded/unusable for the
    'ask' tier even though ARI is undefined (no leakage is measurable when there
    is nothing to leak from).
    """
    if len(set(group_ids)) < 2:
        return True
    ari = adjusted_rand_score(group_ids, labels)
    return bool(ari >= CONFOUND_ARI)


def compute_task(task, limit_clips=None):
    task_id = task["canonical_task_id"]
    clips = wcdata.clips_for_task(task_id)
    if limit_clips:
        clips = clips[:limit_clips]
    result = {
        "n_clips": len(clips),
        "n_families": len({c["independent_repetition_id"] for c in clips}),
        "n_cameras": len({c["camera_id"] for c in clips}),
        "detection_rate": None,
        "silhouette_k2": None,
        "perm_p": None,
        "labels": [],
        "families": [],
        "cameras": [],
        "family_confounded": False,
        "camera_confounded": False,
        "excluded": False,
        "exclude_reason": None,
        "method": None,
    }
    det_rate = task_detection_rate(clips)
    result["detection_rate"] = round(det_rate, 4)
    if det_rate < DETECTION_FLOOR:
        result["excluded"] = True
        result["exclude_reason"] = f"detection_rate {det_rate:.3f} < {DETECTION_FLOOR}"
        return result
    if len(clips) < MIN_CLIPS_FOR_CLUSTERING:
        result["excluded"] = True
        result["exclude_reason"] = f"only {len(clips)} clips, need >= {MIN_CLIPS_FOR_CLUSTERING}"
        return result

    usable, series = [], []
    for c in clips:
        s = clip_landmark_series(c)
        if s is not None:
            usable.append(c)
            series.append(s)
    method = "landmarks"
    if len(usable) < MIN_CLIPS_FOR_CLUSTERING:
        usable, series = [], []
        for c in clips:
            s = clip_imu_series(c)
            if s is not None:
                usable.append(c)
                series.append(s)
        method = "imu"
    result["method"] = method
    if len(usable) < MIN_CLIPS_FOR_CLUSTERING:
        result["excluded"] = True
        result["exclude_reason"] = f"insufficient usable clips after {method} extraction ({len(usable)})"
        return result

    dm = pairwise_dtw(series, ndim=(method == "landmarks"))
    labels = cluster_k2(dm)
    if labels is None or len(set(labels.tolist())) < 2:
        result["excluded"] = True
        result["exclude_reason"] = "clustering degenerate (single cluster)"
        return result
    try:
        sil = float(silhouette_score(dm, labels, metric="precomputed"))
    except Exception as exc:
        result["excluded"] = True
        result["exclude_reason"] = f"silhouette failed: {exc}"
        return result
    p = permutation_p(dm, labels, sil)

    families = [c["independent_repetition_id"] for c in usable]
    cameras = [c["camera_id"] for c in usable]
    result.update({
        "silhouette_k2": round(sil, 4),
        "perm_p": round(p, 4),
        "labels": [int(x) for x in labels],
        "families": families,
        "cameras": cameras,
        "family_confounded": confound_flag(labels, families),
        "camera_confounded": confound_flag(labels, cameras),
    })
    return result


def write_ranking(out, path):
    rows = []
    for tid, r in out.items():
        if r["excluded"] or r["silhouette_k2"] is None:
            continue
        rows.append((r["silhouette_k2"], tid, r))
    rows.sort(reverse=True)
    header = (f"{'task_id':<38}{'silhouette':>11}{'perm_p':>9}{'n_clips':>9}"
              f"{'n_fam':>7}{'n_cam':>7}{'fam_conf':>10}{'cam_conf':>10}{'method':>12}")
    lines = [header]
    for sil, tid, r in rows:
        lines.append(
            f"{tid:<38}{sil:>11.4f}{r['perm_p']:>9.4f}{r['n_clips']:>9}"
            f"{r['n_families']:>7}{r['n_cameras']:>7}{str(r['family_confounded']):>10}"
            f"{str(r['camera_confounded']):>10}{r['method']:>12}"
        )
    excluded = sorted(tid for tid, r in out.items() if r["excluded"])
    lines.append("")
    lines.append(f"excluded ({len(excluded)}):")
    for tid in excluded:
        lines.append(f"  {tid}: {out[tid]['exclude_reason']}")

    computed_clean = {
        tid for tid, r in out.items()
        if not r["excluded"] and not r["family_confounded"] and not r["camera_confounded"]
    }
    only_in_plan = sorted(PLAN_CLEAN_TASKS - computed_clean)
    only_in_computed = sorted(computed_clean - PLAN_CLEAN_TASKS)
    lines.append("")
    lines.append("PLAN_v2 clean-list cross-check (metadata heuristic vs computed clustering):")
    if len(out) < 50:
        lines.append(f"  SKIPPED (partial run over {len(out)} tasks, not all 50)")
    elif not only_in_plan and not only_in_computed:
        lines.append(f"  AGREE: {len(computed_clean)} tasks clean by both metadata heuristic and clustering")
    else:
        lines.append(f"  DISAGREEMENT: plan-only={only_in_plan} computed-only={only_in_computed}")
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", help="restrict to these task_ids (smoke test)")
    ap.add_argument("--limit-clips", type=int, help="cap clips per task (smoke test speed)")
    ap.add_argument("--out", default=str(REPO / "variance/results/variance.json"))
    ap.add_argument("--ranking-out", default=str(REPO / "variance/results/ranking.txt"))
    args = ap.parse_args()

    all_tasks = wcdata.tasks()
    if args.tasks:
        all_tasks = [t for t in all_tasks if t["canonical_task_id"] in args.tasks]

    out = {}
    for t in all_tasks:
        tid = t["canonical_task_id"]
        print(f"computing {tid}...", flush=True)
        out[tid] = compute_task(t, limit_clips=args.limit_clips)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    write_ranking(out, args.ranking_out)
    print(f"wrote {out_path} ({len(out)} tasks) and {args.ranking_out}")


if __name__ == "__main__":
    main()
