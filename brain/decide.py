#!/usr/bin/env python3
"""Decision brain CLI. Frozen contract:

    python brain/decide.py "<task query>"

prints one JSON object to stdout:

    {query, tier: "act"|"ask"|"abstain", matched_task_id|null,
     match_score, coverage_detail, evidence: {...}, utterance_slots: {...}}

No hardcoded task names anywhere in this module -- every task name,
alias, clip count, and hour figure comes from meta/tasks.jsonl and
meta/clips.jsonl via pipeline/wcdata.py.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from coverage import COVERAGE_MATCH_THRESHOLD, content_words, rank_tasks  # noqa: E402
from metadata import clips_for_task, corpus_stats, load_clips, load_tasks  # noqa: E402
from variance_io import load_live_index, load_variance  # noqa: E402


def cluster_sizes(labels):
    if not labels:
        return None, None
    a = sum(1 for label in labels if label == 0)
    b = sum(1 for label in labels if label == 1)
    return a, b


def _live_for_query(query, include_live, live_path):
    if not include_live:
        return None
    live_index = load_live_index(live_path)
    if not live_index:
        return None
    return live_index.get(query)


def _abstain_result(query, q_words, ranked, stats, live):
    best = ranked[0] if ranked else None
    coverage_detail = (
        {"covered": best["covered"], "uncovered": best["uncovered"]}
        if best is not None
        else {"covered": [], "uncovered": [{"word": w, "closest": None, "score": 0.0} for w in q_words]}
    )
    match_score = round(best["coverage"], 3) if best is not None else 0.0
    nearest = ranked[:3]

    evidence = {
        "n_matching_demos": 0,
        "nearest_tasks": [
            {"task_id": r["task_id"], "display_name": r["display_name"], "coverage": round(r["coverage"], 3)}
            for r in nearest
        ],
        "corpus_total_clips": stats["n_clips"],
        "corpus_total_tasks": stats["n_tasks"],
        "corpus_total_hours": stats["hours"],
    }
    utterance_slots = {
        "query": query,
        "hours": stats["hours"],
        "n_tasks": stats["n_tasks"],
        "near1": nearest[0]["display_name"] if len(nearest) > 0 else None,
        "near2": nearest[1]["display_name"] if len(nearest) > 1 else None,
    }

    if live:
        n_live_demos = live.get("n_live_demos", 0)
        n_live_demonstrators = live.get("n_live_demonstrators", 0)
        live_stat_line = live.get("live_stat_line")
        evidence["n_live_demos"] = n_live_demos
        evidence["n_live_demonstrators"] = n_live_demonstrators
        evidence["live_stat_line"] = live_stat_line
        utterance_slots["n_live_demos"] = n_live_demos
        utterance_slots["n_live_demonstrators"] = n_live_demonstrators
        utterance_slots["live_stat_line"] = live_stat_line

    return {
        "query": query,
        "tier": "abstain",
        "matched_task_id": None,
        "match_score": match_score,
        "coverage_detail": coverage_detail,
        "evidence": evidence,
        "utterance_slots": utterance_slots,
    }


def _matched_result(query, best, clips, variance, live):
    task_id = best["task_id"]
    n_clips = len(clips_for_task(clips, task_id))
    coverage_detail = {"covered": best["covered"], "uncovered": best["uncovered"]}
    task_variance = variance.get(task_id) if variance else None

    if task_variance and not task_variance.get("excluded", False):
        family_confounded = task_variance.get("family_confounded")
        camera_confounded = task_variance.get("camera_confounded")
        perm_p = task_variance.get("perm_p")
        silhouette = task_variance.get("silhouette_k2")
        usable_for_ask = (family_confounded is False) and (camera_confounded is False)

        if usable_for_ask and perm_p is not None and perm_p <= 0.05:
            tier = "ask"
            cluster_a_n, cluster_b_n = cluster_sizes(task_variance.get("labels"))
            evidence = {
                "silhouette_k2": silhouette,
                "perm_p": perm_p,
                "n_clips": task_variance.get("n_clips", n_clips),
                "n_families": task_variance.get("n_families"),
                "n_cameras": task_variance.get("n_cameras"),
                "cluster_a_n": cluster_a_n,
                "cluster_b_n": cluster_b_n,
            }
            utterance_slots = {
                "task": best["display_name"],
                "n_clips": evidence["n_clips"],
                "k": 2,
                "silhouette": silhouette,
                "perm_p": perm_p,
                "cluster_a_n": cluster_a_n,
                "cluster_b_n": cluster_b_n,
            }
        else:
            tier = "act"
            evidence = {
                "silhouette_k2": silhouette,
                "perm_p": perm_p,
                "n_clips": task_variance.get("n_clips", n_clips),
                "family_confounded": family_confounded,
                "camera_confounded": camera_confounded,
            }
            utterance_slots = {"task": best["display_name"], "n_clips": evidence["n_clips"]}
    else:
        # variance.json missing, or this task excluded/not yet scored --
        # default to silent execution rather than manufacturing a variance
        # claim we cannot back with a number.
        tier = "act"
        evidence = {"n_clips": n_clips, "variance_available": False}
        utterance_slots = {"task": best["display_name"], "n_clips": n_clips}

    if live:
        evidence["n_live_demos"] = live.get("n_live_demos", 0)
        evidence["n_live_demonstrators"] = live.get("n_live_demonstrators", 0)

    return {
        "query": query,
        "tier": tier,
        "matched_task_id": task_id,
        "match_score": round(best["coverage"], 3),
        "coverage_detail": coverage_detail,
        "evidence": evidence,
        "utterance_slots": utterance_slots,
    }


def decide(query, include_live=False, variance_path=None, live_path=None, tasks=None, clips=None):
    tasks = tasks if tasks is not None else load_tasks()
    clips = clips if clips is not None else load_clips()
    stats = corpus_stats(tasks, clips)

    q_words = content_words(query)
    ranked = rank_tasks(q_words, tasks)
    best = ranked[0] if ranked else None
    live = _live_for_query(query, include_live, live_path)

    if not q_words or best is None or best["coverage"] < COVERAGE_MATCH_THRESHOLD:
        return _abstain_result(query, q_words, ranked, stats, live)

    variance = load_variance(variance_path)
    return _matched_result(query, best, clips, variance, live)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Decision brain: coverage + variance -> act/ask/abstain")
    parser.add_argument("query", help="task query, e.g. \"bottle flip\"")
    parser.add_argument("--include-live", action="store_true", help="fold in feedback/live_index.json if present")
    args = parser.parse_args(argv)

    result = decide(args.query, include_live=args.include_live)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
