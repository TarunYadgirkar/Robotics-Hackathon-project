#!/usr/bin/env python3
"""Smoke test for brain/decide.py. Plain assertions, no pytest dependency.
Exits 0 iff every case passes. This is also the regression guard for the
known token_set_ratio bug: if the bottle-task cases below ever fail, BEAT 3
(and its inversion risk) is broken -- treat any failure here as stop-the-line.
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from coverage import FUZZY_MATCH_THRESHOLD, task_vocab, word_similarity  # noqa: E402
from decide import decide  # noqa: E402
from metadata import load_clips, load_tasks  # noqa: E402

FIXTURE_VARIANCE = HERE / "fixtures" / "variance_fixture.json"
FIXTURE_LIVE = HERE / "fixtures" / "live_index_fixture.json"

FAILURES = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(label)


def test_bottle_flip_abstains():
    result = decide("bottle flip")
    check("bottle flip -> tier=abstain", result["tier"] == "abstain", result)
    check("bottle flip -> matched_task_id=null", result["matched_task_id"] is None, result)
    uncovered_words = {u["word"] for u in result["coverage_detail"]["uncovered"]}
    check("bottle flip -> 'flip' is uncovered", "flip" in uncovered_words, result)


def test_flip_never_matches_emergently():
    """Prove abstain is emergent, not a hardcoded 'bottle flip' special case:
    the word 'flip' must not fuzzy-match >=0.85 against ANY real task's
    vocabulary, computed generically over the actual metadata."""
    tasks = load_tasks()
    best = 0.0
    best_against = None
    for task in tasks:
        for vocab_word in task_vocab(task):
            score = word_similarity("flip", vocab_word)
            if score > best:
                best, best_against = score, vocab_word
    check(
        "'flip' never fuzzy-matches any task vocab word (emergent, not special-cased)",
        best < FUZZY_MATCH_THRESHOLD,
        f"closest was {best_against!r} at {best:.3f}",
    )


def test_three_bottle_tasks_match_themselves():
    cases = [
        ("bottle cleaning", "bottle-cleaning"),
        ("bottle surface buffing", "bottle-surface-buffing"),
        ("water filtration bottle filling", "water-filtration-bottle-filling"),
    ]
    for query, expected_task_id in cases:
        result = decide(query)
        check(f"{query!r} -> tier != abstain", result["tier"] != "abstain", result)
        check(f"{query!r} -> matched_task_id={expected_task_id}",
              result["matched_task_id"] == expected_task_id, result)


def test_near_miss_spelling_matches():
    result = decide("buttonhole stiching")  # typo: missing 't'
    check("typo 'stiching' -> tier != abstain", result["tier"] != "abstain", result)
    check("typo 'stiching' -> matched_task_id=buttonhole-stitching",
          result["matched_task_id"] == "buttonhole-stitching", result)


def test_gibberish_abstains():
    result = decide("zxqvbnm qplsdkfj wobblegorp")
    check("gibberish -> tier=abstain", result["tier"] == "abstain", result)
    check("gibberish -> matched_task_id=null", result["matched_task_id"] is None, result)
    check("gibberish -> match_score below threshold", result["match_score"] < 0.8, result)


def test_ask_tier_from_fixture_variance():
    """Amended rule (orchestrator, in-session): ask iff deconfounded, both
    clusters non-empty, and silhouette_k2>=0.1 -- perm_p<=0.05 never fires
    on the real corpus (best real case is ~0.11), so it is no longer a
    gate. Framed majority-vs-outlier: cluster_maj_n/cluster_min_n, not
    two-comparable-methods."""
    result = decide("belly band assembly", variance_path=FIXTURE_VARIANCE)
    check("belly-band-assembly (clean, silhouette>=0.1) -> tier=ask", result["tier"] == "ask", result)
    check("ask evidence has silhouette_k2", result["evidence"].get("silhouette_k2") == 0.42, result)
    check("ask evidence cluster_maj_n/min_n sum to n_clips",
          (result["evidence"].get("cluster_maj_n") or 0) + (result["evidence"].get("cluster_min_n") or 0) == 8,
          result)
    check("ask evidence cluster_maj_n is the larger split",
          result["evidence"].get("cluster_maj_n") >= result["evidence"].get("cluster_min_n"), result)
    slots = result["utterance_slots"]
    for key in ("task", "n_clips", "cluster_maj_n", "cluster_min_n", "silhouette", "perm_p"):
        check(f"ask utterance_slots has {key!r}", key in slots, slots)
    check("ask utterance_slots silhouette rounded to 2dp", slots.get("silhouette") == 0.42, slots)
    check("ask utterance_slots perm_p rounded to 2dp", slots.get("perm_p") == 0.11, slots)


def test_confounded_task_falls_back_to_act():
    result = decide("axle shaft cutting", variance_path=FIXTURE_VARIANCE)
    check("axle-shaft-cutting (confounded, despite silhouette>=0.1) -> tier=act",
          result["tier"] == "act", result)


def test_degenerate_single_cluster_falls_back_to_act():
    """Deconfounded + high silhouette is not enough if the k=2 labeling is
    degenerate (every clip in one cluster) -- there is no minority to ask
    about, so 'both clusters non-empty' must independently gate ask."""
    result = decide("garment quality checking", variance_path=FIXTURE_VARIANCE)
    check("garment-quality-checking (single-cluster labels) -> tier=act despite silhouette>=0.1",
          result["tier"] == "act", result)


def test_excluded_task_falls_back_to_act():
    result = decide("processing fabric cut", variance_path=FIXTURE_VARIANCE)
    check("processing-fabric-cut (excluded in variance.json) -> tier=act", result["tier"] == "act", result)
    check("processing-fabric-cut -> variance_available=False", result["evidence"].get("variance_available") is False, result)


def test_real_variance_ask_cases():
    """Verify against the real variance/results/variance.json (now present,
    default path -- no variance_path override): these three deconfounded
    tasks have the corpus's highest silhouettes (0.152/0.124/0.122), all as
    8-1 or 7-1 majority/outlier splits."""
    for query, expected_task_id in [
        ("garment inside out", "garment-inside-out"),
        ("fabric layering", "fabric-layering"),
        ("garment carton packing", "garment-carton-packing"),
    ]:
        result = decide(query)
        check(f"{query!r} (real variance.json) -> tier=ask", result["tier"] == "ask", result)
        check(f"{query!r} -> matched_task_id={expected_task_id}",
              result["matched_task_id"] == expected_task_id, result)
        slots = result["utterance_slots"]
        check(f"{query!r} -> utterance_slots has cluster_maj_n and cluster_min_n",
              "cluster_maj_n" in slots and "cluster_min_n" in slots, slots)
        check(f"{query!r} -> minority cluster is non-empty",
              slots.get("cluster_min_n", 0) > 0, slots)


def test_real_variance_act_cases():
    """garment-iron-press (silhouette 0.0625) and drilling (silhouette
    0.026) both fall below the 0.1 gate on the real corpus -> tier=act."""
    for query, expected_task_id in [
        ("garment iron press", "garment-iron-press"),
        ("drilling", "drilling"),
    ]:
        result = decide(query)
        check(f"{query!r} (real variance.json) -> tier=act", result["tier"] == "act", result)
        check(f"{query!r} -> matched_task_id={expected_task_id}",
              result["matched_task_id"] == expected_task_id, result)


def test_missing_variance_defaults_to_act_not_abstain():
    result = decide("bottle cleaning", variance_path=Path("/nonexistent/variance.json"))
    check("missing variance.json -> matched task still tier=act (not abstain)",
          result["tier"] == "act" and result["matched_task_id"] == "bottle-cleaning", result)


def test_include_live_adds_evidence():
    result = decide("bottle flip", include_live=True, live_path=FIXTURE_LIVE)
    check("bottle flip + fixture live index -> n_live_demos=3",
          result["evidence"].get("n_live_demos") == 3, result)
    check("bottle flip + fixture live index -> live_stat_line present",
          bool(result["utterance_slots"].get("live_stat_line")), result)
    check("bottle flip + fixture live index -> still tier=abstain (data changed, not capability)",
          result["tier"] == "abstain", result)


def test_no_hardcoded_task_names_in_source():
    source = (HERE / "decide.py").read_text() + (HERE / "coverage.py").read_text()
    real_task_ids = {t["canonical_task_id"] for t in load_tasks()}
    hits = [tid for tid in real_task_ids if tid in source]
    check("no canonical task_id string literals in decide.py/coverage.py", not hits, hits)


def test_cli_json_contract():
    for query, expect_abstain in [("bottle flip", True), ("bottle-cleaning", False)]:
        proc = subprocess.run(
            [sys.executable, str(HERE / "decide.py"), query],
            capture_output=True, text=True, cwd=HERE.parent, check=False,
        )
        check(f"CLI exits 0 for {query!r}", proc.returncode == 0, proc.stderr)
        try:
            payload = json.loads(proc.stdout.strip())
            valid_json = True
        except json.JSONDecodeError:
            payload, valid_json = None, False
        check(f"CLI stdout is valid JSON for {query!r}", valid_json, proc.stdout)
        if valid_json:
            for key in ("query", "tier", "matched_task_id", "match_score", "coverage_detail", "evidence", "utterance_slots"):
                check(f"CLI JSON has key {key!r} for {query!r}", key in payload)
            check(f"CLI tier matches expectation for {query!r}",
                  (payload["tier"] == "abstain") == expect_abstain, payload)


def main():
    load_clips()  # sanity: dataset is reachable before running any case
    test_bottle_flip_abstains()
    test_flip_never_matches_emergently()
    test_three_bottle_tasks_match_themselves()
    test_near_miss_spelling_matches()
    test_gibberish_abstains()
    test_ask_tier_from_fixture_variance()
    test_confounded_task_falls_back_to_act()
    test_degenerate_single_cluster_falls_back_to_act()
    test_excluded_task_falls_back_to_act()
    test_missing_variance_defaults_to_act_not_abstain()
    test_real_variance_ask_cases()
    test_real_variance_act_cases()
    test_include_live_adds_evidence()
    test_no_hardcoded_task_names_in_source()
    test_cli_json_contract()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
