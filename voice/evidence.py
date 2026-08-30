"""voice/evidence.py — fullscreen evidence panel matching the spoken decision.

    python voice/evidence.py <decision.json> [--out PATH.png] [--show]

Renders ONE static panel per tier — the numbers on screen are the same ones
voice/speak.py is speaking, read from the decision JSON (and, for the "ask"
tier, variance/results/variance.json when that file exists and covers the
matched task). No live dashboard, no LLM text.

Note on the "ask" panel: the frozen variance.json schema (Agent A) carries
per-clip cluster `labels`, not a raw pairwise distance matrix, so this
renders a cluster-size bar chart from those labels (falling back to the
decision JSON's own cluster_a_n/cluster_b_n if variance.json is absent or
does not cover the task) rather than a literal distance-matrix heatmap.
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VARIANCE_JSON = REPO_ROOT / "variance" / "results" / "variance.json"
DEFAULT_OUT_DIR = REPO_ROOT / "voice" / "evidence_out"


def log(*args):
    print(*args, file=sys.stderr, flush=True)


def _normalize_coverage(coverage_detail):
    if isinstance(coverage_detail, dict) and "words" in coverage_detail:
        return coverage_detail["words"]
    if isinstance(coverage_detail, list):
        return coverage_detail
    raise ValueError(f"unrecognized coverage_detail shape: {type(coverage_detail).__name__}")


def render_abstain(fig, decision):
    ax_cov, ax_near = fig.subplots(1, 2)

    words = _normalize_coverage(decision.get("coverage_detail", []))
    word_labels = [w.get("word", "?") for w in words]
    covered = [bool(w.get("covered")) for w in words]
    colors = ["#2a9d5c" if c else "#c94545" for c in covered]
    bars = ax_cov.bar(word_labels, [1.0] * len(words), color=colors)
    for bar, c in zip(bars, covered):
        ax_cov.text(bar.get_x() + bar.get_width() / 2, 0.5, "covered" if c else "NOT covered",
                    ha="center", va="center", color="white", fontweight="bold", rotation=90)
    ax_cov.set_ylim(0, 1.15)
    ax_cov.set_yticks([])
    ax_cov.set_title(f'content-word coverage — "{decision.get("query", "")}"')

    nearest = decision.get("evidence", {}).get("nearest_tasks", [])
    if nearest:
        n_labels = [n.get("task_id", "?") for n in reversed(nearest)]
        n_scores = [n.get("score", 0.0) for n in reversed(nearest)]
        ax_near.barh(n_labels, n_scores, color="#3f6fb4")
        ax_near.set_xlim(0, 1)
        ax_near.set_title("nearest known tasks")
    else:
        ax_near.axis("off")
        ax_near.text(0.5, 0.5, "no nearest-task data in decision JSON", ha="center", va="center")

    n_matching = decision.get("evidence", {}).get("n_matching_demos", 0)
    fig.suptitle(f"ABSTAIN — {n_matching} matching demonstrations", fontsize=20, fontweight="bold")


def _variance_entry_for(task_id):
    if not task_id or not VARIANCE_JSON.exists():
        return None
    try:
        data = json.loads(VARIANCE_JSON.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        log(f"[evidence] could not read {VARIANCE_JSON}: {exc}")
        return None
    return data.get(task_id)


def render_ask(fig, decision):
    ax = fig.subplots(1, 1)
    evidence = decision.get("evidence", {})
    slots = decision.get("utterance_slots", {})
    task_id = decision.get("matched_task_id") or slots.get("task", "?")

    # Amended framing (orchestrator, in-session): A's real data never produces a
    # balanced two-method split on a deconfounded task -- every clean split is
    # majority-vs-outlier (e.g. 8-1). The panel shows majority vs outlier, not
    # "cluster a / cluster b", and always surfaces p even when p > 0.05 -- the
    # non-significance at this n is the honest point, not a flaw to hide.
    entry = _variance_entry_for(decision.get("matched_task_id"))
    labels = entry.get("labels") if entry else None

    maj_n = evidence.get("cluster_maj_n", slots.get("cluster_maj_n"))
    min_n = evidence.get("cluster_min_n", slots.get("cluster_min_n"))
    source = "decision JSON"

    if labels:
        counts = {}
        for lab in labels:
            counts[lab] = counts.get(lab, 0) + 1
        sizes = sorted(counts.values(), reverse=True)
        maj_n, min_n = sizes[0], sum(sizes[1:])
        source = "variance.json"

    if maj_n is not None and min_n is not None:
        ax.bar(["majority", "outlier"], [maj_n, min_n], color=["#3f6fb4", "#c94545"])
        ax.set_title(f"{task_id} — {maj_n} did it one way, {min_n} did something different (from {source})")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "no cluster data available", ha="center", va="center")

    silhouette = evidence.get("silhouette", evidence.get("silhouette_k2", slots.get("silhouette")))
    perm_p = evidence.get("perm_p", slots.get("perm_p"))
    fig.suptitle(f"ASK — majority vs. outlier — silhouette {silhouette}, p = {perm_p} (n too small to rule out chance)",
                 fontsize=18, fontweight="bold")


def render_act(fig, decision):
    ax = fig.subplots(1, 1)
    evidence = decision.get("evidence", {})
    slots = decision.get("utterance_slots", {})
    silhouette = evidence.get("silhouette", slots.get("silhouette"))
    perm_p = evidence.get("perm_p", slots.get("perm_p"))
    n_clips = evidence.get("n_clips", slots.get("n_clips"))
    ax.axis("off")
    ax.text(0.5, 0.62, f"silhouette = {silhouette}", ha="center", va="center", fontsize=30)
    ax.text(0.5, 0.42, f"p = {perm_p}   ·   n_clips = {n_clips}", ha="center", va="center", fontsize=20)
    ax.text(0.5, 0.22, "low variance — executing silently", ha="center", va="center",
            fontsize=14, style="italic", color="#555555")
    task = decision.get("matched_task_id") or decision.get("query", "?")
    fig.suptitle(f"ACT — {task}", fontsize=20, fontweight="bold")


RENDERERS = {"abstain": render_abstain, "ask": render_ask, "act": render_act}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("decision_json")
    ap.add_argument("--out", help="save panel to this path instead of the tier-named default")
    ap.add_argument("--show", action="store_true",
                     help="show fullscreen on the demo screen instead of only saving to disk")
    args = ap.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    decision = json.loads(Path(args.decision_json).read_text())
    tier = decision.get("tier")
    renderer = RENDERERS.get(tier)
    if renderer is None:
        log(f"[evidence] unknown tier {tier!r} — expected one of {sorted(RENDERERS)}")
        return 1

    fig = plt.figure(figsize=(16, 9))
    renderer(fig, decision)

    out_path = Path(args.out) if args.out else DEFAULT_OUT_DIR / f"{tier}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    log(f"[evidence] saved {out_path}")

    if args.show:
        try:
            plt.get_current_fig_manager().full_screen_toggle()
        except Exception:
            pass
        plt.show()
    else:
        plt.close(fig)

    return 0


if __name__ == "__main__":
    sys.exit(main())
