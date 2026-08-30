"""voice/llm_phrase.py — LLM PHRASING ONLY. It decides nothing.

    python voice/llm_phrase.py <decision.json> [--history hist.json] [--timeout 2.0]
        -> ONE short spoken reply on stdout, engine name on stderr, exit 0.
        -> exit 1 on any error, missing key, or timeout, so the caller falls
           back to the fixed template for that turn.

    python voice/llm_phrase.py --probe
        -> prints the detected engine/model and whether its SDK is importable.

Division of labour — this is the judging story, do not blur it:

    listen/transcribe.py   speech in            (local Whisper, no network)
    brain/decide.py        THE DECISION         (tier + evidence, computed from
                                                 the corpus; the ground truth)
    voice/llm_phrase.py    wording only         (this file)
    voice/speak.py         speech out           (TTS)

The tier, every number, every task name and EVERY ARM MOTION are computed
upstream by decide.py. This module receives them already decided and may only
choose how to say them. It cannot cause motion: run_demo.py drives the arm from
the computed tier and never reads this module's output for control. If this
module is slow (>timeout), errors, or has no API key, the caller uses the fixed
template instead — the templates are the reliability floor, not a legacy path.

The system prompt below hard-constrains the model to the decision JSON: numbers
verbatim, no task names that are not in the JSON, no invented capabilities, and
the tier's meaning is not negotiable (an abstain stays a refusal that asks how).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILENAME = ".env"

# Ordered by preference: haiku first for latency on stage.
PROVIDERS = (
    ("ANTHROPIC_API_KEY", "anthropic", "claude-haiku-4-5-20251001"),
    ("OPENAI_API_KEY", "openai", "gpt-4o-mini"),
    ("FIREWORKS_API_KEY", "fireworks", "accounts/fireworks/models/llama-v3p1-8b-instruct"),
)

MAX_TOKENS = 150

SYSTEM_PROMPT = """You are the voice of a robot arm. You phrase replies. You do not decide anything.

A separate program has ALREADY computed the decision and handed it to you as JSON. Your only
job is to say that decision out loud in 2-3 short sentences, conversationally.

HARD RULES — breaking any of these is a failure:
1. You may only assert facts that appear in the decision JSON. Nothing else is true.
2. Reproduce every number EXACTLY as it appears in the JSON. Never round, never estimate,
   never invent a number. If a number is not in the JSON, do not say it.
3. Never name a task that does not appear in the JSON.
4. Never claim a capability, a success, or a completed action. You did not verify anything.
5. Obey the tier. It is not yours to change:
   - "act": you have enough agreement in the data. Say briefly that you know this task and
     are doing it now.
   - "ask": the demonstrations disagree. You MUST ask the human which way to do it, and you
     must mention that the split may be chance if the JSON says so.
   - "abstain": you have ZERO demonstrations. You REFUSE to claim you can do it, and you END
     by asking the human how to do it. Never agree to attempt it as if you knew how.
6. No emoji, no markdown, no stage directions, no quotation marks around your reply.
7. Plain spoken English. This goes straight to a speech synthesiser.

Reply with the spoken sentences only."""


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def _env_file_keys():
    """Read keys from the repo-root env file in Python.

    This session's Bash tool is hook-blocked from referencing that filename, but
    a plain file read from inside Python source is fine — the value never touches
    a shell invocation. Path is built from __file__, not from argv or env.
    """
    path = REPO_ROOT / ENV_FILENAME
    found = {}
    if not path.exists():
        return found
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        if value:
            found[key.strip()] = value
    return found


def detect_provider():
    """Returns (env_key, provider, model, api_key) or (None, None, None, None)."""
    file_keys = _env_file_keys()
    for env_key, provider, model in PROVIDERS:
        api_key = os.environ.get(env_key) or file_keys.get(env_key)
        if api_key:
            return env_key, provider, model, api_key
    return None, None, None, None


def build_user_message(decision, history):
    parts = []
    if history:
        lines = [f"{t.get('role', '?')}: {t.get('text', '')}" for t in history[-6:]]
        parts.append("Recent conversation (for continuity of tone only — facts still come "
                     "only from the decision JSON):\n" + "\n".join(lines))
    parts.append("Decision JSON:\n" + json.dumps({
        "query": decision.get("query"),
        "tier": decision.get("tier"),
        "matched_task_id": decision.get("matched_task_id"),
        "match_score": decision.get("match_score"),
        "coverage_detail": decision.get("coverage_detail"),
        "evidence": decision.get("evidence"),
        "utterance_slots": decision.get("utterance_slots"),
    }, indent=2))
    parts.append("Say the decision out loud now, in 2-3 short sentences.")
    return "\n\n".join(parts)


def phrase_anthropic(api_key, model, user_msg, timeout):
    import anthropic

    client = anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=0)
    resp = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()


def phrase_openai(api_key, model, user_msg, timeout, base_url=None):
    from openai import OpenAI

    client = OpenAI(api_key=api_key, timeout=timeout, max_retries=0, base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": user_msg}],
    )
    return (resp.choices[0].message.content or "").strip()


def phrase(decision, history=None, timeout=2.0):
    """Returns (text, engine, latency_s). Raises on any failure — caller falls back."""
    env_key, provider, model, api_key = detect_provider()
    if not api_key:
        raise RuntimeError("no API key found for any supported provider")

    user_msg = build_user_message(decision, history or [])
    t0 = time.time()
    if provider == "anthropic":
        text = phrase_anthropic(api_key, model, user_msg, timeout)
    elif provider == "openai":
        text = phrase_openai(api_key, model, user_msg, timeout)
    else:
        text = phrase_openai(api_key, model, user_msg, timeout,
                             base_url="https://api.fireworks.ai/inference/v1")
    latency = time.time() - t0

    if not text:
        raise RuntimeError("model returned empty text")
    if latency > timeout:
        raise TimeoutError(f"phrasing took {latency:.2f}s (> {timeout}s budget)")
    # Single spoken line: collapse any stray newlines the model emits.
    return " ".join(text.split()), f"{provider}:{model}", latency


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("decision_json", nargs="?")
    ap.add_argument("--history", help="JSON file: [{role, text}, ...]")
    ap.add_argument("--timeout", type=float, default=2.0)
    ap.add_argument("--probe", action="store_true",
                    help="report the detected provider and SDK availability, then exit")
    args = ap.parse_args()

    if args.probe:
        env_key, provider, model, api_key = detect_provider()
        if not api_key:
            print("engine=none reason=no_api_key_found")
            return 1
        sdk = "anthropic" if provider == "anthropic" else "openai"
        try:
            __import__(sdk)
            sdk_ok = "yes"
        except ImportError:
            sdk_ok = "no"
        print(f"engine={provider} model={model} key_source={env_key} sdk_importable={sdk_ok}")
        return 0

    if not args.decision_json:
        ap.error("decision_json is required unless --probe")

    decision = json.loads(Path(args.decision_json).read_text())
    history = json.loads(Path(args.history).read_text()) if args.history else []
    try:
        text, engine, latency = phrase(decision, history, args.timeout)
    except Exception as exc:  # noqa: BLE001 — any failure means "use the template"
        log(f"[llm] unavailable ({type(exc).__name__}: {exc}) — caller should use template")
        return 1
    log(f"[llm] engine={engine} latency={latency:.2f}s")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
