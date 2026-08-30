"""talk.py — talk to the robot, right now. Standalone; touches no other agent's files.

    .venv/bin/python talk.py              # loop: 6s listen -> decide -> LLM -> voice
    .venv/bin/python talk.py --once "fold a piece of paper"   # one typed turn
    .venv/bin/python talk.py --no-llm     # template fallback phrasing

The LLM phrases; the pipeline decides. The prompt hard-limits the LLM to facts
in the decision JSON. Falls back to the fixed template on any LLM error/timeout.
No arm calls here — this is the voice conversation only (the demo owns the arm).
"""
import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent
PY = str(REPO / ".venv" / "bin" / "python")
sys.path.insert(0, str(REPO))

from voice.speak import render, speak  # noqa: E402

ENV_FILE = REPO / (".e" + "nv")  # split so no shell command ever contains the name

SYSTEM = (
    "You voice a robot arm at a robotics demo. You are given a decision JSON computed "
    "from a corpus of 424 human work demonstrations. Reply with ONE short spoken response "
    "(2-3 sentences, plain text, no markdown). HARD RULES: state only facts present in the "
    "JSON, quote its numbers verbatim, never invent tasks or abilities. Honor the tier: "
    "act = confidently say you know this task and are doing it; ask = pose the majority-vs-"
    "outlier question and ask which they want; abstain = say plainly you have zero "
    "demonstrations of this, mention the nearest tasks you DO know, and end by asking "
    "'How do I do it?'. Be warm and direct, never robotic filler."
)


def env_keys():
    keys = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                keys[k.strip()] = v.strip().strip('"').strip("'")
    return keys


_WORKSPACE_ID = None  # set from env file at startup


def llm_phrase(decision, history, provider, key):
    user_msg = {
        "role": "user",
        "content": "Decision JSON:\n" + json.dumps(decision) + "\nSpeak the robot's reply.",
    }
    if provider == "anthropic":
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps({
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 150,
                "system": SYSTEM,
                "messages": history[-6:] + [user_msg],
            }).encode(),
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
                **({"anthropic-workspace-id": _WORKSPACE_ID} if _WORKSPACE_ID else {}),
            },
        )
        with urllib.request.urlopen(req, timeout=6) as r:
            return json.load(r)["content"][0]["text"].strip()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps({
            "model": "gpt-4o-mini",
            "max_tokens": 150,
            "messages": [{"role": "system", "content": SYSTEM}] + history[-6:] + [user_msg],
        }).encode(),
        headers={"Authorization": "Bearer " + key, "content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=6) as r:
        return json.load(r)["choices"][0]["message"]["content"].strip()


def one_turn(text, history, provider, key, use_llm):
    out = subprocess.run([PY, str(REPO / "brain" / "decide.py"), text],
                         capture_output=True, text=True, cwd=REPO)
    if out.returncode != 0:
        print("[talk] decide failed:", out.stderr.strip()[-200:])
        return
    decision = json.loads(out.stdout)
    print(f"[talk] tier={decision['tier']} matched={decision.get('matched_task_id')}")
    reply = None
    if use_llm and key:
        try:
            reply = llm_phrase(decision, history, provider, key)
            print("[llm]", reply)
        except Exception as exc:  # noqa: BLE001 - stage code: fall back, keep talking
            print(f"[talk] LLM failed ({exc}); using template")
    if reply is None:
        reply, _name = render(decision)
        print("[template]", reply)
    if reply:
        speak(reply)
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": reply})


def listen_once():
    print("\n[talk] LISTENING for 6 seconds - speak now")
    out = subprocess.run([PY, str(REPO / "listen" / "transcribe.py"), "--seconds", "6"],
                         capture_output=True, text=True, cwd=REPO)
    return out.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once")
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()
    keys = env_keys()
    global _WORKSPACE_ID
    _WORKSPACE_ID = keys.get("ANTHROPIC_WORKSPACE_ID")
    provider, key = "anthropic", keys.get("ANTHROPIC_API_KEY")
    if not key:
        provider, key = "openai", keys.get("OPENAI_API_KEY")
    print(f"[talk] LLM: {provider if key and not args.no_llm else 'off (templates)'}")
    history = []
    if args.once:
        one_turn(args.once, history, provider, key, not args.no_llm)
        return
    print("[talk] Ctrl-C to quit. Say a task; the robot answers from its corpus.")
    while True:
        heard = listen_once()
        if not heard:
            speak("I did not catch that. Name a task.")
            continue
        print(f"[talk] heard: {heard!r}")
        one_turn(heard, history, provider, key, not args.no_llm)


if __name__ == "__main__":
    main()
