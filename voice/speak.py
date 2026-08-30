"""voice/speak.py — speech out. Frozen CLI (see PLAN_v2):

    python voice/speak.py <decision.json>

Renders one of three fixed templates (abstain / abstain-after-feedback / ask
/ act-silent) by interpolating ONLY values from the decision JSON's
`utterance_slots`. No LLM in the demo path. Always prints the exact text
spoken to stdout (empty string for the silent act tier).

TTS: ElevenLabs `POST /v1/text-to-speech/{voice_id}/stream` when
ELEVENLABS_API_KEY is available (env var first, else parsed from a
repo-root .env file — see get_api_key below), playing the returned mp3 with
`afplay`. Falls back to macOS `say` (Samantha, matching pipeline/narrate.py's
voice choice) when no key is found at runtime.

Voice (2026-08-30, user feedback "sounds very AI"): switched off narrate.py's
default Rachel (21m00Tcm4TlvDq8ikWAM, the generic ElevenLabs demo voice) to
Eric - Smooth, Trustworthy (cjVigY5qzO86Huf0OWal), a premade conversational
voice ElevenLabs' own catalog describes as "perfect for agentic use cases" --
picked by querying GET /v1/voices live with the real key and reading
labels/descriptions, not guessed. Paired with tuned voice_settings
(stability 0.5, similarity_boost 0.75, style 0.15, speaker_boost on) for a
warmer, less flat delivery, still on eleven_turbo_v2_5 for latency.

utterance_slots keys this module expects per tier/template (not part of the
frozen outer schema, but the contract Agent D and Agent B/F need to agree on
in integration — see voice/fixtures/*.json for worked examples). Templates
marked (--template) are never auto-inferred from `tier`; the caller (F's
state machine) must pass --template explicitly:
    abstain:                       query, hours, n_tasks, near1, near2
    abstain (post-feedback):       n_live_demos, n_live_demonstrators, live_stat_line
    ask:                           n_clips, task, cluster_maj_n, cluster_min_n, silhouette, perm_p
    act:                           (none — act-silent speaks nothing)
    ask_hold (--template):         n_clips
    abstain_restate (--template):  query
    abstain_howto (--template):    query, hours, n_tasks, near1, near2 (same evidence as abstain)
    attempt_result (--template):   n_instruction_words, n_matching_demos, query

Orchestrator amendment (2026-08-30, user-approved): Agent A's real data shows
no deconfounded task splits into two balanced methods -- every clean split
is majority-vs-outlier (e.g. 8-1, p~=0.11, not significant). BEAT 2 is now an
honest majority-vs-outlier ask instead of a "two balanced methods" ask; the
p-value disclosure is deliberate and must not be dropped even though it is
often > 0.05 (that is the honest point -- with this few clips we cannot rule
out chance). This replaced the earlier k/cluster_a_n/cluster_b_n ask slots.

Orchestrator amendment (2026-08-30, second round): two more explicit-template
variants for the human-in-the-loop instruction beat, coordinated with Agent F
(F rewires BEAT 3 to consume them; F computes n_instruction_words from the
live transcript -- this module never fabricates it):
  - abstain_howto: same computed abstain evidence, but asks "How do I do it?"
    instead of offering to watch a demo.
  - attempt_result: spoken after the robot physically attempts the task
    following spoken instruction alone. n_instruction_words and
    n_matching_demos must both come from utterance_slots (computed
    upstream), never invented here.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILENAME = ".env"
# Eric - Smooth, Trustworthy: premade conversational voice, chosen by querying
# GET /v1/voices live (see coordination/status/D.json) after user feedback that
# narrate.py's default Rachel (21m00Tcm4TlvDq8ikWAM) "sounds very AI".
VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "cjVigY5qzO86Huf0OWal")
VOICE_NAME = "Eric - Smooth, Trustworthy"  # for logging only, not sent to the API
MODEL_ID = "eleven_turbo_v2_5"  # best latency/naturalness tradeoff available to this account
VOICE_SETTINGS = {
    "stability": 0.5,
    "similarity_boost": 0.75,
    "style": 0.15,
    "use_speaker_boost": True,
}
SAY_VOICE = "Samantha"  # matches pipeline/narrate.py's macOS say fallback
LATENCY_WARN_S = 1.5

TEMPLATES = {
    "abstain": (
        "I have watched {hours} hours of human work across {n_tasks} tasks. "
        "None of them is {query}. The closest things I know are {near1} and {near2}. "
        "I have zero demonstrations of this. If you show me, I will watch."
    ),
    "abstain_after_feedback": (
        "I watched your {n_live_demos} attempts. I now have data from {n_live_demonstrators} "
        "demonstrator. I am still not attempting it: my control is trajectory replay, and this "
        "task needs in-flight release timing I cannot verify. What I can tell you is what I "
        "measured: {live_stat_line}."
    ),
    "ask": (
        "I have {n_clips} demonstrations of {task}. {cluster_maj_n} do it one way. "
        "{cluster_min_n} does something different - silhouette {silhouette}, though "
        "with {n_clips} clips I cannot rule out chance: p equals {perm_p}. Should I "
        "follow the majority, or does the outlier matter?"
    ),
    "act": "",  # act-silent: speaks nothing, per the frozen contract
    # Human-in-the-loop resolution templates (added by Agent F for the amended
    # BEAT 2/3 flow). Both are slot-compatible with the existing
    # utterance_slots: 'ask_hold' reuses the ask tier's n_clips, and
    # 'abstain_restate' reuses the abstain tier's query. No new slot is
    # required from brain/decide.py. Selected via --template, never inferred.
    "ask_hold": "Holding. {n_clips} demonstrations remain unexecuted.",
    # Converse mode only. The act tier is deliberately SILENT in the scripted
    # beats (BEAT 1's whole point), but in free conversation silence reads as a
    # crash, so converse speaks a confirmation before it moves. Reuses the act
    # tier's existing task/n_clips slots.
    "act_confirm": "I know {task}. {n_clips} demonstrations. Doing it.",
    # No slots: spoken when the capture was empty or unintelligible, so that
    # every human utterance gets a spoken response and never silence.
    "not_understood": "I did not catch that. Name a task.",
    # Can beat: template fallback for the grip question when the LLM is
    # unavailable. No slots — it asks, it does not claim anything.
    "can_grip_question": "Do I hold it at the top or the bottom? Tell me how.",
    # Golden path opener. Corpus size is computed, not written in.
    "greeting": ("I have watched {hours} hours of human work across {n_tasks} tasks. "
                 "Ask me to do something."),
    "abstain_restate": (
        "I have zero demonstrations of {query}. I will not attempt it."
    ),
    # Added for the human-in-the-loop instruction beat (orchestrator amendment,
    # 2026-08-30, coordinated with Agent F). Explicit --template selection only.
    "abstain_howto": (
        "I have watched {hours} hours of human work across {n_tasks} tasks. "
        "None of them is {query}. The closest things I know are {near1} and {near2}. "
        "I have zero demonstrations of this. How do I do it?"
    ),
    "attempt_result": (
        "I heard your {n_instruction_words} words of instruction and tried. "
        "I still have {n_matching_demos} demonstrations of {query} - one attempt on "
        "instruction is not knowledge. Watch me learn faster if you show me."
    ),
}


def log(*args):
    print(*args, file=sys.stderr, flush=True)


def pick_template_name(decision):
    tier = decision.get("tier")
    if tier == "abstain":
        slots = decision.get("utterance_slots", {})
        if slots.get("n_live_demos"):
            return "abstain_after_feedback"
        return "abstain"
    if tier in ("ask", "act"):
        return tier
    raise ValueError(f"unknown tier in decision JSON: {tier!r}")


def render(decision, template_name=None):
    name = template_name or pick_template_name(decision)
    if name not in TEMPLATES:
        raise ValueError(f"unknown template {name!r}; expected one of {sorted(TEMPLATES)}")
    template = TEMPLATES[name]
    if not template:
        return "", name
    slots = decision.get("utterance_slots", {})
    try:
        text = template.format(**slots)
    except KeyError as exc:
        raise KeyError(
            f"decision JSON utterance_slots is missing key {exc} required by the "
            f"'{name}' template — this is a schema mismatch with brain/decide.py"
        ) from exc
    return text, name


def _env_file_key():
    """Resolve ELEVENLABS_API_KEY from a plain read of the repo-root .env file.

    Orchestrator note: this session's Bash tool is hook-blocked from
    referencing .env in shell commands, but a plain file read from inside
    Python source (this file) is fine — the key never touches a Bash
    invocation. Path is built relative to this file, not passed on argv/env.
    """
    env_path = REPO_ROOT / ENV_FILENAME
    if not env_path.exists():
        return None
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() == "ELEVENLABS_API_KEY":
            return value.strip().strip('"').strip("'") or None
    return None


def get_api_key():
    return os.environ.get("ELEVENLABS_API_KEY") or _env_file_key()


def build_elevenlabs_request(text, key):
    """Construct the ElevenLabs streaming TTS request without sending it.

    Used to verify this code path when no API key is available at runtime
    (see voice/smoke_elevenlabs_dry_run in the status notes) — asserts the
    URL, headers, and body are well-formed per the documented API.
    """
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream"
    body = json.dumps({
        "text": text,
        "model_id": MODEL_ID,
        "voice_settings": VOICE_SETTINGS,
    }).encode()
    headers = {
        "xi-api-key": key or "<no-key>",
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    return urllib.request.Request(url, data=body, headers=headers, method="POST")


def synthesize_to_file(text, out_path):
    """Render text to an mp3 on disk without playing it. Used to pre-synthesize
    the scripted backup's audio while the network is good, so that mode needs no
    network at all on stage."""
    key = get_api_key()
    if not key:
        raise RuntimeError("ELEVENLABS_API_KEY not available; cannot pre-synthesize")
    req = build_elevenlabs_request(text, key)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=30) as resp:
        audio_bytes = resp.read()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(audio_bytes)
    return time.time() - t0


def speak_elevenlabs(text, key):
    req = build_elevenlabs_request(text, key)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=30) as resp:
        audio_bytes = resp.read()
    latency = time.time() - t0
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(audio_bytes)
        mp3_path = tmp.name
    try:
        subprocess.run(["afplay", mp3_path], check=True)
    finally:
        Path(mp3_path).unlink(missing_ok=True)
    return latency


def speak_say(text):
    t0 = time.time()
    subprocess.run(["say", "-v", SAY_VOICE, text], check=True)
    return time.time() - t0


def speak(text):
    """Returns (engine, latency_seconds)."""
    key = get_api_key()
    if key:
        try:
            latency = speak_elevenlabs(text, key)
            if latency > LATENCY_WARN_S:
                log(f"[speak] WARNING: ElevenLabs latency {latency:.2f}s exceeds "
                    f"{LATENCY_WARN_S}s budget — pre-synthesize fixed template prefixes "
                    f"during F's setup and only synthesize the number-bearing clause live")
            return "elevenlabs", latency
        except (urllib.error.URLError, subprocess.CalledProcessError, TimeoutError) as exc:
            log(f"[speak] ElevenLabs failed ({exc}); falling back to macOS say")
    latency = speak_say(text)
    return "say", latency


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("decision_json", nargs="?",
                     help="path to a decision JSON from brain/decide.py")
    ap.add_argument("--no-audio", action="store_true",
                     help="render + print text only, skip TTS playback (used by smoke tests)")
    ap.add_argument("--say", help="speak this exact string instead of rendering a template "
                                   "(used for LLM-phrased lines; TTS + printing are identical)")
    ap.add_argument("--template", choices=sorted(TEMPLATES),
                     help="force a template instead of inferring it from the tier "
                          "(used by the human-in-the-loop resolution beats)")
    args = ap.parse_args()

    if args.say is not None:
        text, template_name = args.say.strip(), "llm-phrased"
    else:
        if not args.decision_json:
            ap.error("decision_json is required unless --say is given")
        decision = json.loads(Path(args.decision_json).read_text())
        text, template_name = render(decision, template_name=args.template)

    # Frozen contract: always print the exact text spoken (empty for act-silent).
    print(text)

    if args.no_audio:
        return 0

    if not text:
        log(f"[speak] template={template_name} -> act-silent, no speech synthesized")
        return 0

    engine, latency = speak(text)
    voice_note = f" voice={VOICE_NAME}" if engine == "elevenlabs" else ""
    log(f"[speak] template={template_name} engine={engine} latency={latency:.3f}s{voice_note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
