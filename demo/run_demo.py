"""demo/run_demo.py — the 3-minute demo state machine. Keyboard-stepped, never auto-advances.

===============================================================================
DEMO_RUNBOOK  (read this before you go on stage)
===============================================================================

LAUNCH — from the USER'S OWN Terminal.app, not from a Claude/agent session.
Terminal.app already holds macOS Camera permission; agent-spawned processes do
not (FACTS.md WEBCAM_STATUS=blocked), and BEAT 4 needs the webcam.

    cd /Users/tarunyadgirkar/TarunsCode/hackathons/Robotics-Hackathon-project
    WC_VIDEOS=/Users/tarunyadgirkar/TarunsCode/wc-hack ARM_SIM_RENDER=0 \
      .venv/bin/python demo/run_demo.py

(run_demo sets both of those itself if they are unset — the explicit form above
is just so you can see what is going on. ARM_SIM_RENDER=0 skips the ~2s/motion
GIF render; the stills in arm/sim_out/ are already rendered for the screen.)

Useful flags:
    --text-mode      start in typed-input mode (loud room; you can also switch live)
    --no-audio       render + print the spoken text without playing it
    --beat N         start at beat N (1-4), for a partial re-run
    --sim-arm        print what the arm would do instead of commanding it
    --rehearse       drive all 4 beats through --text end-to-end, non-interactively

KEYS (single keypress, no Enter):
    SPACE   advance to the next step. A beat never starts without this.
    n       same as SPACE (in case the space bar is awkward on a podium)
    t       toggle input mode SPOKEN <-> TYPED, mid-demo, without breaking flow.
            This is the stage fallback for room noise. Same code path either way
            (listen/transcribe.py --text), so nothing else changes. It also
            covers the spoken REPLIES in beats 2 and 3, not just the queries.
    r       re-run the current step (e.g. STT misheard you)
    h       recover-home after a freeze (asks the arm for a second confirmation)
    q       quit. Arm is left where it is; it is not homed behind your back.

ARM MOTIONS ARE NO LONGER INDIVIDUALLY GATED (user override). A motion runs when
its beat's logic decides to run it — which now includes running in response to a
SPOKEN REPLY in beat 2. SPACE still gates the start of every step, and the
terminal prints a red ">>> ARM MOVING NOW <<<" banner immediately before any
motion. Ctrl-C during a motion FREEZES AND HOLDS — it does not home. Press 'h'
afterwards to home at 25% speed. Keep the workspace clear for the whole demo,
not just around the keypresses.

HUMAN-IN-THE-LOOP RESOLUTION (beats 2 and 3). The point is ambiguity that YOU
resolve out loud, not the robot merely declining:
    beat 2  it asks majority-vs-outlier, then WAITS for your reply.
            "go on" / "yes" / "majority" / "continue"  -> it executes the motion
            "stop" / "no" / "wait" / "hold"            -> it holds and says
                                                          "Holding. 9 demonstrations
                                                           remain unexecuted."
    beat 3  after the refusal it listens once more. Tell it to "go on" or "try
            anyway" and it restates the boundary and STILL does not move:
            "I have zero demonstrations of do a bottle flip. I will not attempt it."
            Say "stop" (or nothing recognizable twice) and it moves to beat 4.
Replies are classified by token overlap against two fixed word sets — no LLM. A
reply matching both sets or neither is treated as unrecognized: the robot says it
did not understand and re-listens, twice, then switches to a typed prompt. When
intent is unclear it asks again rather than guessing and moving.

SAY OUT LOUD, EARLY (the state machine prints this too, at start):
    "The sentences are fixed templates. Every number inside them is computed
     live from the corpus, right now, on this machine."

BACKUP VIDEO — record this yourself before the demo (a human must do this;
it cannot be automated from an agent session):
    1. QuickTime Player > File > New Screen Recording. Record the whole screen
       (not a window) so the evidence panels are captured. Enable the microphone
       input in the record menu so your narration lands on the same track.
    2. Point a phone at the laptop screen / arm area as a second angle.
    3. Start both recordings, then run the launch command above and do a clean
       full pass of all 4 beats, narrating exactly as you will on stage.
    4. Stop, trim the head/tail in QuickTime (Edit > Trim), export 1080p.
    5. Keep it UNDER 3 MINUTES. Upload to YouTube unlisted, put the link in the
       submission form and in BUILD_CARD.md.
    6. Do this BEFORE any further code changes. It is the non-negotiable
       fallback if the live demo dies on stage.

BEATS
    1  act      "garment iron press"   -> tier=act    -> arm replays, says nothing
    2  ask      "garment inside out"   -> tier=ask    -> majority (8) vs outlier (1)
    3  abstain  "do a bottle flip"     -> tier=abstain -> the centerpiece
    4  feedback ingest 3 live takes    -> abstain-after-feedback, cites live numbers

TIMING — measured on the F rehearsal, machine time only (no keypress pauses):
    home 0.6s | beat1 19.4s | beat2 45.3s | beat3 28.1s | beat4 35.2s  => ~2m09s
Speech is the second cost after the arm: ElevenLabs synthesis is sub-second, but
afplay then plays the whole utterance (beat2 22.5s, beat3 20.7s, beat4 27.8s).
Add BEAT 4's real recording (3 takes, ~30-60s) and you are OVER 3 minutes.
Cut order if you are long, per PLAN_v2, cutting from the top:
    1. BEAT 2's task_demo replay (-19.4s) — keep the gesture, voice and screen
    2. BEAT 1 entirely (-19.4s)
BEAT 3 and BEAT 4 are never cut.

BEAT 4 records from the webcam. If the camera is not authorized, ingest exits 1
with an actionable message and the state machine tells you to fall back to
narrating the beat instead of dying. Grant Camera to Terminal.app in
System Settings > Privacy & Security > Camera before the demo.
===============================================================================
"""
import argparse
import json
import os
import subprocess
import sys
import termios
import time
import tty
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PY = str(REPO_ROOT / ".venv" / "bin" / "python")

os.environ.setdefault("WC_VIDEOS", "/Users/tarunyadgirkar/TarunsCode/wc-hack")
os.environ.setdefault("ARM_SIM_RENDER", "0")  # live path: no per-motion GIF render

sys.path.insert(0, str(REPO_ROOT))

DECISION_DIR = REPO_ROOT / "demo" / "decisions"
PANEL_DIR = REPO_ROOT / "demo" / "panels"

BEAT1_QUERY = "garment iron press"
BEAT2_QUERY = "garment inside out"
BEAT3_QUERY = "do a bottle flip"

DISCLOSURE = (
    "The sentences are fixed templates. Every number inside them is computed live "
    "from the corpus, right now, on this machine."
)

BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
YEL = "\033[33m"
RED = "\033[31m"
GRN = "\033[32m"
OFF = "\033[0m"


class Quit(Exception):
    pass


class State:
    def __init__(self, args):
        self.text_mode = args.text_mode or args.rehearse
        self.rehearse = args.rehearse
        self.no_audio = args.no_audio
        self.transcript = []
        self.panel_proc = None
        self.sim_arm = args.sim_arm
        # Rehearsal-only scripted replies, keyed by the capture label. This is how
        # --rehearse walks BOTH the negative and affirmative resolution branches
        # (and the unrecognized-retry path) through the same --text code path the
        # stage fallback uses.
        self.scripted = {
            "beat2-reply": "mmhrgh",          # unrecognized -> robot re-asks
            "beat2-reply1": "stop",           # negative     -> arm holds
            "beat2b-reply": "go on",          # affirmative  -> arm executes
            "beat3-reply": "go on",           # affirmative  -> restates the boundary
        }
        # BEAT 3's exact transcript string is reused verbatim as BEAT 4's ingest
        # --task and as the --include-live query. feedback/live_index.json is keyed
        # on the raw query string, so anything less than verbatim reuse silently
        # produces a live-index miss and BEAT 4 degrades to a plain abstain.
        self.beat3_query = None

    def log(self, line):
        self.transcript.append(line)


# --------------------------------------------------------------------------- io

def say(msg=""):
    print(msg, flush=True)


def banner(text, color=CYAN):
    say()
    say(f"{color}{'=' * 78}{OFF}")
    say(f"{color}{BOLD}{text}{OFF}")
    say(f"{color}{'=' * 78}{OFF}")


def read_key():
    """One keypress, no Enter. cbreak is entered and left around each read so
    subprocesses (transcribe's push-to-talk, ingest's cv2 window) get a normal
    terminal."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def gate(st, prompt, is_motion=False):
    """Block until the operator advances. Returns 'go' or 'again'. Never auto-advances."""
    if st.rehearse:
        say(f"{DIM}[rehearse] auto-advance: {prompt}{OFF}")
        return "go"
    mark = f"{YEL}[ARM]{OFF} " if is_motion else ""
    while True:
        say()
        say(f"{mark}{BOLD}{prompt}{OFF}")
        say(f"{DIM}  SPACE/n=go  r=redo  t=input mode ({'TYPED' if st.text_mode else 'SPOKEN'})"
            f"  h=recover-home  q=quit{OFF}")
        k = read_key()
        if k in (" ", "n", "\r", "\n"):
            return "go"
        if k == "r":
            return "again"
        if k == "t":
            st.text_mode = not st.text_mode
            say(f"{GRN}  input mode -> {'TYPED' if st.text_mode else 'SPOKEN'}{OFF}")
            continue
        if k == "h":
            recover_home()
            continue
        if k == "q":
            raise Quit()


def recover_home():
    from arm import arm_io, safety

    if not safety.is_frozen():
        say(f"{DIM}  arm is not frozen; nothing to recover{OFF}")
        return
    say("  arm is frozen. arm_io.recover_home() will ask you to confirm before it moves.")
    try:
        arm_io.recover_home()
    except Exception as exc:  # noqa: BLE001 - stage code, surface and continue
        say(f"{RED}  recover_home failed: {exc}{OFF}")


# ------------------------------------------------------------------- components

def capture_query(st, expected, label):
    """Query in through listen/transcribe.py. Spoken by default; --text is the
    identical code path, which is why the stage fallback costs nothing."""
    cmd = [PY, str(REPO_ROOT / "listen" / "transcribe.py")]
    if st.text_mode:
        typed = st.scripted.get(label, expected) if st.rehearse else expected
        if not st.rehearse:
            say(f"{DIM}  typed-input mode. Enter overrides; blank uses {expected!r}{OFF}")
            try:
                entered = input("  query> ").strip()
            except EOFError:
                entered = ""
            typed = entered or expected
        cmd += ["--text", typed]
    out = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if out.stderr.strip():
        say(f"{DIM}{out.stderr.strip()}{OFF}")
    transcript = out.stdout.strip()
    if out.returncode != 0 or not transcript:
        say(f"{RED}  STT produced nothing (exit {out.returncode}). Press t to type it.{OFF}")
        return None
    say(f"{GRN}  heard: {transcript!r}{OFF}")
    st.log(f"{label} query heard: {transcript!r}")
    return transcript


def decide(st, query, label, include_live=False):
    cmd = [PY, str(REPO_ROOT / "brain" / "decide.py"), query]
    if include_live:
        cmd.append("--include-live")
    out = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        say(f"{RED}  decide.py exit {out.returncode}: {out.stderr.strip()}{OFF}")
        return None
    decision = json.loads(out.stdout)
    DECISION_DIR.mkdir(parents=True, exist_ok=True)
    path = DECISION_DIR / f"{label}.json"
    path.write_text(json.dumps(decision, indent=2))
    ev = decision["evidence"]
    say(f"{GRN}  tier={decision['tier']} matched={decision['matched_task_id']} "
        f"score={decision['match_score']}{OFF}")
    say(f"{DIM}  evidence: {json.dumps(ev)[:300]}{OFF}")
    st.log(f"{label} decision: tier={decision['tier']} matched={decision['matched_task_id']} "
           f"slots={json.dumps(decision['utterance_slots'])}")
    return path, decision


AFFIRMATIVE = {
    "go", "ahead", "yes", "yeah", "yep", "majority", "continue", "proceed",
    "ok", "okay", "sure", "affirmative", "try", "anyway", "do",
}
NEGATIVE = {
    "stop", "no", "nope", "wait", "hold", "cancel", "abort", "halt",
    "negative", "don't", "dont", "never",
}


def classify_reply(text):
    """Computed intent match, no LLM. Token overlap against two fixed word sets.

    A reply that hits BOTH sets ("don't go on") or NEITHER is unrecognized on
    purpose — the safe default when the human's intent is unclear is to ask
    again, not to guess and move an arm.
    """
    tokens = {
        "".join(c for c in w if c.isalpha() or c == "'")
        for w in (text or "").lower().split()
    }
    tokens.discard("")
    aff = tokens & AFFIRMATIVE
    neg = tokens & NEGATIVE
    if aff and not neg:
        return "affirmative", sorted(aff)
    if neg and not aff:
        return "negative", sorted(neg)
    return "unrecognized", sorted(aff | neg)


def resolve_with_human(st, expected, label, max_retries=2):
    """Capture a spoken reply and classify it. Retries on an unrecognized reply,
    then falls back to a typed prompt. Returns 'affirmative' or 'negative'."""
    for attempt in range(max_retries + 1):
        reply = capture_query(st, expected, f"{label}-reply{attempt or ''}")
        verdict, hits = classify_reply(reply or "")
        say(f"{DIM}  reply {reply!r} -> {verdict}"
            f"{' on ' + ', '.join(hits) if hits else ''}{OFF}")
        st.log(f"{label} reply {reply!r} -> {verdict} (matched {hits})")
        if verdict != "unrecognized":
            return verdict
        if attempt < max_retries:
            say(f"{YEL}  robot: \"I did not understand that. Stop, or go on?\"{OFF}")
        else:
            say(f"{YEL}  two retries used — switching to typed input for this reply{OFF}")
            st.text_mode = True
    reply = capture_query(st, expected, f"{label}-reply-typed")
    verdict, _ = classify_reply(reply or "")
    return "negative" if verdict != "affirmative" else "affirmative"


def speak(st, decision_path, label, template=None):
    cmd = [PY, str(REPO_ROOT / "voice" / "speak.py"), str(decision_path)]
    if template:
        cmd += ["--template", template]
    if st.no_audio:
        cmd.append("--no-audio")
    t0 = time.time()
    out = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    dt = time.time() - t0
    spoken = out.stdout.strip()
    if out.returncode != 0:
        say(f"{RED}  speak.py exit {out.returncode}: {out.stderr.strip()}{OFF}")
        return
    if spoken:
        say(f"{BOLD}  SPOKEN ({dt:.2f}s): {spoken}{OFF}")
    else:
        say(f"{DIM}  (act tier: says nothing — that is the point){OFF}")
    st.log(f"{label} spoken ({dt:.2f}s): {spoken!r}")


def panel(st, decision_path, label):
    """Evidence screen. Live: a fullscreen window in its own process, so plt.show()
    blocking there never blocks the state machine. Rehearse: rendered to disk."""
    close_panel(st)
    out_png = PANEL_DIR / f"{label}.png"
    cmd = [PY, str(REPO_ROOT / "voice" / "evidence.py"), str(decision_path), "--out", str(out_png)]
    if st.rehearse:
        r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        ok = r.returncode == 0 and out_png.exists()
        say(f"{DIM}  [rehearse] evidence panel -> {out_png} ({'ok' if ok else 'FAILED'}){OFF}")
        if not ok:
            say(f"{RED}  {r.stderr.strip()}{OFF}")
        st.log(f"{label} panel: {out_png} ok={ok}")
        return
    st.panel_proc = subprocess.Popen(cmd + ["--show"], cwd=REPO_ROOT)
    say(f"{DIM}  evidence panel up (pid {st.panel_proc.pid}){OFF}")


def close_panel(st):
    if st.panel_proc and st.panel_proc.poll() is None:
        st.panel_proc.terminate()
    st.panel_proc = None


def arm_motion(st, kind, name, seconds):
    """kind is 'gesture' or 'replay'.

    Per user override, motions are NO LONGER gated on their own keypress — the
    beat's logic decides when they run (including in response to a spoken reply).
    Ctrl-C freeze-and-hold remains the e-stop, so the banner below is printed
    immediately before any motion starts.
    """
    if st.sim_arm:
        say(f"{DIM}  [sim-arm] would run {kind} {name} (~{seconds}s) — "
            f"hardware not commanded{OFF}")
        st.log(f"arm {kind} {name}: SKIPPED (--sim-arm)")
        time.sleep(0.1)
        return

    from arm import arm_io
    from arm.safety import ArmFrozen, MotionAborted

    say(f"{RED}{BOLD}  >>> ARM MOVING NOW: {kind} {name} (~{seconds}s) — "
        f"Ctrl-C freezes and holds <<<{OFF}")
    say(f"{YEL}  arm: {kind} {name} (~{seconds}s, {arm_io.backend_name()}){OFF}")
    t0 = time.time()
    try:
        if kind == "gesture":
            arm_io.gesture(name)
        else:
            arm_io.replay(arm_io.TASK_DEMO_PATH)
    except (KeyboardInterrupt, MotionAborted):
        say(f"{RED}  MOTION FROZEN AND HOLDING. It did NOT home. Press h to home at 25%.{OFF}")
        st.log(f"arm {name}: FROZEN by operator")
        return
    except ArmFrozen:
        say(f"{RED}  arm is frozen from an earlier abort. Press h to home first.{OFF}")
        return
    dt = time.time() - t0
    say(f"{GRN}  arm: {name} complete ({dt:.1f}s){OFF}")
    st.log(f"arm {kind} {name}: complete in {dt:.1f}s")


def _reset_live_entry(query):
    """ingest.py APPENDS takes to live_index.json by design (a real corpus grows).
    On stage that means a rehearsal's takes are still counted and beat 4 says
    "I watched your 6 attempts" after recording 3. Each demo run starts this
    query's live evidence from empty so the spoken number is this run's number."""
    path = REPO_ROOT / "feedback" / "live_index.json"
    if not path.exists():
        return
    try:
        idx = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return
    if idx.pop(query, None) is not None:
        path.write_text(json.dumps(idx, indent=2))


def ingest_live(st, query):
    """BEAT 4's ingest. Live: the real webcam path. Rehearse: the synthetic path,
    driven through feedback/'s own capture/extract/index functions so the parquet
    and live_index.json come out of the identical code, keyed on the real query."""
    _reset_live_entry(query)
    if not st.rehearse:
        cmd = [PY, str(REPO_ROOT / "feedback" / "ingest.py"),
               "--task", query, "--record", "3", "--demonstrators", "1"]
        r = subprocess.run(cmd, cwd=REPO_ROOT)
        if r.returncode != 0:
            say(f"{RED}  ingest exited {r.returncode} — most likely camera permission.{OFF}")
            say(f"{RED}  Narrate the beat instead: the pipeline is the same one that "
                f"produced all 424 corpus clips.{OFF}")
            return False
        st.log("beat4 ingest: 3 live webcam takes recorded")
        return True

    import time as _t

    sys.path.insert(0, str(REPO_ROOT / "feedback"))
    import capture
    import extract_shim
    import index as live_index
    import pyarrow.parquet as pq

    say(f"{DIM}  [rehearse] synthetic ingest via feedback/capture.synthetic_frame_sequence "
        f"(no camera in this session; see FACTS.md WEBCAM_STATUS){OFF}")
    slug = "".join(c if c.isalnum() else "_" for c in query.lower()).strip("_")
    out_dir = REPO_ROOT / "feedback" / "live_corpus" / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(1, 4):
        frames = capture.synthetic_frame_sequence(duration_s=5.0)
        rows = extract_shim.add_speeds(extract_shim.process_frames(frames))
        clip_id = f"{slug}_rehearse_{int(_t.time())}_{i}"
        table = extract_shim.build_table(rows, clip_id=clip_id, task_id=slug)
        p = out_dir / f"{clip_id}.parquet"
        pq.write_table(table, p, compression="zstd")
        records.append((str(p.relative_to(REPO_ROOT)), live_index.compute_take_stats(table)))
    entry = live_index.update_index(
        REPO_ROOT / "feedback" / "live_index.json", query, records, n_demonstrators=1)
    say(f"{DIM}  [rehearse] live_stat_line: {entry['live_stat_line']}{OFF}")
    st.log(f"beat4 ingest (synthetic): 3 takes, live_stat_line={entry['live_stat_line']!r}")
    return True


# ------------------------------------------------------------------------ beats

def step(st, prompt, fn, is_motion=False):
    while True:
        if gate(st, prompt, is_motion=is_motion) == "go":
            result = fn()
            if result is not False:
                return result
            if st.rehearse:
                return result
            say(f"{YEL}  step did not complete. r retries, SPACE moves on anyway.{OFF}")
        # 'again' falls through to re-prompt


def beat1(st):
    banner("BEAT 1 / act — a task it has enough agreement about to just do it")
    q = step(st, "capture the query (say: 'garment iron press')",
             lambda: capture_query(st, BEAT1_QUERY, "beat1"))
    if q is None:
        return
    got = step(st, "compute the decision", lambda: decide(st, q, "beat1"))
    if got is None:
        return
    path, dec = got
    step(st, "the arm acts, silently — replay task_demo (~19.4s), then evidence screen",
         lambda: (arm_motion(st, "replay", "task_demo", 19.4),
                  speak(st, path, "beat1"), panel(st, path, "beat1")))


def beat2(st):
    banner("BEAT 2 / ask — majority vs outlier, and it refuses to overclaim")
    q = step(st, "capture the query (say: 'garment inside out')",
             lambda: capture_query(st, BEAT2_QUERY, "beat2"))
    if q is None:
        return
    got = step(st, "compute the decision", lambda: decide(st, q, "beat2"))
    if got is None:
        return
    path, dec = got
    step(st, "gesture 'attention', ask the question, show the split (8 vs 1, silhouette, p)",
         lambda: (arm_motion(st, "gesture", "attention", 3.4),
                  speak(st, path, "beat2"), panel(st, path, "beat2")))
    say(f"{DIM}  Say out loud: p is 0.11. With 9 clips it cannot rule out chance, and it "
        f"says so rather than dressing up a k=2 split as a finding.{OFF}")
    _resolve_ask(st, path, "beat2")

    if st.rehearse:
        # Rehearsal-only: walk the OTHER branch too, so one pass proves both.
        say(f"{DIM}  [rehearse] replaying the resolution on the affirmative branch{OFF}")
        _resolve_ask(st, path, "beat2b")


def _resolve_ask(st, path, label):
    """The ambiguity is resolved by the human, out loud. Stop or go on."""
    verdict = resolve_with_human(st, "go on", label)
    if verdict == "affirmative":
        say(f"{DIM}  Say out loud: strategy selection is real; the arm's repertoire is "
            f"one trajectory. Same replay as beat 1, honestly.{OFF}")
        arm_motion(st, "replay", "task_demo", 19.4)
    else:
        say(f"{GRN}  human said stop — the arm holds and does not execute{OFF}")
        speak(st, path, f"{label}-hold", template="ask_hold")


def beat3(st):
    banner("BEAT 3 / abstain — the centerpiece", RED)
    q = step(st, "capture the query (say: 'do a bottle flip')",
             lambda: capture_query(st, BEAT3_QUERY, "beat3"))
    if q is None:
        return
    st.beat3_query = q
    got = step(st, "compute the decision", lambda: decide(st, q, "beat3"))
    if got is None:
        return
    path, dec = got
    if dec["tier"] != "abstain":
        say(f"{RED}  *** tier is {dec['tier']}, expected abstain. This is the "
            f"stop-the-line case. ***{OFF}")
    step(st, "gesture 'decline' (~7.4s), speak the refusal, show the coverage panel",
         lambda: (arm_motion(st, "gesture", "decline", 7.4),
                  speak(st, path, "beat3"), panel(st, path, "beat3")))
    say(f"{DIM}  Say out loud: it checks whether every word in the request is something it "
        f"has seen, not whether the sentence looks similar. 'flip' appears in zero of the "
        f"50 task names.{OFF}")

    def _push_back():
        verdict = resolve_with_human(st, "go on", "beat3")
        if verdict == "affirmative":
            # Pushed to try anyway. The boundary does not move — and crucially the
            # arm does NOT execute here. This is the honesty centerpiece.
            speak(st, path, "beat3-restate", template="abstain_restate")
            say(f"{DIM}  Say out loud: it was told to try anyway and still did not. "
                f"Zero demonstrations is not a confidence problem, it is a data problem.{OFF}")
        else:
            say(f"{GRN}  human accepted the refusal — moving on{OFF}")

    step(st, "push back: tell it to go on anyway, and watch the boundary hold", _push_back)


def beat4(st):
    banner("BEAT 4 / feedback — more data changes what it knows, not what it can do")
    q = st.beat3_query or BEAT3_QUERY
    ok = step(st, f"record 3 live demonstrations of {q!r}", lambda: ingest_live(st, q))
    got = step(st, "recompute with the live demos folded in",
               lambda: decide(st, q, "beat4", include_live=True))
    if got is None:
        return
    path, dec = got
    n_live = dec["evidence"].get("n_live_demos")
    if not n_live:
        say(f"{RED}  live_index had no entry for {q!r} — beat 4 will speak the plain "
            f"abstain. (ingest --task must use this exact string.){OFF}")
    step(st, "gesture 'point_screen' (~7.4s), speak the post-feedback refusal, show the panel",
         lambda: (arm_motion(st, "gesture", "point_screen", 7.4),
                  speak(st, path, "beat4"), panel(st, path, "beat4")))
    say()
    say(f"{BOLD}  CLOSE: \"More data changed what I know, not what I can do. "
        f"That is the honest boundary.\"{OFF}")


BEATS = [beat1, beat2, beat3, beat4]


# ------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-mode", action="store_true", help="start in typed-input mode")
    ap.add_argument("--no-audio", action="store_true", help="print the spoken text, do not play it")
    ap.add_argument("--beat", type=int, default=1, help="start at this beat (1-4)")
    ap.add_argument("--rehearse", action="store_true",
                    help="drive all 4 beats through --text non-interactively")
    ap.add_argument("--sim-arm", action="store_true",
                    help="print what the arm would do instead of commanding it "
                         "(auto-enabled by --rehearse when the backend is real hardware)")
    args = ap.parse_args()

    # A rehearsal must never command a live arm as a side effect. arm/ picks its
    # backend from FACTS.md HARDWARE_PRESENT at import, and that flag is now 'yes',
    # so --rehearse stubs the motions unless the operator explicitly opts in.
    if args.rehearse and not args.sim_arm:
        from arm import facts as _facts
        if _facts.hardware_present():
            args.sim_arm = True

    st = State(args)
    DECISION_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)

    banner("I DON'T KNOW HOW TO DO THAT YET")
    if st.sim_arm:
        say(f"  arm backend : {YEL}STUBBED (--sim-arm){OFF} — no motion will be commanded")
    else:
        from arm import arm_io
        d = arm_io.describe()
        say(f"  arm backend : {d['backend']}  (simulated={d['simulated']}, "
            f"FACTS HARDWARE_PRESENT={d['hardware_present_facts']})")
        if not d["simulated"]:
            say(f"{RED}{BOLD}  REAL ARM. Motions are NOT individually gated: they run when "
                f"the beat decides, including in response to a spoken reply.{OFF}")
            say(f"{RED}  Ctrl-C freezes and holds. Keep the workspace clear.{OFF}")
    say(f"  input mode  : {'TYPED' if st.text_mode else 'SPOKEN (listen/transcribe.py)'}")
    say(f"  sim render  : ARM_SIM_RENDER={os.environ.get('ARM_SIM_RENDER')} "
        f"(0 = no per-motion GIF, stills already in arm/sim_out/)")
    say()
    say(f"{BOLD}  SAY THIS OUT LOUD, ONCE, EARLY:{OFF}")
    say(f"{BOLD}  \"{DISCLOSURE}\"{OFF}")

    rc = 0
    try:
        step(st, "home the arm before anything else (~1s)", lambda: _home(st), is_motion=True)
        for i, beat in enumerate(BEATS, start=1):
            if i < args.beat:
                continue
            beat(st)
        banner("DEMO COMPLETE", GRN)
    except Quit:
        say(f"\n{YEL}quit — arm left where it is, not homed behind your back{OFF}")
    except KeyboardInterrupt:
        say(f"\n{RED}interrupted{OFF}")
        rc = 130
    finally:
        close_panel(st)
        if st.rehearse:
            out = REPO_ROOT / "demo" / "rehearsal_transcript.txt"
            out.write_text("\n".join(st.transcript) + "\n")
            say(f"\n{DIM}rehearsal transcript -> {out}{OFF}")
    return rc


def _home(st):
    if st.sim_arm:
        say(f"{DIM}  [sim-arm] would home — hardware not commanded{OFF}")
        return
    from arm import arm_io
    t0 = time.time()
    arm_io.home()
    say(f"{GRN}  arm: home complete ({time.time() - t0:.1f}s){OFF}")
    st.log(f"arm home: complete in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
