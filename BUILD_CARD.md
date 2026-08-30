# Build card — The Hands Index

Berkeley Robotics Hackathon, 30 August 2026 · Visualization track
Repo: https://github.com/TarunYadgirkar/Robotics-Hackathon-project

## What existed before the event

- **The dataset.** WORLD_CONTEXT_HACKATHON_V3_PUBLIC v3.1.1 — 424 five-minute clips, 50
  industrial tasks, 35.3 hours, 1920×1080 @ 29.97 fps, no audio, with ~200 Hz IMU sidecars,
  thumbnails, 300 eight-second preview proxies, and full metadata.
- **The dataset's own tooling.** Its Python SDK, and `launch/serve.py` — a zero-dependency
  HTTP server with range support. We use that server **verbatim** as our media backend
  rather than writing one.
- **MediaPipe Hands** (Google, Apache 2.0), **ffmpeg**, **React / Vite / Tailwind**,
  **pyarrow**, **numpy**, **Pillow**.
- **Four measurements we made before the event started**, which set the direction:
  1. The camera is torso-mounted, not head-mounted — so image coordinates are body-relative
     hand position with no calibration. Everything here rests on that.
  2. IMU work-cycle mining is dead: gyro autocorrelation median peak 0.16, only 2 of 50
     tasks above 0.30.
  3. IMU machine-vibration signatures are dead: whitened SNR ~2.1 on all 50 tasks, exactly
     the noise floor. The sidecars document why — GPMF timestamps are reconstructed by
     spreading samples uniformly inside ~1 s packets.
  4. Hand tracking works at 2 fps / 640 px, spot-checked on four tasks.

## What we built during the event

- `pipeline/extract.py` — corpus-wide two-hand tracking. ffmpeg pipes rawvideo straight into
  numpy (no intermediate frames); per sampled frame it records 21 landmarks per hand, palm
  centroid, grip aperture, finite-difference speed, plus gyro RMS and torso lean from the
  IMU. One Parquet per clip, atomically renamed, resumable.
- `pipeline/segment.py` — per-second reduction, and a corpus payload small enough that the
  browser classifies states itself (~250 KB for all 424 clips).
- `pipeline/validate.py` — random-sample contact sheets for checking the index by eye.
- `pipeline/narrate.py` — spoken clip summaries whose script is generated from the measured
  timeline.
- `web/` — the explorer: headline, 424-clip corpus wall, click-to-play against the local
  media server, reel builder with export, per-task cards with body-relative heatmaps and
  grip-aperture distributions, the two-handed/asymmetry quadrant, index-picked montage,
  and a limitations tab.

## Central claim and evidence

**Most of a demonstration corpus is not demonstration.**

Of the 35.3 hours collected, 30.5 hours are measurable at all — and only **43.1% of that is
two-handed manipulation**. One-handed handling is 26.8%, transit between objects 17.9%, and
12.2% has no hands in frame. The teachable part of this corpus is **13.2 hours, not 35.3**.

The spread across tasks is **6.2×**: oil-seal pressing is 77.9% two-handed, water-filtration
bottle filling is 12.6%. Every second of all 424 clips was labelled by tracking both hands —
no annotations, no training, one threshold for all 50 tasks.

Evidence, in the order a skeptic should check it:

1. **The montage tab.** For each task the index picks the 8-second window it scores as most
   two-handed. Nobody screened these. All **25 of 25** tiles score a perfect 8 out of 8 seconds, and every one of them visibly
shows hands working.
2. **The threshold slider.** The state boundary is one global number — the 75th percentile
   of pooled hand speed, which lands at 0.60. Across the band **0.35–1.00 the task ordering
   holds at Spearman ρ ≥ 0.90** against the default, while the absolute percentages move a
   lot. Below 0.35 it genuinely breaks down (ρ = 0.52 at 0.15), because nearly every second
   gets called transit and the labelling is degenerate rather than merely stricter. We claim
   the ordering, and only inside that band. A per-task threshold would have been fitting the
   answer.
3. **Eyeball validation.** 60 random seconds from 36 tasks, each rendered as the two frames of that second so a
motion call is checkable, judged against the state the index assigned: **54 agreed (90%)**.
Of the 6 that did not, **5 are the index missing hands that are visible** and 1 is it
over-calling two hands. The error runs one way, so 43.1% is a floor, not a boast. Full
sample table and caveats in [VALIDATION.md](VALIDATION.md) — including that the reviewer is
the same agent that wrote the pipeline, so this is a sanity check, not an independent eval.
4. **The reel export.** The filtered segment list is a real artifact — clip ids and time
   ranges you can hand to a data loader.

## Limitations and failures

- **Gloves defeat the tracker.** Bottle-cleaning detection runs at **3.6%** across all
  nine clips because the worker wears blue nitrile gloves. Read naively it looks like 45
  idle minutes; it is a busy sink. Any task below 50% detection is held out of every
  headline number instead of being averaged in.
  Evidence: `assets/generated/gloved-hands-detection-failure.jpg`.
- **2 fps sampling** hides events shorter than about 1.5 s.
- **Our own pre-event spot checks were too small.** We had recorded 77% detection on lathe
  operation from one clip. Across all seven of its clips it is 35% — the first clip really
  does score 72%, the rest run 13-49%. Within-task variance is large enough that no
  single-clip number should be trusted, including a reassuring one.
- **"Hands absent" means the tracker found nothing**, not that hands were gone. Detection
  rate is printed beside every number for exactly this reason.
- **No ground truth.** Nothing was hand-labelled, so no accuracy figure is claimed — only
  consistency, threshold stability, and checkable picks.
- **Single-source tasks.** **9** of 50 tasks come from one independent recording family;
  their numbers describe that recording, not the task. The dataset ships this warning and
  the task cards repeat it.
- **Seven tasks are excluded entirely**, at detection rates from 3.6% to 48.2%: bottle
  cleaning, plaster ceiling tile, processing fabric spread, lathe operation, processing
  fabric cut, fabric cutting (scissor), fabric cutting (machine). Three distinct causes,
  each confirmed by looking at footage rather than guessed at — gloves, hands caked in wet
  plaster, and occlusion by the machine or workpiece. Evidence:
  `assets/generated/excluded-tasks-failure-modes.jpg`.
- **Body-relative, not metric.** Mount angle varies across the 36 cameras, so reach
  envelopes compare well within a camera and only roughly across them. No depth, no 3D
  pose, no metric scale.
- **Two dead ends before this one.** Both IMU angles above were measured and abandoned. The
  IMU survives here only as gross motion: gyro RMS and torso lean.
- **The narration is not perception.** It is text-to-speech reading statistics this pipeline
  computed. No model watched a video to describe it, and it extends no data.

## External code and assets

| Thing | Source | Use |
| --- | --- | --- |
| WORLD_CONTEXT_HACKATHON_V3_PUBLIC v3.1.1 | World Context | all video, IMU, metadata |
| `launch/serve.py` | ships with the dataset | media server, used verbatim |
| MediaPipe Hands 0.10.21 | Google, Apache 2.0 | hand landmarks |
| ffmpeg | FFmpeg project | decode |
| React, Vite, Tailwind, pyarrow, numpy, Pillow | respective projects | app and pipeline |
| macOS `say` / ElevenLabs | Apple / ElevenLabs | narration voice, and the demo voiceover |

Version pin worth recording: `mediapipe==1.0.1` removes `mp.solutions` and crashes on macOS
arm64 with a Metal service error, and will not install on Python 3.14. We pin 0.10.21 on a
3.12 venv.

## Note on this repository

The repo is shared with a teammate's separate project, so **The Hands Index lives on the
`hands-index` branch** — that is the branch to review:
https://github.com/TarunYadgirkar/Robotics-Hackathon-project/tree/hands-index

On that branch, this submission is `pipeline/`, `web/`, `assets/` and the root markdown.
`yam/`, `scripts/` and `web/enroll/` are the teammate's YAM arm-control work, carried along
by shared history and not part of this submission. `main` holds their work only.

## Reproduction facts

424 of 424 clips extracted, zero errors, 2023 s wall clock on an M5 Pro with 5 worker
processes. The whole corpus index is 248 KB (`states.bin`), which is why the threshold can
be a live control in the browser instead of a decision baked in at build time.

## Arm demo addendum

This section covers the "I don't know how to do that yet" demo built on top of the Hands
Index corpus. It is additive to everything above; nothing above was changed.

### Reused, not built today

- `pipeline/extract.py` and its output: one Parquet per clip (21 landmarks/hand, palm
  centroid, grip aperture, speed, gyro RMS) for all 424 clips across 50 tasks. That Parquet
  IS the trajectory representation the demo reasons over. No video was re-extracted; the raw
  video files are not even present on this machine.
- The corpus metadata (`meta/tasks.jsonl`, `meta/clips.jsonl`) and the `<50% detection`
  exclusion rule, taken unchanged from the existing pipeline.
- MediaPipe extraction settings for live webcam takes: `feedback/extract_shim.py` imports
  `pipeline/extract.py`'s own `_get_hands()`, `add_speeds()` and landmark constants rather
  than re-specifying them, so live demos and corpus clips are measured identically.

### Built today

- `variance/` — per-task DTW distance matrices over within-clip z-scored features,
  k=2 hierarchical clustering, silhouette, and a 1000-shuffle permutation test (seed 42).
  Plus a double confound check: cluster labels tested against both the
  `independent_repetition_id` and the `camera_id` partitions.
- `brain/decide.py` — content-word coverage (not string similarity) driving three tiers:
  act / ask / abstain. No task names are hardcoded anywhere; everything is read from
  metadata at runtime.
- `listen/transcribe.py` (local Whisper, push-to-talk), `voice/speak.py` (fixed templates,
  ElevenLabs TTS with a macOS `say` fallback), `voice/evidence.py` (static evidence panels).
- `feedback/ingest.py` — webcam capture of human demonstrations through the same extraction
  path, writing Parquet plus `feedback/live_index.json`.
- `arm/` — the motion layer, and `demo/run_demo.py` — the keyboard-stepped state machine.

### The sentences are templates; the numbers are not

Every spoken line is a fixed template. Every number inside it is computed at runtime from
the corpus — there is no LLM anywhere in the demo path, and no number is written into the
templates. This is said out loud on stage rather than left for a judge to discover.

### The arm: real hardware, deliberately small motions

The demo runs on a real YAM arm (Damiao actuators over a CANable2 gs_usb adapter), driven
by this repo's own `yam/` package. The build began sim-first — `HARDWARE_PRESENT: no` at
kickoff — and the arm arrived mid-build; `coordination/FACTS.md` records the flip, and
`arm/arm_io.py` selects the backend from that file at import. The simulator branch is
retained and selectable via `ARM_FORCE_SIM=1` (the variable can only force *toward* sim,
never toward hardware).

Bring-up was verified live: enable, home, all three gestures, and the task motion, twice
(reduced speed then full), zero failures, with the velocity cap holding every motion to
1.00x its scheduled duration. Two bugs were found and fixed **on the hardware** — a lock
covering the CAN exchange that stretched an 18s gesture to 86s, and a limit check that
rejected the arm's own resting pose (j2/j3 rest 0.01° below their bound).

The hardware gestures are deliberately minimal by design decision, not limitation-hiding:
the arm stays within ~3° of its resting pose and expression is carried by the calibrated
gripper (gentle closing pulses, ≤10%/s against a 12%/s limit). `home()` on hardware settles
and holds the current pose rather than sweeping to a stored one. The sim-authored
pick-and-place is **refused** by the hardware branch (joint1 would sweep 27.9°) — that
refusal is a tested negative control, not an accident.

In BEAT 2 the strategy *selection* is real — cluster sizes, silhouette and p-value are
computed from the corpus — but the motion played afterwards is the same small gripper cycle
regardless of which answer is given. That is stated out loud during the beat.

What is real in `arm/` regardless of backend: the velocity cap (30% of the driver's own
limit, applied by time dilation rather than position clipping), soft-limit rejection of
out-of-range waypoints, and freeze-and-hold on interrupt — an aborted motion does **not**
auto-home, because homing is itself autonomous motion after someone hit stop.

### Hardware safety: what is verified, what is structural, what is known-flaky

`arm/model.py` mirrors `yam.arm.ARM_JOINTS` limits and derives the velocity cap (34.4°/s =
30% of `SafetyLimits.max_joint_speed`); `hw_backend.verify_against_yam()` re-checks that
mirror at connect time and refuses to run if it has drifted — it fired once for real during
bring-up when upstream renamed the governing constant, and blocked before any bus traffic.

- **Self-collision remains formally unverified.** `yam/arm.py` records that roughly 10% of
  in-limit poses self-collide; `mujoco` and the i2rt URDF are absent, so
  `yam.environment.ArmSafetyChecker` could not be run. The mitigation is structural rather
  than checked: every hardware pose sits within ~5° of a pose the arm already rests in, and
  joint2 (the ~105° base-collision trap) moves ≤3°. Conservative authoring is not a
  collision check, and the build card says so.
- **Comms-lost (0xD) latch: handled.** `hwsupport/keepalive.py` holds position at low rate
  through the talking gaps between beats, with a single-pass recovery policy (one
  `recover_stale_motors()` + `clear_errors()` attempt for a comms fault, then surface —
  never a retry loop into a faulted bus). The keep-alive and the motion streamer alternate
  via an explicit handoff; they are never both on the bus.
- **Known-flaky adapter.** The CANable2 dropped off the USB bus once after a full successful
  bring-up session (independently documented by the driver's author). `hwsupport/triage.py`
  encodes 26 failure modes mined from the driver's own code and commits, including how to
  distinguish host-side bus-off (fixable via `reconnect()`) from a real power/CAN fault, and
  when the only fix is a physical reseat.

`arm/hw_bringup.py --read-only` verifies the link (7 motors, temperatures, resting pose)
without enabling torque, and is the required first step of any session.

### Limitation: no variance claim is made from live data

BEAT 4 ingests three live human demonstrations and cites what was measured from them —
peak hand speed and the fraction of frames tracked. It deliberately does **not** compute a
silhouette, a cluster split, or any strategy-variance claim from those takes.
`n_live_demonstrators` defaults to 1: three takes by the same person is not evidence of
cross-worker strategy diversity, and presenting it as such would be exactly the overclaim
the rest of this build card exists to avoid. Strategy variance is computed on the corpus
only, and only for tasks that survive both confound checks.

### Limitation: no task reaches significance, and the demo says so

Of the 50 tasks, none that is both deconfounded and non-excluded has `perm_p <= 0.05`. The
strongest clean case is `garment-inside-out` at silhouette 0.152, p 0.113. Rather than
raise a threshold to manufacture a result, BEAT 2 was reframed as majority-vs-outlier: 9
demonstrations, 8 one way, 1 different — and the spoken line keeps the p-value disclosure
("with 9 clips I cannot rule out chance: p equals 0.11") even though it is above 0.05. The
non-significance at n=9 is the honest point, not a defect to be hidden. Every clean split in
this corpus is majority-vs-one-outlier; not one splits into two balanced methods.

Related: 27 of 50 tasks carry a `diversity_warning` and are confounded by construction, and
7 more are excluded outright for sub-50% hand detection (gloves). The camera is torso-mounted
and mount position varies, so a k=2 split will happily recover "which camera" and present it
as "which strategy" — which is why cluster labels are tested against both the repetition-family
and the camera partition, and a task must clear both to be eligible for the ask tier.

### Conversational layer: an LLM phrases, it does not decide

The conversational mode added late in the build uses a small LLM (gpt-4o-mini, or
claude-haiku-4-5 when an Anthropic key is present) for **wording only**. The division is
strict and is the whole point:

| stage | component | what it decides |
| --- | --- | --- |
| speech in | `listen/transcribe.py` | nothing (local Whisper, no network) |
| **decision** | **`brain/decide.py`** | **tier, matched task, every number — computed from the corpus** |
| wording | `voice/llm_phrase.py` | how to say the decision, in 2-3 sentences |
| speech out | `voice/speak.py` | nothing |

`voice/llm_phrase.py` receives the decision JSON already decided. Its system prompt
hard-constrains it: it may only assert facts present in that JSON, must reproduce numbers
verbatim, may not name a task that is not in the JSON, may not claim a capability or a
success, and may not change what a tier means — an `abstain` stays a refusal that ends by
asking how. The last six conversation turns are passed for continuity of tone only.

**The LLM cannot cause motion.** Arm motions are dispatched in `demo/run_demo.py` from the
computed tier; the LLM's output is never read for control. If the call errors or exceeds a
2-second budget, the fixed template is used for that turn — the templates are the
reliability floor, and every line is tagged `[llm]` or `[template]` in the operator's
terminal. `--no-llm` disables the layer entirely and the demo runs identically on templates.

So the claim "every utterance is backed by a number computed at runtime" still holds: the
numbers come from `decide.py` either way. Only the sentence around them may vary.

### Scripted backup mode

`demo/run_demo.py --scripted` is a deterministic backup: no microphone, no LLM, no network,
no TTS call. It steps through the golden path on the spacebar, running the real arm gestures
in a fixed order (`wake` → `decline` → `attempt`) and playing pre-synthesized audio from
`demo/audio_cache/`. The event judge explicitly sanctioned a hardcoded demo **as a backup**,
so it is labelled `[SCRIPTED BACKUP]` in the terminal for its whole duration and is disclosed
out loud when used. The numbers inside that cached audio are still real: the decision JSONs
in `demo/decisions/scripted/` were generated by the actual `brain/decide.py`, not written by
hand. The live golden path remains the primary demo.

### The arm is the weakest link, and the demo is built to survive it

Every arm call in every mode is wrapped. If the arm raises anything — including the CAN
adapter disappearing from USB mid-run, which happened during this build — the terminal prints
`[ARM OFFLINE — continuing without motion]` and the spoken line and evidence panel continue
uninterrupted. The failure is deliberately not latched, so a later motion retries and picks
the arm back up if it reconnects. This was verified by forcing the exception in simulation.
