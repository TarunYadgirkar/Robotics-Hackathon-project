# Robotics-Hackathon-project

## Ongoing

Updated: 2026-08-30T17:20:00-07:00 by claude session

Branch: `idk-demo` (off `hands-index`; `main` is Boris's YAM/scan work — never merge main in, cherry-pick `yam/` + `scripts/` via `git checkout origin/main -- yam/ scripts/`).

Done (all pushed to origin/idk-demo, latest `c93c1a8`):
- Full "I Don't Know How To Do That Yet" demo per PLAN_v2 (in ~/Downloads), heavily evolved by user direction. Dataset = LOCAL copy at `~/TarunsCode/wc-hack` (WC23 volume not mounted; set `WC_VIDEOS=$HOME/TarunsCode/wc-hack`). All 424 parquets pre-extracted.
- `brain/decide.py` — content-word coverage matcher (act/ask/abstain), majority-vs-outlier ask tier (sil>=0.1, deconfounded); no task reaches p<=0.05 — that's a corpus property, disclosed. `brain/test_decide.py` green.
- `variance/` — DTW + permutation + double confound check over 50 tasks (`dd19e39`-era).
- `listen/transcribe.py` (mlx-whisper, `--seconds N` timed capture — push-to-talk is dead, it was invisible under captured stdout), `voice/speak.py` (ElevenLabs voice "Eric" `cjVigY5qzO86Huf0OWal`, `say` fallback, templates incl. abstain_howto/attempt_result/ask_hold/abstain_restate), `voice/evidence.py` panels.
- REAL YAM ARM WORKING: `arm/` hardware backend over Boris's `yam.arm.YamArm` (Damiao/CANable2 gs_usb VID 0x1D50 PID 0x606F). Gestures live-verified full range on j2–j6: wake(20s), attention, decline, point_screen, attempt, task_demo, approach_can, can_grip_top, can_grip_bottom. **joint1 HARD-LOCKED ±2° in validate_hardware_motion (physical clamps at base sides — user constraint, overrides collision checker).** Official i2rt yam_pro URDF+MJCF in `hwresearch/` (symlinked to `~/TarunsCode/hackathons/i2rt/i2rt/` where Boris's code expects); mujoco ArmSafetyChecker runs automatically on every setpoint stream. Jaw force lowered to 0.25Nm (~17N) so a can can't be crushed. Auto-recovery: one-shot reconnect on CAN failure, `ArmUnavailable` raised fast otherwise; demo survives arm death (`[ARM OFFLINE]`, keeps talking). Keepalive (`hwsupport/keepalive.py`) prevents the 0xD comms-lost latch between beats. `hwsupport/triage.py --symptom` = 26 failure modes mined from Boris's code.
- `demo/run_demo.py` — golden path: wake gesture → converse mode ("fold a piece of paper" → abstain+"How do I do it?" → spoken instruction → attempt gesture → honest close; "put this can upside down" → 0.0 coverage → approach_can → asks top/bottom → keyword-matched answer picks can_grip_top/bottom). LLM phrasing live (anthropic SDK 1.2.0, claude-haiku-4-5-20251001, ~1.4s; ANTHROPIC_API_KEY in .env works, no workspace header needed). LLM only words the computed decision JSON; motion is dispatched from computed tier/keywords, never LLM output. `--scripted` = judge-sanctioned deterministic backup (pre-synth mp3s in `demo/audio_cache/`, real decision snapshots, real gestures). `--rehearse` (sim), `--text-mode`, `--timeout`, `t` toggle.
- `talk.py` (repo root) — standalone voice conversation loop, no arm.
- `feedback/ingest.py` — live webcam ingest reusing pipeline/extract.py; `CAMERA_INDEX` env selects device.
- BUILD_CARD.md "Arm demo addendum" — current through real-hardware + LLM disclosure.

In flight: nothing (all subagents delivered).

Blocked / user-side only:
- **Camera**: user's iPhone connected via Continuity (friend's phone won't work — wrong Apple ID). cv2 sees the phone from the USER'S Terminal only (agent processes are TCC-blocked). User was mid-diagnosis: capture grabbed Mac camera; next step is snapshot test per index (`/tmp/camN.jpg`) and launch with `CAMERA_INDEX=<phone index>`; phone must be locked/landscape/near, and no other app holding a camera.
- **BACKUP VIDEO NOT RECORDED** — non-negotiable per plan; flow changed 4x since last clean pass. QuickTime steps in run_demo.py header.
- Live run with the physical can never done (gestures verified on arm, but no can present). Place can under `approach_can` hover (~15cm out, straight ahead of the locked base).
- Repo public + YouTube + submission form still pending (hard stop was 17:30).

Next (cold-start order):
1. `.venv/bin/python arm/precheck.py` (16s READY/NOT READY; CANable dropped off USB twice today — reseat/different port fixes it; `hwsupport/triage.py` for faults).
2. Camera snapshot test from USER's Terminal, pick CAMERA_INDEX.
3. Full live run: `CAMERA_INDEX=N .venv/bin/python demo/run_demo.py` (golden path; `--timeout 3` if wifi slow).
4. Record backup video on second clean pass; `--scripted` is the stage fallback.
5. Repo public, upload, submit.

Gotchas: bash-guard hook blocks ANY `.env` reference in Bash commands (read keys inside Python only); never run two processes on the CAN bus; `yes '' | ... --text-mode` drives the demo headlessly (non-tty read_key fallback exists); sim gesture counterparts have older choreography — take durations from C.json `gesture_durations_s`.
