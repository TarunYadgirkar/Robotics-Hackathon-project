# Robotics-Hackathon-project

## Ongoing

Updated: 2026-08-30T23:59:00-07:00 by claude session

Branch: `idk-demo` (off `hands-index`; `main` is Boris's YAM/scan work — never merge main in, cherry-pick `yam/` + `scripts/` via `git checkout origin/main -- yam/ scripts/`).

Done (all pushed to origin/idk-demo, latest `81277f4`):
- Everything from the pre-flip build (through `c93c1a8`): golden path + converse + can beat + `--scripted` backup, real YAM arm over Boris's yam.arm (joint1 HARD-LOCKED ±2°, jaw torque 0.25Nm/~17N, mujoco safety checks on every stream, keepalive, triage). LLM (claude-haiku-4-5) words computed decisions only; keywords dispatch all motion. Dataset local at `~/TarunsCode/wc-hack` (`WC_VIDEOS` override).
- **FLIP BEAT (`81277f4`), user-directed new flow**: `demo/run_demo.py --flip`, or in converse via "pick/grab/lift/hold/take ... can" (PICKUP_WORDS routes to `flip_beat` before `can_beat`). Sequence: `can_pickup` gesture (REAL grip, unlike can_grip_top's pantomime — jaws close to 84% absolute = 63.8mm < 66mm can, stall at 17N; 2.5s hover pause for the operator to center the can; ends HOLDING the can lifted, keepalive maintains grip through the conversation) → listen, expect "flip" (FLIP_WORDS) → decide("flip") = honest abstain, spoken via abstain_howto ("None of them is flip... How do I do it?") → listen, expect "hold it at the top" (RELEASE_TOP keywords; NEGATIVE without "top" aborts and keeps holding) → `can_fling` (dip in j4, rise j3+26/j4+27 at ~23 deg/s, jaws open 84→88% in the last 0.35s of the rise at 11.4%/s — **hardware gripper cap is 12%/s**, opening more would stretch the segment and kill the fling; finish to 98% at apex; ends AT ITS OWN START POSE) → attempt_result close. Can not landing = expected + disclosed.
- Both gestures LIVE-VERIFIED on the real arm (pickup + 8s keepalive hold + fling ran clean; the retimed 88%-release segment verified in the real motion pipeline via ARM_FORCE_SIM, not yet on hardware — CANable died first, see Blocked).
- Gesture-authoring landmines learned on hardware, recorded in the JSONs' notes: the arm SAGS a few degrees under gravity vs commanded (j2 27.6 measured vs 30 commanded, j3 to its 0.0 floor), so any relative offset that "undoes" a previous gesture underflows the j2/j3 soft floors and is REFUSED — end chained gestures at their own start pose, keep j2/j3 offsets >= 0. `arm/verify_poses.py` now verifies `can_fling` from `can_pickup`'s end pose (`follows` map). All 11 gestures PASS collision verification.
- Killed a stale `run_demo.py` that was holding the CAN bus (that's what a "CanOperationError: message could not be sent" precheck FAIL with adapter present looks like).

In flight: nothing. Code review of `81277f4` landed as `4b90fe8`: release-question loop now mirrors classify_reply (both keyword sets = unrecognized, never guess toward a throw; "hold" is the beat's grip word, excluded from NEGATIVE there), a NEGATIVE reply at either prompt or 3 unrecognized answers runs the new gentle `can_release` gesture (lower, open past the can at 9.3%/s, end at own start pose) instead of flinging or holding forever.

Blocked:
- **CANable dropped off USB mid-session (count 0, did not self-recover) — needs a physical reseat/other port**, then `.venv/bin/python arm/precheck.py`. Third drop today.
- Full live `--flip` run with a physical can never done (no can present during verification). Also the retimed release segment not yet replayed on hardware.
- Camera (unchanged): user's iPhone via Continuity; cv2 sees it from the USER'S Terminal only (agent TCC-blocked). Snapshot test per index → `CAMERA_INDEX=N`. Only beat4 ingest needs it; the flip beat doesn't.
- Backup video, repo public, submission — still pending from before.

Next (cold-start order):
1. Reseat CANable → `.venv/bin/python arm/precheck.py`.
2. Dry hardware pass, no can: `.venv/bin/python demo/run_demo.py --flip --text-mode --no-llm --no-audio` with `yes ''` if headless.
3. Live with the can: place it ~straight ahead of the locked base where the jaws hover (the 2.5s "center the can" pause is for exactly this), run `CAMERA_INDEX=N .venv/bin/python demo/run_demo.py --flip` from the USER'S Terminal (mic + camera TCC).
4. Record backup video; `--scripted` remains the stage fallback (flip beat is NOT in the scripted cache).
5. Repo public, upload, submit.

Gotchas: bash-guard hook blocks ANY `.env` reference in Bash commands (read keys inside Python only); never run two processes on the CAN bus; `yes '' | ... --text-mode` drives the demo headlessly; sim gesture counterparts of the can flow are straight copies of the hardware relative files; hardware gripper velocity cap is 12%/s (sim 120) — author jaw moves accordingly.
