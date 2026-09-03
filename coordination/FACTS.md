REPO_ROOT: /Users/tarunyadgirkar/TarunsCode/hackathons/Robotics-Hackathon-project
GIT_BRANCH: idk-demo
DATASET_ROOT: /Users/tarunyadgirkar/TarunsCode/ds-hack
DATASET_VERSION: 3.1.1
DS23_VOLUME_MOUNTED: no
DS_VIDEOS_ENV_REQUIRED: DS_VIDEOS=/Users/tarunyadgirkar/TarunsCode/ds-hack
WC_DATA_ENV_DEFAULT_OK: yes (pipeline/dsdata.py defaults WC_DATA to ~/TarunsCode/ds-hack already)
RAW_VIDEO_FILES_LOCAL: absent (no videos/ dir under ds-hack; do not attempt re-extraction, all 424 clips already extracted)
SERVE_PY_PRESENT: yes
SERVE_PY_COMPILES: yes
PARQUET_OUTPUT_DIR: work/frames/<task_id>/<clip_id>.parquet (relative to REPO_ROOT; dsdata.WORK defaults to REPO/work)
PARQUET_COUNT: 424
PARQUET_EXPECTED: 424
PARQUET_TASK_DIRS: 50
PARQUET_MISSING_TASKS: none
HARDWARE_PRESENT: yes
HARDWARE_DEVICE_PATH: gs_usb:0 (CANable2, USB VID 0x1D50 PID 0x606F, serial 0035005B594E501820313332, bus 1 addr 1; CAN clock 170 MHz; not a /dev/tty or /dev/cu node — gs_usb is a raw-USB CAN adapter, so the old "no serial device" probe was looking for the wrong thing)
HARDWARE_AMENDED_AT: 15:12 PDT 2026-08-30 — amended by orchestrator instruction at hardware arrival; original P0 value was HARDWARE_PRESENT: no. Verified by Agent C with pyusb (1 device matching 0x1D50:0x606F) and gs_usb.GsUsb.scan() (1 adapter). Everything else in this file is P0's, untouched.
HARDWARE_DRIVER: yam.arm.YamArm (in-repo, Boris) over python-can gs_usb. NOT lerobot.
LEROBOT_IMPORT: fail (ModuleNotFoundError: No module named 'lerobot') — irrelevant now; this arm is the YAM, driven by yam/arm.py
PYTHON_CAN_INSTALLED: yes (4.6.1, installed by Agent C at hardware arrival; gs_usb + pyusb were already present)
VENV_PATH: .venv/bin/python -> /Users/tarunyadgirkar/.local/share/uv/python/cpython-3.12-macos-aarch64-none/bin/python3.12
MEDIAPIPE_VERSION: 0.10.21
MEDIAPIPE_OK: yes (matches required pin)
CV2_INSTALLED: yes (4.11.0, was already present, no install needed)
WEBCAM_ISOPENED: false
WEBCAM_STATUS: blocked (OpenCV: "not authorized to capture video" — macOS camera permission not granted to this terminal/process; NOT a hardware-absence signal, likely fixable via System Settings > Privacy & Security > Camera)
SOUNDDEVICE_INSTALLED: yes (0.5.6, was already present, no install needed)
MIC_DEFAULT_DEVICE_INDEX: 0
MIC_DEFAULT_DEVICE_NAME: MacBook Pro Microphone
MIC_OTHER_INPUT_DEVICES: index 2, name "... Microphone" (Manufacturer Apple, Transport Unknown — likely a Continuity/virtual device with a corrupted display name, NOT a confirmed external/headset mic)
EXTERNAL_HEADSET_MIC_PRESENT: no (only built-in array + one ambiguous virtual device confirmed)
ELEVENLABS_API_KEY: absent (macOS say fallback) [env | grep -c ELEVEN -> 0]
PROBE_TIME: 12:29:05 PDT (2026-08-30)
MINUTES_TO_1645: ~256
MINUTES_TO_1730: ~301
INSTALLS_USED: 0 of 2 allowed (opencv-python and sounddevice were both already installed in .venv)

---

## Verified dataset facts (copied verbatim from PLAN_v2, dataset root corrected)

Dataset root: `/Users/tarunyadgirkar/TarunsCode/ds-hack`

`meta/tasks.jsonl` fields (confirmed present): `canonical_task_id`, `display_name`,
`aliases`, `clip_count`, `independent_repetition_count`, `camera_count`,
`collection_group_count`, `coverage_camera_ids`, `coverage_collection_groups`,
`coverage_viewpoints`, `diversity_warning`, `sequence_count`.

`meta/clips.jsonl` fields (confirmed present): `clip_id`, `canonical_task_id`,
`sequence_id`, `independent_repetition_id`, `camera_id`, `collection_group_id`,
`viewpoint`, `duration_s`, `frames`, `fps`, `relative_path`, `thumbnail_path`,
`imu_path`, `imu_rate_hz`, `quality_status`, `quality_flags`, `calibration_status`.

Diversity reality: **27/50 tasks carry a `diversity_warning`** and are confounded by
construction. 9/50 have only one recording family. **21/50 are clean** (≥2 repetition
families AND ≥2 cameras AND ≥8 clips AND no warning). BEAT 2 must draw from these:

| task | clips | families | cameras |
| --- | --- | --- | --- |
| garment-folding-cardboard-insert | 9 | 6 | 6 |
| garment-iron-press | 8 | 6 | 5 |
| belly-band-assembly | 9 | 5 | 3 |
| garment-back-panel-attachment | 9 | 5 | 4 |
| garment-carton-packing | 9 | 5 | 4 |
| garment-inside-out | 9 | 5 | 4 |
| fabric-cutting-machine | 8 | 4 | 4 |
| fabric-layering | 9 | 4 | 4 |
| garment-button-attachment | 9 | 4 | 4 |
| garment-loop-attachment | 9 | 4 | 4 |
| garment-quality-checking | 9 | 4 | 4 |
| garment-belly-band-wrapping | 9 | 4 | 3 |

IMU dead ends (already tested, do not retry): gyro autocorrelation gives no work-cycle
periodicity (median peak 0.16, 2/50 above 0.30); no machine-vibration tone exists
(whitened SNR ≈2.1 on all 50 tasks — the noise floor) because GPMF timestamps are
reconstructed by spreading samples uniformly within ~1s packets. IMU is usable for gross
low-frequency motion only.

---

## Prose notes (ambiguous items)

1. **Dataset volume**: `/Volumes/DS23` is not mounted (`ls /Volumes` shows only
   `Macintosh HD`). The orchestrator-provided local copy at
   `/Users/tarunyadgirkar/TarunsCode/ds-hack` is verified: VERSION file reads `3.1.1`,
   `meta/clips.jsonl` has 424 lines, `meta/tasks.jsonl` has 50 lines, `launch/serve.py`
   exists (20612 bytes) and compiles cleanly with `.venv/bin/python -m py_compile`.
   All agents MUST run with `DS_VIDEOS=/Users/tarunyadgirkar/TarunsCode/ds-hack` — note
   that `pipeline/dsdata.py`'s `WC_DATA` env var already defaults to
   `~/TarunsCode/ds-hack`, so only `DS_VIDEOS` needs an explicit override; `DS_VIDEOS`
   itself defaults to the unmounted `/Volumes/DS23/...` path and will not resolve without
   the override.

2. **Raw video files are NOT present locally.** There is no `videos/` directory anywhere
   under `ds-hack` (checked `find . -iname videos` — no results, and a sample clip's
   `relative_path` resolves to a nonexistent file). This does not block anything right now
   because extraction is already 100% done (424/424 parquets present, verified below), but
   it means **no agent can re-run `pipeline/extract.py` on any clip** — if a parquet is
   ever deleted or corrupted it cannot be regenerated in this environment. IMU JSONs and
   thumbnails ARE present locally (`imu/`, `thumbnails/` each have ~426 files).

3. **Extraction is already complete.** `work/frames/<task_id>/<clip_id>.parquet` (relative
   to REPO_ROOT; the path convention comes from `dsdata.frames_path()`, `WORK` defaults to
   `REPO_ROOT/work`) contains exactly 424 `.parquet` files across 50 task subdirectories —
   full match against the expected 424 clips / 50 tasks. No task_ids are incomplete. Agent
   A can proceed immediately without waiting on any extraction step.

4. **Arm hardware: HARDWARE_PRESENT = no.** No USB serial device is present
   (`ls /dev/tty.* /dev/cu.*` only shows Bluetooth-Incoming-Port and debug-console — no
   `usbmodem`/`usbserial` device that would indicate an SO-10x controller). `lerobot` is
   not installed in the venv (`ModuleNotFoundError`). Agent C should build the simulator
   fallback per its prompt and disclose "sim fallback" in its status/build card.

5. **Webcam: status is BLOCKED, not confirmed-absent.** `cv2.VideoCapture(0).isOpened()`
   returned `False` with the message `OpenCV: not authorized to capture video (status 0),
   requesting...`. This is a macOS TCC camera-permission prompt/denial for the process
   running this shell, not evidence of missing hardware — this MacBook Pro certainly has a
   built-in camera. Whoever runs Agent E's webcam capture (or the live demo) needs to grant
   Camera permission to the terminal app (or whatever process launches `feedback/ingest.py`)
   in System Settings > Privacy & Security > Camera, then re-test. Recorded as-is per the
   "do not fix, record what IS" rule — this is a real risk for Agent E and Beat 4 that the
   orchestrator/user should resolve in-session before the feedback-ingest rehearsal.

6. **Microphone: no confirmed external/headset mic.** `sounddevice.query_devices()` shows
   only two real input-capable devices: index 0 "MacBook Pro Microphone" (built-in array,
   default) and index 2 "… Microphone" (Manufacturer Apple, Transport "Unknown" — name is
   truncated/corrupted in both `sounddevice` and `system_profiler SPAudioDataType` output,
   so its real identity is unconfirmed; it may be a Continuity/virtual device rather than a
   physical headset). No unambiguous external/headset mic was found. Per PLAN_v2's known
   risk ("room noise defeats STT on stage"), the operator should plan on `--text` fallback
   or bring/pair an external mic before the live demo — the built-in array is the only
   confirmed input path right now.

7. **ELEVENLABS_API_KEY is absent** (`env | grep -c ELEVEN` → `0`). Agent D must default to
   the macOS `say` fallback for TTS; do not attempt to read a `.env` file to source it (that
   path is hook-blocked in this environment for the orchestrator, and API keys are out of
   scope for a probe agent regardless).

8. **Timing**: probe ran at 12:29:05 PDT on 2026-08-30. ~256 minutes remain to the 16:45
   target, ~301 minutes remain to the 17:30 hard stop — comfortably on schedule at T+0.
