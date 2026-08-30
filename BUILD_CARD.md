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

FILL_HEADLINE

Evidence, in the order a skeptic should check it:

1. **The montage tab.** For each task the index picks the 8-second window it scores as most
   two-handed. Nobody screened these. FILL_MONTAGE
2. **The threshold slider.** The state boundary is one global number, the 75th percentile of
   pooled hand speed. Drag it across its whole range: absolute percentages move, the
   ordering of tasks holds. A per-task threshold would have been fitting the answer.
3. **Eyeball validation.** FILL_VALIDATION
4. **The reel export.** The filtered segment list is a real artifact — clip ids and time
   ranges you can hand to a data loader.

## Limitations and failures

- **Gloves defeat the tracker.** Bottle-cleaning detection runs at FILL_GLOVE across all
  nine clips because the worker wears blue nitrile gloves. Read naively it looks like 45
  idle minutes; it is a busy sink. Any task below 50% detection is held out of every
  headline number instead of being averaged in.
  Evidence: `assets/generated/gloved-hands-detection-failure.jpg`.
- **2 fps sampling** hides events shorter than about 1.5 s.
- **"Hands absent" means the tracker found nothing**, not that hands were gone. Detection
  rate is printed beside every number for exactly this reason.
- **No ground truth.** Nothing was hand-labelled, so no accuracy figure is claimed — only
  consistency, threshold stability, and checkable picks.
- **Single-source tasks.** FILL_SINGLE of 50 tasks come from one independent recording
  family; their numbers describe that recording, not the task.
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
