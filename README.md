# The Hands Index

An actionable explorer over 35 hours of industrial egocentric video, built for the
Berkeley Robotics Hackathon visualization track. Live at https://hands-index.vercel.app

> This is the `hands-index` branch. The repo is shared with a teammate's YAM arm-control
> project, which lives on `main`; `yam/`, `scripts/` and `web/enroll/` here are theirs.

**The claim:** most of a demonstration corpus is not demonstration. We track both hands
across every clip, collapse each second into one of four states, and show what fraction of
the clock is actual two-handed manipulation versus transit, one-handed handling, and hands
out of frame. No labels, no training, one threshold for all 50 tasks.

The camera in this dataset is **torso-mounted**, not head-mounted. That single property is
what makes this cheap: image coordinates are body-relative hand position with zero
calibration, so a 2D hand tracker gives you a workspace envelope and a grip aperture for
free.

## Run it

Two servers. The first serves media straight off the dataset drive using the launcher that
ships with the package — full HTTP range support, so seeking into a 5-minute 1080p clip is
instant and nothing is copied.

```bash
python3 ~/TarunsCode/ds-hack/launch/serve.py --root /Volumes/DS23/DATASET_V3 --port 8765
```

```bash
cd web && pnpm install && pnpm dev
```

Open http://localhost:5173. Point the explorer somewhere else — an S3 bucket, a CDN — by
setting `VITE_MEDIA_BASE`; it defaults to `http://127.0.0.1:8765`. The index itself is
static data, so the app works without the media server; only playback needs it.

## Rebuild the index

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python mediapipe==0.10.21 numpy pyarrow pillow
.venv/bin/python pipeline/extract.py --all --workers 5   # ~35 min for 424 clips
.venv/bin/python pipeline/segment.py                     # per-second payload for the web app
.venv/bin/python pipeline/validate.py --n 60             # contact sheets to check the index by eye
.venv/bin/python pipeline/narrate.py                     # spoken summaries, generated from the index
```

`extract.py` is resumable: it skips any clip whose Parquet already exists and opens, so a
crash costs one clip, never the run. Set `WC_DATA` / `DS_VIDEOS` if your copies live
elsewhere.

**Pin `mediapipe==0.10.21`.** Version 1.0.1 removes `mp.solutions` entirely and crashes on
macOS arm64 with a Metal service error. It also will not install on Python 3.14 — hence the
3.12 venv.

## How it works

| Stage | What it does |
| --- | --- |
| `extract.py` | ffmpeg pipes rawvideo at 2 fps / 640px straight into numpy; MediaPipe Hands returns 21 landmarks per hand; writes one Parquet per clip with landmarks, palm centroid, grip aperture, finite-difference speed, plus gyro RMS and torso lean from the IMU sidecar. |
| `segment.py` | Reduces to per-second observations and packs the whole corpus into ~250 KB: one byte of hand count and one byte of quantised speed per second. |
| the browser | Classifies states from those raw bytes at render time, which is why the motion threshold is a live slider rather than a baked-in decision. |

States: **hands absent**, **transit** (a hand moving faster than the threshold),
**one-handed**, **two-handed**. A median-of-3 filter over seconds removes single-second
flicker.

## Honesty

The app has a Limitations tab and it is not decorative. The short version: 2 fps sampling
hides sub-1.5-second events; "hands absent" means the tracker found nothing, not that the
hands were gone; tasks where detection falls below 50% are held out of every headline
number rather than averaged in (gloves defeat MediaPipe — see
`assets/generated/gloved-hands-detection-failure.jpg`); many tasks come from a single
recording family, so their numbers describe that recording; and there is no ground truth
here, so no accuracy is claimed.

See [BUILD_CARD.md](BUILD_CARD.md) for what existed before this event and what did not.
