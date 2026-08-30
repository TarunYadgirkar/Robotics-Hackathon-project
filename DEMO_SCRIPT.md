# Demo — 3 minutes

Setup before recording: media server on 8765, `pnpm dev` on 5173, browser full screen,
Headline tab open, threshold at its default. Have the Reel tab pre-warmed once so video is
cached.

**0:00–0:25 — the claim.**
"This is 35 hours of factory video from 50 industrial tasks. The camera is on the worker's
chest, not their head — which means image coordinates are body-relative hand position, for
free, with no calibration. We tracked both hands through every frame of all 424 clips and
labelled every second of the corpus. Here's what it says: only 43.1% of it is two-handed
manipulation." *(point at the headline number)*

**0:25–0:50 — why it matters.**
"If you're training an imitation policy on this, most of what it sees isn't manipulation —
it's transit, one-handed handling, and stretches where the hands aren't in frame at all. The
teachable part is 13.2 hours, not 35. No labels, no training, one threshold for all 50 tasks."

**0:50–1:20 — the wall.** *(Corpus wall tab)*
"Every clip in the corpus, one second per pixel, sorted by how hands-on it is. Bright amber
is two-handed work." *(click into a bright run)* "Click any second and you're in the video at
that instant — full resolution, byte-range seek." *(let it play 3 seconds)*

**1:20–1:50 — the montage.** *(Montage tab)*
"The index picked these. For each task, the eight seconds it scores as most two-handed.
Nobody screened them first — 25 of 25 score a perfect 8 out of 8. That's the check: don't
believe the chart, look at the footage it chose."

**1:50–2:20 — the reel and the export.** *(Reel tab)*
"Now use it: every two-handed run longer than eight seconds, in stitching tasks." *(let it
play)* "the segments. Export gives you clip ids and time ranges — the subset you'd actually
train on."

**2:20–2:50 — honesty.** *(Limitations tab)*
"This fails, and we show where. Bottle cleaning reads as an idle worker. It isn't — he's
wearing blue nitrile gloves and MediaPipe won't fire on them. Detection is 3.6%. Any task
under 50% detection is held out of every number rather than averaged in. And the ordering
holds: sweep the threshold across its whole range and rank correlation never drops below
0.90 across 0.35 to 1.00."

**2:50–3:00 — close.**
"Two IMU approaches died before this one — we've written up both. Everything here was built
today on top of the dataset's own server and MediaPipe. Repo and build card are linked."

## Numbers (full corpus, current build)

43.1% two-handed · 13.2 h teachable of 35.3 h collected / 30.5 h measurable · 6.2× spread ·
top Oil Seal Pressing 77.9%, bottom Water Filtration Bottle Filling 12.6% · montage 25/25 at
8/8 · validation 54/60 = 90% · stability ρ ≥ 0.90 over 0.35-1.00 · 7 tasks excluded.

Say "the ordering holds between 0.35 and 1.00" — not "the ordering always holds". Below 0.35
it does not, and a judge who drags the slider to the far left will see that.
