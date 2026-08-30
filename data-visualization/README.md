# Factory Atlas data visualization

This directory contains the current one-video proof for exploring an existing
factory POV video with precomputed hand landmarks and a relative-depth 3D
representation.

## Run the site

```bash
cd site
npm install
npm run dev
```

Open the local URL printed by the development server. Use `npm run build` to
verify a production build.

## Rebuild the one-video artifacts

The Python proof script is in `worker/process_one_video.py`. Its dependencies
are listed in `worker/requirements.txt`. The script expects a local source clip
and a separately downloaded MediaPipe Hand Landmarker model; large model files,
raw footage, Python environments, credentials, and HOPformer/MANO assets are
intentionally excluded from Git.

See `PROJECT_INSTRUCTIONS.md` for the broader S3 catalog, AWS batch processing,
and browser-based 3D exploration plan.
