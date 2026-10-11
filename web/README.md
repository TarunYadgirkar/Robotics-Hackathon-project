# The Hands Index: web explorer

```bash
pnpm install
pnpm dev     # http://localhost:5173
pnpm build   # typecheck + production build into dist/
pnpm lint
```

The index ships as static data in `public/data/`, so every view works with no server and no keys.
Full-resolution playback reads the dataset drive through the media server described in the
repo README. A local dev build probes `http://127.0.0.1:8765`; a deployed build only probes when
`VITE_MEDIA_BASE` is set. Without a media server the player falls back to the bundled
8-second sample for that task.

Theme tokens (colors, state palette, fonts) live in `src/index.css`.
