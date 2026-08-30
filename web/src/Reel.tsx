import { useEffect, useMemo, useRef, useState } from 'react'
import {
  STATE_COLORS, STATE_NAMES, clipStates, clock, hours, media, runsOf,
  type Corpus, type Run, type State,
} from './data'

interface Props { corpus: Corpus; vHi: number; state: State }

export default function Reel({ corpus, vHi, state: initial }: Props) {
  const [query, setQuery] = useState('stitching')
  const [minSec, setMinSec] = useState(8)
  const [state, setState] = useState<State>(initial)
  const [cursor, setCursor] = useState(0)
  const [playing, setPlaying] = useState(false)
  const video = useRef<HTMLVideoElement>(null)

  const runs = useMemo(() => {
    const q = query.trim().toLowerCase()
    const out: Run[] = []
    for (const [tid, idxs] of corpus.byTask) {
      const task = corpus.tasks.find((t) => t.id === tid)
      const hay = `${tid} ${task?.name ?? ''}`.toLowerCase()
      if (q && !hay.includes(q)) continue
      for (const i of idxs) out.push(...runsOf(clipStates(corpus, i, vHi), i, state, minSec))
    }
    return out.sort((a, b) => b.end - b.start - (a.end - a.start))
  }, [corpus, vHi, query, minSec, state])

  const total = runs.reduce((a, r) => a + (r.end - r.start), 0)
  const run = runs[Math.min(cursor, runs.length - 1)]

  useEffect(() => { setCursor(0) }, [query, minSec, state, vHi])

  useEffect(() => {
    const v = video.current
    if (!v || !run) return
    const seek = () => { v.currentTime = run.start; if (playing) void v.play() }
    if (v.readyState >= 1) seek()
    else v.addEventListener('loadedmetadata', seek, { once: true })
  }, [run?.clip, run?.start, playing])

  const advance = () => setCursor((c) => (runs.length ? (c + 1) % runs.length : 0))

  const exportJson = () => {
    const payload = {
      generated_by: 'the-hands-index',
      filter: { query, state: STATE_NAMES[state], min_seconds: minSec, motion_threshold: vHi },
      note: 'Time ranges are clip-relative seconds, derived from 2 fps hand tracking.',
      segments: runs.map((r) => ({
        clip_id: corpus.clips[r.clip].id,
        task: corpus.clips[r.clip].task,
        path: corpus.clips[r.clip].path,
        start_s: r.start,
        end_s: r.end,
      })),
    }
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }),
    )
    const a = document.createElement('a')
    a.href = url
    a.download = `hands-index-${state}-${minSec}s.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="flex h-full">
      <div className="flex min-w-0 flex-1 flex-col gap-3 p-4">
        <div className="flex flex-wrap items-center gap-3 text-xs">
          <span className="text-dim">show every</span>
          <select
            value={state}
            onChange={(e) => setState(+e.target.value as State)}
            className="rounded border border-line bg-panel px-2 py-1"
          >
            {STATE_NAMES.map((n, i) => (
              <option key={n} value={i}>{n}</option>
            ))}
          </select>
          <span className="text-dim">run longer than</span>
          <input
            type="range" min={2} max={30} value={minSec}
            onChange={(e) => setMinSec(+e.target.value)}
            className="w-28 accent-[#ffb020]"
          />
          <span className="num w-8">{minSec}s</span>
          <span className="text-dim">in tasks matching</span>
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="all tasks"
            className="w-40 rounded border border-line bg-panel px-2 py-1"
          />
          <button
            onClick={exportJson}
            className="ml-auto rounded bg-accent px-3 py-1 font-semibold text-ink"
          >
            Export {runs.length} segments
          </button>
        </div>

        <p className="text-xs text-dim">
          <span className="num text-fg">{runs.length}</span> segments ·{' '}
          <span className="num text-fg">{hours(total)}</span> of footage, assembled from{' '}
          {new Set(runs.map((r) => corpus.clips[r.clip].task)).size} tasks with zero labels.
        </p>

        {run ? (
          <>
            <video
              ref={video}
              key={corpus.clips[run.clip].id}
              src={media(corpus.clips[run.clip].path)}
              poster={media(corpus.clips[run.clip].thumb)}
              controls autoPlay muted playsInline
              onPlay={() => setPlaying(true)}
              onTimeUpdate={(e) => {
                if (e.currentTarget.currentTime >= run.end) advance()
              }}
              className="w-full max-h-[52vh] rounded bg-black"
            />
            <div className="flex items-center gap-3 text-xs">
              <button onClick={advance} className="rounded border border-line px-2 py-1 hover:bg-panel">
                next ›
              </button>
              <span
                className="num rounded px-2 py-0.5 font-semibold text-ink"
                style={{ background: STATE_COLORS[state] }}
              >
                {run.end - run.start}s
              </span>
              <span className="num text-dim">
                {clock(run.start)}–{clock(run.end)}
              </span>
              <span>{corpus.tasks.find((t) => t.id === corpus.clips[run.clip].task)?.name}</span>
              <span className="num text-dim">
                {cursor + 1}/{runs.length}
              </span>
            </div>
          </>
        ) : (
          <p className="text-sm text-dim">No segments match. Loosen the filter.</p>
        )}
      </div>

      <aside className="w-72 shrink-0 overflow-auto border-l border-line bg-panel">
        {runs.slice(0, 400).map((r, i) => (
          <button
            key={`${r.clip}-${r.start}`}
            onClick={() => setCursor(i)}
            className={`flex w-full items-center gap-2 border-b border-line/60 px-3 py-1.5 text-left text-xs ${
              i === cursor ? 'bg-line' : 'hover:bg-line/40'
            }`}
          >
            <span className="num w-8 text-accent">{r.end - r.start}s</span>
            <span className="flex-1 truncate">
              {corpus.tasks.find((t) => t.id === corpus.clips[r.clip].task)?.name}
            </span>
            <span className="num text-dim">{clock(r.start)}</span>
          </button>
        ))}
      </aside>
    </div>
  )
}
