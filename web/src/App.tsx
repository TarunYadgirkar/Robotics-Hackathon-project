import { useEffect, useMemo, useState } from 'react'
import {
  BIMANUAL, STATE_COLORS, STATE_NAMES, hours, loadCorpus, pct, taskStats,
  type Corpus,
} from './data'
import Wall, { type Hit } from './Wall'
import Player from './Player'
import Reel from './Reel'
import TaskPanel from './TaskPanel'
import Montage from './Montage'
import Headline from './Headline'
import Limitations from './Limitations'

type View = 'headline' | 'wall' | 'reel' | 'tasks' | 'montage' | 'limits'

const VIEWS: [View, string][] = [
  ['headline', 'Headline'],
  ['wall', 'Corpus wall'],
  ['reel', 'Reel builder'],
  ['tasks', 'Tasks'],
  ['montage', 'Montage'],
  ['limits', 'Limitations'],
]

export default function App() {
  const [corpus, setCorpus] = useState<Corpus | null>(null)
  const [err, setErr] = useState<string>()
  const [vHi, setVHi] = useState(0.43)
  const [view, setView] = useState<View>('headline')
  const [hit, setHit] = useState<Hit | null>(null)
  const [sortBy, setSortBy] = useState<'bimanual' | 'name'>('bimanual')

  useEffect(() => {
    loadCorpus()
      .then((c) => { setCorpus(c); setVHi(c.config.v_hi) })
      .catch((e) => setErr(String(e)))
  }, [])

  const stats = useMemo(() => (corpus ? taskStats(corpus, vHi) : null), [corpus, vHi])

  const order = useMemo(() => {
    if (!corpus || !stats) return []
    const ids = corpus.tasks.map((t) => t.id)
    return sortBy === 'name'
      ? ids.sort()
      : ids.sort((a, b) => (stats.get(b)?.bimanual ?? 0) - (stats.get(a)?.bimanual ?? 0))
  }, [corpus, stats, sortBy])

  if (err) return <div className="p-8 text-red-400">Failed to load: {err}</div>
  if (!corpus || !stats) return <div className="p-8 text-dim">Loading corpus…</div>

  const totals = [0, 0, 0, 0]
  for (const s of stats.values()) s.tallies.forEach((v, i) => (totals[i] += v))
  const grand = totals.reduce((a, b) => a + b, 0) || 1

  return (
    <div className="flex h-full flex-col">
      <header className="border-b border-line px-5 py-3">
        <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
          <h1 className="text-lg font-semibold tracking-tight">The Hands Index</h1>
          <p className="text-xs text-dim">
            {corpus.clips.length} clips · {corpus.tasks.length} tasks ·{' '}
            {hours(grand)} of industrial egocentric video, segmented with zero labels
          </p>
        </div>
        <nav className="mt-3 flex flex-wrap items-center gap-1">
          {VIEWS.map(([v, label]) => (
            <button
              key={v}
              onClick={() => setView(v)}
              className={`rounded px-3 py-1 text-xs transition-colors ${
                view === v ? 'bg-accent font-semibold text-ink' : 'text-dim hover:text-fg'
              }`}
            >
              {label}
            </button>
          ))}
          <div className="ml-auto flex items-center gap-3">
            <label className="flex items-center gap-2 text-xs text-dim">
              motion threshold
              <input
                type="range" min={0.1} max={1.2} step={0.01} value={vHi}
                onChange={(e) => setVHi(+e.target.value)}
                className="w-40 accent-[#ffb020]"
              />
              <span className="num w-10 text-fg">{vHi.toFixed(2)}</span>
            </label>
            <button
              onClick={() => setVHi(corpus.config.v_hi)}
              className="rounded border border-line px-2 py-0.5 text-[11px] text-dim hover:text-fg"
            >
              reset
            </button>
          </div>
        </nav>
        <div className="mt-2 flex flex-wrap gap-4 text-[11px] text-dim">
          {STATE_NAMES.map((n, i) => (
            <span key={n} className="flex items-center gap-1.5">
              <span className="h-2.5 w-4 rounded-sm" style={{ background: STATE_COLORS[i] }} />
              {n}
              <span className="num text-fg">{pct(totals[i] / grand)}</span>
            </span>
          ))}
        </div>
      </header>

      <main className="min-h-0 flex-1 overflow-hidden">
        {view === 'headline' && (
          <Headline corpus={corpus} stats={stats} order={order} vHi={vHi} onPick={setHit} />
        )}
        {view === 'wall' && (
          <div className="flex h-full">
            <div className="min-w-0 flex-1 overflow-auto p-4">
              <div className="mb-3 flex items-center gap-2 text-xs text-dim">
                sort
                {(['bimanual', 'name'] as const).map((s) => (
                  <button
                    key={s}
                    onClick={() => setSortBy(s)}
                    className={`rounded px-2 py-0.5 ${
                      sortBy === s ? 'bg-line text-fg' : 'hover:text-fg'
                    }`}
                  >
                    {s === 'bimanual' ? 'two-handed share' : 'task name'}
                  </button>
                ))}
                <span className="ml-2">click any second to play that instant</span>
              </div>
              <Wall corpus={corpus} vHi={vHi} stats={stats} order={order} onPick={setHit} />
            </div>
            <aside className="w-[420px] shrink-0 overflow-auto border-l border-line bg-panel p-4">
              {hit ? (
                <Player corpus={corpus} clipIdx={hit.clipIdx} second={hit.second} vHi={vHi} />
              ) : (
                <p className="text-sm text-dim">
                  Pick a moment on the wall. Bright amber is two-handed work.
                </p>
              )}
            </aside>
          </div>
        )}
        {view === 'reel' && <Reel corpus={corpus} vHi={vHi} state={BIMANUAL} />}
        {view === 'tasks' && <TaskPanel corpus={corpus} stats={stats} order={order} vHi={vHi} />}
        {view === 'montage' && <Montage corpus={corpus} vHi={vHi} />}
        {view === 'limits' && <Limitations corpus={corpus} vHi={vHi} />}
      </main>
    </div>
  )
}
