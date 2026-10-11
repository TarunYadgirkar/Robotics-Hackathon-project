import { useEffect, useMemo, useState } from 'react'
import {
  BIMANUAL, STATE_COLORS, STATE_NAMES, hours, loadCorpus, measurableTotals, mediaReachable,
  pct, taskStats, type Corpus,
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
  ['headline', 'Finding'],
  ['wall', 'Every clip'],
  ['reel', 'Reel builder'],
  ['tasks', 'Tasks'],
  ['montage', 'Montage'],
  ['limits', 'Limitations'],
]

const LEGEND_ORDER = [3, 2, 1, 0] as const

export default function App() {
  const [corpus, setCorpus] = useState<Corpus | null>(null)
  const [err, setErr] = useState<string>()
  const [vHi, setVHi] = useState(0.6)
  const [view, setView] = useState<View>('headline')
  const [hit, setHit] = useState<Hit | null>(null)
  const [sortBy, setSortBy] = useState<'bimanual' | 'name'>('bimanual')
  const [media, setMedia] = useState<boolean | null>(null)

  useEffect(() => {
    void mediaReachable().then(setMedia)
    loadCorpus()
      .then((c) => { setCorpus(c); setVHi(c.config.v_hi) })
      .catch((e: unknown) => setErr(String(e)))
  }, [])

  const stats = useMemo(() => (corpus ? taskStats(corpus, vHi) : null), [corpus, vHi])

  const order = useMemo(() => {
    if (!corpus || !stats) return []
    const ids = corpus.tasks.map((t) => t.id)
    return sortBy === 'name'
      ? ids.sort()
      : ids.sort((a, b) => (stats.get(b)?.bimanual ?? 0) - (stats.get(a)?.bimanual ?? 0))
  }, [corpus, stats, sortBy])

  if (err) {
    return (
      <p className="p-8 text-critical">
        The index data didn't load ({err}). Run <span className="num">pnpm build</span> again
        or check that <span className="num">public/data</span> is present.
      </p>
    )
  }
  if (!corpus || !stats) return <p className="p-8 text-dim">Loading the index…</p>

  const { totals, grand } = measurableTotals(corpus, stats)
  const collected = corpus.clips.length * corpus.config.clip_seconds

  return (
    <div className="flex min-h-full flex-col lg:h-full">
      <header className="flex flex-col gap-3 bg-panel px-4 pt-3 sm:px-6">
        <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
          <h1 className="font-display text-2xl font-extrabold tracking-wide uppercase">
            The Hands Index
          </h1>
          <p className="text-xs text-dim">
            <span className="num text-fg">{corpus.clips.length}</span> clips across{' '}
            <span className="num text-fg">{corpus.tasks.length}</span> tasks,{' '}
            <span className="num text-fg">{hours(collected)}</span> of industrial egocentric
            video, segmented with zero labels
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
          <label className="flex items-center gap-2 text-xs text-dim">
            <span title="Hand speed above this counts as transit between objects">
              Motion threshold
            </span>
            <input
              type="range" min={0.1} max={1.2} step={0.01} value={vHi}
              onChange={(e) => setVHi(+e.target.value)}
              className="w-32 sm:w-40"
            />
            <span className="num w-10 text-fg">{vHi.toFixed(2)}</span>
          </label>
          <button
            onClick={() => setVHi(corpus.config.v_hi)}
            disabled={vHi === corpus.config.v_hi}
            className="rounded bg-raised px-2 py-1 text-xs text-fg hover:bg-rule disabled:invisible"
          >
            Reset to {corpus.config.v_hi.toFixed(2)}
          </button>
          <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-dim" aria-label="Share of measurable time">
            {LEGEND_ORDER.map((i) => (
              <li key={i} className="flex items-center gap-1.5">
                <span className="h-2.5 w-4 rounded-sm" style={{ background: STATE_COLORS[i] }} />
                {STATE_NAMES[i]}
                <span className="num text-fg">{pct(totals[i] / grand)}</span>
              </li>
            ))}
          </ul>
        </div>

        <nav aria-label="Views" className="-mx-4 flex overflow-x-auto px-4 sm:-mx-6 sm:px-6">
          {VIEWS.map(([v, label]) => (
            <button
              key={v}
              onClick={() => setView(v)}
              aria-current={view === v ? 'page' : undefined}
              className={`shrink-0 border-b-2 px-3 py-2 text-sm transition-colors ${
                view === v
                  ? 'border-accent font-medium text-fg'
                  : 'border-transparent text-dim hover:text-fg'
              }`}
            >
              {label}
            </button>
          ))}
        </nav>
      </header>

      <main className="flex-1 lg:min-h-0 lg:overflow-hidden">
        {view === 'headline' && (
          <Headline
            corpus={corpus} stats={stats} order={order}
            totals={totals} grand={grand} collected={collected}
          />
        )}
        {view === 'wall' && (
          <div className="flex flex-col lg:h-full lg:flex-row">
            <aside className="bg-panel p-4 lg:order-2 lg:w-[420px] lg:shrink-0 lg:overflow-auto">
              {hit ? (
                <Player
                  corpus={corpus} clipIdx={hit.clipIdx} second={hit.second} vHi={vHi}
                  media={media !== false}
                />
              ) : (
                <p className="text-sm leading-relaxed text-dim">
                  Each row is one 5-minute clip, coloured second by second. Pick any
                  moment to see what the index called it
                  {media === false ? ' and a bundled sample from that task.' : ' and play it.'}
                </p>
              )}
            </aside>
            <section className="min-w-0 flex-1 p-4 lg:overflow-auto" aria-label="Every clip">
              <div className="mb-3 flex flex-wrap items-center gap-2 text-xs text-dim">
                Sort by
                {(['bimanual', 'name'] as const).map((s) => (
                  <button
                    key={s}
                    onClick={() => setSortBy(s)}
                    aria-pressed={sortBy === s}
                    className={`rounded px-2 py-1 ${
                      sortBy === s ? 'bg-raised text-fg' : 'hover:text-fg'
                    }`}
                  >
                    {s === 'bimanual' ? 'Two-handed share' : 'Task name'}
                  </button>
                ))}
              </div>
              <div className="overflow-x-auto">
                <Wall corpus={corpus} vHi={vHi} stats={stats} order={order} onPick={setHit} />
              </div>
            </section>
          </div>
        )}
        {view === 'reel' && (
          <Reel corpus={corpus} vHi={vHi} state={BIMANUAL} media={media !== false} />
        )}
        {view === 'tasks' && <TaskPanel corpus={corpus} stats={stats} order={order} vHi={vHi} />}
        {view === 'montage' && <Montage corpus={corpus} vHi={vHi} />}
        {view === 'limits' && <Limitations corpus={corpus} vHi={vHi} />}
      </main>
    </div>
  )
}
