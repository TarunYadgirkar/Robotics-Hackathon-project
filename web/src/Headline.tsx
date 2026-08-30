import { useMemo } from 'react'
import {
  BIMANUAL, STATE_COLORS, hours, measurable, pct, stability,
  type Corpus, type TaskStats,
} from './data'
import type { Hit } from './Wall'

interface Props {
  corpus: Corpus
  stats: Map<string, TaskStats>
  order: string[]
  vHi: number
  onPick: (hit: Hit) => void
}

export default function Headline({ corpus, stats, order }: Props) {
  const ranked = order.map((id) => ({
    task: corpus.tasks.find((t) => t.id === id)!,
    st: stats.get(id)!,
  }))
  const ok = ranked.filter((r) => measurable(r.task))
  const failed = ranked.filter((r) => !measurable(r.task))

  const totals = [0, 0, 0, 0]
  for (const r of ok) r.st.tallies.forEach((v, i) => (totals[i] += v))
  const grand = totals.reduce((a, b) => a + b, 0) || 1

  const stab = useMemo(() => stability(corpus, corpus.config.v_hi), [corpus])
  const top = ok[0], bottom = ok[ok.length - 1]
  const spread = top && bottom && bottom.st.bimanual > 0
    ? top.st.bimanual / bottom.st.bimanual : 0

  return (
    <div className="h-full overflow-auto px-6 py-6">
      <div className="mx-auto max-w-5xl">
        <p className="text-xs uppercase tracking-widest text-dim">The finding</p>
        <h2 className="mt-2 text-3xl leading-tight font-semibold tracking-tight">
          Only <span className="num text-accent">{pct(totals[BIMANUAL] / grand)}</span> of{' '}
          {hours(grand)} of factory work is two-handed manipulation.
        </h2>
        <p className="mt-3 max-w-3xl text-sm leading-relaxed text-dim">
          The rest is one-handed handling ({pct(totals[2] / grand)}), transit between objects
          ({pct(totals[1] / grand)}), and stretches with no hands in frame ({pct(totals[0] / grand)})
          — walking, waiting, watching a machine run. Train an imitation policy on raw clips and
          most of what it sees is not manipulation. The teachable part of this corpus is{' '}
          <span className="num text-fg">{hours(totals[BIMANUAL])}</span>, not{' '}
          <span className="num">{hours(grand)}</span>. Every second was labelled by tracking both
          hands — no annotations, no training, one threshold for all {corpus.tasks.length} tasks.
        </p>

        <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-4">
          {[
            ['Teachable footage', hours(totals[BIMANUAL])],
            ['Spread across tasks', spread ? `${spread.toFixed(1)}×` : '—'],
            ['Most hands-on', top ? pct(top.st.bimanual) : '—'],
            ['Least hands-on', bottom ? pct(bottom.st.bimanual) : '—'],
          ].map(([k, v]) => (
            <div key={k} className="rounded border border-line bg-panel px-3 py-2">
              <div className="num text-xl text-accent">{v}</div>
              <div className="text-[11px] text-dim">{k}</div>
            </div>
          ))}
        </div>
        {top && bottom && (
          <p className="mt-2 text-[11px] text-dim">
            Most hands-on: {top.task.name}. Least: {bottom.task.name}. The ordering, not the
            absolute number, is the claim — drag the motion threshold and watch it hold.
          </p>
        )}

        <div className="mt-4 flex flex-wrap items-center gap-4 rounded border border-line bg-panel px-4 py-3">
          <div>
            <div className="num text-xl text-accent">ρ ≥ {stab.worst.toFixed(2)}</div>
            <div className="text-[11px] text-dim">ordering stability</div>
          </div>
          <svg viewBox="0 0 220 44" className="h-11 w-56" role="img" aria-label="rank correlation against threshold">
            <line x1={0} y1={40} x2={220} y2={40} stroke="#2a2f38" />
            <polyline
              fill="none" stroke="#ffb020" strokeWidth={1.5}
              points={stab.rhos
                .map((r, i) => `${(i / (stab.rhos.length - 1)) * 220},${40 - Math.max(0, (r - 0.5) / 0.5) * 36}`)
                .join(' ')}
            />
          </svg>
          <p className="max-w-md text-[11px] leading-relaxed text-dim">
            Sweeping the motion threshold across its whole range — {stab.thresholds[0]} to{' '}
            {stab.thresholds[stab.thresholds.length - 1]} — the rank ordering of tasks never
            falls below Spearman ρ = {stab.worst.toFixed(2)} against the default. The
            percentages move; who is at the top does not. That is the part we are willing to
            defend.
          </p>
        </div>

        {failed.length > 0 && (
          <div className="mt-4 rounded border border-red-900/60 bg-red-950/20 px-4 py-3 text-xs">
            <span className="font-semibold text-red-300">
              {failed.length} task{failed.length > 1 ? 's' : ''} excluded: the tracker fails there,
              so the index has nothing to say about them.
            </span>{' '}
            <span className="text-dim">
              {failed.map((r) => `${r.task.name} (${pct(r.task.det1)} detection)`).join(', ')}.
              Gloves are the usual cause — MediaPipe's hand model does not fire on a blue nitrile
              glove. Counting those seconds as "hands absent" would have made idle-looking tasks
              out of busy ones, so they are held out of every number above.
            </span>
          </div>
        )}

        <table className="mt-6 w-full text-xs">
          <thead className="text-dim">
            <tr className="border-b border-line">
              <th className="py-1.5 text-left font-normal">Task</th>
              <th className="w-[46%] text-left font-normal">Where the clock goes</th>
              <th className="text-right font-normal">Two-handed</th>
              <th className="text-right font-normal">Detection</th>
              <th className="text-right font-normal">Sources</th>
            </tr>
          </thead>
          <tbody>
            {[...ok, ...failed].map(({ task, st }) => {
              const bad = !measurable(task)
              return (
                <tr
                  key={task.id}
                  className={`border-b border-line/60 hover:bg-panel ${bad ? 'opacity-45' : ''}`}
                >
                  <td className="py-1 pr-3">
                    {task.name}
                    {bad && <span className="ml-2 text-[10px] text-red-400">not measurable</span>}
                  </td>
                  <td>
                    <div className="flex h-3 w-full overflow-hidden rounded-sm">
                      {[3, 2, 1, 0].map((s) => (
                        <div
                          key={s}
                          style={{
                            width: `${(st.tallies[s] / st.seconds) * 100}%`,
                            background: STATE_COLORS[s],
                          }}
                        />
                      ))}
                    </div>
                  </td>
                  <td className="num pl-3 text-right text-accent">
                    {bad ? '—' : pct(st.bimanual)}
                  </td>
                  <td className={`num pl-3 text-right ${bad ? 'text-red-400' : 'text-dim'}`}>
                    {pct(task.det1)}
                  </td>
                  <td className="num pl-3 text-right text-dim">{task.reps}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
        <p className="mt-3 text-[11px] leading-relaxed text-dim">
          Detection is the share of sampled frames where at least one hand was found. Sources counts
          independent recording families; where it is 1, the number describes that recording, not
          the task in general.
        </p>
      </div>
    </div>
  )
}
