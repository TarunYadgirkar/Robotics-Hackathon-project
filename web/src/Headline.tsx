import { BIMANUAL, STATE_COLORS, hours, pct, type Corpus, type TaskStats } from './data'
import type { Hit } from './Wall'

interface Props {
  corpus: Corpus
  stats: Map<string, TaskStats>
  order: string[]
  vHi: number
  onPick: (hit: Hit) => void
}

export default function Headline({ corpus, stats, order }: Props) {
  const totals = [0, 0, 0, 0]
  for (const s of stats.values()) s.tallies.forEach((v, i) => (totals[i] += v))
  const grand = totals.reduce((a, b) => a + b, 0) || 1
  const ranked = order.map((id) => ({
    task: corpus.tasks.find((t) => t.id === id)!,
    st: stats.get(id)!,
  }))
  const top = ranked[0], bottom = ranked[ranked.length - 1]
  const spread = bottom.st.bimanual > 0 ? top.st.bimanual / bottom.st.bimanual : 0

  return (
    <div className="h-full overflow-auto px-6 py-6">
      <div className="mx-auto max-w-5xl">
        <p className="text-xs uppercase tracking-widest text-dim">The finding</p>
        <h2 className="mt-2 text-3xl leading-tight font-semibold tracking-tight">
          Only{' '}
          <span className="num text-accent">{pct(totals[BIMANUAL] / grand)}</span>{' '}
          of {hours(grand)} of factory work is two-handed manipulation.
        </h2>
        <p className="mt-3 max-w-3xl text-sm leading-relaxed text-dim">
          The rest is one-handed handling ({pct(totals[2] / grand)}), transit between
          objects ({pct(totals[1] / grand)}), and stretches with no hands in frame
          ({pct(totals[0] / grand)}) — walking, waiting, watching a machine run. Train an
          imitation policy on raw clips and most of what it sees is not manipulation.
          The teachable part of this corpus is{' '}
          <span className="num text-fg">{hours(totals[BIMANUAL])}</span>, not{' '}
          <span className="num">{hours(grand)}</span>. Every second below was labelled by
          tracking both hands — no annotations, no training, one threshold for all 50 tasks.
        </p>

        <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-4">
          {[
            ['Teachable footage', hours(totals[BIMANUAL])],
            ['Spread across tasks', `${spread.toFixed(1)}×`],
            ['Most hands-on', `${pct(top.st.bimanual)}`],
            ['Least hands-on', `${pct(bottom.st.bimanual)}`],
          ].map(([k, v]) => (
            <div key={k} className="rounded border border-line bg-panel px-3 py-2">
              <div className="num text-xl text-accent">{v}</div>
              <div className="text-[11px] text-dim">{k}</div>
            </div>
          ))}
        </div>
        <p className="mt-2 text-[11px] text-dim">
          Most hands-on: {top.task.name}. Least: {bottom.task.name}. The ordering, not the
          absolute number, is the claim — drag the motion threshold and watch it hold.
        </p>

        <table className="mt-6 w-full text-xs">
          <thead className="text-dim">
            <tr className="border-b border-line">
              <th className="py-1.5 text-left font-normal">Task</th>
              <th className="w-[46%] text-left font-normal">
                Where the clock goes
              </th>
              <th className="text-right font-normal">Two-handed</th>
              <th className="text-right font-normal" title="Fraction of sampled frames with at least one hand detected">
                Detection
              </th>
              <th className="text-right font-normal">Sources</th>
            </tr>
          </thead>
          <tbody>
            {ranked.map(({ task, st }) => (
              <tr key={task.id} className="border-b border-line/60 hover:bg-panel">
                <td className="py-1 pr-3">{task.name}</td>
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
                <td className="num pl-3 text-right text-accent">{pct(st.bimanual)}</td>
                <td className={`num pl-3 text-right ${task.det1 < 0.6 ? 'text-red-400' : 'text-dim'}`}>
                  {pct(task.det1)}
                </td>
                <td className="num pl-3 text-right text-dim" title="independent recording families">
                  {task.reps}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="mt-3 text-[11px] leading-relaxed text-dim">
          Detection is the share of sampled frames where at least one hand was found. A
          task with low detection has an inflated "hands absent" share — the tracker failed,
          the worker did not stop. Sources counts independent recording families; where it
          is 1, the number describes that recording, not the task in general.
        </p>
      </div>
    </div>
  )
}
