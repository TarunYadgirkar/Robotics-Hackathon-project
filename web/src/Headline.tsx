import { useMemo } from 'react'
import {
  BIMANUAL, STATE_COLORS, TRANSIT, UNIMANUAL, ABSENT, hours, measurable, pct, stability,
  type Corpus, type TaskStats,
} from './data'
import Clock from './Clock'
import Spread from './Spread'
import Evidence from './Evidence'

interface Props {
  corpus: Corpus
  stats: Map<string, TaskStats>
  order: string[]
  totals: number[]
  grand: number
  collected: number
}

const BAR_ORDER = [3, 2, 1, 0] as const

export default function Headline({ corpus, stats, order, totals, grand, collected }: Props) {
  const ranked = order.map((id) => ({
    task: corpus.tasks.find((t) => t.id === id)!,
    st: stats.get(id)!,
  }))
  const ok = ranked.filter((r) => measurable(r.task))
  const failed = ranked.filter((r) => !measurable(r.task))
  const stab = useMemo(() => stability(corpus, corpus.config.v_hi), [corpus])
  const share = totals[BIMANUAL] / grand

  return (
    <div className="px-4 py-8 sm:px-6 lg:h-full lg:overflow-auto lg:py-12">
      <div className="mx-auto flex max-w-6xl flex-col gap-14">
        <section className="flex flex-col gap-6" aria-labelledby="finding">
          <div className="flex flex-col gap-x-10 gap-y-3 lg:flex-row lg:items-end">
            <p aria-hidden className="font-display text-hero font-extrabold tabular-nums text-accent">
              {pct(share)}
            </p>
            <h2 id="finding" className="max-w-[22ch] text-2xl leading-tight font-semibold sm:text-4xl lg:pb-2">
              <span className="sr-only">{pct(share)} </span>
              of measurable factory time is{' '}
              <span className="whitespace-nowrap">two-handed</span> manipulation
            </h2>
          </div>
          <p className="max-w-[65ch] text-base leading-relaxed text-dim">
            The other {pct(1 - share)} is one hand working ({pct(totals[UNIMANUAL] / grand)}),
            hands moving between objects ({pct(totals[TRANSIT] / grand)}), or no hands in frame
            ({pct(totals[ABSENT] / grand)}): walking, waiting, watching a machine run. A robot
            policy trained on raw clips spends most of its data on footage without two-handed
            work. Every second was labelled by tracking both hands, with no
            annotations, no training and one threshold for all {corpus.tasks.length} tasks.
          </p>
          <Clock totals={totals} grand={grand} collected={collected} />
        </section>

        <Spread ok={ok} share={share} />

        <Evidence corpus={corpus} stab={stab} />

        {failed.length > 0 && (
          <details className="rounded-md bg-critical-bg px-4 py-3 text-sm">
            <summary className="cursor-pointer font-medium text-critical">
              {failed.length} tasks held out because the hand tracker fails on them
            </summary>
            <p className="mt-2 max-w-[65ch] leading-relaxed text-dim">
              {failed.map((r) => `${r.task.name} (${pct(r.task.det1)} detection)`).join(', ')}.
              We checked footage from each and found three causes: gloves, hands caked in wet
              plaster, and hands hidden by the machine or workpiece. Counting those seconds as
              "hands absent" would turn busy tasks into idle-looking ones, so they sit outside
              every number above. The Limitations view shows the frames.
            </p>
          </details>
        )}

        <section className="flex flex-col gap-3" aria-labelledby="by-task">
          <h3 id="by-task" className="text-lg font-semibold">Where the clock goes, task by task</h3>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[320px] text-xs">
              <thead className="text-dim">
                <tr className="border-b border-rule">
                  <th className="py-2 text-left font-normal">Task</th>
                  <th className="w-[40%] py-2 text-left font-normal">
                    <span className="sr-only">Share of time by state</span>
                  </th>
                  <th className="py-2 text-right font-normal">Two-handed</th>
                  <th className="hidden py-2 text-right font-normal sm:table-cell">Detection</th>
                  <th className="hidden py-2 text-right font-normal sm:table-cell">Sources</th>
                </tr>
              </thead>
              <tbody>
                {[...ok, ...failed].map(({ task, st }) => {
                  const bad = !measurable(task)
                  return (
                    <tr key={task.id} className="border-b border-rule/60 hover:bg-panel">
                      <td className={`py-1.5 pr-3 ${bad ? 'text-dim' : ''}`}>
                        {task.name}
                        {bad && <span className="ml-2 text-critical">not measurable</span>}
                      </td>
                      <td>
                        <div className={`flex h-3 w-full overflow-hidden rounded-sm ${bad ? 'opacity-35' : ''}`}>
                          {BAR_ORDER.map((s) => (
                            <div
                              key={s}
                              className="h-full border-r border-surface last:border-r-0"
                              style={{
                                width: `${(st.tallies[s] / st.seconds) * 100}%`,
                                background: STATE_COLORS[s],
                              }}
                            />
                          ))}
                        </div>
                      </td>
                      <td className="num pl-3 text-right">{bad ? 'n/a' : pct(st.bimanual)}</td>
                      <td className={`num hidden pl-3 text-right sm:table-cell ${bad ? 'text-critical' : 'text-dim'}`}>
                        {pct(task.det1)}
                      </td>
                      <td className="num hidden pl-3 text-right text-dim sm:table-cell">{task.reps}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          <p className="max-w-[65ch] text-xs leading-relaxed text-dim">
            Bars run two-handed, one-handed, transit, then hands absent. Detection is the share
            of sampled frames where the tracker found at least one hand. Sources counts
            independent recording families; where it is 1, the number describes that one
            recording, not the task in general. Total measured: {hours(grand)}.
          </p>
        </section>
      </div>
    </div>
  )
}
