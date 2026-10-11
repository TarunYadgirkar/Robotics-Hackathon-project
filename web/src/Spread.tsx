import { useState } from 'react'
import { pct, type Task, type TaskStats } from './data'

interface Row { task: Task; st: TaskStats }
interface Props { ok: Row[]; share: number }

const TICKS = [0, 0.25, 0.5, 0.75, 1]

/** Every measurable task on one 0 to 100% axis, so the spread is drawn to scale. */
export default function Spread({ ok, share }: Props) {
  const [hoverId, setHoverId] = useState<string | null>(null)
  const byShare = [...ok].sort((a, b) => b.st.bimanual - a.st.bimanual)
  const top = byShare[0]
  const bottom = byShare[byShare.length - 1]
  const hover = ok.find((r) => r.task.id === hoverId) ?? null
  if (!top || !bottom) return null
  const ratio = bottom.st.bimanual > 0 ? top.st.bimanual / bottom.st.bimanual : 0

  return (
    <section className="flex flex-col gap-4" aria-labelledby="spread">
      <div className="flex max-w-[65ch] flex-col gap-2">
        <h3 id="spread" className="text-lg font-semibold">
          Tasks run from {pct(bottom.st.bimanual)} to {pct(top.st.bimanual)} two-handed
        </h3>
        <p className="text-sm leading-relaxed text-dim">
          {top.task.name} is the most hands-on task and {bottom.task.name} the least, a{' '}
          <span className="num text-fg">{ratio.toFixed(1)}×</span> spread. Drag the motion
          threshold: the percentages move, the order of tasks barely does. Point at a dot to name
          it; the table below lists every task.
        </p>
      </div>

      <div className="relative mx-2 h-28 sm:mx-0" aria-hidden>
        <div className="absolute inset-x-0 top-12 h-px bg-rule" />
        <div
          className="absolute top-6 h-12 w-px bg-dim"
          style={{ left: `${share * 100}%` }}
        />
        <span
          className="num absolute top-0 -translate-x-1/2 text-[11px] whitespace-nowrap text-dim"
          style={{ left: `${share * 100}%` }}
        >
          all tasks {pct(share)}
        </span>
        {ok.map((r) => (
          <span
            key={r.task.id}
            onPointerEnter={() => setHoverId(r.task.id)}
            onPointerLeave={() => setHoverId(null)}
            className={`absolute top-12 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full bg-accent ring-2 ring-surface transition-[left] duration-300 ${
              hoverId && hoverId !== r.task.id ? 'opacity-40' : ''
            }`}
            style={{ left: `${r.st.bimanual * 100}%` }}
          />
        ))}
        {TICKS.map((t) => (
          <span
            key={t}
            className="num absolute top-[3.75rem] -translate-x-1/2 text-[11px] text-dim"
            style={{ left: `${t * 100}%` }}
          >
            {t * 100}%
          </span>
        ))}
        <p className="absolute top-[5.25rem] inset-x-0 text-xs">
          {hover ? (
            <span
              className="absolute -translate-x-1/2 whitespace-nowrap"
              style={{ left: `clamp(25%, ${hover.st.bimanual * 100}%, 75%)` }}
            >
              {hover.task.name} <span className="num text-accent">{pct(hover.st.bimanual)}</span>
            </span>
          ) : (
            <>
              <span className="absolute left-0 text-dim">
                {bottom.task.name} <span className="num text-fg">{pct(bottom.st.bimanual)}</span>
              </span>
              <span className="absolute right-0 hidden text-right text-dim sm:inline">
                {top.task.name} <span className="num text-fg">{pct(top.st.bimanual)}</span>
              </span>
            </>
          )}
        </p>
      </div>
    </section>
  )
}
