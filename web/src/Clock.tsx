import { STATE_COLORS, STATE_NAMES, hours, pct } from './data'

interface Props { totals: number[]; grand: number; collected: number }

const BAR_ORDER = [3, 2, 1, 0] as const
const LABEL_MIN_SHARE = 0.09
const TICK_STEP_HOURS = 5
const MOBILE_TICK_HOURS = 10
const MOBILE_END_GAP_HOURS = 8

interface Segment { key: string; name: string; seconds: number; color?: string; share?: number }

/** All collected footage as one bar, drawn to scale: where the clock actually goes. */
export default function Clock({ totals, grand, collected }: Props) {
  const segments: Segment[] = [
    ...BAR_ORDER.map((s) => ({
      key: String(s), name: STATE_NAMES[s], seconds: totals[s],
      color: STATE_COLORS[s], share: totals[s] / grand,
    })),
    { key: 'excluded', name: 'Not measurable', seconds: Math.max(collected - grand, 0) },
  ]
  let offset = 0
  const placed = segments.map((s) => {
    const left = offset / collected
    offset += s.seconds
    return { ...s, left, width: s.seconds / collected }
  })
  const ticks: number[] = []
  for (let h = 0; (h + TICK_STEP_HOURS / 2) * 3600 < collected; h += TICK_STEP_HOURS) ticks.push(h)
  const bimanual = placed[0]

  return (
    <figure className="flex flex-col gap-3">
      <div className="relative hidden h-5 text-xs sm:block" aria-hidden>
        {placed.filter((s) => s.width >= LABEL_MIN_SHARE).map((s) => (
          <span
            key={s.key}
            className="absolute bottom-0 truncate pr-2 text-dim"
            style={{ left: `${s.left * 100}%`, maxWidth: `${s.width * 100}%` }}
          >
            {s.name}
          </span>
        ))}
      </div>

      <div
        className="clock-grow flex h-12 w-full overflow-hidden rounded-md sm:h-16"
        role="img"
        aria-label={`${hours(collected)} collected. ${placed
          .map((s) => `${s.name} ${hours(s.seconds)}`)
          .join(', ')}.`}
      >
        {placed.map((s) => (
          <div
            key={s.key}
            className={`h-full border-r-2 border-surface transition-[width] duration-300 last:border-r-0 ${
              s.color ? '' : 'hatch bg-panel'
            }`}
            style={{ width: `${s.width * 100}%`, background: s.color }}
          />
        ))}
      </div>

      <div className="relative h-14" aria-hidden>
        <div
          className="absolute top-0 h-2 border-x border-b border-accent transition-[width] duration-300"
          style={{ left: 0, width: `${bimanual.width * 100}%` }}
        />
        <p className="absolute top-3.5 left-0 max-w-[28ch] text-sm text-fg">
          <span className="num font-medium text-accent">{hours(bimanual.seconds)}</span> of
          two-handed footage out of <span className="num">{hours(collected)}</span> collected
        </p>
      </div>

      <div className="relative h-4 border-t border-rule text-[11px] text-dim" aria-hidden>
        {ticks.map((h) => (
          <span
            key={h}
            className={`num absolute top-1 -translate-x-1/2 first:translate-x-0 ${
              h % MOBILE_TICK_HOURS || collected / 3600 - h < MOBILE_END_GAP_HOURS ? 'hidden sm:inline' : ''
            }`}
            style={{ left: `${((h * 3600) / collected) * 100}%` }}
          >
            {h}{'\u2009'}h
          </span>
        ))}
        <span className="num absolute top-1 right-0">{hours(collected)}</span>
      </div>

      <figcaption>
        <table className="mt-4 w-full max-w-xl text-sm">
          <caption className="sr-only">Where the collected footage goes</caption>
          <thead className="text-xs text-dim">
            <tr>
              <th className="pb-1 text-left font-normal">State</th>
              <th className="pb-1 text-right font-normal">Hours</th>
              <th className="pb-1 text-right font-normal">Share of measurable</th>
            </tr>
          </thead>
          <tbody>
            {placed.map((s) => (
              <tr key={s.key}>
                <td className="py-0.5">
                  <div className="flex items-center gap-2">
                  <span
                    className={`h-3 w-3 shrink-0 rounded-sm ${s.color ? '' : 'hatch bg-panel'}`}
                    style={{ background: s.color }}
                  />
                  {s.name}
                  </div>
                </td>
                <td className="num text-right">{hours(s.seconds)}</td>
                <td className="num text-right text-dim">
                  {s.share === undefined ? 'held out' : pct(s.share)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </figcaption>
    </figure>
  )
}
