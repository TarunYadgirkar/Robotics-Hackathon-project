import type { Corpus, Stability } from './data'

const SPARK_W = 240, SPARK_H = 56, RHO_MIN = 0.5

function StabilityChart({ stab, reference }: { stab: Stability; reference: number }) {
  const n = stab.thresholds.length
  const x = (i: number) => (n > 1 ? (i / (n - 1)) * SPARK_W : 0)
  const nearest = (v: number) =>
    stab.thresholds.reduce((best, t, i) =>
      Math.abs(t - v) < Math.abs(stab.thresholds[best] - v) ? i : best, 0)
  const xv = (v: number) => x(nearest(v))
  const y = (r: number) => SPARK_H - Math.max(0, (r - RHO_MIN) / (1 - RHO_MIN)) * (SPARK_H - 4)
  return (
    <svg
      viewBox={`0 0 ${SPARK_W} ${SPARK_H + 14}`}
      className="h-auto w-full max-w-60"
      role="img"
      aria-label={`Rank correlation of task ordering against the motion threshold. It stays at or above ${stab.bandFloor} between ${stab.band[0]} and ${stab.band[1]}.`}
    >
      <rect
        x={xv(stab.band[0])} y={0}
        width={xv(stab.band[1]) - xv(stab.band[0])} height={SPARK_H}
        className="fill-raised"
      />
      <line x1={0} y1={y(stab.bandFloor)} x2={SPARK_W} y2={y(stab.bandFloor)} className="stroke-rule" strokeDasharray="3 3" />
      <line x1={xv(reference)} y1={0} x2={xv(reference)} y2={SPARK_H} className="stroke-dim" />
      <polyline
        fill="none" className="stroke-accent" strokeWidth={2} strokeLinejoin="round"
        points={stab.rhos.map((r, i) => `${x(i)},${y(r)}`).join(' ')}
      />
      <text x={0} y={SPARK_H + 12} className="num fill-dim" fontSize={10}>
        {stab.thresholds[0].toFixed(2)}
      </text>
      <text x={SPARK_W} y={SPARK_H + 12} className="num fill-dim" fontSize={10} textAnchor="end">
        {stab.thresholds[n - 1].toFixed(2)}
      </text>
    </svg>
  )
}

/** The two checks a skeptic should run first: by-eye agreement and threshold stability. */
export default function Evidence({ corpus, stab }: { corpus: Corpus; stab: Stability }) {
  const v = corpus.validation
  return (
    <section className="grid gap-4 lg:grid-cols-2" aria-label="How far to trust the number">
      {v && (
        <article className="flex flex-col gap-3 rounded-md bg-panel p-5">
          <h3 className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
            <span className="num text-3xl whitespace-nowrap text-accent">
              {Math.round((v.agreed / v.samples) * 100)}%
            </span>
            <span className="text-sm font-medium">agreement when checked by eye</span>
          </h3>
          <p className="max-w-[60ch] text-sm leading-relaxed text-dim">
            We drew {v.samples} random seconds from {v.tasks} tasks and judged each pair of
            frames against the state the index assigned. {v.agreed} agreed. Of the{' '}
            {v.samples - v.agreed} that didn't, <span className="text-fg">{v.under} were the
            index missing visible hands</span> and {v.over} the index over-calling two hands. The error runs one way, so the two-handed share is a floor. VALIDATION.md in
            the repo has the full sample table.
          </p>
        </article>
      )}
      <article className="flex flex-col gap-3 rounded-md bg-panel p-5">
        <h3 className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <span className="num text-3xl whitespace-nowrap text-accent">ρ ≥ {stab.bandFloor.toFixed(2)}</span>
          <span className="text-sm font-medium">
            task order holds from {stab.band[0].toFixed(2)} to {stab.band[1].toFixed(2)}
          </span>
        </h3>
        <StabilityChart stab={stab} reference={corpus.config.v_hi} />
        <p className="max-w-[60ch] text-sm leading-relaxed text-dim">
          Spearman rank correlation of the task ordering against the default threshold
          ({corpus.config.v_hi.toFixed(2)}, the vertical line). Inside the shaded band the
          absolute percentages move a lot while the ranking barely changes. Below it the
          ordering breaks down, reaching ρ = {stab.worst.toFixed(2)} at{' '}
          {stab.thresholds[0].toFixed(2)}, where almost every second gets called transit. We
          defend the ordering, and only inside that band.
        </p>
      </article>
    </section>
  )
}
