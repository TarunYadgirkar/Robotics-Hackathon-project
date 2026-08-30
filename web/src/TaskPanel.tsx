import { useEffect, useRef } from 'react'
import Quadrant from './Quadrant'
import { measurable, pct, type Corpus, type Task, type TaskStats } from './data'

function Heatmap({ corpus, index, task }: { corpus: Corpus; index: number; task: Task }) {
  const ref = useRef<HTMLCanvasElement>(null)
  const { heat_w: W, heat_h: H } = corpus.config
  useEffect(() => {
    const cv = ref.current!
    const g = cv.getContext('2d')!
    const img = g.createImageData(W, H)
    const off = index * W * H
    for (let i = 0; i < W * H; i++) {
      const v = Math.pow(corpus.heat[off + i] / 65535, 0.45)
      img.data[i * 4] = 255 * v
      img.data[i * 4 + 1] = 176 * v
      img.data[i * 4 + 2] = 32 * v
      img.data[i * 4 + 3] = 255
    }
    g.putImageData(img, 0, 0)
  }, [corpus, index, W, H])

  const [x0, y0, x1, y1] = task.envelope
  return (
    <div className="relative">
      <canvas
        ref={ref}
        width={W}
        height={H}
        className="w-full rounded-sm bg-black"
        style={{ imageRendering: 'pixelated', aspectRatio: '16/9' }}
      />
      <div
        className="pointer-events-none absolute border border-white/50"
        style={{
          left: `${x0 * 100}%`, top: `${y0 * 100}%`,
          width: `${(x1 - x0) * 100}%`, height: `${(y1 - y0) * 100}%`,
        }}
      />
    </div>
  )
}

interface Props {
  corpus: Corpus
  stats: Map<string, TaskStats>
  order: string[]
  vHi: number
}

export default function TaskPanel({ corpus, stats, order }: Props) {
  return (
    <div className="h-full overflow-auto p-4">
      <p className="mb-3 max-w-3xl text-xs text-dim">
        The camera is torso-mounted, so image coordinates are body-relative hand position with
        no calibration. The heatmap is where this worker's hands live; the white box is the
        90% containment envelope — the reach a robot would need. Grip aperture is
        thumb-tip to index-tip over hand size: low is a precision pinch, high is a spread grasp.
      </p>
      <Quadrant
        corpus={corpus}
        stats={stats}
        onPick={(id) => document.getElementById(`task-${id}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' })}
      />
      <div className="grid gap-3 [grid-template-columns:repeat(auto-fill,minmax(250px,1fr))]">
        {order.map((tid) => {
          const task = corpus.tasks.find((t) => t.id === tid)!
          const index = corpus.tasks.findIndex((t) => t.id === tid)
          const st = stats.get(tid)!
          const [x0, y0, x1, y1] = task.envelope
          return (
            <div key={tid} id={`task-${tid}`} className="rounded border border-line bg-panel p-3">
              <div className="mb-2 flex items-baseline justify-between gap-2">
                <h3 className="truncate text-sm font-medium">{task.name}</h3>
                <span className="num text-sm text-accent">
                  {measurable(task) ? pct(st.bimanual) : '—'}
                </span>
              </div>
              {!measurable(task) && (
                <p className="mb-2 rounded bg-red-950/40 px-2 py-1 text-[11px] leading-tight text-red-300">
                  Not measurable: hands detected in only {pct(task.det1)} of frames. Every
                  number on this card is unreliable.
                </p>
              )}
              <Heatmap corpus={corpus} index={index} task={task} />
              <dl className="mt-2 grid grid-cols-2 gap-x-3 gap-y-0.5 text-xs">
                {[
                  ['reach', `${((x1 - x0) * 100).toFixed(0)}×${((y1 - y0) * 100).toFixed(0)}`],
                  ['grip aperture', task.aperture[2].toFixed(2)],
                  ['asymmetry', task.asymmetry.toFixed(2)],
                  ['transit', pct(st.transit)],
                  ['detection', pct(task.det1)],
                  ['torso lean', `${task.lean_span.toFixed(0)}°`],
                ].map(([k, v]) => (
                  <div key={k} className="flex justify-between gap-2">
                    <dt className="text-dim">{k}</dt>
                    <dd className="num">{v}</dd>
                  </div>
                ))}
              </dl>
              <div className="mt-2 flex h-6 items-end gap-px" title="grip aperture distribution">
                {task.aperture_hist.map((v, i) => {
                  const max = Math.max(...task.aperture_hist, 1)
                  return (
                    <div
                      key={i}
                      className="flex-1 bg-accent/60"
                      style={{ height: `${(v / max) * 100}%` }}
                    />
                  )
                })}
              </div>
              {task.reps === 1 && (
                <p className="mt-2 text-[11px] leading-tight text-red-400/80">
                  One recording family — this describes that recording, not the task.
                </p>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
