import { measurable, pct, type Corpus, type TaskStats } from './data'

const W = 720, H = 260, PAD = 40

/** Two-handed share against left/right speed asymmetry: which robot the job needs. */
export default function Quadrant({
  corpus, stats, onPick,
}: {
  corpus: Corpus
  stats: Map<string, TaskStats>
  onPick: (taskId: string) => void
}) {
  const pts = corpus.tasks.filter(measurable).map((t) => ({
    t, st: stats.get(t.id)!,
  })).filter((p) => p.st)
  if (!pts.length) return null

  const asyms = pts.map((p) => p.t.asymmetry)
  const lo = Math.min(...asyms), hi = Math.max(...asyms)
  const span = Math.max(hi - lo, 0.05) * 1.15
  const mid = (hi + lo) / 2
  const minA = mid - span / 2, maxA = mid + span / 2
  const x = (v: number) => PAD + v * (W - PAD * 2)
  const y = (v: number) =>
    H - PAD - ((v - minA) / (maxA - minA)) * (H - PAD * 2)

  return (
    <figure className="mb-4 max-w-4xl rounded border border-line bg-panel p-3">
      <figcaption className="mb-1 text-xs text-dim">
        Every measurable task, positioned by how much of the clock is two-handed (right is
        more) and how unevenly the two hands move (up means one hand holds while the other
        works). Top-right needs a bimanual robot with a passive holding arm; bottom-right
        needs two arms that actually cooperate; left needs a single arm.
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img">
        <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="#2a2f38" />
        <line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} stroke="#2a2f38" />
        {[0.25, 0.5, 0.75].map((g) => (
          <line key={g} x1={x(g)} y1={PAD} x2={x(g)} y2={H - PAD} stroke="#171a20" />
        ))}
        {pts.map(({ t, st }) => (
          <g key={t.id} className="cursor-pointer" onClick={() => onPick(t.id)}>
            <circle
              cx={x(st.bimanual)} cy={y(t.asymmetry)}
              r={4 + t.det1 * 3}
              fill="#ffb020" fillOpacity={0.55} stroke="#ffb020" strokeOpacity={0.9}
            />
            <title>{`${t.name} — ${pct(st.bimanual)} two-handed, asymmetry ${t.asymmetry.toFixed(2)}`}</title>
          </g>
        ))}
        {(() => {
          // Label the extremes, skipping any that would collide with one already placed.
          const placed: [number, number][] = []
          return pts
            .slice()
            .sort((a, b) => b.st.bimanual - a.st.bimanual)
            .filter((p) => p.st.bimanual > 0.6 || p.t.asymmetry > hi - span * 0.15 || p.st.bimanual < 0.15)
            .filter((p) => {
              const cx = x(p.st.bimanual), cy = y(p.t.asymmetry)
              if (placed.some(([px, py]) => Math.abs(px - cx) < 90 && Math.abs(py - cy) < 12)) {
                return false
              }
              placed.push([cx, cy])
              return true
            })
            .slice(0, 7)
            .map(({ t, st }) => (
              <text
                key={t.id}
                x={x(st.bimanual) + 8} y={y(t.asymmetry) + 3}
                fill="#9aa1ad" fontSize={9}
              >
                {t.name}
              </text>
            ))
        })()}
        <text x={W - PAD} y={H - PAD + 16} fill="#6b7280" fontSize={10} textAnchor="end">
          two-handed share →
        </text>
        <text x={PAD - 6} y={PAD - 8} fill="#6b7280" fontSize={10}>
          ↑ hand asymmetry
        </text>
      </svg>
    </figure>
  )
}
