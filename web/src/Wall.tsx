import { useEffect, useMemo, useRef } from 'react'
import { STATE_COLORS, clipStates, type Corpus, type TaskStats } from './data'

const ROW_H = 13, ROW_GAP = 2, HEAD_H = 22, LABEL_W = 210, SEC_PX = 2

export interface Hit { clipIdx: number; second: number }

interface Props {
  corpus: Corpus
  vHi: number
  stats: Map<string, TaskStats>
  order: string[]
  highlight?: Set<number>
  onPick: (hit: Hit) => void
}

export default function Wall({ corpus, vHi, stats, order, highlight, onPick }: Props) {
  const ref = useRef<HTMLCanvasElement>(null)
  const S = corpus.config.clip_seconds
  const width = LABEL_W + S * SEC_PX

  const layout = useMemo(() => {
    const rows: { y: number; clipIdx: number }[] = []
    const heads: { y: number; task: string }[] = []
    let y = 0
    for (const tid of order) {
      heads.push({ y, task: tid })
      y += HEAD_H
      for (const clipIdx of corpus.byTask.get(tid) ?? []) {
        rows.push({ y, clipIdx })
        y += ROW_H + ROW_GAP
      }
      y += 6
    }
    return { rows, heads, height: y }
  }, [corpus, order])

  useEffect(() => {
    const cv = ref.current!
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    cv.width = width * dpr
    cv.height = layout.height * dpr
    cv.style.width = `${width}px`
    cv.style.height = `${layout.height}px`
    const g = cv.getContext('2d')!
    g.setTransform(dpr, 0, 0, dpr, 0, 0)
    g.clearRect(0, 0, width, layout.height)

    g.font = '600 11px ui-monospace, Menlo, monospace'
    for (const h of layout.heads) {
      const st = stats.get(h.task)
      const task = corpus.tasks.find((t) => t.id === h.task)
      g.fillStyle = '#e7e9ee'
      g.fillText(task?.name ?? h.task, 0, h.y + 13)
      if (st) {
        g.fillStyle = '#ffb020'
        g.fillText(`${(st.bimanual * 100).toFixed(0)}%`, LABEL_W - 34, h.y + 13)
        g.fillStyle = '#2a2f38'
        g.fillRect(LABEL_W, h.y + 8, S * SEC_PX, 1)
      }
    }

    for (const row of layout.rows) {
      const states = clipStates(corpus, row.clipIdx, vHi)
      const dim = highlight && !highlight.has(row.clipIdx)
      g.globalAlpha = dim ? 0.18 : 1
      for (let s = 0; s < S; s++) {
        g.fillStyle = STATE_COLORS[states[s]]
        g.fillRect(LABEL_W + s * SEC_PX, row.y, SEC_PX, ROW_H)
      }
      g.globalAlpha = 1
    }
  }, [corpus, vHi, layout, stats, width, S, highlight])

  return (
    <canvas
      ref={ref}
      className="cursor-crosshair"
      onClick={(e) => {
        const r = ref.current!.getBoundingClientRect()
        const x = e.clientX - r.left, y = e.clientY - r.top
        if (x < LABEL_W) return
        const second = Math.floor((x - LABEL_W) / SEC_PX)
        const row = layout.rows.find((rw) => y >= rw.y && y < rw.y + ROW_H)
        if (row && second >= 0 && second < S) onPick({ clipIdx: row.clipIdx, second })
      }}
    />
  )
}
