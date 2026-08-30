import { useMemo } from 'react'
import { BIMANUAL, clipStates, measurable, media, type Corpus } from './data'

/** One tile per task: the 8-second proxy whose window the index scores most two-handed. */
export default function Montage({ corpus, vHi }: { corpus: Corpus; vHi: number }) {
  const tiles = useMemo(() => {
    const picks: { clipIdx: number; score: number; task: string }[] = []
    for (const [tid, idxs] of corpus.byTask) {
      const task = corpus.tasks.find((t) => t.id === tid)
      if (!task || !measurable(task)) continue
      let best: { clipIdx: number; score: number; task: string } | null = null
      for (const i of idxs) {
        const clip = corpus.clips[i]
        if (clip.preview == null || clip.preview_start == null) continue
        const states = clipStates(corpus, i, vHi)
        let score = 0
        for (let s = clip.preview_start; s < clip.preview_start + 8; s++) {
          if (states[s] === BIMANUAL) score++
        }
        if (!best || score > best.score) best = { clipIdx: i, score, task: tid }
      }
      if (best) picks.push(best)
    }
    return picks.sort((a, b) => b.score - a.score).slice(0, 25)
  }, [corpus, vHi])

  const perfect = tiles.filter((t) => t.score === 8).length

  return (
    <div className="h-full overflow-auto p-4">
      <p className="mb-3 max-w-3xl text-xs text-dim">
        <span className="num text-fg">{tiles.length}</span> tasks, {tiles.length} moments,
        chosen by the index alone: for each task, the 8-second proxy whose window scores most
        two-handed. Nobody watched these before they were picked.{' '}
        <span className="num text-fg">{perfect}</span> of {tiles.length} score a full 8 out of 8
        seconds — check them yourself, that is the point of showing them.
      </p>
      <div className="grid gap-2 [grid-template-columns:repeat(auto-fill,minmax(220px,1fr))]">
        {tiles.map((t) => {
          const clip = corpus.clips[t.clipIdx]
          const task = corpus.tasks.find((x) => x.id === t.task)
          return (
            <figure key={clip.id} className="overflow-hidden rounded border border-line bg-panel">
              <video
                src={media(clip.preview!)}
                poster={media(clip.thumb)}
                autoPlay loop muted playsInline
                className="aspect-video w-full bg-black"
              />
              <figcaption className="flex items-baseline justify-between gap-2 px-2 py-1 text-[11px]">
                <span className="truncate">{task?.name}</span>
                <span className="num text-accent">{t.score}/8</span>
              </figcaption>
            </figure>
          )
        })}
      </div>
    </div>
  )
}
