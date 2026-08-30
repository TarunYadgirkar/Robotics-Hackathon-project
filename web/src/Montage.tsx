import { useMemo } from 'react'
import { BIMANUAL, clipStates, measurable, media, type Corpus } from './data'

const WINDOW = 8

/** One tile per task: the 8-second window the index scores as most two-handed. */
export default function Montage({ corpus, vHi }: { corpus: Corpus; vHi: number }) {
  const tiles = useMemo(() => {
    const picks: {
      task: string; src: string; score: number; start: number; bundled: boolean
    }[] = []
    for (const [tid, idxs] of corpus.byTask) {
      const task = corpus.tasks.find((t) => t.id === tid)
      if (!task || !measurable(task)) continue

      const bundled = corpus.tiles?.[tid]
      if (bundled) {
        picks.push({
          task: tid,
          src: `${import.meta.env.BASE_URL}${bundled.tile}`,
          score: bundled.score, start: bundled.start, bundled: true,
        })
        continue
      }
      // Without bundled tiles, fall back to the shipped 8-second proxies.
      let best: typeof picks[number] | null = null
      for (const i of idxs) {
        const clip = corpus.clips[i]
        if (clip.preview == null || clip.preview_start == null) continue
        const states = clipStates(corpus, i, vHi)
        let score = 0
        for (let s = clip.preview_start; s < clip.preview_start + WINDOW; s++) {
          if (states[s] === BIMANUAL) score++
        }
        if (!best || score > best.score) {
          best = {
            task: tid, src: media(clip.preview), score,
            start: clip.preview_start, bundled: false,
          }
        }
      }
      if (best) picks.push(best)
    }
    return picks.sort((a, b) => b.score - a.score).slice(0, 25)
  }, [corpus, vHi])

  const perfect = tiles.filter((t) => t.score === WINDOW).length

  return (
    <div className="h-full overflow-auto p-4">
      <p className="mb-3 max-w-3xl text-xs text-dim">
        <span className="num text-fg">{tiles.length}</span> tasks,{' '}
        <span className="num text-fg">{tiles.length}</span> moments, chosen by the index
        alone: for each task, the {WINDOW}-second window it scores as most two-handed. Nobody
        watched these before they were picked.{' '}
        <span className="num text-fg">{perfect}</span> of {tiles.length} score a full{' '}
        {WINDOW} out of {WINDOW} seconds. Check them — that is the point of showing them
        rather than describing them.
      </p>
      <div className="grid gap-2 [grid-template-columns:repeat(auto-fill,minmax(220px,1fr))]">
        {tiles.map((t) => {
          const task = corpus.tasks.find((x) => x.id === t.task)
          return (
            <figure key={t.task} className="overflow-hidden rounded border border-line bg-panel">
              <video
                src={t.src}
                autoPlay loop muted playsInline
                className="aspect-video w-full bg-black"
              />
              <figcaption className="flex items-baseline justify-between gap-2 px-2 py-1 text-xs">
                <span className="truncate">{task?.name}</span>
                <span className="num text-accent">
                  {t.score}/{WINDOW}
                </span>
              </figcaption>
            </figure>
          )
        })}
      </div>
    </div>
  )
}
