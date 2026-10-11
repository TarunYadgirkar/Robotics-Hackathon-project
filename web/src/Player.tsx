import { useEffect, useRef } from 'react'
import { STATE_COLORS, STATE_INK, STATE_NAMES, clipStates, clock, media, type Corpus } from './data'
import MediaFallback from './MediaFallback'

interface Props { corpus: Corpus; clipIdx: number; second: number; vHi: number; media: boolean }

export default function Player({ corpus, clipIdx, second, vHi, media: hasMedia }: Props) {
  const clip = corpus.clips[clipIdx]
  const video = useRef<HTMLVideoElement>(null)
  const states = clipStates(corpus, clipIdx, vHi)
  const state = states[second]
  const task = corpus.tasks.find((t) => t.id === clip.task)
  const narration = corpus.narration?.[clip.id]

  useEffect(() => {
    const v = video.current
    if (!v) return
    const seek = () => { v.currentTime = second }
    if (v.readyState >= 1) seek()
    else v.addEventListener('loadedmetadata', seek, { once: true })
  }, [clipIdx, second])

  return (
    <div className="flex flex-col gap-3">
      {hasMedia ? (
        <video
          ref={video}
          key={clip.id}
          src={`${media(clip.path)}#t=${second}`}
          poster={media(clip.thumb)}
          controls autoPlay muted playsInline
          className="aspect-video w-full rounded-md bg-black"
        />
      ) : (
        <MediaFallback corpus={corpus} taskId={clip.task} />
      )}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
        <span
          className="rounded px-2 py-0.5 font-semibold"
          style={{ background: STATE_COLORS[state], color: STATE_INK[state] }}
        >
          {STATE_NAMES[state]}
        </span>
        <span className="num text-dim">at {clock(second)}</span>
        <span className="font-medium text-fg">{task?.name}</span>
      </div>
      <div className="flex h-3 w-full overflow-hidden rounded-sm" aria-hidden>
        {Array.from(states).map((s, i) => (
          <div
            key={i}
            className="h-full flex-1"
            style={{
              background: STATE_COLORS[s],
              outline: i === second ? '2px solid var(--color-fg)' : undefined,
            }}
          />
        ))}
      </div>
      <p className="text-xs text-dim">
        Clip {clip.idx + 1} of {task?.clips} · camera{' '}
        <span className="num">{clip.cam.slice(4, 10)}</span> · repetition{' '}
        <span className="num">{clip.rep.slice(4, 10)}</span>
      </p>
      {narration && (
        <details className="rounded-md bg-surface p-3 text-xs">
          <summary className="cursor-pointer font-medium text-accent">Hear the index read this clip</summary>
          <audio
            controls
            src={`${import.meta.env.BASE_URL}${narration.audio}`}
            className="mt-2 w-full"
          />
          <p className="mt-2 leading-relaxed text-dim">{narration.text}</p>
          <p className="mt-1 text-[11px] text-dim">
            Synthesised speech over computed statistics. The script comes from the per-second
            timeline above; no model watched this video to describe it.
          </p>
        </details>
      )}
    </div>
  )
}
