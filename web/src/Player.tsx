import { useEffect, useRef } from 'react'
import { STATE_COLORS, STATE_NAMES, clipStates, clock, media, type Corpus } from './data'

interface Props { corpus: Corpus; clipIdx: number; second: number; vHi: number }

export default function Player({ corpus, clipIdx, second, vHi }: Props) {
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
    <div className="flex flex-col gap-2">
      <video
        ref={video}
        key={clip.id}
        src={`${media(clip.path)}#t=${second}`}
        poster={media(clip.thumb)}
        controls
        autoPlay
        muted
        playsInline
        className="w-full rounded bg-black aspect-video"
      />
      <div className="flex items-center gap-3 text-xs">
        <span
          className="num rounded px-2 py-0.5 font-semibold text-ink"
          style={{ background: STATE_COLORS[state] }}
        >
          {STATE_NAMES[state]}
        </span>
        <span className="num text-dim">{clock(second)}</span>
        <span className="text-fg">{task?.name}</span>
        <span className="num text-dim truncate">{clip.id}</span>
      </div>
      <div className="flex h-3 w-full overflow-hidden rounded-sm">
        {Array.from(states).map((s, i) => (
          <div
            key={i}
            className="h-full flex-1"
            style={{
              background: STATE_COLORS[s],
              outline: i === second ? '1px solid #fff' : undefined,
            }}
          />
        ))}
      </div>
      <p className="text-xs text-dim">
        Camera {clip.cam.slice(4, 10)} · repetition {clip.rep.slice(4, 10)} · clip{' '}
        {clip.idx + 1} of {task?.clips}
      </p>
      {narration && (
        <details className="rounded border border-line bg-ink p-2 text-xs">
          <summary className="cursor-pointer text-accent">Hear the index read this clip</summary>
          <audio
            controls
            src={`${import.meta.env.BASE_URL}${narration.audio}`}
            className="mt-2 w-full"
          />
          <p className="mt-2 leading-relaxed text-dim">{narration.text}</p>
          <p className="mt-1 text-[11px] text-dim">
            Synthesised speech over computed statistics. The script is generated from the
            per-second timeline above — no model watched this video to describe it.
          </p>
        </details>
      )}
    </div>
  )
}
