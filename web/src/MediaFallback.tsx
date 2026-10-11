import { useReducedMotion } from './hooks'
import type { Corpus } from './data'

interface Props { corpus: Corpus; taskId: string }

/** Stands in for full-resolution playback when the dataset drive isn't mounted. */
export default function MediaFallback({ corpus, taskId }: Props) {
  const tile = corpus.tiles?.[taskId]
  const reduced = useReducedMotion()
  return (
    <figure className="flex flex-col gap-2">
      {tile ? (
        <video
          key={tile.tile}
          aria-label="Bundled 8-second sample of two-handed work from this task"
          src={`${import.meta.env.BASE_URL}${tile.tile}`}
          autoPlay={!reduced}
          controls={reduced}
          loop muted playsInline
          className="aspect-video w-full rounded-md bg-black"
        />
      ) : (
        <div className="hatch grid aspect-video w-full place-items-center rounded-md bg-panel px-6 text-center text-sm text-dim">
          No bundled sample for this task.
        </div>
      )}
      <figcaption className="text-xs leading-relaxed text-dim">
        Full playback needs the dataset drive, which this build can't reach.
        {tile
          ? ' Above is the 8-second window the index scored as most two-handed in this task.'
          : ' The timeline below still shows what the index measured.'}{' '}
        To play any second, run the media server from the README.
      </figcaption>
    </figure>
  )
}
