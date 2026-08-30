import { pct, type Corpus } from './data'

interface Props { corpus: Corpus; vHi: number }

export default function Limitations({ corpus, vHi }: Props) {
  const single = corpus.tasks.filter((t) => t.reps === 1)
  const weak = corpus.tasks.filter((t) => t.det1 < 0.7).sort((a, b) => a.det1 - b.det1)
  const noPreview = corpus.clips.filter((c) => !c.preview).length

  const items: [string, React.ReactNode][] = [
    ['Sampling is 2 fps', <>
      Every state is decided from two frames per second. A grasp or release shorter than
      about 1.5 s is invisible to this index. Runs are therefore slightly coarse, and very
      fast alternating work reads as one continuous state.
    </>],
    ['"Hands absent" is not proof the hands were gone', <>
      It means the tracker found nothing. Detection degrades on motion blur, occlusion by
      the workpiece, gloves, and hands leaving the bottom of the frame. Detection rate is
      printed next to every number for this reason.{' '}
      {weak.length > 0 && <>Weakest: {weak.slice(0, 4).map((t) => `${t.name} (${pct(t.det1)})`).join(', ')}.</>}
    </>],
    ['One threshold, chosen from the data, not fitted per task', <>
      The motion threshold defaults to the 75th percentile of hand speed pooled across the
      whole corpus ({corpus.config.v_hi.toFixed(2)}; currently {vHi.toFixed(2)}). A per-task
      threshold would have let us fit the answer we wanted. Drag it: the absolute
      percentages move, the ordering of tasks largely does not. That stability is the claim.
    </>],
    ['Body-relative, not metric', <>
      The camera is torso-mounted, so image coordinates track hand position relative to the
      body with no calibration — that is the property this whole index rests on. But mount
      angle varies across cameras, so the reach envelope is comparable within a camera and
      only roughly across them. No depth, no 3D pose, no metric scale is claimed.
    </>],
    ['Many tasks are a single recording', <>
      {single.length} of {corpus.tasks.length} tasks come from one independent recording
      family. For those, the number describes that worker on that day, not the task in
      general. The dataset ships this warning; the task cards repeat it.
    </>],
    ['No ground truth', <>
      Nothing here was hand-labelled, so there is no accuracy figure to quote. What is
      testable is consistency: the same rule ran over all {corpus.clips.length} clips, and
      the montage view lets you check the picks by eye against footage nobody pre-screened.
    </>],
    ['Dead ends we hit first', <>
      Two IMU angles were measured and abandoned before this index existed. Gyro-magnitude
      autocorrelation for per-repetition work cycles: median peak 0.16, only 2 of 50 tasks
      above 0.30. Whitened spectral tonality for machine-vibration signatures: SNR ~2.1 on
      every task, exactly the noise floor — the sidecars document that GPMF timestamps are
      reconstructed by spreading samples uniformly inside ~1 s packets, which destroys the
      fine timing those methods need. The IMU is used here only for gross motion: gyro RMS
      and torso lean.
    </>],
    ['Coverage gaps', <>
      {noPreview} of {corpus.clips.length} clips ship no 8-second proxy, so the montage view
      can only draw from the {corpus.clips.length - noPreview} that do. Full-resolution
      playback needs the local media server; the deployed build shows the index without it.
    </>],
  ]

  return (
    <div className="h-full overflow-auto px-6 py-6">
      <div className="mx-auto max-w-3xl">
        <h2 className="text-xl font-semibold">What this does not show</h2>
        <p className="mt-2 text-sm text-dim">
          Written before the demo, not after the questions.
        </p>
        <dl className="mt-5 space-y-4">
          {items.map(([title, body]) => (
            <div key={title} className="rounded border border-line bg-panel p-4">
              <dt className="text-sm font-medium text-accent">{title}</dt>
              <dd className="mt-1 text-sm leading-relaxed text-dim">{body}</dd>
            </div>
          ))}
        </dl>
      </div>
    </div>
  )
}
