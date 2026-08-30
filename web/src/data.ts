export const ABSENT = 0, TRANSIT = 1, UNIMANUAL = 2, BIMANUAL = 3
export type State = 0 | 1 | 2 | 3

export const STATE_NAMES = ['Hands absent', 'Transit', 'One-handed', 'Two-handed'] as const
export const STATE_COLORS = ['#171a20', '#2f4d78', '#a06a1c', '#ffb020'] as const

export interface Clip {
  id: string; task: string; idx: number; cam: string; rep: string; seq: string
  path: string; thumb: string; preview: string | null; preview_start: number | null
}
export interface Task {
  id: string; name: string; clips: number; minutes: number; reps: number; cameras: number
  warnings: string[]; det1: number; det2: number
  palm: [number, number]; envelope: [number, number, number, number]
  aperture: number[]; aperture_hist: number[]
  asymmetry: number; gyro_rms: number; lean_span: number
}
export interface Config {
  clip_seconds: number; fps: number; speed_scale: number; missing: number
  v_hi: number; heat_w: number; heat_h: number; aperture_bins: number
}

export interface Corpus {
  config: Config; clips: Clip[]; tasks: Task[]
  counts: Uint8Array; speeds: Uint8Array; heat: Uint16Array
  clipIndex: Map<string, number>; byTask: Map<string, number[]>
  narration: Record<string, { text: string; audio: string }>
}

export const MEDIA_BASE = (
  import.meta.env.VITE_MEDIA_BASE ?? 'http://127.0.0.1:8765'
).replace(/\/$/, '')

export const media = (p: string) => `${MEDIA_BASE}/${p}`

export async function loadCorpus(): Promise<Corpus> {
  const base = import.meta.env.BASE_URL
  const [meta, statesBuf, heatBuf, narration] = await Promise.all([
    fetch(`${base}data/corpus.json`).then((r) => r.json()),
    fetch(`${base}data/states.bin`).then((r) => r.arrayBuffer()),
    fetch(`${base}data/heatmaps.bin`).then((r) => r.arrayBuffer()),
    fetch(`${base}data/narration.json`).then((r) => (r.ok ? r.json() : {})).catch(() => ({})),
  ])
  const S = meta.config.clip_seconds
  const raw = new Uint8Array(statesBuf)
  const n = meta.clips.length
  const counts = new Uint8Array(n * S)
  const speeds = new Uint8Array(n * S)
  for (let i = 0; i < n; i++) {
    counts.set(raw.subarray(i * 2 * S, i * 2 * S + S), i * S)
    speeds.set(raw.subarray(i * 2 * S + S, (i + 1) * 2 * S), i * S)
  }
  const clipIndex = new Map<string, number>()
  const byTask = new Map<string, number[]>()
  meta.clips.forEach((c: Clip, i: number) => {
    clipIndex.set(c.id, i)
    const list = byTask.get(c.task) ?? []
    list.push(i)
    byTask.set(c.task, list)
  })
  return {
    ...meta, counts, speeds, heat: new Uint16Array(heatBuf), clipIndex, byTask, narration,
  }
}

/** One rule for all 50 tasks. Threshold stays a live control, never baked in. */
export function classify(n: number, sp: number, vHi: number, cfg: Config): State {
  if (n === 0) return ABSENT
  if (sp !== cfg.missing && sp / cfg.speed_scale >= vHi) return TRANSIT
  return n >= 2 ? BIMANUAL : UNIMANUAL
}

/** Median-of-3 over seconds: kills single-second flicker, keeps real runs. */
function smooth(s: Uint8Array): Uint8Array {
  const out = new Uint8Array(s.length)
  out[0] = s[0]
  out[s.length - 1] = s[s.length - 1]
  for (let i = 1; i < s.length - 1; i++) {
    const a = s[i - 1], b = s[i], c = s[i + 1]
    out[i] = a === c ? a : b
  }
  return out
}

export function clipStates(c: Corpus, clipIdx: number, vHi: number): Uint8Array {
  const S = c.config.clip_seconds
  const out = new Uint8Array(S)
  for (let s = 0; s < S; s++) {
    const i = clipIdx * S + s
    out[s] = classify(c.counts[i], c.speeds[i], vHi, c.config)
  }
  return smooth(out)
}

export interface Run { clip: number; start: number; end: number; state: State }

export function runsOf(states: Uint8Array, clip: number, state: State, minSec: number): Run[] {
  const out: Run[] = []
  let start = -1
  for (let s = 0; s <= states.length; s++) {
    const on = s < states.length && states[s] === state
    if (on && start < 0) start = s
    if (!on && start >= 0) {
      if (s - start >= minSec) out.push({ clip, start, end: s, state })
      start = -1
    }
  }
  return out
}

export function tally(states: Uint8Array): number[] {
  const t = [0, 0, 0, 0]
  for (const s of states) t[s]++
  return t
}

export interface TaskStats {
  id: string; tallies: number[]; seconds: number
  manip: number; bimanual: number; transit: number; absent: number
}

export function taskStats(c: Corpus, vHi: number): Map<string, TaskStats> {
  const out = new Map<string, TaskStats>()
  for (const [tid, idxs] of c.byTask) {
    const t = [0, 0, 0, 0]
    for (const i of idxs) {
      const counts = tally(clipStates(c, i, vHi))
      for (let s = 0; s < 4; s++) t[s] += counts[s]
    }
    const total = t[0] + t[1] + t[2] + t[3] || 1
    out.set(tid, {
      id: tid, tallies: t, seconds: total,
      manip: (t[BIMANUAL] + t[UNIMANUAL]) / total,
      bimanual: t[BIMANUAL] / total,
      transit: t[TRANSIT] / total,
      absent: t[ABSENT] / total,
    })
  }
  return out
}

/** Below this detection rate the tracker, not the worker, is the story. */
export const DETECTION_FLOOR = 0.5
export const measurable = (t: Task) => t.det1 >= DETECTION_FLOOR

export const pct = (x: number) => `${(x * 100).toFixed(1)}%`
export const clock = (s: number) =>
  `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`
export const hours = (s: number) => `${(s / 3600).toFixed(1)} h`
