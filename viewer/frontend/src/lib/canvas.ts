// Hand-rolled canvas charts for the swarm dashboards, ported from the legacy
// swarm.html. Each function empties nothing — callers (CanvasBox) clear the
// parent — and appends one canvas sized to the parent width at device DPR.
import { C } from './colors'
import { fmt } from './format'

interface Ctx {
  g: CanvasRenderingContext2D
  w: number
  h: number
}

function cv(parent: HTMLElement, hCss: number): Ctx {
  const c = document.createElement('canvas')
  parent.appendChild(c)
  const w = c.clientWidth || parent.clientWidth || 600
  const dpr = window.devicePixelRatio || 1
  c.width = w * dpr
  c.height = hCss * dpr
  c.style.height = hCss + 'px'
  const g = c.getContext('2d')!
  g.scale(dpr, dpr)
  return { g, w, h: hCss }
}

function axis(g: CanvasRenderingContext2D, x0: number, y0: number, x1: number, y1: number) {
  g.strokeStyle = C.border
  g.lineWidth = 1
  g.strokeRect(x0 + 0.5, y1 + 0.5, x1 - x0, y0 - y1)
}

export interface HistogramData {
  bins: number[]
  pattern: number[]
  control: number[]
  refs: {
    start?: number | null
    bnh?: number | null
    ctl_p95?: number | null
    ctl_p99?: number | null
  }
}

export function histogram(parent: HTMLElement, H: HistogramData) {
  const { g, w, h } = cv(parent, 240)
  const P = { l: 40, r: 12, t: 12, b: 26 }
  const n = H.pattern.length
  const ymax = Math.max(1, ...H.pattern, ...H.control) * 1.08
  const X = (i: number) => P.l + ((w - P.l - P.r) * i) / n
  const Y = (v: number) => h - P.b - ((h - P.t - P.b) * v) / ymax
  axis(g, P.l, h - P.b, w - P.r, P.t)
  // control: outline steps
  g.strokeStyle = C.muted
  g.lineWidth = 1.4
  g.beginPath()
  H.control.forEach((v, i) => {
    g.lineTo(X(i), Y(v))
    g.lineTo(X(i + 1), Y(v))
  })
  g.stroke()
  // pattern: filled bars
  g.fillStyle = 'rgba(38,166,154,.45)'
  H.pattern.forEach((v, i) => {
    if (v) g.fillRect(X(i) + 0.5, Y(v), X(i + 1) - X(i) - 1, h - P.b - Y(v))
  })
  // reference lines on the multiplier axis
  const bx = (m: number) => P.l + ((w - P.l - P.r) * (m - H.bins[0])) / (H.bins[n] - H.bins[0])
  const ref = (
    m: number | null | undefined,
    col: string,
    dash: number[],
    label: string,
    lvl?: number,
  ) => {
    if (m == null || m < H.bins[0] || m > H.bins[n]) return
    g.strokeStyle = col
    g.setLineDash(dash)
    g.beginPath()
    g.moveTo(bx(m), h - P.b)
    g.lineTo(bx(m), P.t)
    g.stroke()
    g.setLineDash([])
    g.fillStyle = col
    g.font = '10px sans-serif'
    g.textAlign = 'left'
    g.fillText(label, bx(m) + 3, P.t + 9 + (lvl || 0) * 11)
  }
  ref(H.refs.start, C.text, [4, 3], 'start', 0)
  ref(H.refs.bnh, C.warn, [], 'B&H x' + fmt(H.refs.bnh), 0)
  ref(H.refs.ctl_p95, C.muted, [2, 3], 'luck p95', 1)
  ref(H.refs.ctl_p99, C.muted, [2, 3], 'p99', 2)
  // x ticks
  g.fillStyle = C.muted
  g.font = '10px sans-serif'
  g.textAlign = 'center'
  for (let i = 0; i <= 4; i++) {
    const m = H.bins[0] + ((H.bins[n] - H.bins[0]) * i) / 4
    g.fillText('x' + fmt(m), bx(m), h - 8)
  }
}

export interface LineSeriesSpec {
  v: (number | null)[]
  c: string
  wd?: number
  dash?: number[]
}

export function lines(
  parent: HTMLElement,
  days: string[],
  series: LineSeriesSpec[],
  hCss: number,
  yfmt: (v: number) => string,
  band?: [LineSeriesSpec, LineSeriesSpec],
  splitIdx?: number | null,
) {
  const { g, w, h } = cv(parent, hCss)
  const P = { l: 44, r: 10, t: 10, b: 24 }
  let lo = Infinity
  let hi = -Infinity
  series.forEach((s) =>
    s.v.forEach((v) => {
      if (v != null) {
        lo = Math.min(lo, v)
        hi = Math.max(hi, v)
      }
    }),
  )
  if (band)
    band.forEach((b) =>
      b.v.forEach((v) => {
        if (v != null) {
          lo = Math.min(lo, v)
          hi = Math.max(hi, v)
        }
      }),
    )
  if (!(hi > lo)) hi = lo + 1
  const pad = (hi - lo) * 0.06
  lo -= pad
  hi += pad
  const X = (i: number) => P.l + ((w - P.l - P.r) * i) / (days.length - 1)
  const Y = (v: number) => h - P.b - ((h - P.t - P.b) * (v - lo)) / (hi - lo)
  axis(g, P.l, h - P.b, w - P.r, P.t)
  if (band) {
    // interquartile band between band[0] and band[1]
    g.fillStyle = 'rgba(38,166,154,.13)'
    g.beginPath()
    band[0].v.forEach((v, i) => (i ? g.lineTo(X(i), Y(v ?? 0)) : g.moveTo(X(i), Y(v ?? 0))))
    for (let i = days.length - 1; i >= 0; i--) g.lineTo(X(i), Y(band[1].v[i] ?? 0))
    g.closePath()
    g.fill()
  }
  series.forEach((s) => {
    g.strokeStyle = s.c
    g.lineWidth = s.wd || 1.6
    g.setLineDash(s.dash || [])
    g.beginPath()
    let started = false
    s.v.forEach((v, i) => {
      if (v == null) return
      started ? g.lineTo(X(i), Y(v)) : g.moveTo(X(i), Y(v))
      started = true
    })
    g.stroke()
    g.setLineDash([])
  })
  if (splitIdx != null && splitIdx >= 0) {
    g.strokeStyle = C.warn
    g.setLineDash([5, 4])
    g.beginPath()
    g.moveTo(X(splitIdx), h - P.b)
    g.lineTo(X(splitIdx), P.t)
    g.stroke()
    g.setLineDash([])
    g.fillStyle = C.warn
    g.font = '10px sans-serif'
    g.textAlign = 'center'
    g.fillText('train | test', X(splitIdx), P.t - 1)
  }
  g.fillStyle = C.muted
  g.font = '10px sans-serif'
  for (let i = 0; i <= 3; i++) {
    const v = lo + ((hi - lo) * i) / 3
    g.textAlign = 'right'
    g.fillText(yfmt(v), P.l - 5, Y(v) + 3)
  }
  g.textAlign = 'left'
  g.fillText(days[0], P.l, h - 8)
  g.textAlign = 'right'
  g.fillText(days[days.length - 1], w - P.r, h - 8)
}

export function hbars<T extends Record<string, unknown>>(
  parent: HTMLElement,
  rows: T[],
  valKey: keyof T,
  labKey: keyof T,
  hint: string,
) {
  const hCss = rows.length * 22 + 30
  const { g, w } = cv(parent, hCss)
  const vmax = Math.max(...rows.map((r) => Math.abs(Number(r[valKey]))), 1e-9)
  const mid = w * 0.58
  const half = w * 0.17
  const labelX = mid - half - 10
  rows.forEach((r, i) => {
    const y = 14 + i * 22
    const v = Number(r[valKey])
    g.fillStyle = C.muted
    g.font = '11.5px sans-serif'
    g.textAlign = 'right'
    g.fillText(String(r[labKey]), labelX, y + 4)
    const len = (Math.abs(v) / vmax) * half
    g.fillStyle = v >= 0 ? 'rgba(38,166,154,.7)' : 'rgba(239,83,80,.7)'
    g.fillRect(v >= 0 ? mid : mid - len, y - 6, Math.max(len, 1), 12)
    g.fillStyle = C.text
    if (v >= 0) {
      g.textAlign = 'left'
      g.fillText(fmt(v, 3), mid + len + 5, y + 4)
    } else {
      g.textAlign = 'left'
      g.fillText(fmt(v, 3), mid + 5, y + 4)
    }
  })
  g.strokeStyle = C.border
  g.beginPath()
  g.moveTo(mid + 0.5, 4)
  g.lineTo(mid + 0.5, hCss - 22)
  g.stroke()
  g.fillStyle = C.muted
  g.font = '10px sans-serif'
  g.textAlign = 'center'
  g.fillText(hint, mid, hCss - 6)
}

export function scatter(parent: HTMLElement, pts: [number, number, number][]) {
  const { g, w, h } = cv(parent, 380)
  const P = { l: 46, r: 14, t: 14, b: 30 }
  const X = (v: number) => P.l + (w - P.l - P.r) * v
  const Y = (v: number) => h - P.b - (h - P.t - P.b) * v
  axis(g, P.l, h - P.b, w - P.r, P.t)
  g.strokeStyle = C.border
  g.setLineDash([4, 4])
  g.beginPath()
  g.moveTo(X(0), Y(0))
  g.lineTo(X(1), Y(1))
  g.stroke()
  g.setLineDash([])
  pts.forEach(([a, b, ctl]) => {
    g.fillStyle = ctl ? 'rgba(139,148,158,.55)' : 'rgba(38,166,154,.5)'
    g.beginPath()
    g.arc(X(a), Y(b), 2.1, 0, 7)
    g.fill()
  })
  g.fillStyle = C.muted
  g.font = '11px sans-serif'
  g.textAlign = 'center'
  g.fillText('rank in TRAIN period →', (P.l + w - P.r) / 2, h - 8)
  g.save()
  g.translate(12, (P.t + h - P.b) / 2)
  g.rotate(-Math.PI / 2)
  g.fillText('rank in TEST period →', 0, 0)
  g.restore()
}

// Worst-case panel: per-window Sharpe report card. Green above zero, red
// below; ★ marks the window the bot was selected on (in-sample by
// construction); the reserved test span gets a warn-colored label.
export function regimeBars(
  parent: HTMLElement,
  labels: string[],
  vals: (number | null)[],
  bornIdx: number,
  hCss = 170,
) {
  const { g, w, h } = cv(parent, hCss)
  const P = { l: 40, r: 10, t: 16, b: 22 }
  let lo = 0
  let hi = 0
  vals.forEach((v) => {
    if (v != null) {
      lo = Math.min(lo, v)
      hi = Math.max(hi, v)
    }
  })
  if (!(hi > lo)) hi = lo + 1
  const pad = (hi - lo) * 0.12
  lo -= pad
  hi += pad
  const bw = (w - P.l - P.r) / vals.length
  const Y = (v: number) => h - P.b - ((h - P.t - P.b) * (v - lo)) / (hi - lo)
  axis(g, P.l, h - P.b, w - P.r, P.t)
  // zero line
  g.strokeStyle = C.muted
  g.setLineDash([3, 3])
  g.beginPath()
  g.moveTo(P.l, Y(0) + 0.5)
  g.lineTo(w - P.r, Y(0) + 0.5)
  g.stroke()
  g.setLineDash([])
  vals.forEach((v, i) => {
    const x = P.l + i * bw + 4
    const isTest = labels[i] === 'test'
    if (v != null) {
      g.fillStyle =
        v >= 0 ? (isTest ? 'rgba(240,180,41,.75)' : 'rgba(38,166,154,.7)') : 'rgba(239,83,80,.7)'
      g.fillRect(x, Math.min(Y(v), Y(0)), bw - 8, Math.max(Math.abs(Y(v) - Y(0)), 1))
      g.fillStyle = v >= 0 ? C.text : C.down
      g.font = '10px sans-serif'
      g.textAlign = 'center'
      g.fillText(fmt(v, 1), x + (bw - 8) / 2, v >= 0 ? Y(v) - 3 : Y(v) + 11)
    }
    g.fillStyle = isTest ? C.warn : C.muted
    g.font = '10px sans-serif'
    g.textAlign = 'center'
    g.fillText((i === bornIdx ? '★' : '') + labels[i], x + (bw - 8) / 2, h - 8)
  })
  // y ticks
  g.fillStyle = C.muted
  g.font = '10px sans-serif'
  g.textAlign = 'right'
  for (let i = 0; i <= 2; i++) {
    const v = lo + ((hi - lo) * i) / 2
    g.fillText(fmt(v, 1), P.l - 5, Y(v) + 3)
  }
}

// Worst-case panel: underwater curve — % below the running equity peak over
// the whole walk-forward record. The only chart where flat-at-zero is good.
export function underwater(
  parent: HTMLElement,
  days: string[],
  ddPct: number[],
  splitIdx: number | null,
  hCss = 150,
) {
  const { g, w, h } = cv(parent, hCss)
  const P = { l: 44, r: 10, t: 10, b: 22 }
  const lo = Math.min(-1, ...ddPct) * 1.1
  const X = (i: number) => P.l + ((w - P.l - P.r) * i) / (ddPct.length - 1)
  const Y = (v: number) => P.t + ((h - P.t - P.b) * v) / lo
  axis(g, P.l, h - P.b, w - P.r, P.t)
  g.fillStyle = 'rgba(239,83,80,.35)'
  g.strokeStyle = C.down
  g.lineWidth = 1.2
  g.beginPath()
  g.moveTo(X(0), Y(0))
  ddPct.forEach((v, i) => g.lineTo(X(i), Y(v)))
  g.lineTo(X(ddPct.length - 1), Y(0))
  g.closePath()
  g.fill()
  g.beginPath()
  ddPct.forEach((v, i) => (i ? g.lineTo(X(i), Y(v)) : g.moveTo(X(i), Y(v))))
  g.stroke()
  // deepest point marker
  let mi = 0
  ddPct.forEach((v, i) => {
    if (v < ddPct[mi]) mi = i
  })
  g.fillStyle = C.down
  g.beginPath()
  g.arc(X(mi), Y(ddPct[mi]), 2.5, 0, 7)
  g.fill()
  g.font = '10px sans-serif'
  g.textAlign = mi > ddPct.length / 2 ? 'right' : 'left'
  g.fillText(fmt(ddPct[mi], 1) + '%', X(mi) + (mi > ddPct.length / 2 ? -5 : 5), Y(ddPct[mi]) + 3)
  if (splitIdx != null && splitIdx >= 0) {
    g.strokeStyle = C.warn
    g.setLineDash([5, 4])
    g.beginPath()
    g.moveTo(X(splitIdx), h - P.b)
    g.lineTo(X(splitIdx), P.t)
    g.stroke()
    g.setLineDash([])
    g.fillStyle = C.warn
    g.textAlign = 'center'
    g.fillText('test', X(splitIdx), P.t - 1)
  }
  g.fillStyle = C.muted
  g.textAlign = 'right'
  g.fillText('0%', P.l - 5, Y(0) + 3)
  g.fillText(fmt(lo, 0) + '%', P.l - 5, Y(lo) + 3)
  g.textAlign = 'left'
  g.fillText(days[0] ?? '', P.l, h - 8)
  g.textAlign = 'right'
  g.fillText(days[days.length - 1] ?? '', w - P.r, h - 8)
}

// Worst-case panel: histogram of bootstrap-resampled test outcomes ("alternate
// histories" — same daily returns, shuffled with replacement). Marks 0%, the
// 5th percentile, and what actually happened.
export function outcomeHist(
  parent: HTMLElement,
  finalsPct: number[],
  actualPct: number | null,
  p5Pct: number,
  hCss = 170,
) {
  const { g, w, h } = cv(parent, hCss)
  const P = { l: 14, r: 14, t: 16, b: 22 }
  const lo = Math.min(...finalsPct, 0)
  const hi = Math.max(...finalsPct, actualPct ?? 0)
  const span = hi - lo || 1
  const NB = 41
  const counts = new Array(NB).fill(0)
  finalsPct.forEach((v) => {
    const bi = Math.min(NB - 1, Math.max(0, Math.floor(((v - lo) / span) * NB)))
    counts[bi]++
  })
  const cmax = Math.max(...counts, 1)
  const X = (v: number) => P.l + ((w - P.l - P.r) * (v - lo)) / span
  const Y = (n: number) => h - P.b - ((h - P.t - P.b) * n) / (cmax * 1.08)
  axis(g, P.l, h - P.b, w - P.r, P.t)
  const bw = (w - P.l - P.r) / NB
  counts.forEach((n, i) => {
    if (!n) return
    const vMid = lo + ((i + 0.5) / NB) * span
    g.fillStyle = vMid < 0 ? 'rgba(239,83,80,.45)' : 'rgba(38,166,154,.45)'
    g.fillRect(P.l + i * bw + 0.5, Y(n), bw - 1, h - P.b - Y(n))
  })
  const ref = (v: number | null, col: string, dash: number[], label: string, lvl = 0) => {
    if (v == null || v < lo || v > hi) return
    g.strokeStyle = col
    g.setLineDash(dash)
    g.beginPath()
    g.moveTo(X(v), h - P.b)
    g.lineTo(X(v), P.t)
    g.stroke()
    g.setLineDash([])
    g.fillStyle = col
    g.font = '10px sans-serif'
    g.textAlign = X(v) > w * 0.75 ? 'right' : 'left'
    g.fillText(label, X(v) + (X(v) > w * 0.75 ? -3 : 3), P.t + 9 + lvl * 11)
  }
  ref(actualPct, C.warn, [], 'actual ' + fmt(actualPct ?? 0, 0) + '%', 0)
  ref(p5Pct, C.down, [2, 3], 'p5 ' + fmt(p5Pct, 0) + '%', 1)
  ref(0, C.muted, [4, 3], 'break-even', 2)
  g.fillStyle = C.muted
  g.font = '10px sans-serif'
  g.textAlign = 'center'
  for (let i = 0; i <= 4; i++) {
    const v = lo + (span * i) / 4
    g.fillText(fmt(v, 0) + '%', X(v), h - 8)
  }
}

export function vbars(parent: HTMLElement, vals: number[], labels: string[], color: string) {
  const { g, w, h } = cv(parent, 200)
  const P = { l: 40, r: 10, t: 14, b: 26 }
  const vmax = Math.max(...vals, 1) * 1.1
  const bw = (w - P.l - P.r) / vals.length
  axis(g, P.l, h - P.b, w - P.r, P.t)
  vals.forEach((v, i) => {
    const bh = ((h - P.t - P.b) * v) / vmax
    g.fillStyle = color
    g.fillRect(P.l + i * bw + 3, h - P.b - bh, bw - 6, bh)
    g.fillStyle = C.muted
    g.font = '10px sans-serif'
    g.textAlign = 'center'
    g.fillText(labels[i], P.l + (i + 0.5) * bw, h - 10)
    g.fillStyle = C.text
    g.fillText(String(v), P.l + (i + 0.5) * bw, h - P.b - bh - 4)
  })
}
