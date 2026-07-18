import { useMemo, type ReactNode } from 'react'
import CanvasBox from '../../components/CanvasBox'
import { outcomeHist, regimeBars, underwater } from '../../lib/canvas'
import { fmt } from '../../lib/format'
import type { EvoBotPayload } from '../../types'

// Deterministic PRNG so the bootstrap doesn't flicker between renders.
function mulberry32(seed: number) {
  let a = seed >>> 0
  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const B = 2000 // bootstrap resamples

// Everything here is computed from data the payload already carries: the
// per-window record (gen_perf), the chained daily equity (eq + eq_test) and
// the run's null-cohort ceilings (run_ctx). No extra API calls.
function useStress(data: EvoBotPayload) {
  return useMemo(() => {
    const b = data.bot
    const perf = b.gen_perf || []
    const born = Number(b.born_gen)
    const gens = data.run_ctx?.gens ?? perf.length

    // ---- per-window report card -----------------------------------------
    const labels = [...perf.map((p) => 'g' + p.gen), 'test']
    const sharpes: (number | null)[] = [...perf.map((p) => p.sharpe), b.test?.sharpe ?? null]
    const negWindows = perf.filter((p) => (p.sharpe ?? 0) < 0)
    let worst: { label: string; s: number } | null = null
    perf.forEach((p) => {
      if (p.sharpe != null && (worst === null || p.sharpe < worst.s))
        worst = { label: 'g' + p.gen, s: p.sharpe }
    })
    const oosWindows = isNaN(born) ? null : Math.max(0, gens - 1 - born)

    // ---- chained equity → drawdown --------------------------------------
    const eqAll: number[] = []
    let base = 1
    for (const w of b.eq || []) {
      for (const x of w) eqAll.push(base * x)
      if (w.length) base *= w[w.length - 1]
    }
    const splitIdx = eqAll.length ? eqAll.length : null
    for (const x of b.eq_test || []) eqAll.push(base * x)
    const dd: number[] = []
    let peak = -Infinity
    let maxDD = 0
    let underDays = 0
    let worstUnder = 0
    for (const v of eqAll) {
      peak = Math.max(peak, v)
      const d = (v / peak - 1) * 100
      dd.push(d)
      maxDD = Math.min(maxDD, d)
      underDays = d < -0.001 ? underDays + 1 : 0
      worstUnder = Math.max(worstUnder, underDays)
    }
    const days = [
      ...(data.windows || []).flatMap((w) => w.days),
      ...(data.test.days || []),
    ].slice(0, eqAll.length)

    // ---- test-span daily returns → bootstrap + tail stats ----------------
    const eqT = b.eq_test || []
    const logr: number[] = []
    for (let i = 1; i < eqT.length; i++)
      if (eqT[i - 1] > 0 && eqT[i] > 0) logr.push(Math.log(eqT[i] / eqT[i - 1]))
    let finals: number[] = []
    let pLoss: number | null = null
    let p5 = 0
    let noBest5: number | null = null
    let bestDay: number | null = null
    let worstDay: number | null = null
    if (logr.length >= 30) {
      const rng = mulberry32((b.bot_id + 1) * 2654435761)
      finals = new Array(B)
      for (let k = 0; k < B; k++) {
        let s = 0
        for (let i = 0; i < logr.length; i++) s += logr[(rng() * logr.length) | 0]
        finals[k] = (Math.exp(s) - 1) * 100
      }
      finals.sort((a, z) => a - z)
      pLoss = (finals.filter((v) => v < 0).length / B) * 100
      p5 = finals[Math.floor(B * 0.05)]
      const total = logr.reduce((a, v) => a + v, 0)
      const top5 = [...logr].sort((a, z) => z - a).slice(0, 5)
      noBest5 = (Math.exp(total - top5.reduce((a, v) => a + v, 0)) - 1) * 100
      bestDay = (Math.exp(Math.max(...logr)) - 1) * 100
      worstDay = (Math.exp(Math.min(...logr)) - 1) * 100
    }
    const actual = eqT.length ? (eqT[eqT.length - 1] / eqT[0] - 1) * 100 : null

    // ---- luck context -----------------------------------------------------
    const ceilings = [data.run_ctx?.placebo_max, data.run_ctx?.fresh_max].filter(
      (v): v is number => v != null,
    )
    const luckCeil = ceilings.length ? Math.max(...ceilings) : null
    const testS = b.test?.sharpe ?? null

    return {
      labels, sharpes, negWindows: negWindows.length, nWindows: perf.length, worst,
      oosWindows, days, dd, splitIdx, maxDD, worstUnder,
      finals, pLoss, p5, noBest5, bestDay, worstDay, actual, luckCeil, testS,
    }
  }, [data])
}

function Flag({ bad, children, title }: { bad: boolean; children: ReactNode; title?: string }) {
  return (
    <span className={'chip' + (bad ? ' red' : ' gray')} title={title}>
      {children}
    </span>
  )
}

export default function WorstCase({ data }: { data: EvoBotPayload }) {
  const s = useStress(data)
  const worst = s.worst as { label: string; s: number } | null
  if (!s.sharpes.some((v) => v != null)) return null

  const noOOS = s.oosWindows !== null && s.oosWindows === 0
  const thinOOS = s.oosWindows !== null && s.oosWindows === 1
  const belowLuck = s.testS != null && s.luckCeil != null && s.testS < s.luckCeil

  return (
    <div className="card" style={{ gridColumn: 'span 12' }}>
      <h3>Worst case</h3>
      <div className="sub">
        the same bot, viewed through what could have gone (or already went) wrong — a good
        test-span number alone is exactly what a lucky order statistic looks like.
      </div>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 10 }}>
        <Flag
          bad={s.negWindows >= Math.ceil(s.nWindows / 2)}
          title="how many fitness windows this genome lost in when re-scored on each"
        >
          negative in {s.negWindows}/{s.nWindows} history windows
        </Flag>
        {worst && (
          <Flag bad={worst.s < -1} title="its worst re-scored fitness window">
            worst window {worst.label}: S {fmt(worst.s, 2)}
          </Flag>
        )}
        <Flag
          bad={noOOS || thinOOS}
          title="windows after the one it was selected on but before the test span — the only honest pre-test record"
        >
          {noOOS
            ? '⚠ no out-of-sample window before test'
            : `${s.oosWindows ?? '—'} OOS window${s.oosWindows === 1 ? '' : 's'} before test`}
        </Flag>
        <Flag bad={s.maxDD < -30} title="deepest peak-to-trough of the chained walk-forward equity">
          max drawdown {fmt(s.maxDD, 1)}%
        </Flag>
        <Flag bad={s.worstUnder > 180} title="longest stretch of days below a prior equity peak">
          {s.worstUnder} days underwater
        </Flag>
        {s.pLoss != null && (
          <Flag
            bad={s.pLoss > 10}
            title={`share of ${B} bootstrap resamples of its own test-span daily returns that end below break-even`}
          >
            P(loss) in alternate histories: {fmt(s.pLoss, 1)}%
          </Flag>
        )}
        {s.noBest5 != null && s.actual != null && (
          <Flag
            bad={s.actual > 0 && s.noBest5 < s.actual / 3}
            title="test return recomputed with its 5 best days removed — concentration check"
          >
            without 5 best days: {fmt(s.actual, 0)}% → {fmt(s.noBest5, 0)}%
          </Flag>
        )}
        {s.testS != null && s.luckCeil != null && (
          <Flag
            bad={belowLuck}
            title="best test Sharpe reached by this run's placebo / fresh-random null bots — pure luck reaches this high at this scale"
          >
            test S {fmt(s.testS, 2)} vs luck ceiling {fmt(s.luckCeil, 2)}
          </Flag>
        )}
      </div>
      <div className="grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(12, 1fr)', gap: 10 }}>
        <div style={{ gridColumn: 'span 4', minWidth: 0 }}>
          <div className="sub">
            regime report card — Sharpe per window (★ = selected there, in-sample)
          </div>
          <CanvasBox
            draw={(el) => regimeBars(el, s.labels, s.sharpes, Number(data.bot.born_gen))}
            deps={[s.sharpes.join(','), s.labels.join(',')]}
          />
        </div>
        <div style={{ gridColumn: 'span 4', minWidth: 0 }}>
          <div className="sub">underwater — % below the running peak, whole record</div>
          <CanvasBox
            draw={(el) => underwater(el, s.days, s.dd, s.splitIdx)}
            deps={[data.bot.bot_id, s.dd.length]}
          />
        </div>
        <div style={{ gridColumn: 'span 4', minWidth: 0 }}>
          <div className="sub">
            {s.finals.length
              ? `${B} alternate test spans (its own daily returns, reshuffled)`
              : 'not enough test days for the bootstrap'}
          </div>
          {s.finals.length > 0 && (
            <CanvasBox
              draw={(el) => outcomeHist(el, s.finals, s.actual, s.p5)}
              deps={[data.bot.bot_id, s.finals.length]}
            />
          )}
        </div>
      </div>
    </div>
  )
}
