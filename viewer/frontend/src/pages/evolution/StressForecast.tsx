import { useEffect, useState } from 'react'
import CanvasBox from '../../components/CanvasBox'
import { lines } from '../../lib/canvas'
import { getJSON } from '../../lib/api'
import { fmt } from '../../lib/format'
import type { EvoStressPayload, StressScenario } from '../../types'

// Scenario palette — replay (the anchor) is muted, everything else hostile.
const COLORS: Record<string, string> = {
  replay: '#8b949e',
  crash: '#ef5350',
  meltup: '#26a69a',
  grind: '#f0b429',
  whipsaw: '#ab7df8',
  flatline: '#4dabf7',
  vshape: '#f78fb3',
  gate_flip: '#ff8f40',
}

// A scenario "crashes" the bot if it dies, loses a quarter of the account,
// or draws down 40%+ — thresholds chosen to mean "you would not keep running
// this live", not "it had a bad week".
const crashed = (s: StressScenario) => s.dead || s.ret_pct <= -25 || s.maxdd_pct <= -40

export default function StressForecast({ run, botId }: { run: string; botId: string }) {
  const [data, setData] = useState<EvoStressPayload | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [hidden, setHidden] = useState<Set<string>>(new Set())

  useEffect(() => {
    setData(null)
    setErr(null)
    setHidden(new Set())
    getJSON<EvoStressPayload>(
      `/api/swarm/evo/stress?id=${encodeURIComponent(run)}&bot=${encodeURIComponent(botId)}`,
    )
      .then((p) => (p.error ? setErr(p.error) : setData(p)))
      .catch(() => setErr('stress forecast failed to load'))
  }, [run, botId])

  if (err)
    return (
      <div className="card" style={{ gridColumn: 'span 12' }}>
        <h3>Hostile futures</h3>
        <div className="sub">{err}</div>
      </div>
    )
  if (!data)
    return (
      <div className="card" style={{ gridColumn: 'span 12' }}>
        <h3>Hostile futures</h3>
        <div className="sub">
          engineering hostile futures and replaying this genome through each… (the first
          bot per dataset rebuilds the feature tape and can take a minute; afterwards it's
          cached)
        </div>
      </div>
    )

  const sc = data.scenarios
  const hostile = sc.filter((s) => s.key !== 'replay')
  const crashes = hostile.filter(crashed)
  let worst = hostile[0]
  hostile.forEach((s) => {
    if (s.ret_pct < worst.ret_pct) worst = s
  })
  const nDays = Math.max(...sc.map((s) => s.eq.length))
  const days = Array.from({ length: nDays }, (_, i) => 'day ' + i)
  const visible = sc.filter((s) => !hidden.has(s.key))
  const series = visible.map((s) => ({
    v: [...s.eq, ...Array(Math.max(0, nDays - s.eq.length)).fill(null)] as (number | null)[],
    c: COLORS[s.key] ?? '#8b949e',
    wd: s.key === 'replay' ? 1.2 : 1.7,
    dash: s.key === 'replay' ? [5, 4] : undefined,
  }))

  const toggle = (k: string) =>
    setHidden((h) => {
      const n = new Set(h)
      if (n.has(k)) n.delete(k)
      else n.add(k)
      return n
    })

  return (
    <div className="card" style={{ gridColumn: 'span 12' }}>
      <h3>Hostile futures — can anything crash this bot?</h3>
      <div className="sub">
        the same genome, replayed by the engine over {hostile.length} synthetic 90-day
        futures engineered to hurt (plus the last 90 real days as anchor, dashed).
        Price is synthetic; volume, funding and positioning replay the current season, so
        the bot's gates behave exactly as they do today. These are engineered attacks, not
        predictions.
      </div>
      <div style={{ margin: '2px 0 8px', fontSize: 13 }}>
        {crashes.length ? (
          <span className="neg">
            💥 crashed in {crashes.length} of {hostile.length} hostile futures — worst:{' '}
            {worst.label} → bot {fmt(worst.ret_pct, 0)}% (maxDD {fmt(worst.maxdd_pct, 0)}%
            {worst.dead ? ', ruined ☠' : ''}). This bot has a kryptonite season.
          </span>
        ) : (
          <span className="pos">
            ✅ survived all {hostile.length} hostile futures — worst was {worst.label} →
            bot {fmt(worst.ret_pct, 0)}% (maxDD {fmt(worst.maxdd_pct, 0)}%). Robust across
            every season tested here; that is evidence of sturdiness, not of edge.
          </span>
        )}
      </div>
      <CanvasBox
        draw={(el) => lines(el, days, series, 240, (x) => 'x' + fmt(x, 2))}
        deps={[data.bot_id, data.run_id, visible.map((s) => s.key).join(',')]}
      />
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 8 }}>
        {sc.map((s) => (
          <span
            key={s.key}
            className={'chip' + (crashed(s) && s.key !== 'replay' ? ' red' : '')}
            title={s.desc + ` · ${s.trades} trades · maxDD ${fmt(s.maxdd_pct, 1)}%`}
            onClick={() => toggle(s.key)}
            style={{
              cursor: 'pointer',
              opacity: hidden.has(s.key) ? 0.35 : 1,
              borderLeft: `3px solid ${COLORS[s.key] ?? '#8b949e'}`,
            }}
          >
            {s.dead ? '☠ ' : ''}
            {s.label} <b className={s.ret_pct >= 0 ? 'pos' : 'neg'}>{fmt(s.ret_pct, 0)}%</b>
          </span>
        ))}
      </div>
    </div>
  )
}
