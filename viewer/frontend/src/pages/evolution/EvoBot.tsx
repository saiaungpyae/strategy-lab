import { useEffect, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import CanvasBox from '../../components/CanvasBox'
import { lines } from '../../lib/canvas'
import { C } from '../../lib/colors'
import { getJSON } from '../../lib/api'
import { fmt, money } from '../../lib/format'
import StressForecast from './StressForecast'
import WorstCase from './WorstCase'
import type { EvoBotPayload } from '../../types'

const v = (x: number | string | null | undefined, d = 2) =>
  x == null || x === '' ? '—' : (+x).toFixed(d)

function Chip({ label, value }: { label: string; value: string }) {
  return (
    <span className="chip" title={label}>
      <span className="muted">{label}</span> {value}
    </span>
  )
}

export default function EvoBot() {
  const [data, setData] = useState<EvoBotPayload | null>(null)
  const [eqMode, setEqMode] = useState<'usd' | 'window'>('usd')
  const [params] = useSearchParams()
  const run = params.get('run') ?? ''
  const botId = params.get('bot') ?? ''

  useEffect(() => {
    document.title = `strategy-lab · Bot ${botId}`
  }, [botId])

  useEffect(() => {
    if (!run || botId === '') return
    setData(null)
    getJSON<EvoBotPayload>(
      `/api/swarm/evo/bot?id=${encodeURIComponent(run)}&bot=${encodeURIComponent(botId)}`,
    )
      .then(setData)
      .catch(() =>
        setData({ error: 'failed to load', run_id: run, has_history: false, windows: [], test: {}, bot: { bot_id: +botId } as EvoBotPayload['bot'] }),
      )
  }, [run, botId])

  const b = data?.bot
  const windows = data?.windows || []
  const perf = b?.gen_perf || []
  const born = Number(b?.born_gen)

  // per-generation Sharpe series with the reserved test span as the last point
  const genDays = [...windows.map((w) => w.span[1]), 'test']
  const genSharpes: (number | null)[] = [
    ...perf.map((p) => p.sharpe),
    b?.test?.sharpe ?? null,
  ]

  // walk-forward equity: every fitness window restarts at ×1.00, then the test
  const eqDays = [...windows.flatMap((w) => w.days), ...(data?.test.days || [])]
  const eqGen = (b?.eq || []).flat()
  const eqTest = b?.eq_test || []

  // holdings in $: start capital compounded window by window, then the test
  const startCap = data?.start_capital ?? 10000
  const usdGen: number[] = []
  const winEndUsd: number[] = []
  let base = 1
  for (const w of b?.eq || []) {
    for (const x of w) usdGen.push(base * x * startCap)
    if (w.length) base *= w[w.length - 1]
    winEndUsd.push(base * startCap)
  }
  const usdTest = eqTest.map((x) => base * x * startCap)
  const finalUsd = usdTest.length
    ? usdTest[usdTest.length - 1]
    : usdGen.length
      ? usdGen[usdGen.length - 1]
      : null

  const usdMode = eqMode === 'usd'
  const genVals = usdMode ? usdGen : eqGen
  const testVals = usdMode ? usdTest : eqTest
  const genSeries: (number | null)[] = [...genVals, ...testVals.map(() => null)]
  const testSeries: (number | null)[] = [...genVals.map(() => null), ...testVals]

  return (
    <div className="swarm">
      <div className="top">
        <h1>
          strategy-lab <span>/ Evolution / Bot {botId}</span>
        </h1>
        <span className="muted">{run}</span>
        <span style={{ flex: 1 }} />
        {data?.has_history && (
          <Link
            to={`/chart?run=${encodeURIComponent(run)}&bot=${encodeURIComponent(botId)}`}
            title="replay every position of this bot on the candle chart"
          >
            view in chart 📈
          </Link>
        )}
        <Link to={'/evolution/bots' + (run ? `?run=${encodeURIComponent(run)}` : '')}>
          ← top bots
        </Link>
        <Link to={'/evolution' + (run ? `?run=${encodeURIComponent(run)}` : '')}>evolution</Link>
      </div>
      <div className="wrap">
        {!data ? (
          <div className="empty">loading…</div>
        ) : data.error || !b ? (
          <div className="empty">{data.error || 'bot not found'}</div>
        ) : (
          <div className="grid">
            <div className="card" style={{ gridColumn: 'span 12' }}>
              <h3>Ideology</h3>
              <div style={{ fontSize: 14, margin: '4px 0 10px' }}>{b.rules}</div>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                <Chip label="born gen" value={String(b.born_gen ?? '—')} />
                <Chip label="tf" value={String(b.tf ?? '—')} />
                <Chip label="session" value={String(b.session ?? '—')} />
                <Chip label="dir" value={String(b.dir_bias ?? '—')} />
                <Chip label="risk" value={v(b.risk_pct) + '%'} />
                <Chip label="stop" value={v(b.stop_atr) + ' ATR'} />
                <Chip label="tp" value={b.tp_rr == null ? 'none' : v(b.tp_rr) + 'R'} />
                <Chip label="max hold" value={String(b.max_hold_bars ?? '—') + ' bars'} />
                <Chip label="maker off" value={v(b.maker_off_atr) + ' ATR'} />
                <Chip label="ttl" value={String(b.order_ttl ?? '—')} />
                <Chip label="loss react" value={String(b.loss_react ?? '—')} />
                <Chip label="cooldown" value={String(b.cooldown_bars ?? '—')} />
                <Chip label="reentry gap" value={String(b.reentry_gap ?? '—')} />
              </div>
            </div>
            {data.has_history ? (
              <>
                <div className="card" style={{ gridColumn: 'span 7' }}>
                  <h3>Sharpe by generation window</h3>
                  <div className="sub">
                    this genome re-scored on every fitness window. Gen {isNaN(born) ? '?' : born}{' '}
                    is where it was selected (in-sample); every later window and the reserved
                    test span are out-of-sample for it.
                  </div>
                  <CanvasBox
                    draw={(el) =>
                      lines(el, genDays, [{ v: genSharpes, c: C.up, wd: 2 }], 220, (x) =>
                        fmt(x, 1),
                      )
                    }
                    deps={[genSharpes.join(',')]}
                  />
                </div>
                <div className="card scroll" style={{ gridColumn: 'span 5', padding: 0 }}>
                  <table>
                    <thead>
                      <tr>
                        <th>gen</th>
                        <th className="l">window</th>
                        <th>S</th>
                        <th>ret</th>
                        <th>holdings</th>
                        <th>trades</th>
                      </tr>
                    </thead>
                    <tbody>
                      {perf.map((p) => (
                        <tr key={p.gen}>
                          <td>
                            {p.gen}
                            {p.gen === born ? ' ★' : ''}
                          </td>
                          <td className="l muted">
                            {windows[p.gen]?.span[0]} → {windows[p.gen]?.span[1]}
                            {p.gen === born ? ' · selected here' : ''}
                          </td>
                          <td className={(p.sharpe ?? 0) >= 0 ? 'pos' : 'neg'}>
                            <b>{v(p.sharpe)}</b>
                          </td>
                          <td>{p.ret_pct == null ? '—' : fmt(p.ret_pct, 1) + '%'}</td>
                          <td className={(winEndUsd[p.gen] ?? startCap) >= startCap ? 'pos' : 'neg'}>
                            {money(winEndUsd[p.gen])}
                          </td>
                          <td>{p.dead ? '☠ ' : ''}{p.trades}</td>
                        </tr>
                      ))}
                      <tr>
                        <td className="muted">test</td>
                        <td className="l muted">
                          reserved from {(data.test.start || '').slice(0, 10)}
                        </td>
                        <td className={(b.test?.sharpe ?? 0) >= 0 ? 'pos' : 'neg'}>
                          <b>{v(b.test?.sharpe)}</b>
                        </td>
                        <td>{b.test?.ret_pct == null ? '—' : fmt(b.test.ret_pct, 1) + '%'}</td>
                        <td className={(finalUsd ?? startCap) >= startCap ? 'pos' : 'neg'}>
                          <b>{money(finalUsd)}</b>
                        </td>
                        <td>{b.test?.dead ? '☠ ' : ''}{b.test?.trades ?? '—'}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
                <div className="card" style={{ gridColumn: 'span 12' }}>
                  <div style={{ display: 'flex', alignItems: 'baseline', gap: 12 }}>
                    <h3>{usdMode ? 'Holdings in USD' : 'Walk-forward equity'}</h3>
                    <span style={{ flex: 1 }} />
                    {usdMode && (
                      <span className="chip">
                        {money(startCap)} → <b>{money(finalUsd)}</b>
                      </span>
                    )}
                    <select
                      title="equity view"
                      value={eqMode}
                      onChange={(e) => setEqMode(e.target.value as 'usd' | 'window')}
                    >
                      <option value="usd">holdings $ (compounded)</option>
                      <option value="window">per-window ×</option>
                    </select>
                  </div>
                  <div className="sub">
                    {usdMode
                      ? `${money(startCap)} handed to this bot at the start and left to ride: ` +
                        'each fitness window is simulated independently, so the $ curve chains ' +
                        'the window multipliers in sequence (%-risk sizing makes them chainable). ' +
                        'Teal = fitness windows, orange = reserved test span.'
                      : 'capital multiplier day by day — every fitness window restarts at ×1.00 ' +
                        '(the raw walk-forward record, not compounded). Teal = fitness windows, ' +
                        'orange = reserved test span.'}
                  </div>
                  <CanvasBox
                    draw={(el) =>
                      lines(
                        el,
                        eqDays,
                        [
                          { v: genSeries, c: C.up, wd: 1.6 },
                          { v: testSeries, c: C.warn, wd: 1.6 },
                        ],
                        260,
                        usdMode ? (x) => '$' + fmt(x / 1000, 1) + 'k' : (x) => 'x' + fmt(x, 2),
                        undefined,
                        eqGen.length,
                      )
                    }
                    deps={[b.bot_id, run, eqDays.length, eqMode]}
                  />
                </div>
                <WorstCase data={data} />
                <StressForecast run={run} botId={botId} />
              </>
            ) : (
              <div className="card" style={{ gridColumn: 'span 12' }}>
                <h3>Reserved test span</h3>
                <div className="sub">
                  this run predates per-generation history (hof_history.json) — re-run an
                  evolution to get the full walk-forward record. Test-span result:
                </div>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  <Chip label="test S" value={v(b.test_sharpe)} />
                  <Chip label="test trades" value={String(b.test_trades ?? '—')} />
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
