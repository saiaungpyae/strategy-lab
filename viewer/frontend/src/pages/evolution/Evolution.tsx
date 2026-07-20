import { useCallback, useEffect, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import CanvasBox from '../../components/CanvasBox'
import TopNav from '../../components/TopNav'
import { lines } from '../../lib/canvas'
import { C } from '../../lib/colors'
import { getJSON, postJSON } from '../../lib/api'
import { fmt } from '../../lib/format'
import type { EvoGenStat, EvoRun, EvosPayload, StartResult } from '../../types'

function EvoChart({ gs }: { gs: EvoGenStat[] }) {
  if (gs.length < 2)
    return (
      <div className="muted" style={{ textAlign: 'center', padding: '40px 0' }}>
        chart appears after the second generation…
      </div>
    )
  return (
    <CanvasBox
      draw={(el) =>
        lines(
          el,
          gs.map((x) => x.window[1]),
          [
            { v: gs.map((x) => x.evolved.median_sharpe ?? null), c: C.up, wd: 2 },
            { v: gs.map((x) => x.placebo.median_sharpe ?? null), c: C.muted, wd: 2 },
            { v: gs.map((x) => x.evolved.p90_sharpe ?? null), c: C.up, dash: [4, 3], wd: 1 },
            { v: gs.map((x) => x.placebo.p90_sharpe ?? null), c: C.muted, dash: [4, 3], wd: 1 },
          ],
          240,
          (x) => fmt(x, 1),
        )
      }
      deps={[gs]}
    />
  )
}

function EvoLegend() {
  return (
    <div className="legend" style={{ marginTop: 6 }}>
      <span>
        <i style={{ background: C.up }} />
        evolved median
      </span>
      <span>
        <i style={{ background: C.muted }} />
        placebo median
      </span>
      <span className="muted">dashed = p90</span>
    </div>
  )
}

function GenTable({ gs }: { gs: EvoGenStat[] }) {
  if (!gs.length)
    return <div className="muted" style={{ padding: 14 }}>waiting for the first generation…</div>
  return (
    <table>
      <thead>
        <tr>
          <th>gen</th>
          <th className="l">fitness window</th>
          <th>evolved S</th>
          <th>p90</th>
          <th>dead</th>
          <th>trades</th>
          <th className="muted">placebo S</th>
          <th className="muted">p90</th>
        </tr>
      </thead>
      <tbody>
        {gs.map((r) => (
          <tr key={r.gen}>
            <td>{r.gen}</td>
            <td className="l muted">
              {r.window[0]} → {r.window[1]}
            </td>
            <td className={(r.evolved.median_sharpe ?? 0) >= 0 ? 'pos' : 'neg'}>
              <b>{fmt(r.evolved.median_sharpe)}</b>
            </td>
            <td>{fmt(r.evolved.p90_sharpe)}</td>
            <td>{fmt(r.evolved.dead_pct, 1)}%</td>
            <td>{r.evolved.median_trades}</td>
            <td className="muted">{fmt(r.placebo.median_sharpe)}</td>
            <td className="muted">{fmt(r.placebo.p90_sharpe)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function EvoLive({ e, running }: { e: EvoRun; running: boolean }) {
  const pctDone = Math.round((e.frac || 0) * 100)
  const stage =
    e.stage === 'gen'
      ? `generation ${(e.gen ?? 0) + 1} / ${e.gens}`
      : e.stage === 'final_test'
        ? 'final exam on reserved test span'
        : e.stage === 'hof_history'
          ? 'hall-of-fame walk-forward history'
          : e.stage || 'starting…'
  const stalled = !running && (e.age_s || 0) > 120
  const gs = e.gen_stats || []
  return (
    <div className="grid">
      {stalled && (
        <div className="banner" style={{ gridColumn: 'span 12' }}>
          ⚠ No progress update for {Math.round(e.age_s || 0)}s and no evolve process is attached to
          this server — the run may have died. Check <code>reports/swarm/last_evolve.log</code>.
        </div>
      )}
      <div className="card" style={{ gridColumn: 'span 12' }}>
        <h3>Live · {e.run_id}</h3>
        <div className="sub">
          {e.bots} bots/lineage · {e.gens} walk-forward windows · {e.fitness} fitness · seed{' '}
          {e.seed}
          {e.span ? ` · span ${e.span[0]} → ${e.span[1]}` : ''} · test reserved from{' '}
          {(e.test_start || '').slice(0, 10) || '…'}
          {e.maker_only ? ' · maker-only' : ''}
        </div>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          <span style={{ minWidth: 180 }}>{stage}</span>
          <div className="pbar" style={{ flex: 1 }}>
            <i style={{ width: pctDone + '%' }} />
          </div>
          <span className="muted" style={{ minWidth: 90, textAlign: 'right' }}>
            {pctDone}% · {Math.round(e.elapsed_s || 0)}s
          </span>
        </div>
      </div>
      <div className="card" style={{ gridColumn: 'span 7' }}>
        <h3>Selection vs placebo — live</h3>
        <div className="sub">
          median fitness-window Sharpe per generation, updating as each window finishes. Every
          generation is scored on data that played no role in creating it.
        </div>
        <EvoChart gs={gs} />
        <EvoLegend />
      </div>
      <div className="card scroll" style={{ gridColumn: 'span 5', padding: 0 }}>
        <GenTable gs={gs} />
      </div>
    </div>
  )
}

function EvoDone({ e }: { e: EvoRun }) {
  const nav = useNavigate()
  const ft = e.final_test || {}
  const groups = ['evolved', 'placebo', 'fresh_random', 'hall_of_fame'] as const
  const g = (k: string) => ft[k] || {}
  const v = (x: number | string | null | undefined, d = 2) => (x == null ? '—' : (+x).toFixed(d))
  return (
    <div className="grid">
      <div
        className="card"
        style={{ gridColumn: 'span 12', display: 'flex', gap: 16, alignItems: 'center', flexWrap: 'wrap' }}
      >
        <span className="muted">
          {e.bots} bots/lineage · {e.gens} walk-forward windows · seed {e.seed} ·{' '}
          {e.fitness || 'sharpe'} fitness · costs {e.maker_bps}bp maker / {e.taker_bps}bp taker ·
          test reserved from {(e.test_start || '').slice(0, 10)} · {e.elapsed_s}s
          {e.funding ? ' · funding charged' : ''}
        </span>
        <span className="chip">
          skill vs placebo: {v(e.skill_vs_placebo)}
          {e.skill_ci95 ? (
            <span className="muted"> [{v(e.skill_ci95[0])} … {v(e.skill_ci95[1])}]</span>
          ) : null}
        </span>
        {e.hof_check && (
          <span
            className="muted"
            title="correlation of each score with test Sharpe across HOF bots — born-window fitness is an in-sample order statistic; consistency on later windows is the honest ranking signal"
          >
            HOF predictiveness — born fitness {v(e.hof_check.corr_born_fitness_vs_test)} vs OOS
            consistency {v(e.hof_check.corr_oos_consistency_vs_test)}
          </span>
        )}
      </div>
      <div className="card" style={{ gridColumn: 'span 7' }}>
        <h3>Generation-by-generation divergence</h3>
        <div className="sub">
          median fitness-window Sharpe per generation — selection (teal) vs random-selection
          placebo (gray). Each generation is scored on a window that played no role in creating it.
        </div>
        <EvoChart gs={e.gen_stats || []} />
        <EvoLegend />
      </div>
      <div className="card" style={{ gridColumn: 'span 5' }}>
        <h3>Reserved test span</h3>
        <div className="sub">
          never used for any selection · the placebo row is what breeding-without-selection
          achieves
        </div>
        <div className="scroll">
          <table>
            <thead>
              <tr>
                <th className="l">group</th>
                <th>median S</th>
                <th>p90</th>
                <th>% pos</th>
                <th>ret</th>
                <th>dead</th>
              </tr>
            </thead>
            <tbody>
              {groups.map((k) => (
                <tr key={k}>
                  <td className="l">{k.replaceAll('_', ' ')}</td>
                  <td className={(g(k).median_sharpe ?? 0) >= 0 ? 'pos' : 'neg'}>
                    <b>{v(g(k).median_sharpe)}</b>
                  </td>
                  <td>{v(g(k).p90_sharpe)}</td>
                  <td>{v(g(k).pct_positive, 1)}%</td>
                  <td>{g(k).median_ret_pct == null ? '—' : v(g(k).median_ret_pct, 1) + '%'}</td>
                  <td>{v(g(k).dead_pct, 1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <div className="card scroll" style={{ gridColumn: 'span 12', padding: 0 }}>
        <div style={{ display: 'flex', alignItems: 'baseline', padding: '10px 12px 0' }}>
          <h3 style={{ margin: 0 }}>Hall of fame</h3>
          <span style={{ flex: 1 }} />
          <Link to={`/evolution/bots?run=${encodeURIComponent(e.run_id)}`}>all bots →</Link>
        </div>
        <table>
          <thead>
            <tr>
              <th>gen</th>
              <th className="l">ideology (hall of fame, by test Sharpe)</th>
              <th>tf</th>
              <th>session</th>
              <th>risk</th>
              <th title="mean Sharpe on fitness windows AFTER the bot's birth — its out-of-sample consistency before the test span">
                OOS S
              </th>
              <th>test S</th>
              <th>test trades</th>
            </tr>
          </thead>
          <tbody>
            {(e.hof_top || []).map((h, i) => (
              <tr
                key={i}
                className={h.bot_id != null && h.bot_id !== '' ? 'click' : undefined}
                onClick={() =>
                  h.bot_id != null && h.bot_id !== '' &&
                  nav(`/evolution/bot?run=${encodeURIComponent(e.run_id)}&bot=${h.bot_id}`)
                }
              >
                <td>{h.born_gen}</td>
                <td className="l rules" title={String(h.rules)}>
                  {h.rules}
                </td>
                <td>{h.tf}</td>
                <td>{h.session}</td>
                <td>{v(h.risk_pct, 2)}%</td>
                <td>{h.oos_sharpe == null || h.oos_sharpe === '' ? '—' : v(h.oos_sharpe)}</td>
                <td className={+h.test_sharpe >= 0 ? 'pos' : 'neg'}>
                  <b>{v(h.test_sharpe)}</b>
                </td>
                <td>{h.test_trades}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// "BTC · 15m" style suffix for run labels; omits tfs when both are in play
export function runBadge(x: { pair?: string | null; tfs?: string[] | null }): string {
  const parts: string[] = []
  if (x.pair) parts.push(x.pair)
  if (x.tfs && x.tfs.length === 1) parts.push(x.tfs[0])
  return parts.length ? ' · ' + parts.join(' ') : ''
}

export default function Evolution() {
  const [evos, setEvos] = useState<EvoRun[]>([])
  const [pairs, setPairs] = useState<{ pair: string; has_metrics: boolean }[]>([])
  const [running, setRunning] = useState(false)
  const [loaded, setLoaded] = useState(false)
  const [formOpen, setFormOpen] = useState(false)
  const [starting, setStarting] = useState(false)

  // selected run lives in the URL (?run=evo-…) so results are deep-linkable
  const [params, setParams] = useSearchParams()
  const sel = params.get('run') ?? ''
  const selectRun = (id: string) => setParams(id ? { run: id } : {}, { replace: true })

  // new-evolution form
  const [evPair, setEvPair] = useState('BTC')
  const [evTf5, setEvTf5] = useState(true)
  const [evTf15, setEvTf15] = useState(true)
  const [evBots, setEvBots] = useState('1500')
  const [evGens, setEvGens] = useState('6')
  const [evTest, setEvTest] = useState('20')
  const [evFit, setEvFit] = useState('sharpe')
  const [evHof, setEvHof] = useState('fitness')
  const [evMbps, setEvMbps] = useState('1')
  const [evMk, setEvMk] = useState(false)
  const [evDerivs, setEvDerivs] = useState(false)
  const [evSeed, setEvSeed] = useState('42')
  const [evSince, setEvSince] = useState('2021-01-01')

  useEffect(() => {
    document.title = 'strategy-lab · Evolution'
  }, [])

  const fetchEvos = useCallback(async () => {
    const j = await getJSON<EvosPayload>('/api/swarm/evos')
    setEvos(j.evos || [])
    setPairs(j.pairs || [])
    setRunning(!!j.running)
    setLoaded(true)
    setStarting(false)
  }, [])

  useEffect(() => {
    fetchEvos()
  }, [fetchEvos])

  // keep polling while a run is live (progress.json still being written)
  useEffect(() => {
    if (!(running || starting || evos.some((x) => !x.done))) return
    const t = setTimeout(fetchEvos, starting ? 1200 : 2500)
    return () => clearTimeout(t)
  }, [evos, running, starting, fetchEvos])

  const startEvolution = async () => {
    if (!evTf5 && !evTf15) {
      alert('pick at least one timeframe')
      return
    }
    const body = {
      pair: evPair,
      tfs: [evTf5 && '5m', evTf15 && '15m'].filter(Boolean).join(','),
      bots: +evBots,
      gens: +evGens,
      test_frac: +evTest / 100,
      fitness: evFit,
      hof_metric: evHof,
      maker_bps: +evMbps,
      maker_only: evMk,
      derivs: evDerivs,
      seed: +evSeed,
      since: evSince.trim(),
    }
    const j = await postJSON<StartResult>('/api/swarm/evolve/start', body)
    if (!j.started) {
      alert(j.error || 'failed')
      return
    }
    selectRun('')
    setFormOpen(false)
    setStarting(true)
  }

  const e = evos.find((x) => x.run_id === sel) || evos[0]

  return (
    <div className="swarm">
      <TopNav>
        <span className="refresh-status">
          {starting ? '⏳ starting…' : running ? '⏳ evolution process running' : ''}
        </span>
      </TopNav>
      <div className="toolbar">
        <span className="crumb">Evolution</span>
        <select title="evolution run" value={e?.run_id ?? ''} onChange={(ev) => selectRun(ev.target.value)}>
          {evos.length ? (
            evos.map((x) => (
              <option key={x.run_id} value={x.run_id}>
                {x.run_id}
                {runBadge(x)}
                {x.done
                  ? ` · ${x.fitness || 'sharpe'}`
                  : ` · ⏳ ${Math.round((x.frac || 0) * 100)}%`}
              </option>
            ))
          ) : (
            <option value="">no evolution runs yet</option>
          )}
        </select>
        <button className="primary" onClick={() => setFormOpen((o) => !o)}>
          New evolution…
        </button>
        <span className="spacer" />
        <Link to={'/evolution/bots' + (e ? `?run=${encodeURIComponent(e.run_id)}` : '')}>
          top bots →
        </Link>
      </div>
      <div className={'top newrun' + (formOpen ? ' open' : '')}>
        <label title="asset — resolves candles (and metrics/funding when OI+funding is on) by naming convention">
          pair{' '}
          <select value={evPair} onChange={(ev) => setEvPair(ev.target.value)}>
            {(pairs.length ? pairs : [{ pair: 'BTC', has_metrics: true }]).map((p) => (
              <option key={p.pair} value={p.pair}>
                {p.pair}
                {p.has_metrics ? '' : ' (no metrics)'}
              </option>
            ))}
          </select>
        </label>
        <label title="timeframes the gene pool may use — dropping 5m skips its feature pass and roughly halves eval cost">
          <input type="checkbox" checked={evTf5} onChange={(ev) => setEvTf5(ev.target.checked)} />{' '}
          5m
        </label>
        <label>
          <input type="checkbox" checked={evTf15} onChange={(ev) => setEvTf15(ev.target.checked)} />{' '}
          15m
        </label>
        <label>
          bots{' '}
          <input type="number" min={100} max={200000} style={{ width: 80 }} value={evBots} onChange={(ev) => setEvBots(ev.target.value)} />
        </label>
        <label>
          gens{' '}
          <input type="number" min={2} max={30} style={{ width: 60 }} value={evGens} onChange={(ev) => setEvGens(ev.target.value)} />
        </label>
        <label>
          test %{' '}
          <input type="number" min={5} max={50} style={{ width: 60 }} value={evTest} onChange={(ev) => setEvTest(ev.target.value)} />
        </label>
        <label title="sharpe = risk-adjusted · return = raw return with a participation floor · balanced = mean of up-day and down-day Sharpe (experimental — underperformed in the 2026-07 fitness experiments)">
          fitness{' '}
          <select value={evFit} onChange={(ev) => setEvFit(ev.target.value)}>
            <option value="sharpe">sharpe</option>
            <option value="return">return</option>
            <option value="balanced">balanced</option>
          </select>
        </label>
        <label title="what admits bots to the hall of fame — 'sharpe' keeps regime-riders out of the HOF when breeding on another fitness (the validated hybrid: fitness=return + HOF by sharpe)">
          HOF by{' '}
          <select value={evHof} onChange={(ev) => setEvHof(ev.target.value)}>
            <option value="fitness">fitness</option>
            <option value="sharpe">sharpe</option>
          </select>
        </label>
        <label>
          maker bps{' '}
          <input type="number" min={0} max={50} step={0.5} style={{ width: 60 }} value={evMbps} onChange={(ev) => setEvMbps(ev.target.value)} />
        </label>
        <label>
          <input type="checkbox" checked={evMk} onChange={(ev) => setEvMk(ev.target.checked)} />{' '}
          maker-only
        </label>
        <label title="adds OI/long-short/funding features and charges perp funding to open positions">
          <input type="checkbox" checked={evDerivs} onChange={(ev) => setEvDerivs(ev.target.checked)} />{' '}
          OI + funding
        </label>
        <label>
          seed{' '}
          <input type="number" style={{ width: 70 }} value={evSeed} onChange={(ev) => setEvSeed(ev.target.value)} />
        </label>
        <label title="training span start — 'auto' starts where the pair's metrics coverage begins (needs OI+funding)">
          since{' '}
          <input placeholder="2021-01-01 | auto" style={{ width: 100 }} value={evSince} onChange={(ev) => setEvSince(ev.target.value)} />
        </label>
        <button className="primary" onClick={startEvolution}>
          Start evolution
        </button>
        <span className="muted">runs as its own process · artifacts in reports/swarm/evo-*/</span>
      </div>
      <div className="wrap">
        {!e ? (
          <div className="empty">
            {!loaded ? (
              'loading evolution runs…'
            ) : running || starting ? (
              'Evolution starting — loading data…'
            ) : (
              <>
                No evolution runs yet — hit <b>New evolution…</b> or run <code>sl-swarm evolve</code>.
              </>
            )}
          </div>
        ) : e.done ? (
          <EvoDone e={e} />
        ) : (
          <EvoLive e={e} running={running} />
        )}
      </div>
    </div>
  )
}
