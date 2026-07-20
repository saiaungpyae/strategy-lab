import { Suspense, lazy, useCallback, useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import TopNav from '../../components/TopNav'
import Sparkline from '../../components/Sparkline'
import { getJSON, postJSON } from '../../lib/api'
import { fmtAge, fmtDate, fmtNum, fmtSize } from '../../lib/format'
import type {
  HealthDataset,
  HealthPayload,
  OverviewPair,
  RefreshStatus,
  ReportItem,
  SnapshotDataset,
} from '../../types'

// keeps lightweight-charts out of the initial bundle
const MiniChart = lazy(() => import('./MiniChart'))

const TFS = ['5m', '15m', '1h', '4h']
const SIGS = [
  ['ema_cross', 'EMA'],
  ['sma_cross', 'SMA'],
  ['supertrend', 'ST'],
] as const

const basePair = (symbol: string) => symbol.split('/')[0]
const isPerp = (d: { exchange: string }) => d.exchange.endsWith('usdm')
const chartHref = (file: string) => `/chart?file=${encodeURIComponent(file)}`

// freshness in bars for intraday tapes (health matrix)
const freshClass = (behind: number | null | undefined) =>
  behind == null ? 'none' : behind <= 1 ? 'fresh' : behind <= 50 ? 'stale' : 'old'

// freshness in wall-clock time for the overview's 1h tape
const ageClass = (s: number | undefined) =>
  s == null ? 'none' : s <= 2 * 3600 ? 'fresh' : s <= 86400 ? 'stale' : 'old'

export default function Dashboard() {
  const [overview, setOverview] = useState<OverviewPair[] | null>(null)
  const [health, setHealth] = useState<HealthDataset[] | null>(null)
  const [reports, setReports] = useState<ReportItem[] | null>(null)
  const [snapshot, setSnapshot] = useState<SnapshotDataset[] | null>(null)
  const [refresh, setRefresh] = useState<RefreshStatus | null>(null)
  const [pair, setPair] = useState('')
  const [market, setMarket] = useState<'spot' | 'perp'>('spot')
  const [allReports, setAllReports] = useState(false)
  const [snapVer, setSnapVer] = useState(0)

  useEffect(() => {
    document.title = 'strategy-lab · dashboard'
  }, [])

  const loadOverview = useCallback(async () => {
    const d = await getJSON<{ pairs: OverviewPair[] }>('/api/overview')
    setOverview(d.pairs)
  }, [])
  const loadHealth = useCallback(async () => {
    const d = await getJSON<HealthPayload>('/api/health')
    setHealth(d.datasets)
    setRefresh(d.refresh)
  }, [])
  const loadReports = useCallback(async () => {
    const d = await getJSON<{ reports: ReportItem[] }>('/api/reports')
    setReports(d.reports)
  }, [])

  useEffect(() => {
    loadOverview()
    loadHealth()
    loadReports()
  }, [loadOverview, loadHealth, loadReports])

  // default to BTC once the pair list arrives (fall back to the first pair)
  useEffect(() => {
    if (!overview?.length) return
    if (!overview.some((p) => p.pair === pair))
      setPair(overview.some((p) => p.pair === 'BTC') ? 'BTC' : overview[0].pair)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [overview])

  // signals only for the selected pair — the server computes just those files
  useEffect(() => {
    if (!pair) return
    let alive = true
    setSnapshot(null)
    getJSON<{ datasets: SnapshotDataset[] }>(
      `/api/snapshot?symbol=${encodeURIComponent(pair)}`,
    ).then((d) => {
      if (alive) setSnapshot(d.datasets)
    })
    return () => {
      alive = false
    }
  }, [pair, snapVer])

  // poll while a refresh is running; reload everything when it finishes
  useEffect(() => {
    if (refresh?.state !== 'running') return
    const t = setInterval(async () => {
      const r = await getJSON<RefreshStatus>('/api/refresh')
      setRefresh(r)
      if (r.state !== 'running') {
        loadOverview()
        loadHealth()
        setSnapVer((v) => v + 1)
      }
    }, 2000)
    return () => clearInterval(t)
  }, [refresh?.state, loadOverview, loadHealth])

  const triggerRefresh = async () => {
    setRefresh(await postJSON<RefreshStatus>('/api/refresh'))
  }

  let refreshText = ''
  if (refresh?.state === 'running') refreshText = 'refreshing datasets…'
  else if (refresh?.state === 'done') {
    const added = (refresh.results || []).reduce((a, x) => a + (x.added || 0), 0)
    const errs = (refresh.results || []).filter((x) => x.error).length
    refreshText = `refreshed: +${fmtNum(added)} candles` + (errs ? ` · ${errs} error(s)` : '')
  }

  const gridFiles = useMemo(
    () =>
      (health ?? [])
        .filter(
          (d) =>
            !d.error &&
            basePair(d.symbol) === pair &&
            isPerp(d) === (market === 'perp') &&
            TFS.includes(d.timeframe),
        )
        .sort((a, b) => TFS.indexOf(a.timeframe) - TFS.indexOf(b.timeframe)),
    [health, pair, market],
  )

  const snapByFile = useMemo(() => {
    const m = new Map<string, SnapshotDataset>()
    for (const s of snapshot ?? []) if (!s.error) m.set(s.file, s)
    return m
  }, [snapshot])

  type MatrixRow = {
    pair: string
    cells: Record<'spot' | 'perp', Record<string, HealthDataset | undefined>>
    rows: number
    size: number
    missing: number
  }
  const matrix = useMemo<MatrixRow[]>(() => {
    const rows = new Map<string, MatrixRow>()
    for (const d of health ?? []) {
      const p = basePair(d.symbol)
      let r = rows.get(p)
      if (!r) rows.set(p, (r = { pair: p, cells: { spot: {}, perp: {} }, rows: 0, size: 0, missing: 0 }))
      r.cells[isPerp(d) ? 'perp' : 'spot'][d.timeframe] = d
      r.rows += d.rows ?? 0
      r.size += d.size_bytes ?? 0
      r.missing += d.gaps ?? 0
    }
    return [...rows.values()].sort((a, b) => a.pair.localeCompare(b.pair))
  }, [health])

  const totalSize = matrix.reduce((a, r) => a + r.size, 0)
  const totalMissing = matrix.reduce((a, r) => a + r.missing, 0)
  const sortedReports = useMemo(
    () => [...(reports ?? [])].sort((a, b) => b.mtime - a.mtime),
    [reports],
  )

  return (
    <div className="dash">
      <TopNav>
        <span className={'refresh-status' + (refresh?.state === 'running' ? ' busy' : '')}>
          {refreshText}
        </span>
        <button onClick={triggerRefresh} disabled={refresh?.state === 'running'}>
          ⟳ Refresh data
        </button>
      </TopNav>

      <main>
        <section>
          <div className="sec-head">
            <h2>Market overview</h2>
            {overview && health && (
              <span className="sec-meta">
                {overview.length} pairs · {health.length} datasets · {fmtSize(totalSize)}
              </span>
            )}
          </div>
          {!overview ? (
            <div className="skel block" />
          ) : !overview.length ? (
            <div className="empty">no datasets in data/ — hit “Refresh data” after fetching</div>
          ) : (
            <div className="panel tablewrap">
              <table className="mkt">
                <thead>
                  <tr>
                    <th>Pair</th>
                    <th className="num">Price</th>
                    <th className="num">24h</th>
                    <th>7d</th>
                    <th>Fresh</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {overview.map((p) =>
                    p.error ? (
                      <tr key={p.pair}>
                        <td>
                          <b>{p.pair}</b>
                        </td>
                        <td colSpan={5} className="err">
                          {p.error}
                        </td>
                      </tr>
                    ) : (
                      <tr
                        key={p.pair}
                        className={p.pair === pair ? 'sel' : ''}
                        onClick={() => setPair(p.pair)}
                      >
                        <td>
                          <span className="pairmark">{p.pair.slice(0, 2)}</span>
                          <b>{p.pair}</b>
                          <span className="sub">/USDT{p.has_perp ? ' + perp' : ''}</span>
                        </td>
                        <td className="num strong">{fmtNum(p.last_close!)}</td>
                        <td className={'num ' + ((p.change_24h_pct ?? 0) >= 0 ? 'pos' : 'neg')}>
                          {p.change_24h_pct == null
                            ? '–'
                            : (p.change_24h_pct >= 0 ? '+' : '') +
                              p.change_24h_pct.toFixed(2) +
                              '%'}
                        </td>
                        <td>
                          <Sparkline data={p.spark!} up={(p.change_24h_pct ?? 0) >= 0} />
                        </td>
                        <td>
                          <span
                            className={'dot ' + ageClass(p.age_seconds)}
                            title={fmtAge(p.age_seconds ?? 0)}
                          />
                        </td>
                        <td className="act">
                          <Link to={chartHref(p.file!)} onClick={(e) => e.stopPropagation()}>
                            chart →
                          </Link>
                        </td>
                      </tr>
                    ),
                  )}
                </tbody>
              </table>
            </div>
          )}
        </section>

        <section>
          <div className="sec-head">
            <h2>Timeframes</h2>
            <div className="chips">
              {(overview ?? []).map((p) => (
                <button
                  key={p.pair}
                  className={'chip' + (p.pair === pair ? ' active' : '')}
                  onClick={() => setPair(p.pair)}
                >
                  {p.pair}
                </button>
              ))}
            </div>
            <span className="spacer" />
            <div className="seg">
              <button className={market === 'spot' ? 'active' : ''} onClick={() => setMarket('spot')}>
                Spot
              </button>
              <button className={market === 'perp' ? 'active' : ''} onClick={() => setMarket('perp')}>
                Perp
              </button>
            </div>
          </div>
          {!health ? (
            <div className="skel block" />
          ) : !gridFiles.length ? (
            <div className="empty">
              no {market} datasets for {pair || '—'}
            </div>
          ) : (
            <div className="tfgrid">
              {gridFiles.map((ds) => {
                const snap = snapByFile.get(ds.file)
                return (
                  <div className="pane" key={ds.file}>
                    <div className="bar">
                      <b>{ds.timeframe}</b>
                      <span className="sub">{market}</span>
                      <span className="sigchips">
                        {snap ? (
                          SIGS.map(([key, label]) => {
                            const s = snap.signals[key]
                            return (
                              <span
                                key={key}
                                className={`sigchip ${s.state}`}
                                title={`${label} · ${s.state} · flipped ${fmtNum(s.bars_since_flip)} bars ago`}
                              >
                                {label}
                              </span>
                            )
                          })
                        ) : (
                          <span className="sigchip">…</span>
                        )}
                        {snap && snap.signals.fvg_open > 0 && (
                          <span className="sigchip fvg" title="open fair-value gaps">
                            {snap.signals.fvg_open} FVG
                          </span>
                        )}
                      </span>
                      <Link className="open" to={chartHref(ds.file)}>
                        open →
                      </Link>
                    </div>
                    <Suspense fallback={<div className="minichart skel" />}>
                      <MiniChart file={ds.file} />
                    </Suspense>
                  </div>
                )
              })}
            </div>
          )}
        </section>

        <section>
          <div className="sec-head">
            <h2>Data health</h2>
            {health && (
              <span className="sec-meta">
                {totalMissing ? `${fmtNum(totalMissing)} missing bars total` : 'no missing bars'}
              </span>
            )}
          </div>
          {!health ? (
            <div className="skel block" />
          ) : (
            <div className="panel tablewrap">
              <table className="healthmx">
                <thead>
                  <tr>
                    <th rowSpan={2} className="l">
                      Pair
                    </th>
                    <th colSpan={4}>Spot</th>
                    <th colSpan={4}>Perp</th>
                    <th rowSpan={2} className="num">
                      Rows
                    </th>
                    <th rowSpan={2} className="num">
                      Missing
                    </th>
                    <th rowSpan={2} className="num">
                      Size
                    </th>
                  </tr>
                  <tr>
                    {[...TFS, ...TFS].map((tf, i) => (
                      <th key={i} className="c">
                        {tf}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {matrix.map((r) => (
                    <tr key={r.pair}>
                      <td className="l">
                        <b>{r.pair}</b>
                      </td>
                      {(['spot', 'perp'] as const).flatMap((m) =>
                        TFS.map((tf) => {
                          const d = r.cells[m][tf]
                          return (
                            <td key={m + tf} className="c">
                              {d && !d.error ? (
                                <Link
                                  to={chartHref(d.file)}
                                  className={'dot ' + freshClass(d.bars_behind)}
                                  title={
                                    `${d.symbol} ${tf} (${m}) · ${fmtNum(d.rows ?? 0)} rows · ` +
                                    `${fmtAge(d.age_seconds ?? 0)} · ${fmtNum(d.gaps ?? 0)} missing · ` +
                                    fmtSize(d.size_bytes ?? 0)
                                  }
                                />
                              ) : d ? (
                                <span className="dot err" title={d.error} />
                              ) : (
                                <span className="dot none" />
                              )}
                            </td>
                          )
                        }),
                      )}
                      <td className="num sub2">{fmtNum(r.rows)}</td>
                      <td className={'num ' + (r.missing ? 'warn' : 'sub2')}>{fmtNum(r.missing)}</td>
                      <td className="num sub2">{fmtSize(r.size)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        <section>
          <div className="sec-head">
            <h2>Reports</h2>
            {sortedReports.length > 9 && (
              <button className="linkbtn" onClick={() => setAllReports(!allReports)}>
                {allReports ? 'show latest' : `show all (${sortedReports.length})`}
              </button>
            )}
          </div>
          {!reports ? (
            <div className="skel block short" />
          ) : !reports.length ? (
            <div className="empty">nothing in reports/ yet</div>
          ) : (
            <div className="reports">
              {sortedReports.slice(0, allReports ? undefined : 9).map((r) => (
                <a
                  key={r.file}
                  className="report"
                  href={`/reports/${encodeURIComponent(r.file)}`}
                  target="_blank"
                  rel="noreferrer"
                >
                  <span className={'kind ' + r.kind}>{r.kind}</span>
                  <span className="fname">{r.file}</span>
                  <span className="rmeta">
                    {fmtSize(r.size_bytes)} · {fmtDate(r.mtime)}
                  </span>
                </a>
              ))}
            </div>
          )}
        </section>
      </main>
    </div>
  )
}
