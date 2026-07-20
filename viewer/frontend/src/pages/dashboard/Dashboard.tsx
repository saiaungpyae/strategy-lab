import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { getJSON, postJSON } from '../../lib/api'
import { fmtAge, fmtDate, fmtNum, fmtSize } from '../../lib/format'
import type {
  HealthDataset,
  HealthPayload,
  RefreshStatus,
  ReportItem,
  SnapshotDataset,
} from '../../types'
import MiniChart from './MiniChart'

const SIGS = [
  ['ema_cross', 'EMA 12/26'],
  ['sma_cross', 'SMA 50/200'],
  ['supertrend', 'Supertrend'],
] as const

const tfSec = (tf: string): number => {
  const m = tf.match(/^(\d+)([smhdwM])$/)
  return m ? +m[1] * { s: 1, m: 60, h: 3600, d: 86400, w: 604800, M: 2592000 }[m[2] as 's' | 'm' | 'h' | 'd' | 'w' | 'M']! : 0
}

const chartHref = (file: string) => `/chart?file=${encodeURIComponent(file)}`

export default function Dashboard() {
  const [datasets, setDatasets] = useState<HealthDataset[]>([])
  const [healthLoaded, setHealthLoaded] = useState(false)
  const [snapshot, setSnapshot] = useState<SnapshotDataset[] | null>(null)
  const [reports, setReports] = useState<ReportItem[] | null>(null)
  const [refresh, setRefresh] = useState<RefreshStatus | null>(null)
  const [symbol, setSymbol] = useState('')

  useEffect(() => {
    document.title = 'strategy-lab · dashboard'
  }, [])

  const loadHealth = useCallback(async () => {
    const d = await getJSON<HealthPayload>('/api/health')
    setDatasets(d.datasets)
    setHealthLoaded(true)
    setRefresh(d.refresh)
  }, [])

  const loadSnapshot = useCallback(async () => {
    const d = await getJSON<{ datasets: SnapshotDataset[] }>('/api/snapshot')
    setSnapshot(d.datasets)
  }, [])

  const loadReports = useCallback(async () => {
    const d = await getJSON<{ reports: ReportItem[] }>('/api/reports')
    setReports(d.reports)
  }, [])

  useEffect(() => {
    loadHealth()
    loadSnapshot()
    loadReports()
  }, [loadHealth, loadSnapshot, loadReports])

  // keep the symbol selection valid as datasets arrive/change
  const symbols = [...new Set(datasets.map((d) => d.symbol))]
  useEffect(() => {
    if (!symbols.includes(symbol) && symbols.length) setSymbol(symbols[0])
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasets])

  // poll while a refresh is running; reload everything when it finishes
  useEffect(() => {
    if (refresh?.state !== 'running') return
    const t = setInterval(async () => {
      const r = await getJSON<RefreshStatus>('/api/refresh')
      setRefresh(r)
      if (r.state !== 'running') {
        loadHealth()
        loadSnapshot()
      }
    }, 2000)
    return () => clearInterval(t)
  }, [refresh?.state, loadHealth, loadSnapshot])

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

  const gridFiles = datasets
    .filter((d) => d.symbol === symbol)
    .sort((a, b) => tfSec(a.timeframe) - tfSec(b.timeframe))

  return (
    <div className="dash">
      <header>
        <h1>📊 strategy-lab</h1>
        <label>symbol</label>
        <select value={symbol} onChange={(e) => setSymbol(e.target.value)}>
          {symbols.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
        <Link to="/chart" style={{ fontSize: '12.5px', color: 'var(--muted)' }}>
          open chart view →
        </Link>
        <Link to="/swarm" style={{ fontSize: '12.5px', color: 'var(--muted)' }}>
          bot swarm →
        </Link>
        <Link to="/evolution" style={{ fontSize: '12.5px', color: 'var(--muted)' }}>
          evolution →
        </Link>
        <Link to="/paper" style={{ fontSize: '12.5px', color: 'var(--muted)' }}>
          paper trading →
        </Link>
        <span className="spacer" />
        <span className={'refresh-status' + (refresh?.state === 'running' ? ' busy' : '')}>
          {refreshText}
        </span>
        <button onClick={triggerRefresh}>⟳ refresh data</button>
      </header>
      <main>
        <h2>Data health</h2>
        <div className="cards">
          {!healthLoaded ? (
            <span className="empty">loading…</span>
          ) : !datasets.length ? (
            <span className="empty">no datasets in data/</span>
          ) : (
            datasets.map((ds) => <HealthCard key={ds.file} ds={ds} />)
          )}
        </div>

        <h2>
          Multi-timeframe · <span>{symbol}</span>
        </h2>
        <div className="tfgrid">
          {gridFiles.map((ds) => (
            <div className="pane" key={ds.file}>
              <div className="bar">
                <b>{ds.timeframe}</b>&nbsp;
                <span style={{ color: 'var(--muted)' }}>· {ds.exchange}</span>
                <Link to={chartHref(ds.file)}>open →</Link>
              </div>
              <MiniChart file={ds.file} />
            </div>
          ))}
        </div>

        <h2>Signal snapshot</h2>
        <SnapshotTable snapshot={snapshot} />

        <h2>Reports</h2>
        <div className="reports">
          {!reports ? (
            <span className="empty">loading…</span>
          ) : !reports.length ? (
            <span className="empty">nothing in reports/ yet</span>
          ) : (
            reports.map((r) => (
              <a
                key={r.file}
                className="report"
                href={`/reports/${encodeURIComponent(r.file)}`}
                target="_blank"
                rel="noreferrer"
              >
                <span className="kind">{r.kind}</span>
                {r.file}
                <span className="rmeta">
                  {fmtSize(r.size_bytes)} · {fmtDate(r.mtime)}
                </span>
              </a>
            ))
          )}
        </div>
      </main>
    </div>
  )
}

function HealthCard({ ds }: { ds: HealthDataset }) {
  const behind = ds.bars_behind ?? 0
  const cls = behind <= 1 ? 'fresh' : behind <= 50 ? 'stale' : 'old'
  const badge = behind <= 1 ? 'fresh' : `${fmtNum(behind)} bars`
  return (
    <Link className="card" to={chartHref(ds.file)}>
      <div className="top">
        <span className="name">
          {ds.symbol} · {ds.timeframe}
        </span>
        <span style={{ color: 'var(--muted)', fontSize: '11.5px' }}>{ds.exchange}</span>
        <span className={`badge ${cls}`}>{badge}</span>
      </div>
      <div className="kv">
        <span>candles</span>
        <b>{fmtNum(ds.rows)}</b>
        <span>span</span>
        <b>
          {fmtDate(ds.first)} → {fmtDate(ds.last)}
        </b>
        <span>freshness</span>
        <b>{fmtAge(ds.age_seconds)}</b>
        <span>gaps · size</span>
        <b>
          {ds.gaps} · {fmtSize(ds.size_bytes)}
        </b>
      </div>
    </Link>
  )
}

function SnapshotTable({ snapshot }: { snapshot: SnapshotDataset[] | null }) {
  if (!snapshot) return <span className="empty">loading…</span>
  if (!snapshot.length) return <span className="empty">no datasets</span>
  return (
    <table>
      <thead>
        <tr>
          <th>dataset</th>
          <th>last close</th>
          <th>24h</th>
          {SIGS.map(([, label]) => (
            <th key={label}>{label}</th>
          ))}
          <th>open FVGs</th>
        </tr>
      </thead>
      <tbody>
        {snapshot.map((ds) =>
          ds.error ? (
            <tr key={ds.file}>
              <td>{ds.label}</td>
              <td colSpan={6} style={{ color: 'var(--down)' }}>
                {ds.error}
              </td>
            </tr>
          ) : (
            <tr key={ds.file}>
              <td>
                <Link to={chartHref(ds.file)} style={{ textDecoration: 'none' }}>
                  {ds.symbol} · {ds.timeframe}
                </Link>
              </td>
              <td className="num">{fmtNum(ds.last_close)}</td>
              <td className={'num ' + ((ds.change_24h_pct ?? 0) >= 0 ? 'pos' : 'neg')}>
                {ds.change_24h_pct == null
                  ? '–'
                  : (ds.change_24h_pct >= 0 ? '+' : '') + ds.change_24h_pct.toFixed(2) + '%'}
              </td>
              {SIGS.map(([key]) => {
                const s = ds.signals[key]
                return (
                  <td key={key}>
                    <span className={`sig ${s.state}`}>{s.state}</span>
                    <small className="muted"> {fmtNum(s.bars_since_flip)} bars ago</small>
                  </td>
                )
              })}
              <td className="num">{ds.signals.fvg_open}</td>
            </tr>
          ),
        )}
      </tbody>
    </table>
  )
}
