import { useEffect, useMemo, useRef, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  useReactTable,
  type SortingState,
} from '@tanstack/react-table'
import { C } from '../../lib/colors'
import { getJSON } from '../../lib/api'
import { fmt, money } from '../../lib/format'
import type { EvoBotSummary, EvoBotsPayload, EvosPayload } from '../../types'

const PAGE = 25

function Spark({ v }: { v: (number | null)[] }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const c = ref.current
    if (!c) return
    const w = 90
    const h = 22
    const dpr = window.devicePixelRatio || 1
    c.width = w * dpr
    c.height = h * dpr
    const g = c.getContext('2d')!
    g.scale(dpr, dpr)
    const vals = v.filter((x): x is number => x != null)
    if (!vals.length) return
    let lo = Math.min(...vals, 0)
    let hi = Math.max(...vals, 0)
    if (!(hi > lo)) hi = lo + 1
    const X = (i: number) => 1 + ((w - 2) * i) / Math.max(v.length - 1, 1)
    const Y = (x: number) => h - 2 - ((h - 4) * (x - lo)) / (hi - lo)
    g.strokeStyle = C.border
    g.beginPath()
    g.moveTo(0, Y(0))
    g.lineTo(w, Y(0))
    g.stroke()
    g.strokeStyle = C.up
    g.lineWidth = 1.4
    g.beginPath()
    let started = false
    v.forEach((x, i) => {
      if (x == null) return
      started ? g.lineTo(X(i), Y(x)) : g.moveTo(X(i), Y(x))
      started = true
    })
    g.stroke()
  }, [v])
  return <canvas ref={ref} style={{ width: 90, height: 22, display: 'block' }} />
}

const v = (x: number | string | null | undefined, d = 2) =>
  x == null || x === '' ? '—' : (+x).toFixed(d)

const col = createColumnHelper<EvoBotSummary>()

export default function EvoBots() {
  const [evos, setEvos] = useState<{ run_id: string; done: boolean }[]>([])
  const [data, setData] = useState<EvoBotsPayload | null>(null)
  const [sorting, setSorting] = useState<SortingState>([{ id: 'final_usd', desc: true }])
  const [pagination, setPagination] = useState({ pageIndex: 0, pageSize: PAGE })
  const [params, setParams] = useSearchParams()
  const nav = useNavigate()
  const run = params.get('run') ?? ''

  useEffect(() => {
    document.title = 'strategy-lab · Top bots'
  }, [])

  // list of finished evolution runs for the selector; default to the newest
  useEffect(() => {
    getJSON<EvosPayload>('/api/swarm/evos').then((j) => {
      const done = (j.evos || []).filter((x) => x.done)
      setEvos(done)
      if (!run && done.length)
        setParams({ run: done[0].run_id }, { replace: true })
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (!run) return
    setData(null)
    getJSON<EvoBotsPayload>(`/api/swarm/evo/bots?id=${encodeURIComponent(run)}`)
      .then((j) => {
        setData(j)
        setSorting([{ id: j.has_history ? 'final_usd' : 'test_sharpe', desc: true }])
        setPagination((p) => ({ ...p, pageIndex: 0 }))
      })
      .catch(() =>
        setData({ error: 'failed to load', run_id: run, has_history: false, n_hof: 0, windows: [], bots: [] }),
      )
  }, [run])

  const hasHistory = !!data?.has_history
  const startCap = data?.start_capital ?? 10000

  const columns = useMemo(() => {
    const cols = [
      col.accessor('bot_id', { header: 'bot' }),
      col.accessor('born_gen', { header: 'born' }),
      col.accessor('rules', {
        header: 'ideology',
        enableSorting: false,
        cell: (info) => <span title={info.getValue()}>{info.getValue()}</span>,
        meta: { className: 'l rules' },
      }),
      col.accessor('tf', { header: 'tf' }),
      col.accessor('session', { header: 'session' }),
      col.accessor((r) => r.risk_pct ?? undefined, {
        id: 'risk_pct',
        header: 'risk',
        sortUndefined: 'last' as const,
        cell: (info) => v(info.row.original.risk_pct) + '%',
      }),
      ...(hasHistory
        ? [
            col.display({
              id: 'gen_sharpes',
              header: 'S by generation',
              cell: (info) => {
                const gs = info.row.original.gen_sharpes
                return gs ? <Spark v={gs} /> : '—'
              },
              meta: { className: 'l' },
            }),
          ]
        : []),
      col.accessor((r) => (r.test_sharpe == null ? undefined : +r.test_sharpe), {
        id: 'test_sharpe',
        header: 'test S',
        sortUndefined: 'last' as const,
        cell: (info) => {
          const x = info.row.original.test_sharpe
          return (
            <span className={(Number(x) || 0) >= 0 ? 'pos' : 'neg'}>
              <b>{v(x)}</b>
            </span>
          )
        },
      }),
      ...(hasHistory
        ? [
            col.accessor((r) => r.test_ret_pct ?? undefined, {
              id: 'test_ret_pct',
              header: 'test ret',
              sortUndefined: 'last' as const,
              cell: (info) => {
                const x = info.row.original.test_ret_pct
                return x == null ? '—' : fmt(x, 1) + '%'
              },
            }),
            col.accessor((r) => r.final_usd ?? undefined, {
              id: 'final_usd',
              header: 'holdings $',
              sortUndefined: 'last' as const,
              cell: (info) => {
                const x = info.row.original.final_usd
                return (
                  <span className={(x ?? 0) >= startCap ? 'pos' : 'neg'}>
                    <b>{money(x)}</b>
                  </span>
                )
              },
            }),
          ]
        : []),
      col.accessor((r) => (r.test_trades == null ? undefined : +r.test_trades), {
        id: 'test_trades',
        header: 'trades',
        sortUndefined: 'last' as const,
        cell: (info) => info.row.original.test_trades ?? '—',
      }),
      ...(hasHistory
        ? [
            col.display({
              id: 'chart',
              header: 'chart',
              cell: (info) => (
                <span
                  className="chartlink"
                  title="view every position of this bot on the candle chart"
                  style={{ cursor: 'pointer' }}
                  onClick={(e) => {
                    e.stopPropagation()
                    nav(`/chart?run=${encodeURIComponent(run)}&bot=${info.row.original.bot_id}`)
                  }}
                >
                  📈
                </span>
              ),
            }),
          ]
        : []),
    ]
    return cols
  }, [hasHistory, startCap, run, nav])

  const table = useReactTable({
    data: data?.bots ?? [],
    columns,
    state: { sorting, pagination },
    onSortingChange: setSorting,
    onPaginationChange: setPagination,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    sortDescFirst: true,
    enableSortingRemoval: false,
    autoResetPageIndex: false,
  })

  const pages = Math.max(1, table.getPageCount())
  const pageIndex = Math.min(pagination.pageIndex, pages - 1)

  return (
    <div className="swarm">
      <div className="top">
        <h1>
          strategy-lab <span>/ Evolution / Top bots</span>
        </h1>
        <select
          title="evolution run"
          value={run}
          onChange={(e) => setParams({ run: e.target.value }, { replace: true })}
        >
          {evos.length ? (
            evos.map((x) => (
              <option key={x.run_id} value={x.run_id}>
                {x.run_id}
              </option>
            ))
          ) : (
            <option value="">no finished evolution runs</option>
          )}
        </select>
        <span style={{ flex: 1 }} />
        <Link to={'/evolution' + (run ? `?run=${encodeURIComponent(run)}` : '')}>← evolution</Link>
        <Link to="/">dashboard</Link>
      </div>
      <div className="wrap">
        {!data ? (
          <div className="empty">{run ? 'loading…' : 'no evolution runs yet'}</div>
        ) : data.error ? (
          <div className="empty">{data.error}</div>
        ) : (
          <div className="grid">
            <div
              className="card"
              style={{ gridColumn: 'span 12', display: 'flex', gap: 12, alignItems: 'baseline', flexWrap: 'wrap' }}
            >
              <h3 style={{ margin: 0 }}>
                {data.n_hof} hall-of-fame bots
                {hasHistory ? ` · ranked by holdings (${money(startCap)} at the start)` : ' · ranked by test Sharpe'}
              </h3>
              <span className="muted" style={{ flex: 1 }}>
                {data.gens ? `${data.gens} windows` : ''}
                {data.test_start ? ` · test from ${data.test_start.slice(0, 10)}` : ''} · click a
                column to sort, a row for detail
                {!hasHistory &&
                  ' · this run predates per-generation history — re-run an evolution for holdings and sparklines'}
              </span>
              <span>
                <button
                  disabled={pageIndex === 0}
                  onClick={() => setPagination((p) => ({ ...p, pageIndex: p.pageIndex - 1 }))}
                >
                  ‹
                </button>
                <span className="muted"> {pageIndex + 1}/{pages} </span>
                <button
                  disabled={pageIndex >= pages - 1}
                  onClick={() => setPagination((p) => ({ ...p, pageIndex: p.pageIndex + 1 }))}
                >
                  ›
                </button>
              </span>
            </div>
            <div className="card scroll" style={{ gridColumn: 'span 12', padding: 0 }}>
              <table>
                <thead>
                  {table.getHeaderGroups().map((hg) => (
                    <tr key={hg.id}>
                      <th>#</th>
                      {hg.headers.map((h) => {
                        const meta = h.column.columnDef.meta as { className?: string } | undefined
                        const sorted = h.column.getIsSorted()
                        const sortable = h.column.getCanSort()
                        return (
                          <th
                            key={h.id}
                            className={
                              (sortable ? 'sortable ' : '') +
                              (meta?.className?.includes('l') ? 'l' : '')
                            }
                            onClick={sortable ? h.column.getToggleSortingHandler() : undefined}
                          >
                            {flexRender(h.column.columnDef.header, h.getContext())}
                            {sorted === 'desc' ? ' ▾' : sorted === 'asc' ? ' ▴' : ''}
                          </th>
                        )
                      })}
                    </tr>
                  ))}
                </thead>
                <tbody>
                  {table.getRowModel().rows.map((row, i) => (
                    <tr
                      key={row.id}
                      className="click"
                      onClick={() =>
                        nav(
                          `/evolution/bot?run=${encodeURIComponent(run)}&bot=${row.original.bot_id}`,
                        )
                      }
                    >
                      <td className="muted">{pageIndex * PAGE + i + 1}</td>
                      {row.getVisibleCells().map((cell) => {
                        const meta = cell.column.columnDef.meta as { className?: string } | undefined
                        return (
                          <td key={cell.id} className={meta?.className}>
                            {flexRender(cell.column.columnDef.cell, cell.getContext())}
                          </td>
                        )
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
