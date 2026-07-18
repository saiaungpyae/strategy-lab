import { useEffect, useMemo, useRef, useState } from 'react'
import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  getFilteredRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  useReactTable,
  type FilterFn,
  type SortingState,
} from '@tanstack/react-table'
import { getJSON } from '../../lib/api'
import { fmt, money } from '../../lib/format'
import type { BotRow, SwarmBotsPayload } from '../../types'

const PAGE = 250

// same semantics as the legacy filter box: bot id exact, rules substring,
// tf/session exact, plus the dead / alive / control keywords
const botFilter: FilterFn<BotRow> = (row, _columnId, value) => {
  const f = String(value).trim().toLowerCase()
  if (!f) return true
  const r = row.original
  return (
    String(r.bot_id) === f ||
    r.rules.toLowerCase().includes(f) ||
    r.tf === f ||
    r.session.toLowerCase() === f ||
    (f === 'dead' && r.dead) ||
    (f === 'control' && r.control) ||
    (f === 'alive' && !r.dead)
  )
}

const col = createColumnHelper<BotRow>()

export default function BotsTab({
  runId,
  openBot,
}: {
  runId: string
  openBot: (id: number) => void
}) {
  const [data, setData] = useState<{ run: string; rows: BotRow[]; start: number } | null>(null)
  const [error, setError] = useState('')
  const [sorting, setSorting] = useState<SortingState>([{ id: 'final_usd', desc: true }])
  const [globalFilter, setGlobalFilter] = useState('')
  const [filterInput, setFilterInput] = useState('')
  const [pagination, setPagination] = useState({ pageIndex: 0, pageSize: PAGE })
  const debounceRef = useRef<number>()

  useEffect(() => {
    if (data?.run === runId) return
    let alive = true
    setError('')
    getJSON<SwarmBotsPayload>(`/api/swarm/bots?id=${encodeURIComponent(runId)}`).then((j) => {
      if (!alive) return
      if (j.error) {
        setError(j.error)
        return
      }
      const b = j.bots
      const rows: BotRow[] = []
      for (let i = 0; i < j.n; i++)
        rows.push({
          bot_id: b.bot_id[i],
          control: b.control[i],
          tf: b.tf[i],
          session: b.session[i],
          rules: b.rules[i],
          final_usd: b.final_usd[i],
          ret_a: b.ret_a[i],
          ret_b: b.ret_b[i],
          sharpe_b: b.sharpe_b[i],
          trades: b.trades[i],
          dead: b.dead[i],
        })
      setData({ run: runId, rows, start: j.start_capital })
      setPagination((p) => ({ ...p, pageIndex: 0 }))
    })
    return () => {
      alive = false
    }
  }, [runId, data?.run])

  const start = data?.start ?? 10000

  const columns = useMemo(
    () => [
      col.accessor('bot_id', {
        header: 'bot',
        cell: (info) => (
          <>
            #{info.getValue()}
            {info.row.original.control && <span className="chip gray">C</span>}
            {info.row.original.dead && <span className="chip red">✝</span>}
          </>
        ),
      }),
      col.accessor('rules', {
        header: 'ideology',
        cell: (info) => (
          <span title={info.getValue()}>{info.getValue()}</span>
        ),
        meta: { className: 'l rules' },
      }),
      col.accessor('tf', { header: 'tf' }),
      col.accessor('session', { header: 'session' }),
      col.accessor((r) => r.final_usd ?? undefined, {
        id: 'final_usd',
        header: 'final $',
        sortUndefined: 'last',
        cell: (info) => {
          const v = info.row.original.final_usd
          return (
            <span className={(v ?? 0) >= start ? 'pos' : 'neg'}>
              <b>{money(v)}</b>
            </span>
          )
        },
      }),
      col.accessor((r) => r.ret_a ?? undefined, {
        id: 'ret_a',
        header: 'ret(train)',
        sortUndefined: 'last',
        cell: (info) => <span className="muted">{fmt(info.row.original.ret_a, 1)}%</span>,
      }),
      col.accessor((r) => r.ret_b ?? undefined, {
        id: 'ret_b',
        header: 'ret(test)',
        sortUndefined: 'last',
        cell: (info) => {
          const v = info.row.original.ret_b
          return <span className={(v ?? 0) >= 0 ? 'pos' : 'neg'}>{fmt(v, 1)}%</span>
        },
      }),
      col.accessor((r) => r.sharpe_b ?? undefined, {
        id: 'sharpe_b',
        header: 'S(test)',
        sortUndefined: 'last',
        cell: (info) => {
          const v = info.row.original.sharpe_b
          return v == null ? '—' : fmt(v, 2)
        },
      }),
      col.accessor('trades', { header: 'trades' }),
    ],
    [start],
  )

  const table = useReactTable({
    data: data?.rows ?? [],
    columns,
    state: { sorting, globalFilter, pagination },
    onSortingChange: setSorting,
    onGlobalFilterChange: setGlobalFilter,
    onPaginationChange: setPagination,
    globalFilterFn: botFilter,
    getCoreRowModel: getCoreRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    sortDescFirst: true,
    enableSortingRemoval: false,
    autoResetPageIndex: false,
  })

  if (error) return <div className="empty">{error}</div>
  if (!data || data.run !== runId) return <div className="empty">loading bots…</div>

  const nFiltered = table.getFilteredRowModel().rows.length
  const pages = Math.max(1, table.getPageCount())
  const pageIndex = Math.min(pagination.pageIndex, pages - 1)

  const onFilterInput = (v: string) => {
    setFilterInput(v)
    window.clearTimeout(debounceRef.current)
    debounceRef.current = window.setTimeout(() => {
      setGlobalFilter(v.trim())
      setPagination((p) => ({ ...p, pageIndex: 0 }))
    }, 250)
  }

  return (
    <>
      <div
        className="card"
        style={{ marginBottom: 12, display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}
      >
        <input
          placeholder="filter: rules text, bot id, 5m/15m, session, dead, alive, control"
          style={{ flex: 1, minWidth: 260 }}
          value={filterInput}
          onChange={(e) => onFilterInput(e.target.value)}
          autoFocus
        />
        <span className="muted">
          {nFiltered.toLocaleString()} bots · started at {money(start)} each · click a column to
          sort, a row for detail
        </span>
        <span>
          <button
            disabled={pageIndex === 0}
            onClick={() => setPagination((p) => ({ ...p, pageIndex: p.pageIndex - 1 }))}
          >
            ‹
          </button>
          <span className="muted">
            {' '}
            {pageIndex + 1}/{pages}{' '}
          </span>
          <button
            disabled={pageIndex >= pages - 1}
            onClick={() => setPagination((p) => ({ ...p, pageIndex: p.pageIndex + 1 }))}
          >
            ›
          </button>
        </span>
      </div>
      <div className="card scroll" style={{ padding: 0 }}>
        <table>
          <thead>
            {table.getHeaderGroups().map((hg) => (
              <tr key={hg.id}>
                {hg.headers.map((h) => {
                  const meta = h.column.columnDef.meta as { className?: string } | undefined
                  const sorted = h.column.getIsSorted()
                  return (
                    <th
                      key={h.id}
                      className={'sortable ' + (meta?.className?.includes('l') ? 'l' : '')}
                      onClick={h.column.getToggleSortingHandler()}
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
            {table.getRowModel().rows.map((row) => (
              <tr
                key={row.id}
                className="click"
                onClick={() => openBot(row.original.bot_id)}
              >
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
    </>
  )
}
