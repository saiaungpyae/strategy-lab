// fixed-decimal with em-dash fallback (swarm-style)
export const fmt = (x: number | string | null | undefined, d = 2): string =>
  x == null || Number.isNaN(+x) ? '—' : (+x).toFixed(d)

// locale grouping, up to 2 decimals (chart/dashboard-style)
export const fmtNum = (n: number | string): string =>
  Number(n).toLocaleString(undefined, { maximumFractionDigits: 2 })

export function fmtAge(s: number): string {
  if (s < 90) return 'up to date'
  const d = Math.floor(s / 86400)
  const h = Math.floor((s % 86400) / 3600)
  const m = Math.floor((s % 3600) / 60)
  if (d) return `${d}d ${h}h behind`
  if (h) return `${h}h ${m}m behind`
  return `${m}m behind`
}

export const fmtDate = (t: number): string => new Date(t * 1000).toISOString().slice(0, 10)

export const fmtSize = (b: number): string =>
  b > 1e6 ? (b / 1e6).toFixed(1) + ' MB' : (b / 1e3).toFixed(0) + ' kB'

export const money = (v: number | string | null | undefined): string =>
  v == null
    ? '—'
    : '$' + (+v).toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 })

export const pct = (x: number, digits = 1): string => (x * 100).toFixed(digits) + '%'
