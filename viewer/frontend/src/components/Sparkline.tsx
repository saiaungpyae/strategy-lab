// Tiny inline price sparkline — pure SVG, no chart library.
export default function Sparkline({
  data,
  up,
  w = 110,
  h = 30,
}: {
  data: number[]
  up: boolean
  w?: number
  h?: number
}) {
  if (!data || data.length < 2) return null
  const min = Math.min(...data)
  const max = Math.max(...data)
  const span = max - min || 1
  const x = (i: number) => 1 + (i / (data.length - 1)) * (w - 2)
  const y = (v: number) => h - 3 - ((v - min) / span) * (h - 6)
  const pts = data.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ')
  const color = up ? 'var(--up)' : 'var(--down)'
  return (
    <svg className="spark" width={w} height={h} viewBox={`0 0 ${w} ${h}`} aria-hidden>
      <polygon points={`1,${h - 1} ${pts} ${w - 1},${h - 1}`} fill={color} opacity="0.08" />
      <polyline
        points={pts}
        fill="none"
        stroke={color}
        strokeWidth="1.4"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  )
}
