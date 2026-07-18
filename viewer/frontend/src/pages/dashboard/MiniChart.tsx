import { useEffect, useRef } from 'react'
import { createChart, type CandlestickData } from 'lightweight-charts'
import { getJSON } from '../../lib/api'
import type { CandlesPayload } from '../../types'

export default function MiniChart({ file }: { file: string }) {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const chart = createChart(el, {
      layout: { background: { color: 'transparent' }, textColor: '#8b949e', fontSize: 11 },
      grid: { vertLines: { color: '#1c2027' }, horzLines: { color: '#1c2027' } },
      rightPriceScale: { borderColor: '#2a2f38' },
      timeScale: { borderColor: '#2a2f38', timeVisible: true, secondsVisible: false },
      crosshair: { mode: 1 },
      autoSize: true,
      handleScroll: false,
      handleScale: false,
    })
    const series = chart.addCandlestickSeries({
      upColor: '#26a69a', downColor: '#ef5350',
      borderUpColor: '#26a69a', borderDownColor: '#ef5350',
      wickUpColor: '#26a69a', wickDownColor: '#ef5350',
    })
    let alive = true
    getJSON<CandlesPayload>(`/api/candles?file=${encodeURIComponent(file)}&bars=200`).then((d) => {
      if (alive && !d.error) {
        series.setData(d.candles as unknown as CandlestickData[])
        chart.timeScale().fitContent()
      }
    })
    return () => {
      alive = false
      chart.remove()
    }
  }, [file])

  return <div className="minichart" ref={ref} />
}
