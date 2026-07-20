import { useEffect, useRef } from 'react'
import {
  createChart,
  LineStyle,
  type CandlestickData,
  type IPriceLine,
  type ISeriesApi,
  type UTCTimestamp,
} from 'lightweight-charts'
import { getJSON } from '../../lib/api'
import type { CandlesPayload, PaperBot } from '../../types'

const RELOAD_MS = 30_000        // closed bars from the on-disk tape
const WS_RECONNECT_MS = 3_000   // Binance drops streams ~24h; auto-rejoin

/** Live 15m candles for the selected pair. History = the stored tape (what
 * the bots actually trade on); the last bar rides Binance's public kline
 * WebSocket, so it ticks in real time. Price lines mark every open position
 * (entry / SL / TP) and resting limit order of the pair's bots. */
export default function PairChart({
  pair,
  tape = 'spot',
  bots,
  onLive,
}: {
  pair: string
  tape?: 'spot' | 'perp'
  bots: PaperBot[]
  onLive?: (px: number) => void
}) {
  const ref = useRef<HTMLDivElement>(null)
  const seriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const linesRef = useRef<IPriceLine[]>([])

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
    })
    const series = chart.addCandlestickSeries({
      upColor: '#26a69a', downColor: '#ef5350',
      borderUpColor: '#26a69a', borderDownColor: '#ef5350',
      wickUpColor: '#26a69a', wickDownColor: '#ef5350',
    })
    seriesRef.current = series
    let alive = true
    let ws: WebSocket | null = null
    let reconnect: ReturnType<typeof setTimeout> | null = null
    let lastLive: CandlestickData | null = null
    let lastMsgAt = 0   // socket health; 0 = never delivered

    const paint = (bar: CandlestickData, close: number) => {
      lastLive = bar
      series.update(bar)
      onLive?.(close)
    }

    const connect = () => {
      if (!alive) return
      const host = tape === 'perp' ? 'fstream.binance.com' : 'stream.binance.com:9443'
      ws = new WebSocket(`wss://${host}/ws/${pair.toLowerCase()}usdt@kline_15m`)
      ws.onmessage = (ev) => {
        if (!alive) return
        try {
          const k = JSON.parse(ev.data).k
          if (!k) return
          lastMsgAt = Date.now()
          paint({
            time: (Number(k.t) / 1000) as UTCTimestamp,
            open: +k.o, high: +k.h, low: +k.l, close: +k.c,
          }, +k.c)
        } catch { /* malformed frame: ignore */ }
      }
      ws.onclose = () => {
        if (alive) reconnect = setTimeout(connect, WS_RECONNECT_MS)
      }
      ws.onerror = () => ws?.close()
    }

    // REST fallback: some networks silently drop WebSocket upgrades to the
    // Binance stream hosts while plain HTTPS works. Whenever the socket has
    // been quiet for >10s, poll the forming kline every 5s; the socket keeps
    // reconnecting in the background and takes over as soon as it delivers.
    const restBase = tape === 'perp'
      ? 'https://fapi.binance.com/fapi/v1/klines'
      : 'https://api.binance.com/api/v3/klines'
    const pollLive = () => {
      if (!alive || Date.now() - lastMsgAt <= 10_000) return
      fetch(`${restBase}?symbol=${pair}USDT&interval=15m&limit=1`)
        .then((r) => r.json())
        .then((ks: (string | number)[][]) => {
          if (!alive || Date.now() - lastMsgAt <= 10_000) return
          if (!Array.isArray(ks) || !ks.length) return
          const k = ks[0]
          paint({
            time: (Number(k[0]) / 1000) as UTCTimestamp,
            open: +k[1], high: +k[2], low: +k[3], close: +k[4],
          }, +k[4])
        })
        .catch(() => undefined)
    }

    const file = tape === 'perp'
      ? `ohlcv/${pair}-USDT/binanceusdm_${pair}-USDT-USDT_15m.csv`
      : `ohlcv/${pair}-USDT/binance_${pair}-USDT_15m.csv`
    const load = () =>
      getJSON<CandlesPayload>(`/api/candles?file=${encodeURIComponent(file)}&bars=300`)
        .then((d) => {
          if (!alive || d.error) return
          series.setData(d.candles as unknown as CandlestickData[])
          // setData drops the forming bar — repaint the latest socket tick
          if (lastLive) series.update(lastLive)
        })

    load()
    connect()
    const tLoad = setInterval(load, RELOAD_MS)
    const tPoll = setInterval(pollLive, 5_000)
    return () => {
      alive = false
      if (reconnect) clearTimeout(reconnect)
      ws?.close()
      clearInterval(tLoad)
      clearInterval(tPoll)
      linesRef.current = []
      seriesRef.current = null
      chart.remove()
    }
  }, [pair, tape, onLive])

  useEffect(() => {
    const s = seriesRef.current
    if (!s) return
    linesRef.current.forEach((l) => s.removePriceLine(l))
    linesRef.current = []
    const add = (price: number, color: string, style: LineStyle, title: string, width: 1 | 2 = 1) =>
      linesRef.current.push(s.createPriceLine({ price, color, lineWidth: width, lineStyle: style, title, axisLabelVisible: true }))
    for (const b of bots) {
      const p = b.position
      if (p) {
        add(p.entry, p.side === 'LONG' ? '#26a69a' : '#ef5350', LineStyle.Solid, `${b.label} ${p.side}`, 2)
        add(p.stop_px, '#ef5350', LineStyle.Dashed, `${b.label} SL`)
        if (p.tp_px != null) add(p.tp_px, '#26a69a', LineStyle.Dashed, `${b.label} TP`)
      }
      const o = b.pending_order
      if (o?.px != null) add(o.px, '#f0b429', LineStyle.Dotted, `${b.label} ${o.side} limit`)
    }
  }, [bots, pair])

  return <div ref={ref} className="paper-chart" />
}
