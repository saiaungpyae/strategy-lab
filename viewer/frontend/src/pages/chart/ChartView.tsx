import { useCallback, useEffect, useRef, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import TopNav from '../../components/TopNav'
import {
  createChart,
  type CandlestickData,
  type HistogramData,
  type IChartApi,
  type ISeriesApi,
  type LineData,
  type SeriesMarker,
  type Time,
} from 'lightweight-charts'
import { getJSON } from '../../lib/api'
import { fmtNum, money, pct } from '../../lib/format'
import type {
  BotTrade,
  ChartMarker,
  CandlesPayload,
  DatasetInfo,
  EvoTradesPayload,
  FvgPayload,
  FvgSummary,
  FvgZone,
  HistItem,
  SignalsPayload,
} from '../../types'

// New York wall-clock formatter — Intl applies the EST/EDT switch for us,
// and timeZoneName shows which one is in effect on that bar.
const etFmt = new Intl.DateTimeFormat('en-US', {
  timeZone: 'America/New_York', weekday: 'short', month: 'short', day: '2-digit',
  hour: '2-digit', minute: '2-digit', hour12: false, timeZoneName: 'short',
})

const FVG_FILL: Record<string, string> = {
  win: 'rgba(38,166,154,0.20)',
  loss: 'rgba(239,83,80,0.20)',
  timeout: 'rgba(240,180,41,0.16)',
  open: 'rgba(139,148,158,0.16)',
  untouched: 'rgba(139,148,158,0.09)',
}
const FVG_EDGE: Record<string, string> = {
  win: 'rgba(38,166,154,0.55)',
  loss: 'rgba(239,83,80,0.55)',
  timeout: 'rgba(240,180,41,0.45)',
  open: 'rgba(139,148,158,0.45)',
  untouched: 'rgba(139,148,158,0.25)',
}

interface OhlcReadout {
  open: number
  high: number
  low: number
  close: number
  time: number | null
}

export default function ChartView() {
  const [params] = useSearchParams()
  const [files, setFiles] = useState<DatasetInfo[]>([])
  const [file, setFile] = useState('')
  const [bars, setBars] = useState('5000')
  const [sessionOn, setSessionOn] = useState(true)
  const [signal, setSignal] = useState('none')
  const [rr, setRr] = useState('2')
  const [fvgMarkersOn, setFvgMarkersOn] = useState(false)
  const [status, setStatus] = useState('')
  const [meta, setMeta] = useState('')
  const [ohlc, setOhlc] = useState<OhlcReadout | null>(null)
  const [fvgSummary, setFvgSummary] = useState<FvgSummary | null>(null)

  // bot-position overlay ("view in chart" from the evolution pages)
  const botRun = params.get('run') ?? ''
  const botId = params.get('bot') ?? ''
  const botMode = botRun !== '' && botId !== ''
  const [botData, setBotData] = useState<EvoTradesPayload | null>(null)
  const [segFilter, setSegFilter] = useState('all')

  const containerRef = useRef<HTMLDivElement>(null)
  const fvgCanvasRef = useRef<HTMLCanvasElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const candleRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const volumeRef = useRef<ISeriesApi<'Histogram'> | null>(null)
  const sessionRef = useRef<ISeriesApi<'Histogram'> | null>(null)
  const overlaysRef = useRef<ISeriesApi<'Line'>[]>([])
  const sessionBarsRef = useRef<HistItem[]>([])
  const fvgZonesRef = useRef<FvgZone[]>([])
  const fvgMarkersRef = useRef<ChartMarker[]>([])
  const botTradesRef = useRef<BotTrade[]>([]) // trades drawn on the canvas
  const candleRangeRef = useRef<{ from: number; to: number } | null>(null)
  const loadSeqRef = useRef(0)

  // latest control values, readable from async loaders without stale closures
  const ctrlRef = useRef({ file, bars, signal, rr, sessionOn, fvgMarkersOn, botData, segFilter })
  ctrlRef.current = { file, bars, signal, rr, sessionOn, fvgMarkersOn, botData, segFilter }

  useEffect(() => {
    document.title = botMode
      ? `strategy-lab · Bot ${botId} positions`
      : 'strategy-lab · candle viewer'
  }, [botMode, botId])

  const drawZones = useCallback(() => {
    const el = containerRef.current
    const canvas = fvgCanvasRef.current
    const chart = chartRef.current
    const candleSeries = candleRef.current
    if (!el || !canvas || !chart || !candleSeries) return
    const dpr = window.devicePixelRatio || 1
    canvas.width = el.clientWidth * dpr
    canvas.height = el.clientHeight * dpr
    canvas.style.width = el.clientWidth + 'px'
    canvas.style.height = el.clientHeight + 'px'
    const ctx = canvas.getContext('2d')!
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, el.clientWidth, el.clientHeight)
    const zones = fvgZonesRef.current
    const trades = botTradesRef.current
    if (!zones.length && !trades.length) return

    const ts = chart.timeScale()
    const vr = ts.getVisibleRange()
    if (!vr) return
    const vrFrom = vr.from as unknown as number
    const vrTo = vr.to as unknown as number

    for (const z of zones) {
      if (z.to < vrFrom || z.from > vrTo) continue
      const x1 = ts.timeToCoordinate(Math.max(z.from, vrFrom) as unknown as Time)
      const x2 = ts.timeToCoordinate(Math.min(z.to, vrTo) as unknown as Time)
      const y1 = candleSeries.priceToCoordinate(z.top)
      const y2 = candleSeries.priceToCoordinate(z.bottom)
      if (x1 == null || x2 == null || y1 == null || y2 == null) continue
      const w = Math.max(x2 - x1, 2)
      const h = Math.max(y2 - y1, 1)
      ctx.fillStyle = FVG_FILL[z.outcome] || FVG_FILL.untouched
      ctx.fillRect(x1, y1, w, h)
      ctx.strokeStyle = FVG_EDGE[z.outcome] || FVG_EDGE.untouched
      ctx.lineWidth = 1
      ctx.strokeRect(x1 + 0.5, y1 + 0.5, w - 1, h - 1)
    }

    // entry→exit lines for the bot-position overlay, clipped to the visible
    // range with linear interpolation so a half-visible trade keeps its slope
    for (const t of trades) {
      if (t.xt < vrFrom || t.et > vrTo) continue
      let t1 = t.et
      let p1 = t.ep
      let t2 = t.xt
      let p2 = t.xp
      if (t2 > t1) {
        if (t1 < vrFrom) {
          p1 = p1 + ((p2 - p1) * (vrFrom - t1)) / (t2 - t1)
          t1 = vrFrom
        }
        if (t2 > vrTo) {
          p2 = p1 + ((p2 - p1) * (vrTo - t1)) / (t2 - t1)
          t2 = vrTo
        }
      }
      const x1 = ts.timeToCoordinate(t1 as unknown as Time)
      const x2 = ts.timeToCoordinate(t2 as unknown as Time)
      const y1 = candleSeries.priceToCoordinate(p1)
      const y2 = candleSeries.priceToCoordinate(p2)
      if (x1 == null || x2 == null || y1 == null || y2 == null) continue
      ctx.strokeStyle = t.pnl >= 0 ? 'rgba(38,166,154,0.85)' : 'rgba(239,83,80,0.85)'
      ctx.lineWidth = 1.6
      ctx.setLineDash(t.why === 'eof' ? [4, 3] : [])
      ctx.beginPath()
      ctx.moveTo(x1, y1)
      ctx.lineTo(x2, y2)
      ctx.stroke()
      ctx.setLineDash([])
    }
  }, [])

  const fvgClear = useCallback(() => {
    fvgZonesRef.current = []
    fvgMarkersRef.current = []
    setFvgSummary(null)
    const canvas = fvgCanvasRef.current
    if (canvas) {
      const ctx = canvas.getContext('2d')!
      ctx.clearRect(0, 0, canvas.width, canvas.height)
    }
  }, [])

  const clearOverlays = useCallback(() => {
    const chart = chartRef.current
    if (chart) for (const s of overlaysRef.current) chart.removeSeries(s)
    overlaysRef.current = []
    candleRef.current?.setMarkers([])
    fvgClear()
  }, [fvgClear])

  // markers + canvas lines for every replayed position of the selected bot,
  // limited to the loaded candle range (markers need a matching bar)
  const applyBotOverlay = useCallback(() => {
    const { botData, segFilter } = ctrlRef.current
    const rng = candleRangeRef.current
    if (!botData || !rng) return
    const all = segFilter === 'all'
      ? botData.trades
      : botData.trades.filter((t) => t.seg === segFilter)
    const trs = all.filter((t) => t.et >= rng.from && t.et <= rng.to)
    const markers: ChartMarker[] = []
    for (const t of trs) {
      markers.push({
        time: t.et,
        position: t.side > 0 ? 'belowBar' : 'aboveBar',
        color: t.side > 0 ? '#26a69a' : '#ef5350',
        shape: t.side > 0 ? 'arrowUp' : 'arrowDown',
        text: '',
      })
      markers.push({
        time: t.xt,
        position: t.side > 0 ? 'aboveBar' : 'belowBar',
        color: t.pnl >= 0 ? '#26a69a' : '#ef5350',
        shape: 'circle',
        text: (t.pnl >= 0 ? '+' : '') + Math.round(t.pnl),
      })
    }
    markers.sort((a, b) => a.time - b.time)
    candleRef.current?.setMarkers(markers as unknown as SeriesMarker<Time>[])
    botTradesRef.current = trs
    drawZones()
    setMeta(
      `bot ${botData.bot_id} — ${trs.length} of ${all.length} positions in loaded bars` +
        (trs.length < all.length ? ' · raise "bars" to see more' : ''),
    )
  }, [drawZones])

  const loadFVG = useCallback(async () => {
    const { file, bars, rr, fvgMarkersOn } = ctrlRef.current
    if (!file) return
    setStatus('running FVG study… (first run on a big file takes a while)')
    try {
      const d = await getJSON<FvgPayload>(
        `/api/fvg?file=${encodeURIComponent(file)}&bars=${bars}&rr=${rr}`,
      )
      if (d.error) { setStatus(d.error); return }
      fvgZonesRef.current = d.zones
      fvgMarkersRef.current = d.markers
      candleRef.current?.setMarkers(
        (fvgMarkersOn ? d.markers : []) as unknown as SeriesMarker<Time>[],
      )
      setFvgSummary(d.summary)
      drawZones()
      setMeta(`FVG — ${fmtNum(d.zones.length)} zones in view · stats cover full history`)
      setStatus('ready')
    } catch (e) {
      setStatus('error: ' + (e as Error).message)
    }
  }, [drawZones])

  const loadSignals = useCallback(async () => {
    clearOverlays()
    const { file, bars, signal } = ctrlRef.current
    if (signal === 'none') return
    if (signal === 'fvg') return loadFVG()
    if (!file) return
    try {
      const d = await getJSON<SignalsPayload>(
        `/api/signals?file=${encodeURIComponent(file)}&strategy=${signal}&bars=${bars}`,
      )
      if (d.error) { setStatus(d.error); return }
      const chart = chartRef.current
      if (!chart) return
      for (const ln of d.lines) {
        const s = chart.addLineSeries({
          color: ln.color, lineWidth: 2, priceLineVisible: false, lastValueVisible: false,
        })
        s.setData(ln.data as unknown as LineData[])
        overlaysRef.current.push(s)
      }
      candleRef.current?.setMarkers(d.markers as unknown as SeriesMarker<Time>[])
      setMeta(`${d.label} — ${fmtNum(d.entries)} entries · ${fmtNum(d.exits)} exits`)
    } catch (e) {
      setStatus('error: ' + (e as Error).message)
    }
  }, [clearOverlays, loadFVG])

  const loadCandles = useCallback(async () => {
    const { file, bars, sessionOn } = ctrlRef.current
    if (!file) return
    const seq = ++loadSeqRef.current
    setStatus('loading…')
    try {
      const d = await getJSON<CandlesPayload>(
        `/api/candles?file=${encodeURIComponent(file)}&bars=${bars}`,
      )
      if (seq !== loadSeqRef.current) return // a newer request superseded this one
      if (d.error) { setStatus(d.error); return }
      candleRef.current?.setData(d.candles as unknown as CandlestickData[])
      volumeRef.current?.setData(d.volume as unknown as HistogramData[])
      sessionBarsRef.current = d.session || []
      sessionRef.current?.setData(
        (sessionOn ? sessionBarsRef.current : []) as unknown as HistogramData[],
      )
      candleRangeRef.current = d.candles.length
        ? { from: d.candles[0].time, to: d.candles[d.candles.length - 1].time }
        : null
      chartRef.current?.timeScale().fitContent()
      setMeta(`${fmtNum(d.returned)} of ${fmtNum(d.total)} candles`)
      setStatus('ready')
      if (ctrlRef.current.botData) applyBotOverlay()
      else await loadSignals() // redraw overlay for the new file/bars window
    } catch (e) {
      setStatus('error: ' + (e as Error).message)
    }
  }, [loadSignals, applyBotOverlay])

  // one-time chart construction
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const chart = createChart(el, {
      layout: { background: { color: '#0e1117' }, textColor: '#e6edf3' },
      grid: { vertLines: { color: '#1c2027' }, horzLines: { color: '#1c2027' } },
      rightPriceScale: { borderColor: '#2a2f38' },
      timeScale: { borderColor: '#2a2f38', timeVisible: true, secondsVisible: false },
      crosshair: { mode: 1 },
      autoSize: true,
    })

    // Full-height translucent bands marking the US cash session. Added before
    // the candles so it draws behind them; value=1 on a hidden 0-margin scale
    // makes each bar span the whole pane.
    const sessionSeries = chart.addHistogramSeries({
      priceScaleId: 'session',
      priceLineVisible: false, lastValueVisible: false,
      priceFormat: { type: 'volume' },
    })
    sessionSeries.priceScale().applyOptions({ scaleMargins: { top: 0, bottom: 0 }, visible: false })

    const candleSeries = chart.addCandlestickSeries({
      upColor: '#26a69a', downColor: '#ef5350',
      borderUpColor: '#26a69a', borderDownColor: '#ef5350',
      wickUpColor: '#26a69a', wickDownColor: '#ef5350',
    })
    const volumeSeries = chart.addHistogramSeries({
      priceFormat: { type: 'volume' }, priceScaleId: '',
    })
    volumeSeries.priceScale().applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } })

    chartRef.current = chart
    candleRef.current = candleSeries
    volumeRef.current = volumeSeries
    sessionRef.current = sessionSeries

    chart.subscribeCrosshairMove((param) => {
      const c = param.seriesData?.get(candleSeries) as CandlestickData | undefined
      if (!c) { setOhlc(null); return }
      setOhlc({
        open: c.open, high: c.high, low: c.low, close: c.close,
        time: typeof param.time === 'number' ? param.time : null,
      })
    })
    chart.timeScale().subscribeVisibleTimeRangeChange(drawZones)
    const ro = new ResizeObserver(drawZones)
    ro.observe(el)

    return () => {
      ro.disconnect()
      chart.remove()
      chartRef.current = null
      candleRef.current = null
      volumeRef.current = null
      sessionRef.current = null
      overlaysRef.current = []
    }
  }, [drawZones])

  // dataset list, honoring a ?file= deep link from the dashboard; in bot
  // mode the trades payload decides the file, so skip the auto-pick
  useEffect(() => {
    ;(async () => {
      const list = await getJSON<DatasetInfo[]>('/api/files')
      setFiles(list)
      if (botMode) return
      const want = params.get('file')
      if (want && list.some((f) => f.file === want)) setFile(want)
      else if (list.length) setFile(list[0].file)
      else setStatus('no datasets in data/')
    })()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // bot-position overlay: fetch the replayed trade log, then load the run's
  // own dataset (may live under data/snapshots/) with a wider bar window
  useEffect(() => {
    if (!botMode) return
    setStatus('replaying bot trades… (first request re-simulates the bot)')
    getJSON<EvoTradesPayload>(
      `/api/swarm/evo/trades?id=${encodeURIComponent(botRun)}&bot=${encodeURIComponent(botId)}`,
    )
      .then((d) => {
        if (d.error) {
          setStatus(d.error)
          return
        }
        setBotData(d)
        ctrlRef.current.botData = d // loaders read the ref before re-render
        if (!d.file) {
          setStatus('dataset file of this run not found under data/')
          return
        }
        setBars('20000')
        setFile(d.file)
        setStatus('ready')
      })
      .catch((e) => setStatus('error: ' + (e as Error).message))
  }, [botMode, botRun, botId])

  // re-filter markers + lines when the window filter changes
  useEffect(() => {
    if (botData) applyBotOverlay()
  }, [segFilter, botData, applyBotOverlay])

  useEffect(() => {
    if (file) loadCandles()
  }, [file, bars, loadCandles])

  // skip the initial mount — loadCandles already chains into loadSignals
  const signalsMounted = useRef(false)
  useEffect(() => {
    if (!signalsMounted.current) { signalsMounted.current = true; return }
    if (file) loadSignals()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [signal, rr])

  useEffect(() => {
    sessionRef.current?.setData(
      (sessionOn ? sessionBarsRef.current : []) as unknown as HistogramData[],
    )
  }, [sessionOn])

  useEffect(() => {
    if (ctrlRef.current.signal !== 'fvg') return
    candleRef.current?.setMarkers(
      (fvgMarkersOn ? fvgMarkersRef.current : []) as unknown as SeriesMarker<Time>[],
    )
  }, [fvgMarkersOn])

  const isFvg = signal === 'fvg'
  const dirCls = ohlc && ohlc.close >= ohlc.open ? 'up' : 'down'

  return (
    <div className="chartpage">
      <TopNav>
        <span className="status">{status}</span>
      </TopNav>
      <div className="toolbar">
        <span className="crumb">{botMode ? `Bot ${botId} positions` : 'Candle viewer'}</span>
        <label>dataset</label>
        <select value={file} onChange={(e) => setFile(e.target.value)}>
          {file && !files.some((f) => f.file === file) && (
            <option value={file}>{file} (run snapshot)</option>
          )}
          {files.map((f) => (
            <option key={f.file} value={f.file}>
              {f.label}
            </option>
          ))}
        </select>
        <label>bars</label>
        <select value={bars} onChange={(e) => setBars(e.target.value)}>
          <option value="500">500</option>
          <option value="2000">2,000</option>
          <option value="5000">5,000</option>
          <option value="20000">20,000</option>
          <option value="0">all</option>
        </select>
        <label
          className="check"
          title="Regular US trading hours 09:30–16:00 New York time (blue), opening hour emphasized (amber). Follows EST/EDT daylight saving automatically."
        >
          <input
            type="checkbox"
            checked={sessionOn}
            onChange={(e) => setSessionOn(e.target.checked)}
          />
          US session
        </label>
        {!botMode && (
          <>
            <label>signals</label>
            <select value={signal} onChange={(e) => setSignal(e.target.value)}>
              <option value="none">none</option>
              <option value="ema_cross">EMA cross 12/26 · trend</option>
              <option value="sma_cross">SMA cross 50/200 · trend</option>
              <option value="supertrend">Supertrend · trend</option>
              <option value="fvg">FVG · gap-fill event study</option>
            </select>
          </>
        )}
        {isFvg && !botMode && (
          <>
            <label>target R:R</label>
            <select value={rr} onChange={(e) => setRr(e.target.value)}>
              <option value="1">1 : 1</option>
              <option value="1.5">1.5 : 1</option>
              <option value="2">2 : 1</option>
              <option value="3">3 : 1</option>
            </select>
            <label className="check" title="Arrows at each gap-entry bar: W win, L loss, T timeout">
              <input
                type="checkbox"
                checked={fvgMarkersOn}
                onChange={(e) => setFvgMarkersOn(e.target.checked)}
              />
              entry markers
            </label>
          </>
        )}
        {botMode && botData && (
          <>
            <label>window</label>
            <select
              title="fitness window / test span"
              value={segFilter}
              onChange={(e) => setSegFilter(e.target.value)}
            >
              <option value="all">all windows</option>
              {botData.windows.map((w) => (
                <option key={w.seg} value={w.seg}>
                  {w.seg === 'test' ? 'test span' : w.seg}
                  {w.seg === `g${botData.born_gen}` ? ' ★' : ''} · {w.span[0]} → {w.span[1]}
                </option>
              ))}
            </select>
          </>
        )}
        {botMode && (
          <Link to={`/evolution/bot?run=${encodeURIComponent(botRun)}&bot=${encodeURIComponent(botId)}`}>
            ← bot page
          </Link>
        )}
        <button onClick={loadCandles}>reload</button>
        <span className="meta">{meta}</span>
      </div>
      <div className="chart-container" ref={containerRef}>
        <canvas className="fvg-canvas" ref={fvgCanvasRef} />
      </div>
      {ohlc && (
        <div className="ohlc">
          <span>
            O <b className={dirCls}>{fmtNum(ohlc.open)}</b>
          </span>
          <span>
            H <b className={dirCls}>{fmtNum(ohlc.high)}</b>
          </span>
          <span>
            L <b className={dirCls}>{fmtNum(ohlc.low)}</b>
          </span>
          <span>
            C <b className={dirCls}>{fmtNum(ohlc.close)}</b>
          </span>
          {ohlc.time != null && (
            <span style={{ color: 'var(--muted)' }}>{etFmt.format(new Date(ohlc.time * 1000))}</span>
          )}
        </div>
      )}
      {fvgSummary && <FvgPanel s={fvgSummary} />}
      {botMode && botData && <BotPanel d={botData} seg={segFilter} />}
    </div>
  )
}

function BotPanel({ d, seg }: { d: EvoTradesPayload; seg: string }) {
  const trs = seg === 'all' ? d.trades : d.trades.filter((t) => t.seg === seg)
  const wins = trs.filter((t) => t.pnl > 0).length
  const pnl = trs.reduce((s, t) => s + t.pnl, 0)
  const why = (k: string) => trs.filter((t) => t.why === k).length
  const rows: [string, string][] = [
    ['positions', `${fmtNum(trs.length)} (${trs.filter((t) => t.side > 0).length}▲ / ${trs.filter((t) => t.side < 0).length}▼)`],
    ['wins', `${fmtNum(wins)} · ${trs.length ? pct(wins / trs.length) : '—'}`],
    ['Σ pnl', money(pnl)],
    ['exits stop / tp / time', `${why('stop')} / ${why('tp')} / ${why('time')}`],
    ['avg hold', trs.length ? `${Math.round(trs.reduce((s, t) => s + t.hold, 0) / trs.length)} bars` : '—'],
  ]
  return (
    <div className="fvg-panel">
      <h2>
        bot {d.bot_id} · {d.tf} · born g{d.born_gen} · {seg === 'all' ? 'all windows' : seg}
      </h2>
      <div style={{ fontSize: 11, opacity: 0.8, margin: '2px 0 6px', maxWidth: 320 }} title={d.rules}>
        {d.rules}
      </div>
      <table>
        <tbody>
          {rows.map(([k, v]) => (
            <tr key={k}>
              <td>{k}</td>
              <td>{v}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {d.note && <div style={{ fontSize: 10, opacity: 0.55, marginTop: 6, maxWidth: 320 }}>{d.note}</div>}
    </div>
  )
}

function FvgPanel({ s }: { s: FvgSummary }) {
  const rows: [string, string][] = [
    ['gaps found', `${fmtNum(s.gaps)} (${fmtNum(s.bull)}▲ / ${fmtNum(s.bear)}▼)`],
    ['frequency', `${s.gaps_per_day.toFixed(1)} / day`],
    ['touched (filled)', `${fmtNum(s.touched)} · ${pct(s.touch_rate, 0)}`],
    ['wins / losses / timeouts', `${fmtNum(s.wins)} / ${fmtNum(s.losses)} / ${fmtNum(s.timeouts)}`],
    ['win rate', pct(s.win_rate)],
    ['random baseline',
      s.control_win_rate == null ? '—' : `${pct(s.control_win_rate)} ± ${pct(s.control_sd)}`],
    ['FVG vs random',
      s.percentile_vs_random == null ? '—' : `${Math.round(s.percentile_vs_random * 100)}th pctile`],
    ['avg net / trade', pct(s.avg_net_pct, 3)],
    ['avg R (gross)', s.avg_r.toFixed(2)],
  ]
  return (
    <div className="fvg-panel">
      <h2>FVG event study · full history</h2>
      <table>
        <tbody>
          {rows.map(([k, v]) => (
            <tr key={k}>
              <td>{k}</td>
              <td>{v}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className={`verdict ${s.grade}`}>{s.verdict}</div>
    </div>
  )
}
