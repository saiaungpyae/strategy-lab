import { useCallback, useEffect, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { getJSON } from '../../lib/api'
import { fmt } from '../../lib/format'
import type { PaperBot, PaperPayload } from '../../types'
import PairChart from './PairChart'

const POLL_MS = 10_000

// price formatting across BTC (5 figures) … DOGE (sub-cent)
const fpx = (x: number | null | undefined): string =>
  x == null ? '—' : x >= 1000 ? x.toFixed(2) : x >= 1 ? x.toFixed(4) : x.toFixed(6)

const fusd = (x: number | null | undefined, d = 2): string =>
  x == null ? '—' : (x >= 0 ? '+' : '') + x.toFixed(d)

const pnlCls = (x: number | null | undefined): string =>
  x == null || x === 0 ? 'muted' : x > 0 ? 'pos' : 'neg'

const age = (ms: number): string => {
  const s = Math.max(0, Math.round((Date.now() - ms) / 1000))
  if (s < 120) return `${s}s ago`
  if (s < 7200) return `${Math.round(s / 60)}m ago`
  return `${(s / 3600).toFixed(1)}h ago`
}

const holdFmt = (bars: number): string =>
  bars >= 96 ? `${(bars / 96).toFixed(1)}d` : `${(bars / 4).toFixed(1)}h`

function StatusChip({ bot }: { bot: PaperBot }) {
  if (bot.status === 'dead') return <span className="paper-chip dead">☠ ruined</span>
  if (bot.status === 'position') {
    const side = bot.position!.side
    return (
      <span className={'paper-chip ' + (side === 'LONG' ? 'long' : 'short')}>
        {side} {fmt(bot.position!.leverage, 1)}x
      </span>
    )
  }
  if (bot.status === 'pending') return <span className="paper-chip pending">order resting</span>
  return <span className="paper-chip idle">idle</span>
}

function Row({ k, v, cls }: { k: string; v: React.ReactNode; cls?: string }) {
  return (
    <div className="paper-row">
      <span className="muted">{k}</span>
      <span className={'paper-num ' + (cls ?? '')}>{v}</span>
    </div>
  )
}

function Triggers({ bot, dim }: { bot: PaperBot; dim?: boolean }) {
  if (!bot.triggers?.length) return null
  return (
    <div className={'paper-radar' + (dim ? ' dim' : '')}>
      <div className="paper-eyebrow">
        TRIGGER RADAR{dim ? ' · IN ZONE' : ''}
      </div>
      {bot.triggers.map((t, i) => {
        const gap = t.cur_q == null ? null
          : t.op === '>' ? t.need_q - t.cur_q : t.cur_q - t.need_q
        const close = gap == null ? 0 : Math.max(0.03, Math.min(1, 1 - Math.max(0, gap) / 0.4))
        return (
          <div className="paper-trig" key={i}>
            <div className="paper-trig-top">
              <span className="paper-trig-rule">
                {t.feature} {t.op} q{t.need_q.toFixed(2)}
                <span className={'paper-trig-dir ' + (t.dir === 'LONG' ? 'pos' : 'neg')}> {t.dir}</span>
              </span>
              {t.cur_q == null ? (
                <span className="muted">no data</span>
              ) : t.fired ? (
                <span className="paper-trig-fired">FIRING</span>
              ) : (
                <span className="paper-trig-prox">now q{t.cur_q.toFixed(2)} · {(Math.max(0, gap!) * 100).toFixed(1)}pp away</span>
              )}
            </div>
            <div className="paper-trig-track">
              <i
                className={t.fired ? (dim ? 'fill-pos' : 'fill-fired') : 'fill'}
                style={{ width: `${(t.fired ? 1 : close) * 100}%` }}
              />
            </div>
          </div>
        )
      })}
    </div>
  )
}

function Hero({ eyebrow, big, bigCls, right, rightLbl, accent }: {
  eyebrow: string; big: string; bigCls: string
  right?: string; rightLbl?: string; accent: string
}) {
  return (
    <div className="paper-hero" style={{ borderLeftColor: accent }}>
      <div>
        <div className="paper-eyebrow">{eyebrow}</div>
        <div className={'paper-hero-big ' + bigCls}>{big}</div>
      </div>
      {right != null && (
        <div className="paper-hero-right">
          <div className={'paper-hero-roe ' + bigCls}>{right}</div>
          <div className="paper-eyebrow" style={{ letterSpacing: '0.5px' }}>{rightLbl}</div>
        </div>
      )}
    </div>
  )
}

function BotCard({ bot }: { bot: PaperBot }) {
  const p = bot.position
  const o = bot.pending_order
  const total = (bot.realized_pnl ?? 0) + (bot.unrealized_pnl ?? 0)
  const roe = p && bot.unrealized_pnl != null ? (bot.unrealized_pnl / bot.equity) * 100 : null
  const entryStyle =
    bot.maker_off_atr && bot.maker_off_atr > 0
      ? `maker limit · ${fmt(bot.maker_off_atr)}×ATR offset · TTL ${bot.order_ttl} bars`
      : 'taker market'
  return (
    <div className={'paper-card' + (p ? ' open' : '') + (bot.status === 'dead' ? ' dead' : '')}>
      <div className="paper-card-head">
        <div className="paper-card-id">
          <Link to={`/evolution/bot?run=${encodeURIComponent(bot.run_id)}&bot=${bot.bot_id}`}>
            {bot.label}
          </Link>
          <span className="paper-num muted">test S {fmt(bot.test_sharpe)}</span>
        </div>
        <StatusChip bot={bot} />
      </div>
      <div className="paper-card-rules" title={bot.rules}>{bot.rules}</div>
      <div className="paper-card-entrystyle muted">{entryStyle}</div>

      {p && (
        <>
          <Hero
            eyebrow="UNREALIZED PNL"
            big={fusd(bot.unrealized_pnl)}
            bigCls={pnlCls(bot.unrealized_pnl)}
            right={roe == null ? undefined : `${roe >= 0 ? '+' : ''}${roe.toFixed(2)}%`}
            rightLbl="ROE"
            accent={(bot.unrealized_pnl ?? 0) >= 0 ? '#26a69a' : '#ef5350'}
          />
          <div className="paper-stats">
            <Row k="entry → mark" v={`${fpx(p.entry)} → ${fpx(bot.mark)}`} />
            <Row k="size" v={`${p.qty} (${fmt(p.leverage, 2)}x)`} />
            <Row k="stop / TP" v={
              <>
                <span className="neg">{fpx(p.stop_px)}</span>
                <span className="muted"> / </span>
                <span className="pos">{p.tp_px == null ? 'time exit' : fpx(p.tp_px)}</span>
              </>
            } />
            {bot.mark != null && (
              <Row k="dist SL / TP" v={
                <>
                  <span className="neg">{(((p.stop_px - bot.mark) / bot.mark) * 100).toFixed(2)}%</span>
                  <span className="muted"> / </span>
                  <span className="pos">
                    {p.tp_px == null ? '—' : `${(((p.tp_px - bot.mark) / bot.mark) * 100).toFixed(2)}%`}
                  </span>
                </>
              } />
            )}
            <Row k="liquidation" v={p.liq_px == null ? 'unreachable (<1x)' : `≈ ${fpx(p.liq_px)}`} cls="warn" />
            <Row k="funding / cost" v={
              <>
                <span className={pnlCls(p.funding)}>{fusd(p.funding, 4)}</span>
                <span className="muted"> / </span>
                <span className={pnlCls(p.entry_cost)}>{fusd(p.entry_cost, 4)}</span>
              </>
            } />
            <Row
              k="opened"
              v={`${p.opened_sec ? new Date(p.opened_sec * 1000).toISOString().slice(5, 16).replace('T', ' ') : '—'} · held ${holdFmt(p.held_bars)}`}
            />
          </div>
          <Triggers bot={bot} dim />
        </>
      )}

      {!p && o && (
        <div className="paper-order">
          <div className="paper-order-top">
            <span className={'paper-order-side ' + (o.side === 'LONG' ? 'pos' : 'neg')}>
              {o.side} {o.type}
            </span>
            <span className="paper-num paper-order-px">@ {fpx(o.px)}</span>
          </div>
          <div className="paper-order-meta muted">
            {bot.mark && o.px ? `dist ${(((o.px - bot.mark) / bot.mark) * 100).toFixed(3)}%` : 'dist —'}
            {' · expires '}{o.ttl_bars} bars ({holdFmt(o.ttl_bars)})
          </div>
        </div>
      )}

      {!p && bot.status !== 'dead' && <Triggers bot={bot} />}

      {bot.status === 'dead' && (
        <>
          <Hero
            eyebrow="FINAL EQUITY · RUIN LINE 30%"
            big={`$${bot.equity.toFixed(2)}`}
            bigCls="neg"
            right={`${(((bot.equity - 100) / 100) * 100).toFixed(1)}%`}
            rightLbl="TOTAL"
            accent="#ef5350"
          />
          <div className="paper-radar">
            <div className="paper-eyebrow">TRIGGER RADAR · HALTED</div>
            <div className="paper-halted muted">bot retired — no live triggers</div>
          </div>
        </>
      )}

      <div className="paper-card-foot paper-num">
        equity <b>${bot.equity.toLocaleString()}</b>
        {' · '}<span className={pnlCls(bot.realized_pnl)}>realized {fusd(bot.realized_pnl)}</span>
        {' · '}<span className={pnlCls(total)}>total {fusd(total)}</span>
        {' · '}{bot.n_trades} trades
      </div>
    </div>
  )
}

export default function Paper() {
  const [data, setData] = useState<PaperPayload | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [livePx, setLivePx] = useState<number | null>(null)
  const [params, setParams] = useSearchParams()

  useEffect(() => {
    document.title = 'strategy-lab · paper trading'
  }, [])

  const load = useCallback(() => {
    getJSON<PaperPayload>('/api/paper')
      .then((d) => { setData(d); setErr(d.error ?? null) })
      .catch(() => setErr('failed to load /api/paper'))
  }, [])

  useEffect(() => {
    load()
    const t = setInterval(load, POLL_MS)
    return () => clearInterval(t)
  }, [load])

  const st = data?.state
  const stale = st ? Date.now() - st.generated_ms > 3 * (st.interval_s * 1000) : false
  const pairs = st ? [...new Set(st.bots.map((b) => b.pair))].sort() : []
  const pair = params.get('pair') && pairs.includes(params.get('pair')!) ? params.get('pair')! : pairs[0]
  const pairBots = st ? st.bots.filter((b) => b.pair === pair) : []
  useEffect(() => setLivePx(null), [pair])
  const pairTrades = st ? st.trades.filter((t) => t.pair === pair) : []

  const pairStat = (p: string) => {
    const bots = st!.bots.filter((b) => b.pair === p)
    const open = bots.filter((b) => b.status === 'position').length
    const pnl = bots.reduce((a, b) => a + (b.unrealized_pnl ?? 0) + b.realized_pnl, 0)
    return { open, pnl }
  }

  return (
    <div className="dash">
      <header>
        <h1>🧾 paper trading</h1>
        <Link to="/" style={{ fontSize: '12.5px', color: 'var(--muted)' }}>dashboard →</Link>
        <Link to="/evolution" style={{ fontSize: '12.5px', color: 'var(--muted)' }}>evolution →</Link>
        <span className="spacer" />
        {st && (stale ? (
          <span className="refresh-status busy">
            ⚠ daemon stale — last cycle {age(st.generated_ms)}
          </span>
        ) : (
          <span
            className="paper-heartbeat"
            title={`daemon alive — last cycle ${age(st.generated_ms)}`}
          />
        ))}
      </header>
      <main>
        {err && (
          <p className="empty">
            {err} — freeze a roster with <code>python -m strategylab.paper select</code>
          </p>
        )}
        {!err && data && !st && (
          <p className="empty">
            roster frozen {data.created} ({data.roster?.length} bots) but no live state yet —
            start the loop with <code>python -m strategylab.paper daemon</code>
          </p>
        )}
        {st && (
          <>
            <div className="paper-totals">
              <span>bots <b>{st.totals.bots}</b></span>
              <span>open <b>{st.totals.open_positions}</b></span>
              <span>equity <b>${st.totals.equity.toLocaleString()}</b></span>
              <span>unrealized <b className={pnlCls(st.totals.unrealized_pnl)}>{fusd(st.totals.unrealized_pnl)}</b></span>
              <span>realized <b className={pnlCls(st.totals.realized_pnl)}>{fusd(st.totals.realized_pnl)}</b></span>
              <span className="muted">
                paper since {new Date(st.paper_start_ms).toISOString().slice(0, 16).replace('T', ' ')} UTC
              </span>
            </div>
            {st.errors.length > 0 && (
              <p className="warn" style={{ fontSize: 12 }}>⚠ {st.errors.join(' · ')}</p>
            )}
            <div className="paper-tabs">
              {pairs.map((p) => {
                const s = pairStat(p)
                return (
                  <button
                    key={p}
                    className={'paper-tab' + (p === pair ? ' active' : '')}
                    onClick={() => setParams({ pair: p })}
                  >
                    {p}
                    {s.open > 0 && <span className="paper-tab-open">{s.open}</span>}
                    <span className={'paper-tab-pnl ' + pnlCls(s.pnl)}>{fusd(s.pnl, 0)}</span>
                  </button>
                )
              })}
            </div>
            <h2>
              {pair}/USDT
              <span className="muted" style={{ marginLeft: 10, fontWeight: 400 }}>
                mark {fpx(livePx ?? st.marks[pair])}{livePx != null ? ' · live' : ''} · 15m
              </span>
            </h2>
            <PairChart pair={pair} tape={data?.tape ?? 'spot'} bots={pairBots} onLive={setLivePx} />
            <div className="paper-grid">
              {pairBots.map((b) => <BotCard key={b.label} bot={b} />)}
            </div>
            <h2>closed trades — {pair} ({pairTrades.length})</h2>
            {pairTrades.length === 0 ? (
              <p className="empty">none yet — trades appear once an entry fills and exits</p>
            ) : (
              <table className="paper-table">
                <thead>
                  <tr>
                    <th>closed</th><th>bot</th><th>side</th><th>entry → exit</th>
                    <th>qty</th><th>pnl</th><th>funding</th><th>why</th><th>held</th>
                  </tr>
                </thead>
                <tbody>
                  {pairTrades.slice(0, 40).map((t, i) => (
                    <tr key={i}>
                      <td className="muted">{new Date(t.xt * 1000).toISOString().slice(5, 16).replace('T', ' ')}</td>
                      <td>{t.bot}</td>
                      <td className={t.side > 0 ? 'pos' : 'neg'}>{t.side > 0 ? 'LONG' : 'SHORT'}</td>
                      <td>{fpx(t.ep)} → {fpx(t.xp)}</td>
                      <td>{t.qty}</td>
                      <td className={pnlCls(t.pnl)}>{fusd(t.pnl)}</td>
                      <td className={pnlCls(t.fund)}>{fusd(t.fund ?? 0)}</td>
                      <td className="muted">{t.why}</td>
                      <td className="muted">{holdFmt(t.hold)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </>
        )}
      </main>
    </div>
  )
}
