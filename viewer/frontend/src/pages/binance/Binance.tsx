import { useCallback, useEffect, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { getJSON } from '../../lib/api'
import type {
  BinanceAssetRow, BinanceOrder, BinancePayload, BinancePosition,
  BinanceRestrictions, BinanceWallet,
} from '../../types'

const POLL_MS = 15_000

const usd = (x: number | null | undefined, d = 2): string =>
  x == null ? '—' : x.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d })

const qty = (x: number): string =>
  x.toLocaleString(undefined, { maximumFractionDigits: 8 })

const px = (x: number | null | undefined): string =>
  x == null ? '—' : x >= 1000 ? x.toFixed(2) : x >= 1 ? x.toFixed(4) : x.toFixed(6)

const signed = (x: number, d = 2): string => (x >= 0 ? '+' : '') + x.toFixed(d)

const pnlCls = (x: number | null | undefined): string =>
  x == null || x === 0 ? 'muted' : x > 0 ? 'pos' : 'neg'

const age = (ms: number): string => {
  const s = Math.max(0, Math.round((Date.now() - ms) / 1000))
  return s < 120 ? `${s}s ago` : s < 7200 ? `${Math.round(s / 60)}m ago` : `${(s / 3600).toFixed(1)}h ago`
}

const baseAsset = (symbol: string): string =>
  symbol.replace(/[/:].*$/, '').replace(/USDT$|USDC$|USD$/, '')

// ---- asset icons -----------------------------------------------------------

const ICON_BASE = 'https://cdn.jsdelivr.net/gh/spothq/cryptocurrency-icons@master/32/color/'
const iconFailed = new Set<string>()

const hueFor = (sym: string): string => {
  let h = 0
  for (const c of sym) h = (h * 31 + c.charCodeAt(0)) % 360
  return `hsl(${h}, 42%, 38%)`
}

function AssetIcon({ sym }: { sym: string }) {
  const s = sym.toLowerCase()
  const [failed, setFailed] = useState(iconFailed.has(s))
  if (failed) {
    return <span className="bx-icon-fb" style={{ background: hueFor(sym) }}>{sym.slice(0, 3)}</span>
  }
  return (
    <img
      className="bx-icon" alt="" src={`${ICON_BASE}${s}.png`}
      onError={() => { iconFailed.add(s); setFailed(true) }}
    />
  )
}

function AssetCell({ sym, tag }: { sym: string; tag?: string }) {
  return (
    <span className="bx-asset">
      <AssetIcon sym={sym} />
      {sym}
      {tag && <span className="tag">{tag}</span>}
    </span>
  )
}

// ---- building blocks -------------------------------------------------------

function Tile({ label, value, sub, hero, cls }: {
  label: string; value: React.ReactNode; sub?: React.ReactNode; hero?: boolean; cls?: string
}) {
  return (
    <div className={'bx-tile' + (hero ? ' bx-hero' : '')}>
      <div className="lbl">{label}</div>
      <div className={'val ' + (cls ?? '')}>{value}</div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  )
}

function Card({ title, amount, meta, children }: {
  title: string; amount?: string; meta?: React.ReactNode; children: React.ReactNode
}) {
  return (
    <section className="bx-card">
      <div className="bx-card-head">
        <h3>{title}</h3>
        {amount && <span className="amt">{amount}</span>}
        {meta && <span className="meta">{meta}</span>}
      </div>
      {children}
    </section>
  )
}

function KeyChips({ r }: { r: BinanceRestrictions }) {
  const chips: { label: string; cls: string; title: string }[] = [
    r.enable_reading
      ? { label: '✓ reading', cls: 'ok', title: 'the key can read account data — all the dashboard needs' }
      : { label: '✗ reading off', cls: 'danger', title: 'the dashboard needs Enable Reading' },
    r.ip_restrict
      ? { label: '✓ IP whitelisted', cls: 'ok', title: 'only whitelisted IPs can use this key' }
      : { label: '⚠ no IP restriction', cls: 'warn', title: 'anyone with the key can use it from anywhere' },
    r.enable_spot_trading
      ? { label: '⚠ spot trading enabled', cls: 'warn',
          title: 'this key CAN place spot orders — more permission than a view-only dashboard needs' }
      : { label: '✓ spot trading off', cls: 'ok', title: 'the key cannot place spot orders' },
    r.enable_portfolio_margin
      ? { label: '⚠ portfolio margin trading', cls: 'warn',
          title: 'this key CAN trade on the PM account — required to even read PM data, but it is trading power' }
      : r.enable_futures
        ? { label: '⚠ futures enabled', cls: 'warn', title: 'this key can trade futures' }
        : { label: '✓ futures off', cls: 'ok', title: 'the key cannot trade futures' },
    r.enable_withdrawals
      ? { label: '⛔ withdrawals enabled', cls: 'danger', title: 'this key can move funds out — strongly consider disabling' }
      : { label: '✓ withdrawals off', cls: 'ok', title: 'the key cannot withdraw funds' },
  ]
  if (r.enable_margin) chips.push({ label: '⚠ margin enabled', cls: 'warn', title: 'this key can trade classic margin' })
  if (r.permits_universal_transfer) chips.push({ label: '⚠ universal transfer', cls: 'warn', title: 'this key can move funds between wallets' })
  return (
    <div className="bx-chips">
      {chips.map((c) => (
        <span key={c.label} className={'bx-chip ' + c.cls} title={c.title}>{c.label}</span>
      ))}
      {r.created_ms > 0 && (
        <span className="meta">key created {new Date(r.created_ms).toISOString().slice(0, 10)}</span>
      )}
    </div>
  )
}

function WalletTable({ wallet, total }: { wallet: BinanceWallet; total: number }) {
  if (wallet.assets.length === 0) return <p className="bx-empty">no balances</p>
  return (
    <table className="bx-table">
      <thead>
        <tr>
          <th>asset</th><th>free</th><th>locked</th><th>total</th>
          <th>price $</th><th>value $</th><th>alloc</th>
        </tr>
      </thead>
      <tbody>
        {wallet.assets.map((a: BinanceAssetRow) => {
          const sym = a.earn ? a.asset.slice(2) : a.asset
          const alloc = a.value_usd == null || total <= 0 ? null : (a.value_usd / total) * 100
          return (
            <tr key={a.asset}>
              <td><AssetCell sym={sym} tag={a.earn ? 'earn' : undefined} /></td>
              <td>{qty(a.free)}</td>
              <td className={a.locked > 0 ? 'warn' : 'muted'}>{a.locked > 0 ? qty(a.locked) : '—'}</td>
              <td>{qty(a.total)}</td>
              <td className="muted">{px(a.price_usd)}</td>
              <td><b>{a.value_usd == null ? '?' : usd(a.value_usd)}</b></td>
              <td className="muted">
                {alloc == null ? '—' : `${alloc.toFixed(1)}%`}
                {alloc != null && (
                  <span className="bx-allocbar"><i style={{ width: `${Math.min(100, alloc)}%` }} /></span>
                )}
              </td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

function PositionsTable({ positions, showMarket }: { positions: BinancePosition[]; showMarket?: boolean }) {
  if (positions.length === 0) return <p className="bx-empty">no open positions</p>
  return (
    <table className="bx-table">
      <thead>
        <tr>
          <th>symbol</th>{showMarket && <th>mkt</th>}<th>side</th><th>size</th>
          <th>notional $</th><th>lev</th><th>entry → mark</th><th>liq</th><th>uPnL $</th>
        </tr>
      </thead>
      <tbody>
        {positions.map((p) => (
          <tr key={(p.market ?? '') + p.symbol + p.side}>
            <td><AssetCell sym={baseAsset(p.symbol)} tag={p.symbol} /></td>
            {showMarket && <td className="muted">{p.market}</td>}
            <td className={p.side === 'long' ? 'pos' : 'neg'}>{p.side.toUpperCase()}</td>
            <td>{qty(p.contracts)}</td>
            <td>{usd(p.notional)}</td>
            <td>{p.leverage ? `${p.leverage}x` : '—'}</td>
            <td>{px(p.entry_price)} → {px(p.mark_price)}</td>
            <td className="warn">{px(p.liq_price)}</td>
            <td className={pnlCls(p.unrealized_pnl)}>{signed(p.unrealized_pnl)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function OrdersTable({ orders, showMarket }: { orders: BinanceOrder[]; showMarket?: boolean }) {
  if (orders.length === 0) return <p className="bx-empty">none</p>
  return (
    <table className="bx-table">
      <thead>
        <tr>
          <th>placed</th><th>symbol</th>{showMarket && <th>mkt</th>}<th>side</th>
          <th>type</th><th>price</th><th>amount</th><th>filled</th>
        </tr>
      </thead>
      <tbody>
        {orders.map((o, i) => (
          <tr key={i}>
            <td className="muted">
              {o.created_ms ? new Date(o.created_ms).toISOString().slice(0, 16).replace('T', ' ') : '—'}
            </td>
            <td><AssetCell sym={baseAsset(o.symbol)} tag={o.symbol} /></td>
            {showMarket && <td className="muted">{o.market}</td>}
            <td className={o.side === 'buy' ? 'pos' : 'neg'}>{o.side?.toUpperCase()}</td>
            <td>
              {o.type}
              {o.trigger != null && <span className="muted"> @ {px(o.trigger)}</span>}
              {o.reduce_only ? <span className="muted"> · RO</span> : null}
            </td>
            <td>{px(o.price)}</td>
            <td>{qty(o.amount)}</td>
            <td className="muted">{qty(o.filled)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

// ---- page ------------------------------------------------------------------

export default function Binance() {
  const [data, setData] = useState<BinancePayload | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [params, setParams] = useSearchParams()

  useEffect(() => { document.title = 'strategy-lab · binance account' }, [])

  const load = useCallback(() => {
    getJSON<BinancePayload>('/api/binance')
      .then((d) => { setData(d); setErr(d.error ?? null) })
      .catch(() => setErr('failed to load /api/binance'))
  }, [])

  useEffect(() => {
    load()
    const t = setInterval(load, POLL_MS)
    return () => clearInterval(t)
  }, [load])

  const total = data?.total_value_usd ?? 0
  const fut = data?.futures
  const pm = data?.portfolio_margin
  const spotOrders = data?.open_orders ?? []
  const pmOrders = pm?.open_orders ?? []
  const pmUpnl = pm ? pm.assets.reduce((a, b) => a + b.um_upnl + b.cm_upnl, 0) : null
  const upnl = pmUpnl ?? fut?.unrealized_pnl ?? null
  const nPositions = pm?.positions.length ?? fut?.positions.length ?? 0

  const tab = params.get('tab') === 'spot' ? 'spot' : pm ? 'pm' : 'spot'

  return (
    <div className="dash bx">
      <header>
        <h1>🏦 binance account</h1>
        <Link to="/" style={{ fontSize: '12.5px', color: 'var(--muted)' }}>dashboard →</Link>
        <Link to="/paper" style={{ fontSize: '12.5px', color: 'var(--muted)' }}>paper trading →</Link>
        <span className="spacer" />
        <span className="refresh-status">
          read-only{data?.generated_ms ? ` · snapshot ${age(data.generated_ms)}` : ''}
        </span>
      </header>
      <main>
        {err && (
          <p className="empty">
            {err}
            {data?.configured === false && (
              <> — the <code>.env</code> comments explain the recommended key restrictions</>
            )}
          </p>
        )}

        {data?.configured && !err && (
          <>
            <div className="bx-kpis">
              <Tile
                hero
                label="Total est. value"
                value={`$${usd(total)}`}
                sub={data.generated_ms ? `across all wallets · ${age(data.generated_ms)}` : 'across all wallets'}
              />
              {pm ? (
                <Tile label="Portfolio margin" value={`$${usd(pm.value_usd)}`}
                      sub={`equity $${usd(pm.equity_usd)}`} />
              ) : (
                <Tile label="Futures margin" value={`$${usd(fut?.margin_balance_usd ?? 0)}`} />
              )}
              <Tile label="Spot wallet" value={`$${usd(data.spot?.value_usd ?? 0)}`} />
              <Tile
                label="Unrealized PnL"
                value={upnl == null ? '—' : signed(upnl)}
                cls={pnlCls(upnl)}
                sub={`${nPositions} positions · ${pmOrders.length + spotOrders.length} orders`}
              />
            </div>

            {data.restrictions && <KeyChips r={data.restrictions} />}

            {data.public_ip && (
              <div className="bx-alert">
                ⚠ Binance rejected the key (code -2015). This machine's public IP is{' '}
                <b style={{ userSelect: 'all' }}>{data.public_ip}</b> — if it recently changed,
                add it to the key's IP whitelist (Binance → API Management → Edit restrictions).
                Recovers automatically once whitelisted.
              </div>
            )}
            {(data.errors?.length ?? 0) > 0 && (
              <div className="bx-alert danger">⚠ {data.errors!.join(' · ')}</div>
            )}

            <div className="bx-tabs">
              {pm && (
                <button
                  className={'bx-tab' + (tab === 'pm' ? ' active' : '')}
                  onClick={() => setParams({ tab: 'pm' })}
                >
                  portfolio margin <b>${usd(pm.value_usd)}</b>
                </button>
              )}
              <button
                className={'bx-tab' + (tab === 'spot' ? ' active' : '')}
                onClick={() => setParams({ tab: 'spot' })}
              >
                spot &amp; funding <b>${usd((data.spot?.value_usd ?? 0) + (data.funding?.value_usd ?? 0))}</b>
              </button>
            </div>

            {tab === 'pm' && pm && (
              <>
                <Card
                  title="Portfolio margin" amount={`$${usd(pm.value_usd)}`}
                  meta={
                    <>
                      equity ${usd(pm.equity_usd)}
                      {pm.uni_mmr != null && pm.uni_mmr < 5000 && ` · uniMMR ${pm.uni_mmr.toFixed(2)}`}
                      {pm.status && pm.status !== 'NORMAL' && ` · ${pm.status}`}
                    </>
                  }
                >
                  {pm.assets.length === 0 ? (
                    <p className="bx-empty">no balances</p>
                  ) : (
                    <table className="bx-table">
                      <thead>
                        <tr><th>asset</th><th>wallet</th><th>UM uPnL</th><th>CM uPnL</th><th>value $</th></tr>
                      </thead>
                      <tbody>
                        {pm.assets.map((a) => (
                          <tr key={a.asset}>
                            <td><AssetCell sym={a.asset} /></td>
                            <td>{qty(a.wallet)}</td>
                            <td className={pnlCls(a.um_upnl)}>{a.um_upnl === 0 ? '—' : signed(a.um_upnl)}</td>
                            <td className={pnlCls(a.cm_upnl)}>{a.cm_upnl === 0 ? '—' : signed(a.cm_upnl)}</td>
                            <td><b>{a.value_usd == null ? '?' : usd(a.value_usd)}</b></td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </Card>
                <Card title="Open positions" meta={`${pm.positions.length}`}>
                  <PositionsTable positions={pm.positions} showMarket />
                </Card>
                <Card title="Open orders" meta={`${pmOrders.length} · UM + CM + margin`}>
                  <OrdersTable orders={pmOrders} showMarket />
                </Card>
              </>
            )}

            {tab === 'spot' && (
              <>
                <Card
                  title="Spot wallet" amount={`$${usd(data.spot?.value_usd ?? 0)}`}
                  meta={(data.spot?.dust_hidden ?? 0) > 0 ? `${data.spot!.dust_hidden} dust balances hidden` : undefined}
                >
                  {data.spot && <WalletTable wallet={data.spot} total={total} />}
                </Card>
                {data.funding && data.funding.assets.length > 0 && (
                  <Card title="Funding wallet" amount={`$${usd(data.funding.value_usd)}`}>
                    <WalletTable wallet={data.funding} total={total} />
                  </Card>
                )}
                {fut && (
                  <Card
                    title="USDⓈ-M futures" amount={`$${usd(fut.margin_balance_usd)}`}
                    meta={`wallet $${usd(fut.wallet_usd)} · available $${usd(fut.available_usd)}`}
                  >
                    <PositionsTable positions={fut.positions} />
                  </Card>
                )}
                <Card title="Open spot orders" meta={`${spotOrders.length}`}>
                  <OrdersTable orders={spotOrders} />
                </Card>
              </>
            )}
          </>
        )}
      </main>
    </div>
  )
}
