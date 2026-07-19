#!/usr/bin/env python3
"""
ema_report.py — Detailed, anti-overfit backtest report for the EMA-cross trend
strategy on the full local history (BTC/USDT 1h, 2021-01 → present).

Why this is *not* an overfit:
  • Parameters are FIXED at the canonical MACD default (EMA 12 vs EMA 26). They
    are not searched or tuned to this data.
  • The strategy has NO take-profit / stop-loss knobs to fit. Exit = the death
    cross. Entry = the golden cross. There is nothing to curve-fit.
  • Robustness is shown three ways: (a) per-calendar-year (does it work every
    year or just in one bull run?), (b) a strict chronological train/test split
    (out-of-sample), and (c) a neighbouring-parameter sweep reported in full
    (not cherry-picked) so you can see the result isn't a knife-edge.

Both execution models are reported:
  • MAKER — your zero-fee limit-order case (post inside the market, fill only if
    price comes to you within a wait window, else miss). This is your real edge.
  • TAKER — market orders, guaranteed fill next bar's open, pay 0.10% per side.

Run:
    ./.venv/bin/python ema_report.py
    ./.venv/bin/python ema_report.py --file data/ohlcv/BTC-USDT/binance_BTC-USDT_1h.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from strategylab.core import Indicators, PPY
from strategylab.data.paths import default_candles

# Execution assumptions (identical to maker_backtest.py so numbers reconcile).
TAKER_FEE = 0.001      # 0.10% per side, market orders
MAKER_FEE = 0.0        # zero-fee liquidity provider (no rebate credited = conservative)
OFFSET = 0.0005        # post limit orders 0.05% inside the market
MAX_WAIT = 24          # leave a resting limit order for up to 24 bars, else cancel
# Default family/params: canonical MACD EMAs — FIXED, not optimised. Override on
# the CLI, e.g. --kind sma --fast 50 --slow 200 for the golden/death cross.
KIND, FAST, SLOW = "ema", 12, 26


# ----------------------------------------------------------------------------
# Signals
# ----------------------------------------------------------------------------
def cross_signals(ind: Indicators, kind: str, fast: int, slow: int):
    ma = ind.ema if kind == "ema" else ind.sma
    f, s = ma(fast), ma(slow)
    regime = f > s
    prev = np.zeros_like(regime)
    prev[1:] = regime[:-1]
    entry = regime & ~prev             # bar the fast MA crosses ABOVE the slow
    exit_ = (~regime) & prev           # bar it crosses back BELOW
    return entry, exit_


# Neighbouring parameter pairs to probe per family (reported in full, not tuned).
NEIGHBOURS = {
    "ema": [(8, 21), (12, 26), (20, 50), (50, 100), (50, 200)],
    "sma": [(20, 100), (50, 150), (50, 200), (100, 200), (50, 250)],
}
PARAM_NOTE = {
    "ema": "canonical MACD defaults",
    "sma": "the classic golden/death-cross defaults",
}


# ----------------------------------------------------------------------------
# Execution engines over a bar range [i0, i1). Each starts flat and returns the
# segment equity curve (base 1.0) plus the list of completed round-trip trades.
# ----------------------------------------------------------------------------
def run_taker(o, c, entry, exit_, i0, i1, fee):
    cash, units, position, order = 1.0, 0.0, 0, None
    entry_cash = entry_bar = None
    equity, trades = [], []
    for t in range(i0, i1):
        if order == "buy":
            entry_cash = cash; units = cash / o[t] * (1 - fee); cash = 0.0
            position = 1; entry_bar = t; order = None
        elif order == "sell":
            cash = units * o[t] * (1 - fee)
            trades.append((cash / entry_cash - 1.0, t - entry_bar)); units = 0.0
            position = 0; order = None
        if order is None:
            if position == 0 and entry[t]:
                order = "buy"
            elif position == 1 and exit_[t]:
                order = "sell"
        equity.append(cash + units * c[t])
    return np.asarray(equity), trades


def run_maker(o, h, l, c, entry, exit_, elim, xlim, i0, i1, fee, max_wait):
    cash, units, position, pending = 1.0, 0.0, 0, None
    entry_cash = entry_bar = None
    equity, trades = [], []
    attempts = filled = missed = 0
    for t in range(i0, i1):
        if pending is not None:                        # try to fill a resting order
            side, limit, age = pending
            hit = (side == "buy" and l[t] <= limit) or (side == "sell" and h[t] >= limit)
            if hit:
                if side == "buy":
                    entry_cash = cash; units = cash / limit * (1 - fee); cash = 0.0
                    position = 1; entry_bar = t; filled += 1
                else:
                    cash = units * limit * (1 - fee)
                    trades.append((cash / entry_cash - 1.0, t - entry_bar)); units = 0.0
                    position = 0
                pending = None
            else:
                age += 1
                if age >= max_wait:
                    if side == "buy":
                        missed += 1
                    pending = None                     # cancel; a missed sell keeps holding
                else:
                    pending = (side, limit, age)
        if pending is None:                            # post a new order from this bar's signal
            if position == 0 and entry[t]:
                attempts += 1; pending = ("buy", float(elim[t]), 0)
            elif position == 1 and exit_[t]:
                pending = ("sell", float(xlim[t]), 0)
        equity.append(cash + units * c[t])
    fill_rate = (filled / attempts * 100) if attempts else 0.0
    return np.asarray(equity), trades, dict(attempts=attempts, filled=filled,
                                            missed=missed, fill_rate=fill_rate)


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def portfolio_stats(equity, ppy):
    if len(equity) < 2:
        return dict(total=0.0, cagr=0.0, sharpe=0.0, maxdd=0.0)
    ret = np.empty(len(equity)); ret[0] = 0.0; ret[1:] = equity[1:] / equity[:-1] - 1.0
    end = float(equity[-1]); years = len(equity) / ppy; sd = ret.std()
    return dict(
        total=end - 1.0,
        cagr=(end ** (1 / years) - 1) if end > 0 and years > 0 else -1.0,
        sharpe=(ret.mean() / sd * np.sqrt(ppy)) if sd > 0 else 0.0,
        maxdd=float((equity / np.maximum.accumulate(equity) - 1).min()),
    )


def trade_stats(trades, hours_per_bar):
    n = len(trades)
    if n == 0:
        return dict(n=0, win_rate=0.0, avg_win=0.0, avg_loss=0.0, payoff=0.0,
                    profit_factor=0.0, expectancy=0.0, avg_days=0.0,
                    best=0.0, worst=0.0, max_consec_loss=0)
    r = np.array([t[0] for t in trades]); bars = np.array([t[1] for t in trades])
    wins = r[r > 0]; losses = r[r <= 0]
    gp = wins.sum(); gl = -losses.sum()
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0
    # longest run of consecutive losing trades
    mc = c = 0
    for x in r:
        c = c + 1 if x <= 0 else 0
        mc = max(mc, c)
    return dict(
        n=n,
        win_rate=len(wins) / n * 100,
        avg_win=avg_win * 100,
        avg_loss=avg_loss * 100,
        payoff=(avg_win / abs(avg_loss)) if avg_loss < 0 else float("inf"),
        profit_factor=(gp / gl) if gl > 0 else float("inf"),
        expectancy=r.mean() * 100,
        avg_days=bars.mean() * hours_per_bar / 24.0,
        best=r.max() * 100, worst=r.min() * 100,
        max_consec_loss=mc,
    )


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------
def bh(c, i0, i1):
    return c[i1 - 1] / c[i0] - 1.0


def fmt_pct(x):
    return f"{x:+.1f}%"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", "-f", default=default_candles("1h"))
    p.add_argument("--kind", choices=["ema", "sma"], default=KIND, help="moving-average family")
    p.add_argument("--fast", type=int, default=FAST, help="fast MA length")
    p.add_argument("--slow", type=int, default=SLOW, help="slow MA length")
    p.add_argument("--out", default=None, help="markdown output path (default reports/<kind>_cross_1h_report.md)")
    args = p.parse_args()
    kind, fast, slow = args.kind, args.fast, args.slow
    if args.out is None:
        args.out = f"reports/{kind}_cross_1h_report.md"

    path = Path(args.file)
    df = pd.read_csv(path)
    dt = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    dfi = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                             "close": "Close", "volume": "Volume"})
    o = dfi["Open"].to_numpy(float); h = dfi["High"].to_numpy(float)
    l = dfi["Low"].to_numpy(float); c = dfi["Close"].to_numpy(float)
    n = len(c)
    tf = path.stem.split("_")[-1]
    ppy = PPY.get(tf, 8760)
    hours_per_bar = 8760 / ppy

    ind = Indicators(dfi)
    entry, exit_ = cross_signals(ind, kind, fast, slow)
    elim = c * (1 - OFFSET); xlim = c * (1 + OFFSET)

    def maker_seg(i0, i1):
        eq, tr, fills = run_maker(o, h, l, c, entry, exit_, elim, xlim, i0, i1, MAKER_FEE, MAX_WAIT)
        return eq, tr, fills

    def taker_seg(i0, i1):
        eq, tr = run_taker(o, c, entry, exit_, i0, i1, TAKER_FEE)
        return eq, tr

    out = []
    def w(line=""):
        out.append(line); print(line)

    w(f"# {kind.upper()}-cross ({fast}/{slow}) trend — backtest report")
    w()
    w(f"Data: `{path.name}` · {n:,} bars · {dt.iloc[0]:%Y-%m-%d} → {dt.iloc[-1]:%Y-%m-%d} "
      f"({n/ppy:.1f} years, {tf})")
    w(f"Params: {kind.upper()} fast={fast}, slow={slow} ({PARAM_NOTE[kind]} — fixed, not tuned)")
    w(f"Exit rule: NONE fixed. No take-profit, no stop-loss. In on golden cross, "
      f"out on death cross.")
    w(f"Maker: 0 fee, limit {OFFSET*100:.2f}% inside, {MAX_WAIT}-bar wait · "
      f"Taker: {TAKER_FEE*100:.2f}%/side market")
    w()

    # ---- Full period, maker vs taker -------------------------------------
    eqm, trm, fills = maker_seg(0, n)
    eqt, trt = taker_seg(0, n)
    pm, pt = portfolio_stats(eqm, ppy), portfolio_stats(eqt, ppy)
    sm, st = trade_stats(trm, hours_per_bar), trade_stats(trt, hours_per_bar)
    bh_full = bh(c, 0, n)

    w("## 1) Full period — MAKER (your edge) vs TAKER")
    w()
    w("| metric | MAKER (0 fee) | TAKER (0.10%) | Buy & Hold |")
    w("|---|---:|---:|---:|")
    w(f"| Total return | **{fmt_pct(pm['total']*100)}** | {fmt_pct(pt['total']*100)} | {fmt_pct(bh_full*100)} |")
    w(f"| CAGR | {fmt_pct(pm['cagr']*100)} | {fmt_pct(pt['cagr']*100)} | "
      f"{fmt_pct(((1+bh_full)**(ppy/n)-1)*100)} |")
    w(f"| Sharpe (ann.) | {pm['sharpe']:.2f} | {pt['sharpe']:.2f} | — |")
    w(f"| Max drawdown | {pm['maxdd']*100:.1f}% | {pt['maxdd']*100:.1f}% | "
      f"{portfolio_stats(c / c[0], ppy)['maxdd']*100:.1f}% |")
    w(f"| # trades | {sm['n']} | {st['n']} | 1 |")
    w(f"| Win rate | {sm['win_rate']:.1f}% | {st['win_rate']:.1f}% | — |")
    w(f"| Avg win | {sm['avg_win']:+.2f}% | {st['avg_win']:+.2f}% | — |")
    w(f"| Avg loss | {sm['avg_loss']:+.2f}% | {st['avg_loss']:+.2f}% | — |")
    w(f"| Payoff (win/loss) | {sm['payoff']:.2f} | {st['payoff']:.2f} | — |")
    w(f"| Profit factor | {sm['profit_factor']:.2f} | {st['profit_factor']:.2f} | — |")
    w(f"| Expectancy/trade | {sm['expectancy']:+.3f}% | {st['expectancy']:+.3f}% | — |")
    w(f"| Avg hold | {sm['avg_days']:.1f} d | {st['avg_days']:.1f} d | — |")
    w(f"| Best / Worst trade | {sm['best']:+.1f}% / {sm['worst']:+.1f}% | "
      f"{st['best']:+.1f}% / {st['worst']:+.1f}% | — |")
    w(f"| Max consec. losses | {sm['max_consec_loss']} | {st['max_consec_loss']} | — |")
    w(f"| Maker fill rate | {fills['fill_rate']:.1f}% ({fills['missed']} missed) | 100% | — |")
    w()

    # ---- TP/SL answer ----------------------------------------------------
    w("## 2) \"What's the TP/SL ratio?\"")
    w()
    w("There isn't one — and that's deliberate. This is a *trend-follower*: it "
      "holds until the trend flips (death cross), so winners run and losers are "
      "cut at the reversal, not at a fixed level. The realised economics you're "
      "actually getting (maker, full period):")
    w()
    w(f"- **Win rate {sm['win_rate']:.0f}%** with a **payoff ratio of {sm['payoff']:.2f}** "
      f"(avg win {sm['avg_win']:+.2f}% vs avg loss {sm['avg_loss']:+.2f}%). "
      "Classic trend profile: lose small often, win big occasionally.")
    w(f"- The **worst single trade was {sm['worst']:+.1f}%** — that is your de-facto "
      "per-trade risk with no stop. If you want a hard stop, that's the number to "
      "cap, but adding/tuning a stop on this same data is exactly how you'd start "
      "overfitting, so treat it as risk sizing, not optimisation.")
    w(f"- Position sizing matters more than a TP/SL here: at {sm['worst']:.0f}% worst-case "
      f"and up to {sm['max_consec_loss']} losses in a row, size so that run is survivable.")
    w()

    # ---- Per year --------------------------------------------------------
    w("## 3) Per calendar year (MAKER) — does it work every year, or just one bull run?")
    w()
    w("| year | strat return | Buy & Hold | trades | win rate | max DD |")
    w("|---|---:|---:|---:|---:|---:|")
    years = sorted(dt.dt.year.unique().tolist())
    for y in years:
        mask = (dt.dt.year == y).to_numpy()
        idx = np.where(mask)[0]
        i0, i1 = int(idx[0]), int(idx[-1]) + 1
        eq, tr, _ = maker_seg(i0, i1)
        ps = portfolio_stats(eq, ppy); ts = trade_stats(tr, hours_per_bar)
        w(f"| {y}{' (part)' if y == years[-1] else ''} | {fmt_pct(ps['total']*100)} | "
          f"{fmt_pct(bh(c, i0, i1)*100)} | {ts['n']} | {ts['win_rate']:.0f}% | {ps['maxdd']*100:.0f}% |")
    w()

    # ---- Out-of-sample split --------------------------------------------
    w("## 4) Out-of-sample split (strict chronological, MAKER)")
    w()
    split_year = years[len(years) // 2]
    split_idx = int(np.where((dt.dt.year >= split_year).to_numpy())[0][0])
    for label, i0, i1 in [("TRAIN (in-sample)", 0, split_idx),
                          ("TEST (out-of-sample)", split_idx, n)]:
        eq, tr, _ = maker_seg(i0, i1)
        ps = portfolio_stats(eq, ppy); ts = trade_stats(tr, hours_per_bar)
        seg = f"{dt.iloc[i0]:%Y-%m} → {dt.iloc[i1-1]:%Y-%m}"
        w(f"- **{label}** ({seg}): total {fmt_pct(ps['total']*100)}, "
          f"CAGR {fmt_pct(ps['cagr']*100)}, Sharpe {ps['sharpe']:.2f}, "
          f"maxDD {ps['maxdd']*100:.0f}%, {ts['n']} trades, win {ts['win_rate']:.0f}%, "
          f"vs B&H {fmt_pct(bh(c, i0, i1)*100)}")
    w()
    w("If the TEST half is in the same ballpark as TRAIN, the edge generalises "
      "rather than being fit to one period.")
    w()

    # ---- Parameter robustness -------------------------------------------
    w("## 5) Neighbouring parameters (MAKER, full period) — reported in full, not cherry-picked")
    w()
    w(f"| {kind.upper()} pair | total | Sharpe | trades | win rate |")
    w("|---|---:|---:|---:|---:|")
    for f, s in NEIGHBOURS[kind]:
        e2, x2 = cross_signals(ind, kind, f, s)
        eq, tr, _ = run_maker(o, h, l, c, e2, x2, elim, xlim, 0, n, MAKER_FEE, MAX_WAIT)
        ps = portfolio_stats(eq, ppy); ts = trade_stats(tr, hours_per_bar)
        star = "  ← used" if (f, s) == (fast, slow) else ""
        w(f"| {f}/{s}{star} | {fmt_pct(ps['total']*100)} | {ps['sharpe']:.2f} | "
          f"{ts['n']} | {ts['win_rate']:.0f}% |")
    w()
    w(f"A whole neighbourhood of pairs being positive (not just {fast}/{slow}) means the "
      "result is a property of the *approach*, not a single lucky setting.")
    w()

    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text("\n".join(out) + "\n")
    w(f"\n_Report saved → {args.out}_")


if __name__ == "__main__":
    main()
