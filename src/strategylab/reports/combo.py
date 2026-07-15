#!/usr/bin/env python3
"""
combo_report.py — Validate the trend-filtered dip-buy combo under MAKER execution.

Strategy `combo(sma50, rsi14<45)` — the only mean-reversion variant that survived
the strategy_lab out-of-sample sweep:
    ENTRY : price is ABOVE the 50-period SMA  AND  RSI(14) < 45   (buy the dip,
            but only while the trend is up — no catching falling knives)
    EXIT  : RSI(14) > 60                                          (sell the bounce)

It reuses the exact maker/taker execution engines and metrics from ema_report.py,
so the numbers reconcile with the EMA/SMA cross tearsheets. Reports full period,
per calendar year, a strict chronological train/test split, and a parameter
robustness sweep — the same anti-overfit protocol as the cross reports.

Run:
    ./.venv/bin/python combo_report.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from strategylab.core import Indicators, PPY
from strategylab.reports.ema import (run_maker, run_taker, portfolio_stats, trade_stats,
                        bh, fmt_pct, TAKER_FEE, MAKER_FEE, OFFSET, MAX_WAIT)


def combo_signals(ind: Indicators, ma: int, rsi_n: int, buy: float, exit_rsi: float):
    """entry when above the trend MA and RSI dipped below `buy`; exit when RSI > exit_rsi."""
    above = ind.close > ind.sma(ma)
    r = ind.rsi(rsi_n)
    entry = above & (r < buy)      # a level, not a cross — the engine only acts when flat
    exit_ = r > exit_rsi
    return entry, exit_


# Parameter neighbourhood to probe (reported in full, not cherry-picked).
NEIGHBOURS = [
    (50, 45, 60), (50, 40, 60), (50, 35, 60),
    (100, 45, 60), (200, 45, 60),
    (50, 45, 55), (50, 45, 65),
]
USED = (50, 45, 60)
RSI_N = 14


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", "-f", default="data/binance_BTC-USDT_1h.csv")
    p.add_argument("--out", default="reports/combo_sma50_rsi14_report.md")
    args = p.parse_args()

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
    ma, buy, exit_rsi = USED
    entry, exit_ = combo_signals(ind, ma, RSI_N, buy, exit_rsi)
    elim = c * (1 - OFFSET); xlim = c * (1 + OFFSET)

    def maker_seg(i0, i1):
        return run_maker(o, h, l, c, entry, exit_, elim, xlim, i0, i1, MAKER_FEE, MAX_WAIT)

    def taker_seg(i0, i1):
        return run_taker(o, c, entry, exit_, i0, i1, TAKER_FEE)

    out = []
    def w(line=""):
        out.append(line); print(line)

    w(f"# Trend-filtered dip-buy — combo(sma{ma}, rsi{RSI_N}<{buy}) — backtest report")
    w()
    w(f"Data: `{path.name}` · {n:,} bars · {dt.iloc[0]:%Y-%m-%d} → {dt.iloc[-1]:%Y-%m-%d} "
      f"({n/ppy:.1f} years, {tf})")
    w(f"Entry: price > SMA{ma} AND RSI{RSI_N} < {buy}  ·  Exit: RSI{RSI_N} > {exit_rsi}  "
      f"(fixed — the config that survived the OOS sweep)")
    w(f"Maker: 0 fee, limit {OFFSET*100:.2f}% inside, {MAX_WAIT}-bar wait · "
      f"Taker: {TAKER_FEE*100:.2f}%/side market")
    w()

    # ---- Full period ----------------------------------------------------
    eqm, trm, fills = maker_seg(0, n)
    eqt, trt = taker_seg(0, n)
    pm, pt = portfolio_stats(eqm, ppy), portfolio_stats(eqt, ppy)
    sm, st = trade_stats(trm, hours_per_bar), trade_stats(trt, hours_per_bar)
    bh_full = bh(c, 0, n)

    w("## 1) Full period — MAKER vs TAKER vs Buy & Hold")
    w()
    w("| metric | MAKER (0 fee) | TAKER (0.10%) | Buy & Hold |")
    w("|---|---:|---:|---:|")
    w(f"| Total return | **{fmt_pct(pm['total']*100)}** | {fmt_pct(pt['total']*100)} | {fmt_pct(bh_full*100)} |")
    w(f"| CAGR | {fmt_pct(pm['cagr']*100)} | {fmt_pct(pt['cagr']*100)} | "
      f"{fmt_pct(((1+bh_full)**(ppy/n)-1)*100)} |")
    w(f"| Sharpe (ann.) | {pm['sharpe']:.2f} | {pt['sharpe']:.2f} | — |")
    w(f"| Max drawdown | {pm['maxdd']*100:.1f}% | {pt['maxdd']*100:.1f}% | "
      f"{portfolio_stats(c / c[0], ppy)['maxdd']*100:.1f}% |")
    w(f"| Profit factor | {sm['profit_factor']:.2f} | {st['profit_factor']:.2f} | — |")
    w(f"| Expectancy/trade | {sm['expectancy']:+.3f}% | {st['expectancy']:+.3f}% | — |")
    w(f"| # trades | {sm['n']} | {st['n']} | 1 |")
    w(f"| Win rate | {sm['win_rate']:.1f}% | {st['win_rate']:.1f}% | — |")
    w(f"| Avg win / loss | {sm['avg_win']:+.2f}% / {sm['avg_loss']:+.2f}% | "
      f"{st['avg_win']:+.2f}% / {st['avg_loss']:+.2f}% | — |")
    w(f"| Payoff ratio | {sm['payoff']:.2f} | {st['payoff']:.2f} | — |")
    w(f"| Avg hold | {sm['avg_days']:.1f} d | {st['avg_days']:.1f} d | — |")
    w(f"| Best / worst trade | {sm['best']:+.1f}% / {sm['worst']:+.1f}% | "
      f"{st['best']:+.1f}% / {st['worst']:+.1f}% | — |")
    w(f"| Max consec. losses | {sm['max_consec_loss']} | {st['max_consec_loss']} | — |")
    w(f"| Time in market | {portfolio_exposure(eqm, trm):.0f}% (approx) | — | 100% |")
    w(f"| Maker fill rate | {fills['fill_rate']:.1f}% ({fills['missed']} missed) | 100% | — |")
    w()

    # ---- Per year -------------------------------------------------------
    w("## 2) Per calendar year (MAKER)")
    w()
    w("| year | strat return | Buy & Hold | trades | win rate | max DD |")
    w("|---|---:|---:|---:|---:|---:|")
    years = sorted(dt.dt.year.unique().tolist())
    for y in years:
        idx = np.where((dt.dt.year == y).to_numpy())[0]
        i0, i1 = int(idx[0]), int(idx[-1]) + 1
        eq, tr, _ = maker_seg(i0, i1)
        ps = portfolio_stats(eq, ppy); ts = trade_stats(tr, hours_per_bar)
        w(f"| {y}{' (part)' if y == years[-1] else ''} | {fmt_pct(ps['total']*100)} | "
          f"{fmt_pct(bh(c, i0, i1)*100)} | {ts['n']} | {ts['win_rate']:.0f}% | {ps['maxdd']*100:.0f}% |")
    w()

    # ---- OOS split ------------------------------------------------------
    w("## 3) Out-of-sample split (strict chronological, MAKER)")
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

    # ---- Parameter robustness ------------------------------------------
    w("## 4) Parameter robustness (MAKER, full period) — reported in full")
    w()
    w("| config | total | Sharpe | maxDD | trades | win rate |")
    w("|---|---:|---:|---:|---:|---:|")
    for m2, b2, x2 in NEIGHBOURS:
        e2, ex2 = combo_signals(ind, m2, RSI_N, b2, x2)
        eq, tr, _ = run_maker(o, h, l, c, e2, ex2, elim, xlim, 0, n, MAKER_FEE, MAX_WAIT)
        ps = portfolio_stats(eq, ppy); ts = trade_stats(tr, hours_per_bar)
        star = "  ← used" if (m2, b2, x2) == USED else ""
        w(f"| sma{m2},rsi<{b2},>{x2}{star} | {fmt_pct(ps['total']*100)} | {ps['sharpe']:.2f} | "
          f"{ps['maxdd']*100:.0f}% | {ts['n']} | {ts['win_rate']:.0f}% |")
    w()

    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text("\n".join(out) + "\n")
    w(f"\n_Report saved → {args.out}_")


def portfolio_exposure(equity, trades):
    """Rough time-in-market: sum of trade holding bars / total bars, from trade list."""
    if len(equity) == 0:
        return 0.0
    held = sum(t[1] for t in trades)
    return held / len(equity) * 100


if __name__ == "__main__":
    main()
