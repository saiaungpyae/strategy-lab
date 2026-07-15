#!/usr/bin/env python3
"""
scalp_lab.py — Test frequent low-timeframe SCALPING strategies for a zero-fee
maker (liquidity provider).

Scalping is where zero fees matter most: a normal trader pays a fee on every
one of hundreds of trades and dies by a thousand cuts. With zero maker fees
that obstacle is gone — but a NEW cost takes over: the effective spread +
adverse selection on your limit fills. Bars can't measure it, so we model it
as a tunable per-fill 'edge cost' (in basis points) and sweep it. Whether a
scalp survives a realistic few-bps cost is the entire question.

Execution model (maker, liquidity-provider style):
  • Entry: a resting BUY limit at a computed dip level; fills if a bar trades
    down to it. You pay an adverse-selection haircut on the fill.
  • Take-profit: a resting SELL limit above entry (maker, zero fee, haircut).
  • Stop-loss: a protective exit that fills at market (TAKER — pays fee), because
    when you're wrong you can't wait to be a maker.
  • Optional trend filter: only rest bids while price is above a long MA — the
    standard fix for 'catching a falling knife' in downtrends.

Usage:
    ./.venv/bin/python scalp_lab.py --file data/binance_BTC-USDT_5m.csv
    ./.venv/bin/python scalp_lab.py --file data/binance_BTC-USDT_15m.csv --train 0.7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from strategylab.core import Indicators, PPY

BARS_PER_DAY = {"1m": 1440, "3m": 480, "5m": 288, "15m": 96, "30m": 48,
                "1h": 24, "2h": 12, "4h": 6, "1d": 1}


def run_scalp(open_, high, low, close, buy_level, mean_arr,
              target_pct, stop_pct, use_mean_exit, max_hold,
              haircut, taker_fee):
    """
    Vector inputs, sequential fill simulation. buy_level[t] is the resting bid
    level decided at bar t (used from t+1). NaN = no bid resting.
    Returns equity curve + trade stats.
    """
    n = len(close)
    cash, units, position = 1.0, 0.0, 0
    entry_price, hold = 0.0, 0
    equity = np.empty(n)
    trades = wins = hold_sum = stops = 0
    for t in range(n):
        if position == 0:
            bl = buy_level[t - 1] if t > 0 else np.nan
            if bl == bl and low[t] <= bl:                 # bl==bl skips NaN
                fill = bl * (1 + haircut)                 # adverse selection: pay up a touch
                units = cash / fill; cash = 0.0
                position = 1; entry_price = fill; hold = 0; trades += 1
        else:
            hold += 1
            tp = entry_price * (1 + target_pct)
            sl = entry_price * (1 - stop_pct)
            exit_price = None; taker = False
            if low[t] <= sl:                              # stop first (pessimistic intrabar order)
                exit_price = sl; taker = True; stops += 1
            elif high[t] >= tp:
                exit_price = tp * (1 - haircut)           # maker take-profit
            elif use_mean_exit and high[t] >= mean_arr[t]:
                exit_price = mean_arr[t] * (1 - haircut)  # maker revert-to-mean exit
            elif max_hold and hold >= max_hold:
                exit_price = close[t]; taker = True       # time-stop at market
            if exit_price is not None:
                fee = taker_fee if taker else 0.0
                proceeds = units * exit_price * (1 - fee)
                if proceeds > (units * entry_price):
                    wins += 1
                cash = proceeds; units = 0.0
                position = 0; hold_sum += hold
        equity[t] = cash + units * close[t]
    return equity, dict(trades=trades, wins=wins, stops=stops,
                        avg_hold=(hold_sum / trades if trades else 0.0))


def metrics(equity, ppy):
    ret = np.empty(len(equity)); ret[0] = 0.0; ret[1:] = equity[1:] / equity[:-1] - 1.0
    end = equity[-1]; years = len(equity) / ppy
    sd = ret.std()
    return dict(total=end - 1.0,
                sharpe=ret.mean() / sd * np.sqrt(ppy) if sd > 0 else 0.0,
                maxdd=(equity / np.maximum.accumulate(equity) - 1).min())


def build_scalps(ind: Indicators):
    """Each scalp returns buy_level array, mean array, and exit params."""
    close = ind.close
    S = {}
    sma200 = ind.sma(200)
    uptrend = close > sma200

    # z-score dip: rest bid k std below a short mean, take profit reverting to mean
    for tag, n, k in [("z(20,2)", 20, 2.0), ("z(20,2.5)", 20, 2.5), ("z(50,2)", 50, 2.0)]:
        mean, sd = ind.sma(n), ind.std(n)
        level = mean - k * sd
        S[f"zscore_{tag}"] = dict(buy_level=level, mean=mean, target=0.004, stop=0.02,
                                  use_mean_exit=True, max_hold=0, trend=None)
        # trend-filtered: only rest the bid while above the 200-MA
        lvl_t = np.where(uptrend, level, np.nan)
        S[f"zscore_{tag}_trend"] = dict(buy_level=lvl_t, mean=mean, target=0.004, stop=0.02,
                                        use_mean_exit=True, max_hold=0, trend="up")

    # Bollinger scalp: bid at lower band, take profit at mid band
    mid, sd20 = ind.sma(20), ind.std(20)
    lower = mid - 2 * sd20
    S["bb_scalp"] = dict(buy_level=lower, mean=mid, target=0.006, stop=0.02,
                         use_mean_exit=True, max_hold=0, trend=None)
    S["bb_scalp_trend"] = dict(buy_level=np.where(uptrend, lower, np.nan), mean=mid,
                               target=0.006, stop=0.02, use_mean_exit=True, max_hold=0, trend="up")

    # Fixed grid scalp: bid 0.4% under last close, +0.4% profit target, hard stop
    prev = np.empty_like(close); prev[0] = close[0]; prev[1:] = close[:-1]
    grid_level = prev * (1 - 0.004)
    S["grid_0.4%"] = dict(buy_level=grid_level, mean=mid, target=0.004, stop=0.015,
                          use_mean_exit=False, max_hold=48, trend=None)
    S["grid_0.4%_trend"] = dict(buy_level=np.where(uptrend, grid_level, np.nan), mean=mid,
                                target=0.004, stop=0.015, use_mean_exit=False, max_hold=48, trend="up")
    return S


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", "-f", default="data/binance_BTC-USDT_5m.csv")
    p.add_argument("--train", type=float, default=0.7)
    p.add_argument("--taker-fee", type=float, default=0.001, help="Fee paid on stop-loss market exits")
    p.add_argument("--haircuts-bps", type=float, nargs="+", default=[0, 1, 2, 5, 10],
                   help="Adverse-selection costs to sweep, in basis points (default 0 1 2 5 10)")
    args = p.parse_args()

    path = Path(args.file)
    if not path.is_file():
        sys.exit(f"File not found: {path}")
    tf = path.stem.split("_")[-1]
    ppy = PPY.get(tf, 105120)
    bpd = BARS_PER_DAY.get(tf, 288)

    df = pd.read_csv(path)
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    o = df["Open"].to_numpy(float); h = df["High"].to_numpy(float)
    lo = df["Low"].to_numpy(float); c = df["Close"].to_numpy(float)
    n = len(c); split = int(n * args.train)
    days_test = (n - split) / bpd

    print(f"\nData: {path.name} | {n:,} bars | {tf} | ~{bpd} bars/day")
    print(f"Train: 0–{split:,}  |  Test (out-of-sample): {split:,}–{n:,}  (~{days_test:.0f} days)")
    print(f"Sweeping adverse-selection cost (bps per fill): {args.haircuts_bps}\n")

    bh_test = c[split:][-1] / c[split:][0] - 1
    ind = Indicators(df)
    scalps = build_scalps(ind)

    for name, spec in scalps.items():
        print(f"── {name}  (target {spec['target']*100:.2f}%, stop {spec['stop']*100:.1f}%"
              f"{', trend-filtered' if spec['trend'] else ''})")
        header = f"{'cost(bps)':>9} {'test_ret%':>9} {'sharpe':>7} {'maxDD%':>7} {'trades':>7} {'trades/day':>10} {'win%':>6} {'avg_hold':>8}"
        print(header)
        for bps in args.haircuts_bps:
            haircut = bps / 10000.0
            eq, st = run_scalp(o[split:], h[split:], lo[split:], c[split:],
                               spec["buy_level"][split:], spec["mean"][split:],
                               spec["target"], spec["stop"], spec["use_mean_exit"],
                               spec["max_hold"], haircut, args.taker_fee)
            m = metrics(eq, ppy)
            tpd = st["trades"] / days_test if days_test else 0
            win = (st["wins"] / st["trades"] * 100) if st["trades"] else 0
            print(f"{bps:>9.0f} {m['total']*100:>9.1f} {m['sharpe']:>7.2f} {m['maxdd']*100:>7.1f} "
                  f"{st['trades']:>7d} {tpd:>10.1f} {win:>6.1f} {st['avg_hold']:>8.1f}")
        print()

    print(f"Benchmark — buy & hold over the out-of-sample test window: {bh_test*100:+.1f}%")
    print("\nHow to read this:")
    print("  • cost(bps) = effective spread + adverse selection per fill. 0 is fantasy; 1–5 bps is realistic for liquid BTC.")
    print("  • A scalp is only real if it stays profitable at a realistic cost AND has enough trades to trust.")
    print("  • Watch how fast returns decay as cost rises — that slope is your true margin of safety.")


if __name__ == "__main__":
    main()
