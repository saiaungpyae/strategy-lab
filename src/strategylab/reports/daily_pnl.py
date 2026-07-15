#!/usr/bin/env python3
"""
daily_pnl.py — Turn a scalp equity curve into a DAILY DOLLAR P&L distribution
on a fixed capital base, to answer "can I make $X/day?".

Reuses scalp_lab's execution engine and strategy definitions so the numbers
reconcile with the lab. Runs the chosen strategy out-of-sample, then resamples
the equity curve to calendar-day P&L in dollars on --capital, and reports the
distribution: mean/median day, % of days that clear the target, % green days,
worst day, best day, and how many days had no trade at all.

Usage:
    ./.venv/bin/python daily_pnl.py --strategy "zscore_z(50,2)_trend" \
        --file data/binance_BTC-USDT_15m.csv --capital 3000 --target 1 --cost-bps 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from strategylab.core import Indicators
from strategylab.strategies.scalp import build_scalps, run_scalp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", "-f", default="data/binance_BTC-USDT_15m.csv")
    p.add_argument("--strategy", default="zscore_z(50,2)_trend")
    p.add_argument("--capital", type=float, default=3000.0)
    p.add_argument("--target", type=float, default=1.0, help="Daily $ goal to test against")
    p.add_argument("--train", type=float, default=0.7)
    p.add_argument("--cost-bps", type=float, default=1.0, help="Adverse-selection haircut per fill (bps)")
    p.add_argument("--taker-fee", type=float, default=0.001)
    args = p.parse_args()

    path = Path(args.file)
    if not path.is_file():
        sys.exit(f"File not found: {path}")

    df = pd.read_csv(path)
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                            "close": "Close", "volume": "Volume"})
    ts = pd.to_datetime(df["datetime"], utc=True)
    o = df["Open"].to_numpy(float); h = df["High"].to_numpy(float)
    lo = df["Low"].to_numpy(float); c = df["Close"].to_numpy(float)
    n = len(c); split = int(n * args.train)

    ind = Indicators(df)
    scalps = build_scalps(ind)
    if args.strategy not in scalps:
        sys.exit(f"Unknown strategy {args.strategy!r}. Options: {list(scalps)}")
    spec = scalps[args.strategy]
    haircut = args.cost_bps / 10000.0

    eq, st = run_scalp(o[split:], h[split:], lo[split:], c[split:],
                       spec["buy_level"][split:], spec["mean"][split:],
                       spec["target"], spec["stop"], spec["use_mean_exit"],
                       spec["max_hold"], haircut, args.taker_fee)

    # equity is a growth multiple starting at 1.0 → dollars on the capital base
    dollars = eq * args.capital
    day = ts.iloc[split:].dt.floor("D").to_numpy()

    # Daily P&L = change in dollar equity within each calendar day, chained across
    # the day boundary (P&L of day d = equity at last bar of d − equity at last bar of d-1).
    s = pd.Series(dollars, index=pd.DatetimeIndex(day))
    eod = s.groupby(level=0).last()
    start_equity = args.capital
    prev = pd.concat([pd.Series([start_equity]), eod.iloc[:-1].reset_index(drop=True)]).to_numpy()
    daily_pnl = eod.to_numpy() - prev

    d = pd.Series(daily_pnl, index=eod.index)
    total_days = len(d)
    green = (d > 0).sum()
    flat = (d == 0).sum()
    hit = (d >= args.target).sum()
    losedays = (d < 0).sum()

    print(f"\n=== Daily P&L distribution — {args.strategy} ===")
    print(f"File {path.name} | out-of-sample test window | capital ${args.capital:,.0f} "
          f"| cost {args.cost_bps:.1f} bps/fill")
    print(f"Trades over window: {st['trades']:,} ({st['trades'] / total_days:.2f}/day, "
          f"win rate {st['wins'] / st['trades'] * 100:.1f}%)" if st['trades'] else "No trades")
    print(f"Test window: {total_days} calendar days "
          f"({eod.index[0].date()} → {eod.index[-1].date()})\n")

    end_dollars = args.capital + daily_pnl.sum()
    print(f"End equity: ${end_dollars:,.0f}  (net {end_dollars - args.capital:+,.0f}, "
          f"{(end_dollars/args.capital - 1)*100:+.1f}% over window)")
    print(f"Avg per calendar day: ${d.mean():+.2f}   median: ${d.median():+.2f}")
    print(f"Std of daily P&L:     ${d.std():.2f}")
    print(f"Best day:  ${d.max():+.2f}     Worst day: ${d.min():+.2f}\n")

    print(f"Days ≥ ${args.target:.0f} target : {hit:4d} / {total_days}  ({hit/total_days*100:.1f}%)")
    print(f"Green days (> $0)     : {green:4d} / {total_days}  ({green/total_days*100:.1f}%)")
    print(f"Flat days ($0, no P&L): {flat:4d} / {total_days}  ({flat/total_days*100:.1f}%)")
    print(f"Losing days (< $0)    : {losedays:4d} / {total_days}  ({losedays/total_days*100:.1f}%)")

    # A single bad day can erase how many $1 target days?
    if d.min() < 0:
        print(f"\nOne worst day (${d.min():+.2f}) erases {abs(d.min())/args.target:.0f} "
              f"days of hitting the ${args.target:.0f} target.")

    # percentiles
    qs = d.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    print("\nDaily P&L percentiles ($):")
    for q, v in qs.items():
        print(f"  p{int(q*100):>2}: {v:+8.2f}")


if __name__ == "__main__":
    main()
