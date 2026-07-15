#!/usr/bin/env python3
"""
scalp_optimize.py — Honest search for a scalp with real, surviving edge.

Sweeps entry level / take-profit / stop / trend-filter across a grid, ranks
configs on the TRAIN half at a REALISTIC per-fill cost, then validates the
winners on the UNSEEN TEST half. A scalp only counts if it is profitable
out-of-sample at a realistic cost AND trades often enough to trust.

This is the definitive test of "can I scalp with my zero-fee edge?": no rigged
reward:risk, no peeking at the test set, cost baked in from the start.

Usage:
    ./.venv/bin/python scalp_optimize.py --file data/binance_BTC-USDT_15m.csv
    ./.venv/bin/python scalp_optimize.py --file data/binance_BTC-USDT_5m.csv --cost-bps 2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from strategy_lab import Indicators, PPY
from scalp_lab import run_scalp, metrics, BARS_PER_DAY


def entry_families(ind: Indicators):
    """Return dict: name -> (buy_level, mean, use_mean_exit, max_hold)."""
    close = ind.close
    fam = {}
    for n, k in [(20, 2.0), (20, 2.5), (30, 2.0), (50, 2.0)]:
        mean, sd = ind.sma(n), ind.std(n)
        fam[f"z({n},{k})"] = (mean - k * sd, mean, True, 0)
    prev = np.empty_like(close); prev[0] = close[0]; prev[1:] = close[:-1]
    mid = ind.sma(20)
    for drop in (0.003, 0.005):
        fam[f"grid({drop*100:.1f}%)"] = (prev * (1 - drop), mid, False, 48)
    return fam


def evaluate(o, h, lo, c, level, mean, target, stop, use_mean, max_hold, cost_bps, taker_fee, ppy, bpd, days):
    eq, st = run_scalp(o, h, lo, c, level, mean, target, stop, use_mean, max_hold,
                       cost_bps / 10000.0, taker_fee)
    m = metrics(eq, ppy)
    return dict(ret=m["total"], sharpe=m["sharpe"], maxdd=m["maxdd"],
                trades=st["trades"], win=(st["wins"] / st["trades"] if st["trades"] else 0),
                tpd=st["trades"] / days if days else 0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", "-f", default="data/binance_BTC-USDT_15m.csv")
    p.add_argument("--train", type=float, default=0.7)
    p.add_argument("--cost-bps", type=float, default=2.0, help="Realistic per-fill cost for ranking (default 2 bps)")
    p.add_argument("--taker-fee", type=float, default=0.001)
    p.add_argument("--min-trades", type=int, default=100, help="Min TRAIN trades to trust a config")
    p.add_argument("--top", type=int, default=20)
    args = p.parse_args()

    path = Path(args.file)
    if not path.is_file():
        sys.exit(f"File not found: {path}")
    tf = path.stem.split("_")[-1]
    ppy = PPY.get(tf, 35040)
    bpd = BARS_PER_DAY.get(tf, 96)

    df = pd.read_csv(path)
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    o = df["Open"].to_numpy(float); h = df["High"].to_numpy(float)
    lo = df["Low"].to_numpy(float); c = df["Close"].to_numpy(float)
    n = len(c); split = int(n * args.train)
    tr = slice(0, split); te = slice(split, n)
    days_tr = split / bpd; days_te = (n - split) / bpd

    ind = Indicators(df)
    sma200 = ind.sma(200)
    uptrend = c > sma200
    fams = entry_families(ind)

    targets = [0.003, 0.005, 0.008, 0.012]
    stops = [0.006, 0.012, 0.02]
    trends = [False, True]

    print(f"\nData: {path.name} | {n:,} bars | {tf}")
    print(f"Train 0–{split:,} (~{days_tr:.0f}d)  |  Test {split:,}–{n:,} (~{days_te:.0f}d)")
    print(f"Ranking cost: {args.cost_bps:.0f} bps/fill  |  grid = {len(fams)}×{len(targets)}×{len(stops)}×{len(trends)} "
          f"= {len(fams)*len(targets)*len(stops)*len(trends)} configs\n")

    rows = []
    t0 = time.time()
    for fname, (level, mean, use_mean, max_hold) in fams.items():
        for use_trend in trends:
            lvl = np.where(uptrend, level, np.nan) if use_trend else level
            for target in targets:
                for stop in stops:
                    tr_res = evaluate(o[tr], h[tr], lo[tr], c[tr], lvl[tr], mean[tr],
                                      target, stop, use_mean, max_hold, args.cost_bps, args.taker_fee, ppy, bpd, days_tr)
                    if tr_res["trades"] < args.min_trades:
                        continue
                    te_res = evaluate(o[te], h[te], lo[te], c[te], lvl[te], mean[te],
                                      target, stop, use_mean, max_hold, args.cost_bps, args.taker_fee, ppy, bpd, days_te)
                    # gross (0-cost) train return, to expose expectancy before friction
                    gross = evaluate(o[tr], h[tr], lo[tr], c[tr], lvl[tr], mean[tr],
                                     target, stop, use_mean, max_hold, 0.0, args.taker_fee, ppy, bpd, days_tr)
                    rows.append(dict(
                        family=fname, trend=use_trend, target=target, stop=stop,
                        train_gross=gross["ret"], train_ret=tr_res["ret"], train_sharpe=tr_res["sharpe"],
                        train_trades=tr_res["trades"], train_win=tr_res["win"],
                        test_ret=te_res["ret"], test_sharpe=te_res["sharpe"], test_trades=te_res["trades"],
                        test_win=te_res["win"], test_tpd=te_res["tpd"], test_maxdd=te_res["maxdd"],
                    ))
    res = pd.DataFrame(rows)
    print(f"Backtested {len(res):,} valid configs in {time.time()-t0:.0f}s\n")

    bh_test = c[te][-1] / c[te][0] - 1
    print("=" * 78)
    print(f"BENCHMARK — buy & hold on the test window: {bh_test*100:+.1f}%")
    print("=" * 78)

    # How many even have positive GROSS (0-cost) train expectancy?
    pos_gross = res[res["train_gross"] > 0]
    print(f"\nConfigs with positive gross (0-cost) train edge: {len(pos_gross)}/{len(res)}")
    print(f"Configs still positive on train after {args.cost_bps:.0f} bps cost: {(res['train_ret']>0).sum()}/{len(res)}")

    # The honest test: pick winners on train, judge out-of-sample
    winners = res[res["train_ret"] > 0].sort_values("train_sharpe", ascending=False).head(args.top)
    if winners.empty:
        print("\nNo config was even profitable on TRAIN after realistic cost. Scalping edge: not found.")
    else:
        survivors = winners[winners["test_ret"] > 0]
        beat_bh = winners[winners["test_ret"] > bh_test]
        print(f"\nOf the top {len(winners)} TRAIN winners (by Sharpe), out-of-sample at {args.cost_bps:.0f} bps:")
        print(f"  Still profitable on unseen TEST:  {len(survivors)}/{len(winners)}")
        print(f"  Beat buy & hold on TEST:          {len(beat_bh)}/{len(winners)}")
        print(f"\nTop train-winners and their REAL out-of-sample result:")
        show = winners.copy()
        for col in ("train_gross", "train_ret", "test_ret", "test_maxdd"):
            show[col] = (show[col] * 100).round(1)
        for col in ("train_win", "test_win"):
            show[col] = (show[col] * 100).round(0)
        show["target"] = (show["target"] * 100).round(1)
        show["stop"] = (show["stop"] * 100).round(1)
        cols = ["family", "trend", "target", "stop", "train_gross", "train_ret",
                "test_ret", "test_sharpe", "test_win", "test_tpd", "test_trades"]
        print(show[cols].to_string(index=False,
              header=["family", "trend", "tgt%", "stop%", "trainGross%", "trainNet%",
                      "TEST%", "testSharpe", "testWin%", "t/day", "testTrades"]))

    out = Path("reports") / f"scalp_opt_{path.stem}.csv"
    out.parent.mkdir(exist_ok=True)
    res.to_csv(out, index=False)
    print(f"\nFull results saved -> {out}")
    print("\nVerdict rule: a real scalp is profitable out-of-sample at a realistic cost with many trades.")


if __name__ == "__main__":
    main()
