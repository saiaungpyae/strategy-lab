#!/usr/bin/env python3
"""
regime_lab.py — Can a regime-aware, risk-managed maker strategy average 0.03%/day?

Tests the ideas from the "0.03%/day" discussion as a CUMULATIVE ladder, so each
idea's marginal contribution is visible:

  base            z-score(50,2) mean reversion, bids always resting, full size
  trend           + only rest bids while close > SMA200 (don't catch knives)
  regime          + only in RANGING markets (Kaufman efficiency ratio low) with
                    no vol spike (rolling day-vol below its trailing 90th pct)
  regime_vol      + vol-targeted sizing (risk less when vol is high)
  regime_vol_cb   + drawdown circuit breaker (half size at -3%, halt at -6%)
  ..._rest        + user's rule: after a day >= 10x target, halve size next day

Execution is pessimistic maker: entry is a resting bid that fills only if a
LATER bar trades down to it, at the level plus an adverse-selection haircut
(swept in bps); take-profit / revert-to-mean exits are maker; stop-losses pay
the taker fee. Verdict is measured on the DAILY return distribution of the
out-of-sample window vs the 0.03%/day target.

Usage:
    ./.venv/bin/python regime_lab.py
    ./.venv/bin/python regime_lab.py --file data/binance_BTC-USDT_5m.csv --train 0.7
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from strategy_lab import Indicators
from scalp_lab import BARS_PER_DAY

DAILY_TARGET = 0.0003  # 0.03%/day


def run_sized_scalp(high, low, close, buy_level, mean_arr, size_frac, day_id,
                    target_pct, stop_pct, haircut, taker_fee,
                    cb_half_dd=0.0, cb_halt_dd=0.0, rest_after=0.0,
                    cb_memory_bars=0):
    """
    Sequential maker-fill simulation with fractional position sizing.

    buy_level[t] / size_frac[t] are decided at bar t and acted on from t+1.
    cb_half_dd / cb_halt_dd: equity drawdown levels that halve / block new entries.
    cb_memory_bars: time constant over which the drawdown peak relaxes toward
      current equity, so a halt is a cooling-off period, not a permanent lockout
      (0 = all-time peak, which CAN lock the strategy out forever).
    rest_after: if the previous CALENDAR day's return >= this, halve size today (0 = off).
    """
    n = len(close)
    cash, units, position = 1.0, 0.0, 0
    entry_price = 0.0
    equity = np.empty(n)
    peak = 1.0
    decay = np.exp(-1.0 / cb_memory_bars) if cb_memory_bars else 1.0
    trades = wins = stops = 0
    day_start_eq, prev_day_ret = 1.0, 0.0
    for t in range(n):
        cur = cash + units * close[t]
        if t > 0 and day_id[t] != day_id[t - 1]:
            prev_day_ret = cur / day_start_eq - 1.0
            day_start_eq = cur
        peak = max(cur, cur + (peak - cur) * decay)
        dd = cur / peak - 1.0
        if position == 0:
            mult = 1.0
            if cb_halt_dd and dd <= -cb_halt_dd:
                mult = 0.0
            elif cb_half_dd and dd <= -cb_half_dd:
                mult = 0.5
            if rest_after and prev_day_ret >= rest_after:
                mult *= 0.5
            bl = buy_level[t - 1] if t > 0 else np.nan
            f = (size_frac[t - 1] if t > 0 else 0.0) * mult
            if bl == bl and f > 0 and low[t] <= bl:
                fill = bl * (1 + haircut)               # adverse selection: pay up
                spend = cash * f
                units = spend / fill; cash -= spend
                position = 1; entry_price = fill; trades += 1
        else:
            tp = entry_price * (1 + target_pct)
            sl = entry_price * (1 - stop_pct)
            exit_price = None; taker = False
            if low[t] <= sl:                            # stop first (pessimistic)
                exit_price = sl; taker = True; stops += 1
            elif high[t] >= tp:
                exit_price = tp * (1 - haircut)
            elif high[t] >= mean_arr[t]:
                exit_price = mean_arr[t] * (1 - haircut)
            if exit_price is not None:
                proceeds = units * exit_price * (1 - (taker_fee if taker else 0.0))
                if exit_price * (1 - (taker_fee if taker else 0.0)) > entry_price:
                    wins += 1
                cash += proceeds; units = 0.0; position = 0
        equity[t] = cash + units * close[t]
    return equity, dict(trades=trades, wins=wins, stops=stops)


def daily_stats(equity, day_id, target=DAILY_TARGET):
    """Resample an equity curve to calendar-day returns and score vs target."""
    s = pd.Series(equity, index=pd.DatetimeIndex(day_id))
    eod = s.groupby(level=0).last()
    prev = np.empty(len(eod)); prev[0] = 1.0; prev[1:] = eod.to_numpy()[:-1]
    d = eod.to_numpy() / prev - 1.0
    days = len(d)
    sd = d.std()
    dd = (eod / eod.cummax() - 1.0).min()
    return dict(
        days=days,
        mean_bp=d.mean() * 1e4,
        median_bp=np.median(d) * 1e4,
        hit_pct=(d >= target).mean() * 100,
        green_pct=(d > 0).mean() * 100,
        flat_pct=(d == 0).mean() * 100,
        worst_bp=d.min() * 1e4,
        worst_days_of_target=abs(d.min()) / target if d.min() < 0 else 0.0,
        sharpe=d.mean() / sd * np.sqrt(365) if sd > 0 else 0.0,
        maxdd_pct=dd * 100,
        total_pct=(eod.iloc[-1] - 1.0) * 100,
    )


def build_variants(ind: Indicators, df: pd.DataFrame, bpd: int):
    """Return {name: (buy_level, mean, size_frac, cb, rest)} — a cumulative ladder."""
    close = ind.close
    c = pd.Series(close)

    mean, sd = ind.sma(50), ind.std(50)
    level = mean - 2.0 * sd
    uptrend = close > ind.sma(200)

    # Kaufman efficiency ratio over ~1 day: |net move| / sum |bar moves|.
    n_er = bpd
    net = c.diff(n_er).abs()
    path = c.diff().abs().rolling(n_er).sum()
    er = (net / path).to_numpy()
    ranging = er < 0.35

    # Rolling day-vol and its trailing 30-day percentile (vol-spike detector).
    ret = c.pct_change()
    dayvol = (ret.rolling(bpd).std() * np.sqrt(bpd))
    volpct = dayvol.rolling(30 * bpd).rank(pct=True).to_numpy()
    calm = volpct < 0.90

    # Vol-targeted size: aim ~2% day-vol exposure, capped at full size.
    dv = dayvol.to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        vol_size = np.clip(0.02 / dv, 0.0, 1.0)
    vol_size[~np.isfinite(vol_size)] = 0.0

    full = np.ones_like(close)
    gate_t = np.where(uptrend, level, np.nan)
    gate_r = np.where(uptrend & ranging & calm, level, np.nan)

    return {
        "base":              (level,  mean, full,     (0.0, 0.0),  0.0),
        "trend":             (gate_t, mean, full,     (0.0, 0.0),  0.0),
        "regime":            (gate_r, mean, full,     (0.0, 0.0),  0.0),
        "regime_vol":        (gate_r, mean, vol_size, (0.0, 0.0),  0.0),
        "regime_vol_cb":     (gate_r, mean, vol_size, (0.03, 0.06), 0.0),
        "regime_vol_cb_rest":(gate_r, mean, vol_size, (0.03, 0.06), 10 * DAILY_TARGET),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", "-f", default="data/binance_BTC-USDT_15m.csv")
    p.add_argument("--train", type=float, default=0.7)
    p.add_argument("--taker-fee", type=float, default=0.001)
    p.add_argument("--target-pct", type=float, default=0.004, help="Take-profit above entry")
    p.add_argument("--stop-pct", type=float, default=0.02, help="Stop-loss below entry")
    p.add_argument("--costs-bps", type=float, nargs="+", default=[0, 1, 2, 5])
    args = p.parse_args()

    path = Path(args.file)
    if not path.is_file():
        sys.exit(f"File not found: {path}")
    tf = path.stem.split("_")[-1]
    bpd = BARS_PER_DAY.get(tf, 96)

    df = pd.read_csv(path)
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                            "close": "Close", "volume": "Volume"})
    ts = pd.to_datetime(df["datetime"], utc=True)
    h = df["High"].to_numpy(float); lo = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    day_id = ts.dt.floor("D").to_numpy()
    n = len(c); split = int(n * args.train)

    ind = Indicators(df)
    variants = build_variants(ind, df, bpd)

    print(f"\nData: {path.name} | {n:,} bars | {tf}")
    print(f"Out-of-sample window: {ts.iloc[split].date()} → {ts.iloc[-1].date()} "
          f"({(n - split) // bpd} days)  |  target: {DAILY_TARGET*1e4:.0f} bp/day "
          f"(≈{((1+DAILY_TARGET)**365-1)*100:.1f}%/yr)")
    print(f"TP {args.target_pct*100:.2f}% / SL {args.stop_pct*100:.1f}% | stop exits pay "
          f"{args.taker_fee*100:.2f}% taker fee\n")

    hdr = (f"{'variant':<20} {'mean_bp':>8} {'med_bp':>7} {'hit%':>6} {'green%':>7} "
           f"{'flat%':>6} {'worst_bp':>9} {'=days':>6} {'sharpe':>7} {'maxDD%':>7} "
           f"{'total%':>7} {'trades':>7} {'win%':>6}")
    for bps in args.costs_bps:
        haircut = bps / 10000.0
        print(f"═══ adverse-selection cost: {bps:.0f} bps/fill "
              f"{'(fantasy)' if bps == 0 else '(realistic)' if bps <= 2 else '(harsh)'} ═══")
        print(hdr)
        for name, (bl, mn, sz, (cb1, cb2), rest) in variants.items():
            eq, st = run_sized_scalp(h[split:], lo[split:], c[split:],
                                     bl[split:], mn[split:], sz[split:], day_id[split:],
                                     args.target_pct, args.stop_pct, haircut, args.taker_fee,
                                     cb_half_dd=cb1, cb_halt_dd=cb2, rest_after=rest,
                                     cb_memory_bars=30 * bpd)
            m = daily_stats(eq, day_id[split:])
            win = st["wins"] / st["trades"] * 100 if st["trades"] else 0.0
            print(f"{name:<20} {m['mean_bp']:>8.2f} {m['median_bp']:>7.2f} {m['hit_pct']:>6.1f} "
                  f"{m['green_pct']:>7.1f} {m['flat_pct']:>6.1f} {m['worst_bp']:>9.1f} "
                  f"{m['worst_days_of_target']:>6.0f} {m['sharpe']:>7.2f} {m['maxdd_pct']:>7.1f} "
                  f"{m['total_pct']:>7.1f} {st['trades']:>7d} {win:>6.1f}")
        print()

    print("Verdict columns:")
    print("  • mean_bp  = average daily return in basis points — needs ≥ 3.0 to hit the goal")
    print("  • hit%     = share of days at/above 3 bp (sitting-out days count as misses)")
    print("  • =days    = how many target-days the single worst day erases")
    print("  • Compare rows top→bottom: each adds ONE idea from the discussion.")


if __name__ == "__main__":
    main()
