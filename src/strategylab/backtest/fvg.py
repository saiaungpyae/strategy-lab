#!/usr/bin/env python3
"""
fvg.py — Fair Value Gap event study with a random-entry control.

A Fair Value Gap (FVG) is a 3-candle imbalance: the wicks of candle 1 and
candle 3 don't overlap, leaving an untraded "gap" across the middle candle.
The ICT claim: price tends to return to the gap ("rebalance") and then
continue in the direction of the original impulse.

This module tests that claim mechanically, with no chart-reading bias:

  1. Detect every FVG (min height filter in ATR units, to skip noise).
  2. Wait for price to retrace into the gap. Enter at the proximal edge,
     stop at the distal edge, target = rr x risk. Walk forward bar by bar
     (stop checked before target within a bar — conservative).
  3. CONTROL: replay the same trades (same direction, same relative risk
     geometry, same hold limit) at random entry bars, many trials. If FVG
     entries don't beat the random distribution, the gap added nothing.

Usage:
    ./.venv/bin/python -m strategylab.backtest.fvg
    ./.venv/bin/python -m strategylab.backtest.fvg --file data/binance_BTC-USDT_15m.csv --rr 2
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FVGParams:
    min_atr_frac: float = 0.25   # min gap height as a fraction of ATR(atr_period)
    atr_period: int = 14
    max_age: int = 500           # bars a gap waits for a touch before expiring
    max_hold: int = 96           # bars after entry before a timeout exit at close
    rr: float = 2.0              # target = rr * (entry - stop)
    fee: float = 0.001           # taker fee per side (0.10%)
    control_trials: int = 200    # random-entry resamples for the baseline
    seed: int = 42


@dataclass
class FVGEvent:
    form_idx: int        # index of the 3rd candle that completed the gap
    direction: int       # +1 bullish, -1 bearish
    top: float           # upper price of the gap zone
    bottom: float        # lower price of the gap zone
    touch_idx: int | None = None    # bar where price re-entered the zone
    resolve_idx: int | None = None  # bar where the trade resolved
    entry: float | None = None
    stop: float | None = None
    target: float | None = None
    outcome: str = "untouched"      # win | loss | timeout | untouched | open
    net_pct: float | None = None    # % return net of fees (per unit notional)
    r_mult: float | None = None     # result in R (risk units), gross of fees


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int) -> np.ndarray:
    pc = np.empty_like(close)
    pc[0] = close[0]
    pc[1:] = close[:-1]
    tr = np.maximum(high - low, np.maximum(np.abs(high - pc), np.abs(low - pc)))
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().to_numpy()


def detect_fvgs(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                p: FVGParams) -> list[FVGEvent]:
    """3-candle FVGs: bullish when low[i] > high[i-2], bearish when high[i] < low[i-2]."""
    atr = _atr(high, low, close, p.atr_period)
    events: list[FVGEvent] = []
    n = len(close)
    bull = np.zeros(n, dtype=bool)
    bear = np.zeros(n, dtype=bool)
    bull[2:] = low[2:] > high[:-2]
    bear[2:] = high[2:] < low[:-2]
    warmup = max(p.atr_period * 3, 20)  # let ATR settle before trusting the filter
    for i in np.flatnonzero(bull | bear):
        if i < warmup:
            continue
        if bull[i]:
            top, bottom, d = low[i], high[i - 2], +1
        else:
            top, bottom, d = low[i - 2], high[i], -1
        if (top - bottom) >= p.min_atr_frac * atr[i]:
            events.append(FVGEvent(form_idx=int(i), direction=d,
                                   top=float(top), bottom=float(bottom)))
    return events


def _walk_trade(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                start: int, direction: int, entry: float, stop: float,
                target: float, max_hold: int, fee: float):
    """Walk bars from `start` until stop/target/timeout. Stop wins ties (conservative).

    Returns (resolve_idx, outcome, exit_price) — outcome 'open' if data ran out."""
    n = len(close)
    end = min(start + max_hold, n - 1)
    for k in range(start, end + 1):
        if direction > 0:
            hit_stop, hit_tgt = low[k] <= stop, high[k] >= target
        else:
            hit_stop, hit_tgt = high[k] >= stop, low[k] <= target
        if hit_stop:                 # same-bar ambiguity resolved against us
            return k, "loss", stop
        if hit_tgt:
            return k, "win", target
    if end == n - 1 and start + max_hold > n - 1:
        return end, "open", close[end]      # ran off the end of the data
    return end, "timeout", close[end]


def _net_pct(direction: int, entry: float, exit_: float, fee: float) -> float:
    return direction * (exit_ / entry - 1.0) - 2.0 * fee


def simulate(high: np.ndarray, low: np.ndarray, close: np.ndarray,
             events: list[FVGEvent], p: FVGParams) -> list[FVGEvent]:
    """Fill each event in place: touch, entry at proximal edge, stop at distal edge."""
    n = len(close)
    for ev in events:
        i = ev.form_idx
        hi_lim = min(i + 1 + p.max_age, n)
        if ev.direction > 0:
            proximal, distal = ev.top, ev.bottom       # price retraces DOWN into gap
            touched = low[i + 1:hi_lim] <= proximal
        else:
            proximal, distal = ev.bottom, ev.top       # price retraces UP into gap
            touched = high[i + 1:hi_lim] >= proximal
        if not touched.any():
            ev.outcome = "untouched"
            continue
        j = i + 1 + int(np.argmax(touched))            # first touch bar
        risk = abs(proximal - distal)
        entry = proximal
        stop = distal
        target = entry + ev.direction * p.rr * risk
        k, outcome, exit_ = _walk_trade(high, low, close, j, ev.direction,
                                        entry, stop, target, p.max_hold, p.fee)
        ev.touch_idx, ev.resolve_idx = j, k
        ev.entry, ev.stop, ev.target = entry, stop, target
        ev.outcome = outcome
        ev.net_pct = _net_pct(ev.direction, entry, exit_, p.fee)
        ev.r_mult = ev.direction * (exit_ - entry) / risk if risk > 0 else 0.0
    return events


def random_control(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                   trades: list[FVGEvent], p: FVGParams) -> dict:
    """Same trades — direction, relative risk size, rr, max_hold — at random bars.

    Answers: did the FVG *location/timing* add anything beyond the risk geometry?"""
    n = len(close)
    rng = np.random.default_rng(p.seed)
    lo_idx = max(p.atr_period * 3, 20)
    hi_idx = n - p.max_hold - 2
    if hi_idx <= lo_idx or not trades:
        return {"trials": 0}

    rel_risks = [abs(t.entry - t.stop) / t.entry for t in trades]
    dirs = [t.direction for t in trades]
    win_rates, exps = [], []
    for _ in range(p.control_trials):
        starts = rng.integers(lo_idx, hi_idx, size=len(trades))
        wins = decided = 0
        pnl = []
        for s, d, rr_ in zip(starts, dirs, rel_risks):
            entry = close[s]
            risk = rr_ * entry
            stop = entry - d * risk
            target = entry + d * p.rr * risk
            _, outcome, exit_ = _walk_trade(high, low, close, int(s) + 1, d,
                                            entry, stop, target, p.max_hold, p.fee)
            if outcome in ("win", "loss"):
                decided += 1
                wins += outcome == "win"
            pnl.append(_net_pct(d, entry, exit_, p.fee))
        win_rates.append(wins / decided if decided else 0.0)
        exps.append(float(np.mean(pnl)))
    return {
        "trials": p.control_trials,
        "win_rate_mean": float(np.mean(win_rates)),
        "win_rate_sd": float(np.std(win_rates)),
        "win_rates": win_rates,
        "exp_mean": float(np.mean(exps)),
    }


def summarize(events: list[FVGEvent], control: dict) -> dict:
    trades = [e for e in events if e.outcome in ("win", "loss", "timeout")]
    wins = [e for e in trades if e.outcome == "win"]
    losses = [e for e in trades if e.outcome == "loss"]
    timeouts = [e for e in trades if e.outcome == "timeout"]
    decided = len(wins) + len(losses)
    win_rate = len(wins) / decided if decided else 0.0
    touched = [e for e in events if e.touch_idx is not None]

    pctile = None
    if control.get("trials"):
        wr = np.asarray(control["win_rates"])
        pctile = float((wr < win_rate).mean())

    s = {
        "gaps": len(events),
        "bull": sum(1 for e in events if e.direction > 0),
        "bear": sum(1 for e in events if e.direction < 0),
        "touched": len(touched),
        "touch_rate": len(touched) / len(events) if events else 0.0,
        "wins": len(wins), "losses": len(losses), "timeouts": len(timeouts),
        "win_rate": win_rate,
        "avg_net_pct": float(np.mean([e.net_pct for e in trades])) if trades else 0.0,
        "avg_r": float(np.mean([e.r_mult for e in trades])) if trades else 0.0,
        "control_win_rate": control.get("win_rate_mean"),
        "control_sd": control.get("win_rate_sd"),
        "control_exp": control.get("exp_mean"),
        "percentile_vs_random": pctile,
    }
    profitable = s["avg_net_pct"] > 0
    if pctile is None:
        s["verdict"], s["grade"] = "no trades — nothing to judge", "none"
    elif pctile >= 0.975 and profitable:
        s["verdict"], s["grade"] = "PASS — beats random AND profitable after fees", "pass"
    elif pctile >= 0.975:
        s["verdict"], s["grade"] = "MIXED — beats random timing, but fees eat the edge", "mixed"
    elif pctile >= 0.90:
        s["verdict"], s["grade"] = "WEAK — better than random, not significant", "weak"
    else:
        s["verdict"], s["grade"] = "FAIL — indistinguishable from random entries", "fail"
    return s


def run_fvg_study(df: pd.DataFrame, p: FVGParams = FVGParams()):
    """df needs lowercase open/high/low/close columns. Returns (events, summary)."""
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    events = detect_fvgs(high, low, close, p)
    simulate(high, low, close, events, p)
    trades = [e for e in events if e.outcome in ("win", "loss", "timeout")]
    control = random_control(high, low, close, trades, p)

    if "timestamp" in df.columns and len(df) > 1:
        dt_ms = float(np.median(np.diff(df["timestamp"].to_numpy()[:1000])))
        bars_per_day = 86_400_000 / dt_ms if dt_ms > 0 else 24.0
    else:
        bars_per_day = 24.0
    summary = summarize(events, control)
    summary["gaps_per_day"] = len(events) / (len(df) / bars_per_day) if len(df) else 0.0
    summary["days"] = len(df) / bars_per_day
    return events, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", "-f", default="data/binance_BTC-USDT_15m.csv")
    ap.add_argument("--rr", type=float, default=2.0, help="reward:risk target multiple")
    ap.add_argument("--min-atr", type=float, default=0.25, help="min gap height in ATRs")
    ap.add_argument("--fee", type=float, default=0.001)
    ap.add_argument("--trials", type=int, default=200)
    args = ap.parse_args()

    path = Path(args.file)
    if not path.is_file():
        sys.exit(f"File not found: {path}")
    df = pd.read_csv(path)
    p = FVGParams(rr=args.rr, min_atr_frac=args.min_atr, fee=args.fee,
                  control_trials=args.trials)

    print(f"\nFVG event study — {path.name} ({len(df):,} bars)")
    print(f"params: min gap {p.min_atr_frac} ATR | rr {p.rr} | fee {p.fee*100:.2f}%/side "
          f"| max_age {p.max_age} | max_hold {p.max_hold}\n")

    events, s = run_fvg_study(df, p)
    print(f"Gaps found:      {s['gaps']:,}  ({s['bull']:,} bull / {s['bear']:,} bear)"
          f"  ~{s['gaps_per_day']:.1f}/day")
    print(f"Touched (filled):{s['touched']:,}  ({s['touch_rate']*100:.0f}% of gaps)")
    print(f"Resolved trades: {s['wins']:,} wins / {s['losses']:,} losses / "
          f"{s['timeouts']:,} timeouts")
    print(f"Win rate:        {s['win_rate']*100:.1f}%  "
          f"(needs >{100/(1+p.rr):.0f}% to break even at {p.rr}R before fees)")
    print(f"Avg net/trade:   {s['avg_net_pct']*100:+.3f}%   avg R: {s['avg_r']:+.2f}")
    print(f"\nRANDOM CONTROL ({args.trials} trials, same direction/risk/hold at random bars):")
    print(f"Random win rate: {s['control_win_rate']*100:.1f}% ± {s['control_sd']*100:.1f}%")
    print(f"FVG percentile:  {s['percentile_vs_random']*100:.0f}th of random distribution")
    print(f"\nVERDICT: {s['verdict']}\n")


if __name__ == "__main__":
    main()
