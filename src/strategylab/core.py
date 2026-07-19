#!/usr/bin/env python3
"""
strategy_lab.py — Mass strategy sweep with out-of-sample validation.

Generates thousands of strategy configurations (strategy templates expanded
across parameter grids), backtests every one with a fast vectorized engine
(0.10% fee/trade, no lookahead), and — crucially — evaluates them on a
train/test split so you can tell a real edge from an overfit fluke.

The whole point of testing thousands of strategies is the multiple-testing
problem: on ANY dataset, some strategies look brilliant by pure luck. Only
out-of-sample survival (good on train AND on unseen test data) means anything.

Usage:
    ./.venv/bin/python strategy_lab.py                       # 1h data, default
    ./.venv/bin/python strategy_lab.py --file data/ohlcv/BTC-USDT/binance_BTC-USDT_15m.csv
    ./.venv/bin/python strategy_lab.py --commission 0 --train 0.7
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from strategylab.data.paths import default_candles

# Bars per year, by timeframe — for annualizing Sharpe / CAGR.
PPY = {"1m": 525600, "3m": 175200, "5m": 105120, "15m": 35040, "30m": 17520,
       "1h": 8760, "2h": 4380, "4h": 2190, "6h": 1460, "12h": 730, "1d": 365}


# ----------------------------------------------------------------------------
# Indicator cache — compute each unique indicator once, reuse across configs.
# ----------------------------------------------------------------------------
class Indicators:
    def __init__(self, df: pd.DataFrame):
        self.close = df["Close"].to_numpy(dtype=float)
        self.high = df["High"].to_numpy(dtype=float)
        self.low = df["Low"].to_numpy(dtype=float)
        self._c = df["Close"]
        self._h = df["High"]
        self._l = df["Low"]
        self._sma, self._ema, self._std = {}, {}, {}
        self._rsi, self._roc = {}, {}
        self._dchi, self._dclo = {}, {}
        self._atr = {}

    def sma(self, n):
        if n not in self._sma:
            self._sma[n] = self._c.rolling(n).mean().to_numpy()
        return self._sma[n]

    def ema(self, n):
        if n not in self._ema:
            self._ema[n] = self._c.ewm(span=n, adjust=False).mean().to_numpy()
        return self._ema[n]

    def std(self, n):
        if n not in self._std:
            self._std[n] = self._c.rolling(n).std().to_numpy()
        return self._std[n]

    def rsi(self, n):
        if n not in self._rsi:
            delta = self._c.diff()
            up = delta.clip(lower=0)
            down = -delta.clip(upper=0)
            ru = up.ewm(alpha=1 / n, adjust=False).mean()
            rd = down.ewm(alpha=1 / n, adjust=False).mean()
            self._rsi[n] = (100 - 100 / (1 + ru / rd)).to_numpy()
        return self._rsi[n]

    def roc(self, n):
        if n not in self._roc:
            self._roc[n] = self._c.pct_change(n).to_numpy() * 100
        return self._roc[n]

    def donchian_hi(self, n):
        if n not in self._dchi:
            self._dchi[n] = self._h.rolling(n).max().shift(1).to_numpy()
        return self._dchi[n]

    def donchian_lo(self, n):
        if n not in self._dclo:
            self._dclo[n] = self._l.rolling(n).min().shift(1).to_numpy()
        return self._dclo[n]

    def atr(self, n):
        if n not in self._atr:
            pc = self._c.shift(1)
            tr = pd.concat([self._h - self._l, (self._h - pc).abs(), (self._l - pc).abs()], axis=1).max(axis=1)
            self._atr[n] = tr.ewm(alpha=1 / n, adjust=False).mean().to_numpy()
        return self._atr[n]


def hold_from_signals(entry: np.ndarray, exit_: np.ndarray) -> np.ndarray:
    """Stateful position: 1 on entry, hold until exit, 0 otherwise (vectorized ffill)."""
    raw = np.where(entry, 1.0, np.where(exit_, 0.0, np.nan))
    pos = pd.Series(raw).ffill().to_numpy()
    return np.nan_to_num(pos, nan=0.0)


def supertrend_dir(ind: Indicators, period, mult) -> np.ndarray:
    atr = ind.atr(period)
    hl2 = (ind.high + ind.low) / 2
    close = ind.close
    n = len(close)
    bu, bl = hl2 + mult * atr, hl2 - mult * atr
    fu, fl = bu.copy(), bl.copy()
    d = np.ones(n)
    for i in range(1, n):
        fu[i] = bu[i] if (bu[i] < fu[i - 1] or close[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = bl[i] if (bl[i] > fl[i - 1] or close[i - 1] < fl[i - 1]) else fl[i - 1]
        if close[i] > fu[i - 1]:
            d[i] = 1
        elif close[i] < fl[i - 1]:
            d[i] = -1
        else:
            d[i] = d[i - 1]
    return (d > 0).astype(float)


# ----------------------------------------------------------------------------
# Config generators — each yields (family, label, target_position_array)
# target_position[t] uses only info available at bar t (no lookahead).
# ----------------------------------------------------------------------------
def generate(ind: Indicators):
    close = ind.close

    # 1) SMA crossover  (the bulk of the configs)
    sma_fast = list(range(3, 100, 2))
    sma_slow = list(range(10, 400, 3))
    for f in sma_fast:
        sf = ind.sma(f)
        for s in sma_slow:
            if s <= f:
                continue
            yield "sma_cross", f"sma({f},{s})", (sf > ind.sma(s)).astype(float)

    # 2) EMA crossover
    ema_fast = list(range(3, 60, 2))
    ema_slow = list(range(10, 300, 3))
    for f in ema_fast:
        ef = ind.ema(f)
        for s in ema_slow:
            if s <= f:
                continue
            yield "ema_cross", f"ema({f},{s})", (ef > ind.ema(s)).astype(float)

    # 3) RSI mean reversion (stateful)
    for n in (7, 10, 14, 21, 28):
        r = ind.rsi(n)
        for buy in (15, 20, 25, 30, 35, 40):
            for ex in (50, 55, 60, 65, 70, 75):
                if ex <= buy:
                    continue
                yield "rsi_reversion", f"rsi({n},<{buy},>{ex})", hold_from_signals(r < buy, r > ex)

    # 4) Bollinger reversion (stateful): buy below lower band, exit above mid
    for n in (10, 15, 20, 30, 40, 50):
        mid, sd = ind.sma(n), ind.std(n)
        for k in (1.5, 2.0, 2.5, 3.0):
            lower = mid - k * sd
            yield "bollinger", f"bb({n},{k})", hold_from_signals(close < lower, close > mid)

    # 5) Donchian breakout (stateful): break N-bar high, exit on M-bar low
    for n in range(10, 120, 4):
        hi = ind.donchian_hi(n)
        for m in (n // 2, n):
            lo = ind.donchian_lo(m)
            yield "donchian", f"don({n},{m})", hold_from_signals(close >= hi, close <= lo)

    # 6) MACD regime: long while MACD line above signal
    for fast in (6, 8, 10, 12, 16):
        for slow in (20, 26, 30, 40):
            if slow <= fast:
                continue
            macd = ind.ema(fast) - ind.ema(slow)
            for sig in (7, 9, 12):
                signal = pd.Series(macd).ewm(span=sig, adjust=False).mean().to_numpy()
                yield "macd", f"macd({fast},{slow},{sig})", (macd > signal).astype(float)

    # 7) Supertrend
    for period in (7, 10, 14, 20):
        for mult in (1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
            yield "supertrend", f"st({period},{mult})", supertrend_dir(ind, period, mult)

    # 8) Momentum / ROC regime: long when N-bar return exceeds a threshold
    for n in (12, 24, 48, 96, 168, 336):
        r = ind.roc(n)
        for th in (-5, 0, 2, 5, 10):
            yield "momentum", f"roc({n},>{th})", (r > th).astype(float)

    # 9) Trend filter: long only when price above a long MA (classic "regime filter")
    for n in (20, 50, 100, 150, 200, 300):
        yield "ma_trend", f"above_sma({n})", (close > ind.sma(n)).astype(float)

    # 10) MA trend + RSI pullback combo: above trend MA AND rsi dipped then rose
    for ma in (50, 100, 200):
        above = close > ind.sma(ma)
        for rp in (7, 14):
            r = ind.rsi(rp)
            for buy in (35, 40, 45):
                yield "trend_rsi_combo", f"combo(sma{ma},rsi{rp}<{buy})", \
                    hold_from_signals(above & (r < buy), r > 60)


# ----------------------------------------------------------------------------
# Vectorized backtest
# ----------------------------------------------------------------------------
def net_returns(close: np.ndarray, target: np.ndarray, commission: float):
    """Return (net_return_per_bar, held_position, entry_flags). Lookahead-safe."""
    n = len(close)
    held = np.empty(n)
    held[0] = 0.0
    held[1:] = target[:-1]                       # hold during bar t what we chose at t-1
    aret = np.empty(n)
    aret[0] = 0.0
    aret[1:] = close[1:] / close[:-1] - 1.0
    turn = np.empty(n)
    turn[0] = held[0]
    turn[1:] = np.abs(held[1:] - held[:-1])
    entries = np.zeros(n)
    entries[1:] = (held[1:] > held[:-1]).astype(float)
    net = held * aret - turn * commission
    return net, held, entries


def metrics(net, held, entries, ppy):
    equity = np.cumprod(1.0 + net)
    end = equity[-1]
    total = end - 1.0
    years = len(net) / ppy
    cagr = end ** (1.0 / years) - 1.0 if end > 0 and years > 0 else -1.0
    sd = net.std()
    sharpe = (net.mean() / sd * np.sqrt(ppy)) if sd > 0 else 0.0
    dd = (equity / np.maximum.accumulate(equity) - 1.0).min()
    return dict(total=total, cagr=cagr, sharpe=sharpe, maxdd=dd,
                trades=int(entries.sum()), exposure=float(held.mean()))


def bh_metrics(close, ppy):
    aret = np.empty(len(close)); aret[0] = 0.0; aret[1:] = close[1:] / close[:-1] - 1.0
    equity = close / close[0]
    years = len(close) / ppy
    sd = aret.std()
    return dict(total=equity[-1] - 1.0,
                cagr=equity[-1] ** (1 / years) - 1 if years > 0 else 0.0,
                sharpe=aret.mean() / sd * np.sqrt(ppy) if sd > 0 else 0.0,
                maxdd=(equity / np.maximum.accumulate(equity) - 1).min())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", "-f", default=default_candles("1h"))
    p.add_argument("--commission", "-c", type=float, default=0.001)
    p.add_argument("--train", type=float, default=0.7, help="Fraction of data used for in-sample training")
    p.add_argument("--top", type=int, default=100, help="How many train-winners to validate out-of-sample")
    args = p.parse_args()

    path = Path(args.file)
    if not path.is_file():
        sys.exit(f"File not found: {path}")

    tf = path.stem.split("_")[-1]
    ppy = PPY.get(tf, 8760)

    df = pd.read_csv(path)
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    close = df["Close"].to_numpy(dtype=float)
    n = len(close)
    split = int(n * args.train)

    print(f"\nData: {path.name} | {n:,} bars | timeframe {tf} | fee {args.commission*100:.3f}%/trade")
    print(f"Train: bars 0–{split:,}  |  Test (out-of-sample): bars {split:,}–{n:,}\n")

    bh_train = bh_metrics(close[:split], ppy)
    bh_test = bh_metrics(close[split:], ppy)

    ind = Indicators(df)
    print("Generating + backtesting configs...")
    t0 = time.time()
    rows = []
    count = 0
    for family, label, target in generate(ind):
        count += 1
        # Train segment
        net, held, ent = net_returns(close[:split], target[:split], args.commission)
        mtr = metrics(net, held, ent, ppy)
        # Test segment (out-of-sample)
        net2, held2, ent2 = net_returns(close[split:], target[split:], args.commission)
        mte = metrics(net2, held2, ent2, ppy)
        rows.append(dict(
            family=family, label=label,
            train_ret=mtr["total"], train_sharpe=mtr["sharpe"], train_dd=mtr["maxdd"], train_trades=mtr["trades"],
            test_ret=mte["total"], test_sharpe=mte["sharpe"], test_dd=mte["maxdd"], test_trades=mte["trades"],
        ))
        if count % 2000 == 0:
            print(f"  ...{count:,} configs ({time.time()-t0:.0f}s)")

    res = pd.DataFrame(rows)
    dt = time.time() - t0
    print(f"\nDone: {len(res):,} strategy configs backtested in {dt:.1f}s "
          f"({len(res)/dt:,.0f} backtests/sec)\n")

    # ---- Benchmarks ----
    print("=" * 74)
    print("BENCHMARK — just holding BTC:")
    print(f"  Train return: {bh_train['total']*100:+.1f}%   |   Test (out-of-sample) return: {bh_test['total']*100:+.1f}%")
    print(f"  Test Sharpe: {bh_test['sharpe']:.2f}   |   Test max drawdown: {bh_test['maxdd']*100:.1f}%")
    print("=" * 74)

    # ---- In-sample illusion ----
    beat_train = res[res["train_ret"] > bh_train["total"]]
    print(f"\nIN-SAMPLE (train): {len(beat_train):,} of {len(res):,} configs "
          f"({len(beat_train)/len(res)*100:.1f}%) 'beat' buy & hold.")
    print("  ^ This is the illusion. On the data they were tuned on, tons look great.")

    # ---- The honest test: pick winners on TRAIN, judge them on unseen TEST ----
    valid = res[res["train_trades"] >= 20].copy()
    winners = valid.sort_values("train_sharpe", ascending=False).head(args.top).copy()
    survivors = winners[winners["test_ret"] > bh_test["total"]]
    profitable_oos = winners[winners["test_ret"] > 0]

    print(f"\nOUT-OF-SAMPLE test — take the top {args.top} by TRAIN Sharpe, judge on UNSEEN data:")
    print(f"  Beat buy & hold out-of-sample: {len(survivors)} / {args.top} "
          f"({len(survivors)/args.top*100:.0f}%)")
    print(f"  Merely profitable out-of-sample: {len(profitable_oos)} / {args.top}")
    print(f"  Median out-of-sample return of these 'winners': {winners['test_ret'].median()*100:+.1f}%  "
          f"(vs buy & hold {bh_test['total']*100:+.1f}%)")

    print(f"\nTop 15 train-winners and how they ACTUALLY did out-of-sample:")
    show = winners.head(15)[["family", "label", "train_ret", "train_sharpe", "test_ret", "test_sharpe", "test_trades"]].copy()
    for col in ("train_ret", "test_ret"):
        show[col] = (show[col] * 100).round(1)
    show = show.rename(columns={"train_ret": "train_%", "test_ret": "test_%"})
    pd.options.display.float_format = lambda x: f"{x:,.2f}"
    print(show.to_string(index=False))

    # ---- Best on the TEST set itself (for contrast: cherry-picking the answer) ----
    best_oos = valid.sort_values("test_ret", ascending=False).head(1).iloc[0]
    print(f"\nFor contrast — the single best config ON the test set itself was "
          f"{best_oos['label']} ({best_oos['test_ret']*100:+.1f}%).")
    print("  But you could only know that by peeking at the answer — useless for real trading.")

    out = Path("reports") / f"sweep_{path.stem}.csv"
    out.parent.mkdir(exist_ok=True)
    res.to_csv(out, index=False)
    print(f"\nFull results ({len(res):,} rows) saved -> {out}")


if __name__ == "__main__":
    main()
