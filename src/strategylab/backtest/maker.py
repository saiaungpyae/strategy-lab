#!/usr/bin/env python3
"""
maker_backtest.py — Backtest with realistic MAKER (limit-order) execution.

Models the liquidity-provider case: you pay zero fee (or earn a rebate), but a
limit order only fills if the market actually trades to your price — otherwise
you MISS the trade. This captures the hidden cost of maker-only execution
(unfilled orders + adverse selection) that a naive fee=0 backtest ignores.

Each strategy is run two ways for comparison:
  • TAKER  — market order, guaranteed fill at next bar's open, pays taker fee.
  • MAKER  — limit order at your price, zero fee, fills only if price comes to
             you (within a wait window), else the trade is missed.

Usage:
    ./.venv/bin/python maker_backtest.py
    ./.venv/bin/python maker_backtest.py --file data/binance_BTC-USDT_15m.csv
    ./.venv/bin/python maker_backtest.py --taker-fee 0.001 --rebate 0.0001 --max-wait 24
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from strategylab.core import Indicators, PPY


# ----------------------------------------------------------------------------
# Execution engines (cash/units accounting, mark-to-market at each bar close)
# ----------------------------------------------------------------------------
def taker_execute(open_, close, entry, exit_, fee):
    """Market orders: guaranteed fill at the NEXT bar's open, pay `fee` per side."""
    n = len(close)
    cash, units, position = 1.0, 0.0, 0
    order = None
    equity = np.empty(n)
    fills = 0
    for t in range(n):
        if order == "buy":
            units = cash / open_[t] * (1 - fee); cash = 0.0; position = 1; fills += 1; order = None
        elif order == "sell":
            cash = units * open_[t] * (1 - fee); units = 0.0; position = 0; fills += 1; order = None
        if order is None:
            if position == 0 and entry[t]:
                order = "buy"
            elif position == 1 and exit_[t]:
                order = "sell"
        equity[t] = cash + units * close[t]
    return equity, dict(attempts=fills, filled=fills, missed=0)


def maker_execute(open_, high, low, close, entry, exit_, entry_limit, exit_limit, fee, max_wait):
    """
    Limit orders: post at your price, fill only if a LATER bar trades to it.
      buy limit L  fills if a later bar's low  <= L  (fill at min(open, L))
      sell limit L fills if a later bar's high >= L  (fill at max(open, L))
    Unfilled after `max_wait` bars -> cancelled (missed). `fee` may be negative
    (a maker rebate). Orders are checked starting the bar AFTER they're posted.
    """
    n = len(close)
    cash, units, position = 1.0, 0.0, 0
    pending = None  # (side, limit_price, age)
    equity = np.empty(n)
    attempts = filled = missed = 0
    for t in range(n):
        # 1) try to fill an order posted on a previous bar
        if pending is not None:
            side, limit, age = pending
            hit = (side == "buy" and low[t] <= limit) or (side == "sell" and high[t] >= limit)
            if hit:
                # Conservative: a resting limit fills at exactly its price (no
                # price improvement credited, even if the bar gapped through it).
                if side == "buy":
                    units = cash / limit * (1 - fee); cash = 0.0; position = 1; filled += 1
                else:
                    cash = units * limit * (1 - fee); units = 0.0; position = 0
                pending = None
            else:
                age += 1
                if age >= max_wait:
                    if side == "buy":
                        missed += 1
                    pending = None      # give up; sells that miss just keep holding
                else:
                    pending = (side, limit, age)
        # 2) post a new order from this bar's signal (fills evaluated from t+1)
        if pending is None:
            if position == 0 and entry[t]:
                attempts += 1
                pending = ("buy", float(entry_limit[t]), 0)
            elif position == 1 and exit_[t]:
                pending = ("sell", float(exit_limit[t]), 0)
        equity[t] = cash + units * close[t]
    return equity, dict(attempts=attempts, filled=filled, missed=missed)


def metrics(equity, ppy):
    ret = np.empty(len(equity)); ret[0] = 0.0; ret[1:] = equity[1:] / equity[:-1] - 1.0
    end = equity[-1]
    years = len(equity) / ppy
    sd = ret.std()
    return dict(
        total=end - 1.0,
        cagr=end ** (1 / years) - 1 if end > 0 and years > 0 else -1.0,
        sharpe=ret.mean() / sd * np.sqrt(ppy) if sd > 0 else 0.0,
        maxdd=(equity / np.maximum.accumulate(equity) - 1).min(),
    )


def shift_bool(a):
    out = np.zeros_like(a, dtype=bool)
    out[1:] = a[:-1]
    return out


# ----------------------------------------------------------------------------
# Strategy signal builders. Each returns:
#   entry, exit_ (bool arrays), entry_limit, exit_limit (price arrays), kind
# kind = 'reversion' (maker-natural) or 'trend' (maker-hostile)
# ----------------------------------------------------------------------------
def build_strategies(ind: Indicators, offset: float):
    close = ind.close
    S = {}

    # Bollinger reversion — the natural maker: bid at the lower band, ask at mid.
    mid, sd = ind.sma(20), ind.std(20)
    lower = mid - 2 * sd
    S["bollinger"] = dict(
        entry=close < lower, exit=close > mid,
        entry_limit=lower, exit_limit=mid, kind="reversion",
    )

    # RSI reversion — buy the dip (bid just below), sell the recovery (ask just above).
    r = ind.rsi(14)
    S["rsi_reversion"] = dict(
        entry=r < 30, exit=r > 55,
        entry_limit=close * (1 - offset), exit_limit=close * (1 + offset), kind="reversion",
    )

    # SMA golden/death cross — trend. Entry only on the cross event (miss = truly missed).
    f, s = ind.sma(50), ind.sma(200)
    regime = f > s
    S["sma_cross"] = dict(
        entry=regime & ~shift_bool(regime), exit=(~regime) & shift_bool(regime),
        entry_limit=close * (1 - offset), exit_limit=close * (1 + offset), kind="trend",
    )

    # EMA cross — faster trend.
    f2, s2 = ind.ema(12), ind.ema(26)
    reg2 = f2 > s2
    S["ema_cross"] = dict(
        entry=reg2 & ~shift_bool(reg2), exit=(~reg2) & shift_bool(reg2),
        entry_limit=close * (1 - offset), exit_limit=close * (1 + offset), kind="trend",
    )

    # Supertrend — trend flip.
    from strategylab.core import supertrend_dir
    d = supertrend_dir(ind, 10, 3.0) > 0
    S["supertrend"] = dict(
        entry=d & ~shift_bool(d), exit=(~d) & shift_bool(d),
        entry_limit=close * (1 - offset), exit_limit=close * (1 + offset), kind="trend",
    )
    return S


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", "-f", default="data/binance_BTC-USDT_1h.csv")
    p.add_argument("--taker-fee", type=float, default=0.001, help="Taker fee (default 0.10%%)")
    p.add_argument("--rebate", type=float, default=0.0,
                   help="Maker rebate as a fraction; models 'get paid to provide liquidity'. "
                        "Maker fee used = -rebate (default 0 = free)")
    p.add_argument("--offset", type=float, default=0.0005,
                   help="How far inside the market to post limit orders (default 0.05%%)")
    p.add_argument("--max-wait", type=int, default=24, help="Bars to leave a limit order resting (default 24)")
    args = p.parse_args()

    path = Path(args.file)
    if not path.is_file():
        sys.exit(f"File not found: {path}")
    tf = path.stem.split("_")[-1]
    ppy = PPY.get(tf, 8760)

    df = pd.read_csv(path)
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    open_ = df["Open"].to_numpy(float); high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float); close = df["Close"].to_numpy(float)

    maker_fee = -args.rebate  # rebate => negative fee => you get paid
    print(f"\nData: {path.name} | {len(df):,} bars | timeframe {tf}")
    print(f"Taker fee: {args.taker_fee*100:.3f}%  |  Maker fee: {maker_fee*100:+.3f}% "
          f"({'rebate' if maker_fee < 0 else 'free'})  |  limit offset {args.offset*100:.3f}%  |  "
          f"max wait {args.max_wait} bars\n")

    bh = close[-1] / close[0] - 1
    ind = Indicators(df)
    strategies = build_strategies(ind, args.offset)

    rows = []
    for name, spec in strategies.items():
        eq_t, st_t = taker_execute(open_, close, spec["entry"], spec["exit"], args.taker_fee)
        eq_m, st_m = maker_execute(open_, high, low, close, spec["entry"], spec["exit"],
                                   spec["entry_limit"], spec["exit_limit"], maker_fee, args.max_wait)
        mt, mm = metrics(eq_t, ppy), metrics(eq_m, ppy)
        fill_rate = (st_m["filled"] / st_m["attempts"] * 100) if st_m["attempts"] else 0.0
        rows.append(dict(
            strategy=name, kind=spec["kind"],
            taker_ret=mt["total"] * 100, maker_ret=mm["total"] * 100,
            taker_sharpe=mt["sharpe"], maker_sharpe=mm["sharpe"],
            attempts=st_m["attempts"], filled=st_m["filled"], missed=st_m["missed"],
            fill_rate=fill_rate,
        ))

    res = pd.DataFrame(rows)
    show = res.copy()
    for c in ("taker_ret", "maker_ret"):
        show[c] = show[c].round(1)
    for c in ("taker_sharpe", "maker_sharpe", "fill_rate"):
        show[c] = show[c].round(2)
    show = show.rename(columns={"taker_ret": "taker_%", "maker_ret": "maker_%",
                                "taker_sharpe": "tk_sharpe", "maker_sharpe": "mk_sharpe",
                                "fill_rate": "fill_%"})
    pd.options.display.float_format = lambda x: f"{x:,.2f}"
    print(show[["strategy", "kind", "taker_%", "maker_%", "tk_sharpe", "mk_sharpe",
                "attempts", "filled", "missed", "fill_%"]].to_string(index=False))

    print(f"\nBuy & hold over this period: {bh*100:+.1f}%")
    print("\nRead it like this:")
    print("  • taker_%  = pay the fee, but every trade fills (guaranteed execution)")
    print("  • maker_%  = zero fee, but only trades that price came back to filled")
    print("  • fill_%   = share of entry signals that actually got filled as a maker")
    print("  • 'missed' = entry signals the market ran away from — your hidden cost")


if __name__ == "__main__":
    main()
