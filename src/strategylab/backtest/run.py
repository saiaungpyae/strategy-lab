#!/usr/bin/env python3
"""
run_backtest.py — Backtest classic community strategies on local OHLCV data.

Runs each strategy in bt_strategies.py against a CSV of candles, applies
realistic fees, and prints a comparison table sorted by return — always next
to a Buy & Hold benchmark so you can see whether the strategy actually beats
just holding the coin.

Examples
--------
  # All strategies on the 1h data (default)
  python run_backtest.py

  # A specific file and just two strategies
  python run_backtest.py --file data/ohlcv/BTC-USDT/binance_BTC-USDT_15m.csv -s supertrend ema_cross

  # Higher fees, and save an interactive chart of the best run
  python run_backtest.py --commission 0.0015 --plot supertrend
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd
from backtesting import Backtest

from strategylab.data.paths import default_candles
from strategylab.strategies.community import STRATEGIES

warnings.filterwarnings("ignore")  # backtesting emits noisy user-warnings on some stats

# Large cash so whole-unit rounding is negligible vs. equity (BTC can cost >$100k).
CASH = 100_000_000


def load_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime")
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                            "close": "Close", "volume": "Volume"})
    return df[["Open", "High", "Low", "Close", "Volume"]]


def run_one(df, strat_cls, commission):
    bt = Backtest(df, strat_cls, cash=CASH, commission=commission)
    return bt.run()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", "-f", default=default_candles("1h"), help="OHLCV CSV to test on")
    p.add_argument("--strategy", "-s", nargs="+", default=["all"],
                   help=f"Which strategies to run. Options: {', '.join(STRATEGIES)} (default: all)")
    p.add_argument("--commission", "-c", type=float, default=0.001,
                   help="Per-trade commission as a fraction (default 0.001 = 0.10%%)")
    p.add_argument("--plot", metavar="STRATEGY", default=None,
                   help="Save an interactive HTML chart for this one strategy")
    args = p.parse_args()

    path = Path(args.file)
    if not path.is_file():
        sys.exit(f"File not found: {path}. Fetch data first with fetch_ohlcv.py.")

    names = list(STRATEGIES) if args.strategy == ["all"] else args.strategy
    for n in names:
        if n not in STRATEGIES:
            sys.exit(f"Unknown strategy '{n}'. Options: {', '.join(STRATEGIES)}")

    df = load_ohlcv(path)
    print(f"\nData: {path.name}  |  {len(df):,} bars  |  "
          f"{df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d}  |  fee {args.commission*100:.3f}%/trade\n")

    rows = []
    bh_return = None
    for name in names:
        try:
            stats = run_one(df, STRATEGIES[name], args.commission)
        except Exception as e:
            print(f"  ! {name} failed: {e}", file=sys.stderr)
            continue
        bh_return = stats["Buy & Hold Return [%]"]
        rows.append({
            "strategy": name,
            "return_%": stats["Return [%]"],
            "ann_%": stats.get("Return (Ann.) [%]", float("nan")),
            "sharpe": stats["Sharpe Ratio"],
            "maxDD_%": stats["Max. Drawdown [%]"],
            "winrate_%": stats["Win Rate [%]"],
            "trades": int(stats["# Trades"]),
            "exposure_%": stats["Exposure Time [%]"],
        })

    if not rows:
        sys.exit("No results.")

    table = pd.DataFrame(rows).sort_values("return_%", ascending=False).reset_index(drop=True)
    # "beat" = did the strategy's total return exceed buy & hold?
    table["vs_B&H"] = table["return_%"].apply(lambda r: "beat" if r > bh_return else "—")

    pd.options.display.float_format = lambda x: f"{x:,.2f}"
    print(table.to_string(index=False))
    print(f"\nBuy & Hold return over this period: {bh_return:,.2f}%")
    beat = table[table["return_%"] > bh_return]["strategy"].tolist()
    if beat:
        print(f"Beat Buy & Hold: {', '.join(beat)}")
    else:
        print("Beat Buy & Hold: none — holding BTC outperformed every strategy here.")

    if args.plot:
        if args.plot not in STRATEGIES:
            print(f"\n! Can't plot unknown strategy '{args.plot}'", file=sys.stderr)
        else:
            bt = Backtest(df, STRATEGIES[args.plot], cash=CASH, commission=args.commission)
            bt.run()
            out = Path("reports") / f"{path.stem}_{args.plot}.html"
            out.parent.mkdir(exist_ok=True)
            bt.plot(filename=str(out), open_browser=False)
            print(f"\nSaved chart -> {out}")


if __name__ == "__main__":
    main()
