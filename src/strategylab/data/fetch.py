#!/usr/bin/env python3
"""
fetch_ohlcv.py — Download historical crypto candles (OHLCV) to a local file.

Pulls candle data from any exchange supported by ccxt (Binance, Bybit, OKX,
Coinbase, Kraken, ...) using only free public endpoints — no API key, no
scraping, no TradingView. Handles pagination and rate limits automatically so
you can grab years of history in one command, then backtest fully offline.

Examples
--------
  # 1 year of BTC/USDT hourly candles from Binance -> data/binance_BTC-USDT_1h.csv
  python fetch_ohlcv.py --symbol BTC/USDT --timeframe 1h --since 2024-01-01

  # Several pairs at once, 15m candles, save as Parquet
  python fetch_ohlcv.py -s BTC/USDT ETH/USDT SOL/USDT -t 15m --since 2023-06-01 --format parquet

  # A different exchange and an explicit end date
  python fetch_ohlcv.py --exchange bybit -s BTC/USDT -t 4h --since 2022-01-01 --until 2023-01-01

List the timeframes an exchange supports:
  python fetch_ohlcv.py --exchange binance --list-timeframes
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import ccxt
    import pandas as pd
except ImportError:
    sys.exit(
        "Missing dependencies. Install them first:\n"
        "    pip install -r requirements.txt\n"
        "(or: pip install ccxt pandas pyarrow)"
    )

# One candle request is capped by each exchange (often 500-1500 rows). We ask
# for a big page and let ccxt clamp it to the exchange maximum.
PAGE_LIMIT = 1000


def parse_date_to_ms(date_str: str) -> int:
    """Parse 'YYYY-MM-DD' or a full ISO timestamp into epoch milliseconds (UTC)."""
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(
        f"Unrecognized date '{date_str}'. Use YYYY-MM-DD (e.g. 2024-01-01)."
    )


def make_exchange(name: str) -> "ccxt.Exchange":
    if name not in ccxt.exchanges:
        sys.exit(
            f"Unknown exchange '{name}'.\n"
            f"Some common ones: binance, bybit, okx, coinbase, kraken, kucoin.\n"
            f"Full list: python -c \"import ccxt; print(ccxt.exchanges)\""
        )
    exchange = getattr(ccxt, name)({"enableRateLimit": True})
    return exchange


def fetch_symbol(
    exchange: "ccxt.Exchange",
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int | None,
) -> "pd.DataFrame":
    """Page through the exchange's OHLCV endpoint until we reach `until` (or now)."""
    all_rows: list[list] = []
    cursor = since_ms
    tf_ms = exchange.parse_timeframe(timeframe) * 1000  # seconds -> ms
    end = until_ms if until_ms is not None else exchange.milliseconds()

    while cursor < end:
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=PAGE_LIMIT)
        except ccxt.RateLimitExceeded:
            time.sleep(2)
            continue
        except ccxt.NetworkError as e:
            print(f"    network hiccup ({e}); retrying in 3s...", file=sys.stderr)
            time.sleep(3)
            continue

        if not batch:
            break

        # Guard against exchanges that return the `since` candle repeatedly.
        batch = [c for c in batch if c[0] < end]
        if not batch:
            break

        all_rows.extend(batch)
        last_ts = batch[-1][0]

        # Advance past the last candle. If the exchange returned a single
        # already-seen candle, step forward one bar to avoid an infinite loop.
        cursor = last_ts + tf_ms if last_ts >= cursor else cursor + tf_ms

        got = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc)
        print(f"    {symbol} {timeframe}: {len(all_rows):>7} candles (through {got:%Y-%m-%d %H:%M})")

        # Partial page => we've caught up to the present.
        if len(batch) < PAGE_LIMIT:
            break

    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    if until_ms is not None:
        df = df[df["timestamp"] < until_ms]
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df[["timestamp", "datetime", "open", "high", "low", "close", "volume"]]


def save(df: "pd.DataFrame", out_dir: Path, exchange_name: str, symbol: str, timeframe: str, fmt: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_symbol = symbol.replace("/", "-").replace(":", "-")
    stem = f"{exchange_name}_{safe_symbol}_{timeframe}"
    if fmt == "parquet":
        path = out_dir / f"{stem}.parquet"
        df.to_parquet(path, index=False)
    else:
        path = out_dir / f"{stem}.csv"
        df.to_csv(path, index=False)
    return path


def main() -> None:
    p = argparse.ArgumentParser(
        description="Download historical crypto OHLCV candles to a local CSV/Parquet file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--exchange", "-e", default="binance", help="ccxt exchange id (default: binance)")
    p.add_argument("--symbol", "-s", nargs="+", metavar="PAIR",
                   help="One or more pairs, e.g. BTC/USDT ETH/USDT")
    p.add_argument("--timeframe", "-t", default="1h",
                   help="Candle size: 1m,5m,15m,1h,4h,1d,... (default: 1h)")
    p.add_argument("--since", type=parse_date_to_ms, metavar="YYYY-MM-DD",
                   help="Start date (UTC). Default: 1 year ago.")
    p.add_argument("--until", type=parse_date_to_ms, metavar="YYYY-MM-DD",
                   help="End date (UTC, exclusive). Default: now.")
    p.add_argument("--format", "-f", choices=["csv", "parquet"], default="csv",
                   help="Output format (default: csv)")
    p.add_argument("--out", "-o", default="data", help="Output directory (default: data/)")
    p.add_argument("--list-timeframes", action="store_true",
                   help="Print the timeframes this exchange supports and exit.")
    args = p.parse_args()

    exchange = make_exchange(args.exchange)

    if args.list_timeframes:
        tfs = getattr(exchange, "timeframes", None)
        if tfs:
            print(f"{args.exchange} timeframes: {', '.join(tfs)}")
        else:
            print(f"{args.exchange} does not advertise its timeframes via ccxt.")
        return

    if not args.symbol:
        p.error("at least one --symbol is required (e.g. -s BTC/USDT)")

    # Default window: last 365 days.
    since_ms = args.since if args.since is not None else exchange.milliseconds() - 365 * 24 * 60 * 60 * 1000

    print(f"Loading markets from {args.exchange}...")
    exchange.load_markets()

    for symbol in args.symbol:
        if symbol not in exchange.markets:
            print(f"  ! {symbol} not listed on {args.exchange}; skipping.", file=sys.stderr)
            continue
        print(f"  Fetching {symbol} [{args.timeframe}] ...")
        df = fetch_symbol(exchange, symbol, args.timeframe, since_ms, args.until)
        if df.empty:
            print(f"  ! No data returned for {symbol}.", file=sys.stderr)
            continue
        path = save(df, Path(args.out), args.exchange, symbol, args.timeframe, args.format)
        span = f"{df['datetime'].iloc[0]:%Y-%m-%d} -> {df['datetime'].iloc[-1]:%Y-%m-%d}"
        print(f"  ✓ {len(df):,} candles ({span})  ->  {path}")


if __name__ == "__main__":
    main()
