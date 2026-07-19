#!/usr/bin/env python3
"""
fetch_ohlcv.py — Download historical crypto candles (OHLCV) to a local file.

Pulls candle data from any exchange supported by ccxt (Binance, Bybit, OKX,
Coinbase, Kraken, ...) using only free public endpoints — no API key, no
scraping, no TradingView. Handles pagination and rate limits automatically so
you can grab years of history in one command, then backtest fully offline.

On Binance-family exchanges the saved files also carry two extra kline
columns — `taker_buy_volume` (taker buy base volume) and `n_trades` — which
unlock the order-flow features in the swarm feature pool.

Examples
--------
  # 1 year of BTC/USDT hourly candles from Binance -> data/ohlcv/BTC-USDT/binance_BTC-USDT_1h.csv
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
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import paths

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

# Binance's raw kline endpoints return fields that ccxt's parsed OHLCV drops:
# number of trades (index 8) and taker buy base volume (index 9). Exchanges
# with such an endpoint get these saved as extra columns; others fall back to
# plain OHLCV.
RAW_KLINE_METHODS = {
    "binance": "public_get_klines",
    "binanceus": "public_get_klines",
    "binanceusdm": "fapipublic_get_klines",
    "binancecoinm": "dapipublic_get_klines",
}
EXTRA_COLS = ["taker_buy_volume", "n_trades"]


def _raw_kline_fetcher(exchange: "ccxt.Exchange", symbol: str, timeframe: str):
    """Return a callable(since_ms, limit) yielding OHLCV rows with taker buy
    volume and trade count appended, or None if the exchange has no raw
    Binance-style kline endpoint."""
    method = getattr(exchange, RAW_KLINE_METHODS.get(exchange.id, ""), None)
    if method is None:
        return None
    exchange.load_markets()  # cached after the first call
    market_id = exchange.market(symbol)["id"]
    interval = exchange.timeframes.get(timeframe, timeframe)

    def fetch(since_ms: int, limit: int) -> list[list]:
        raw = method({"symbol": market_id, "interval": interval,
                      "startTime": since_ms, "limit": limit})
        # kline: [openTime, open, high, low, close, volume, closeTime,
        #         quoteVolume, nTrades, takerBuyBase, takerBuyQuote, ...]
        return [[int(k[0]), float(k[1]), float(k[2]), float(k[3]),
                 float(k[4]), float(k[5]), float(k[9]), int(k[8])]
                for k in raw]

    return fetch


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
    """Page through the exchange's OHLCV endpoint until we reach `until` (or now).

    On Binance-family exchanges the returned frame also carries the extra
    kline columns `taker_buy_volume` and `n_trades` (see EXTRA_COLS).
    """
    raw_fetch = _raw_kline_fetcher(exchange, symbol, timeframe)
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    if raw_fetch is not None:
        cols += EXTRA_COLS

    all_rows: list[list] = []
    cursor = since_ms
    tf_ms = exchange.parse_timeframe(timeframe) * 1000  # seconds -> ms
    end = until_ms if until_ms is not None else exchange.milliseconds()

    while cursor < end:
        try:
            if raw_fetch is not None:
                batch = raw_fetch(cursor, PAGE_LIMIT)
            else:
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
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(all_rows, columns=cols)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    if until_ms is not None:
        df = df[df["timestamp"] < until_ms]
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df[["timestamp", "datetime"] + cols[1:]]


def save(df: "pd.DataFrame", out_dir: Path, exchange_name: str, symbol: str, timeframe: str, fmt: str) -> Path:
    safe_symbol = symbol.replace("/", "-").replace(":", "-")
    path = paths.candles(safe_symbol, timeframe, exchange_name,
                         root=out_dir).with_suffix(f".{fmt}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    return path


# ----------------------------------------------------------------------------
# Incremental updates — bring an existing dataset file up to the present by
# fetching only the candles after the last one on disk. The last saved candle
# is re-fetched too (it may have been captured mid-bar), and the still-forming
# current candle is dropped so every stored row is final.
# ----------------------------------------------------------------------------
DATASET_RE = re.compile(r"^(?P<exchange>.+?)_(?P<symbol>.+?)_(?P<tf>[0-9]+[smhdwM])\.(?P<ext>csv|parquet)$")


def parse_dataset_filename(path: Path) -> dict | None:
    """binance_BTC-USDT_15m.csv -> {exchange, symbol, timeframe, format}."""
    m = DATASET_RE.match(path.name)
    if not m:
        return None
    return {
        "exchange": m["exchange"],
        "symbol": m["symbol"].replace("-", "/"),
        "timeframe": m["tf"],
        "format": m["ext"],
    }


def read_dataset(path: Path) -> "pd.DataFrame":
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def update_file(path: Path, exchange: "ccxt.Exchange | None" = None) -> dict:
    """Extend `path` with candles from its last saved timestamp through now.

    Returns {"file", "added", "rows", "last"} on success or {"file", "error"}.
    """
    info = parse_dataset_filename(path)
    if info is None:
        return {"file": path.name, "error": "unrecognized filename pattern"}

    try:
        old = read_dataset(path)
        if old.empty:
            return {"file": path.name, "error": "existing file is empty"}
        last_ts = int(old["timestamp"].iloc[-1])

        if exchange is None:
            exchange = make_exchange(info["exchange"])
        tf_ms = exchange.parse_timeframe(info["timeframe"]) * 1000

        # Re-fetch from the last saved candle so a partial capture gets finalized.
        new = fetch_symbol(exchange, info["symbol"], info["timeframe"], last_ts, None)
        # Drop the still-forming current candle: keep only fully closed bars.
        now_ms = exchange.milliseconds()
        new = new[new["timestamp"] + tf_ms <= now_ms]

        if new.empty:
            return {"file": path.name, "added": 0, "rows": len(old),
                    "last": str(old["datetime"].iloc[-1])}

        # Preserve the on-disk schema: files written before the extra kline
        # columns (EXTRA_COLS) existed stay plain OHLCV — a mostly-NaN
        # order-flow column would silently enable order-flow features
        # downstream. A full re-fetch upgrades them.
        new = new.drop(columns=[c for c in EXTRA_COLS if c not in old.columns],
                       errors="ignore")

        merged = pd.concat([old, new], ignore_index=True)
        # keep="last" so the re-fetched (final) version of the last candle wins
        merged = merged.drop_duplicates(subset="timestamp", keep="last")
        merged = merged.sort_values("timestamp").reset_index(drop=True)

        if info["format"] == "parquet":
            merged.to_parquet(path, index=False)
        else:
            merged.to_csv(path, index=False)
        return {"file": path.name, "added": len(merged) - len(old), "rows": len(merged),
                "last": str(merged["datetime"].iloc[-1])}
    except Exception as e:  # network down, exchange error, bad file — report, don't raise
        return {"file": path.name, "error": f"{type(e).__name__}: {e}"}


def update_all(data_dir: Path) -> list[dict]:
    """Incrementally update every recognized dataset file under `data_dir`,
    recursing into the per-pair ohlcv/ tree. Pinned snapshots/ stay frozen —
    their whole point is to never change under a running comparison."""
    targets = sorted(p for p in data_dir.rglob("*")
                     if p.is_file() and parse_dataset_filename(p)
                     and "snapshots" not in p.relative_to(data_dir).parts)
    # A .parquet sharing a stem with a .csv is the swarm's transparent sidecar
    # cache (swarm/run._load), not a primary dataset. Its `datetime` column is
    # raw csv strings, so concatenating fetched datetime64 rows onto it yields a
    # mixed object column that fails to serialize. Skip it: updating the csv
    # bumps its mtime, so the swarm regenerates the sidecar on next load.
    csv_stems = {p.with_suffix("") for p in targets if p.suffix == ".csv"}
    targets = [p for p in targets
               if not (p.suffix == ".parquet" and p.with_suffix("") in csv_stems)]
    exchanges: dict[str, "ccxt.Exchange"] = {}  # one instance per exchange id
    results = []
    for p in targets:
        name = parse_dataset_filename(p)["exchange"]
        try:
            ex = exchanges.get(name) or exchanges.setdefault(name, make_exchange(name))
        except SystemExit:  # make_exchange sys.exits on unknown ids
            results.append({"file": p.name, "error": f"unknown exchange '{name}'"})
            continue
        results.append(update_file(p, ex))
    return results


def main_update() -> None:
    p = argparse.ArgumentParser(
        description="Incrementally update existing dataset files (fetch only missing candles).",
    )
    p.add_argument("--data", "-d", default="data", help="Data directory (default: data/)")
    args = p.parse_args()

    data_dir = Path(args.data)
    if not data_dir.is_dir():
        sys.exit(f"No data directory at {data_dir}")

    for r in update_all(data_dir):
        if "error" in r:
            print(f"  ! {r['file']}: {r['error']}", file=sys.stderr)
        else:
            print(f"  ✓ {r['file']}: +{r['added']:,} candles ({r['rows']:,} total, through {r['last']})")


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
    p.add_argument("--out", "-o", default="data",
                   help="Data root (default: data/); files land under <out>/ohlcv/<PAIR>/")
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
