#!/usr/bin/env python3
"""
server.py — Tiny local web viewer for the OHLCV CSVs in ../data/.

Renders candlestick + volume charts in your browser using TradingView's
open-source lightweight-charts (vendored locally, so it works offline).
No external services, no TradingView account.

Run:
    ./.venv/bin/python viewer/server.py
then open http://127.0.0.1:8000

Endpoints:
    /                       -> the chart page
    /api/files              -> JSON list of datasets found in ../data/
    /api/candles?file=..&bars=N   -> candle + volume JSON (last N bars)
"""

from __future__ import annotations

import json
import os
import re
import sys
from functools import lru_cache
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
STATIC_DIR = HERE / "static"
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))

# Reuse the exact indicator + signal code the backtests use, so overlaid
# samples match what the backtests in strategylab.backtest actually traded.
from strategylab.core import Indicators, supertrend_dir  # noqa: E402

# Cache raw CSV loads so repeated requests / bar-count changes are instant.
# Keyed by (path, mtime) so a re-fetched file is picked up automatically.
@lru_cache(maxsize=16)
def _load(path_str: str, mtime: float) -> pd.DataFrame:
    df = pd.read_csv(path_str)
    # seconds (UTC) is what lightweight-charts wants for intraday data
    df["time"] = (df["timestamp"] // 1000).astype("int64")
    return df


def load_df(path: Path) -> pd.DataFrame:
    return _load(str(path), path.stat().st_mtime)


def list_datasets() -> list[dict]:
    out = []
    if not DATA_DIR.exists():
        return out
    for p in sorted(DATA_DIR.glob("*.csv")):
        m = re.match(r"(.+?)_(.+?)_([0-9]+[smhdwM])\.csv$", p.name)
        exchange, symbol, tf = (m.groups() if m else ("", p.stem, ""))
        out.append({
            "file": p.name,
            "exchange": exchange,
            "symbol": symbol.replace("-", "/"),
            "timeframe": tf,
            "label": f"{symbol.replace('-', '/')}  ·  {tf}" + (f"  ({exchange})" if exchange else ""),
        })
    return out


def safe_data_path(filename: str) -> Path | None:
    """Resolve a requested filename strictly inside DATA_DIR (no traversal)."""
    if not filename:
        return None
    candidate = (DATA_DIR / filename).resolve()
    if candidate.parent != DATA_DIR.resolve() or not candidate.is_file():
        return None
    return candidate


def candles_payload(filename: str, bars: int) -> dict:
    path = safe_data_path(filename)
    if path is None:
        return {"error": f"file not found: {filename}"}
    df = load_df(path)
    total = len(df)
    if bars > 0:
        df = df.tail(bars)

    candles = [
        {"time": int(t), "open": float(o), "high": float(h), "low": float(l), "close": float(c)}
        for t, o, h, l, c in zip(df["time"], df["open"], df["high"], df["low"], df["close"])
    ]
    up = "rgba(38,166,154,0.5)"
    down = "rgba(239,83,80,0.5)"
    volume = [
        {"time": int(t), "value": float(v), "color": up if c >= o else down}
        for t, v, o, c in zip(df["time"], df["volume"], df["open"], df["close"])
    ]
    return {
        "file": filename,
        "returned": len(candles),
        "total": total,
        "candles": candles,
        "volume": volume,
    }


# ----------------------------------------------------------------------------
# Strategy signal overlays — the *trend* families from the maker backtest, so
# you can eyeball the medium-frequency entries/exits on the candles themselves.
# ----------------------------------------------------------------------------
def _shift_bool(a: np.ndarray) -> np.ndarray:
    out = np.zeros_like(a, dtype=bool)
    out[1:] = a[:-1]
    return out


def _cross_signals(regime: np.ndarray):
    """Entry on the bar the regime turns True, exit on the bar it turns False."""
    entry = regime & ~_shift_bool(regime)
    exit_ = (~regime) & _shift_bool(regime)
    return entry, exit_


# label shown in the UI; kept in sync with the overlay <select> in index.html
STRATEGIES = {
    "ema_cross": "EMA cross (12/26)",
    "sma_cross": "SMA cross (50/200)",
    "supertrend": "Supertrend (10, 3.0)",
}


def _build_signals(dfi: pd.DataFrame, strategy: str):
    """Return (entry, exit_, lines) where lines = [(name, color, array), ...]."""
    ind = Indicators(dfi)
    if strategy == "ema_cross":
        fast, slow = ind.ema(12), ind.ema(26)
        entry, exit_ = _cross_signals(fast > slow)
        return entry, exit_, [("EMA 12", "#f0b429", fast), ("EMA 26", "#3b82f6", slow)]
    if strategy == "sma_cross":
        fast, slow = ind.sma(50), ind.sma(200)
        entry, exit_ = _cross_signals(fast > slow)
        return entry, exit_, [("SMA 50", "#f0b429", fast), ("SMA 200", "#3b82f6", slow)]
    if strategy == "supertrend":
        entry, exit_ = _cross_signals(supertrend_dir(ind, 10, 3.0) > 0)
        return entry, exit_, []
    return None


def signals_payload(filename: str, bars: int, strategy: str) -> dict:
    path = safe_data_path(filename)
    if path is None:
        return {"error": f"file not found: {filename}"}
    if strategy not in STRATEGIES:
        return {"error": f"unknown strategy: {strategy}"}
    df = load_df(path)
    # Indicators expects capitalized OHLCV columns (same as the backtests).
    dfi = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                             "close": "Close", "volume": "Volume"})
    entry, exit_, lines = _build_signals(dfi, strategy)  # computed on full history

    times = df["time"].to_numpy()
    n = len(times)
    lo = n - bars if 0 < bars < n else 0  # only return the visible window

    line_out = []
    for name, color, arr in lines:
        data = [{"time": int(times[i]), "value": float(arr[i])}
                for i in range(lo, n) if arr[i] == arr[i]]  # drop NaN warmup
        line_out.append({"name": name, "color": color, "data": data})

    markers, entries, exits = [], 0, 0
    for i in range(lo, n):
        if entry[i]:
            markers.append({"time": int(times[i]), "position": "belowBar",
                            "color": "#26a69a", "shape": "arrowUp", "text": "B"})
            entries += 1
        elif exit_[i]:
            markers.append({"time": int(times[i]), "position": "aboveBar",
                            "color": "#ef5350", "shape": "arrowDown", "text": "S"})
            exits += 1
    markers.sort(key=lambda m: m["time"])  # lightweight-charts wants them ascending
    return {
        "file": filename, "strategy": strategy, "label": STRATEGIES[strategy],
        "lines": line_out, "markers": markers, "entries": entries, "exits": exits,
    }


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        route = parsed.path

        if route == "/" or route == "/index.html":
            self.path = "/index.html"
            return super().do_GET()

        if route == "/api/files":
            return self._send_json(list_datasets())

        if route == "/api/candles":
            q = parse_qs(parsed.query)
            filename = (q.get("file") or [""])[0]
            try:
                bars = int((q.get("bars") or ["5000"])[0])
            except ValueError:
                bars = 5000
            payload = candles_payload(filename, bars)
            status = 404 if "error" in payload else 200
            return self._send_json(payload, status)

        if route == "/api/signals":
            q = parse_qs(parsed.query)
            filename = (q.get("file") or [""])[0]
            strategy = (q.get("strategy") or ["ema_cross"])[0]
            try:
                bars = int((q.get("bars") or ["5000"])[0])
            except ValueError:
                bars = 5000
            payload = signals_payload(filename, bars, strategy)
            status = 404 if "error" in payload else 200
            return self._send_json(payload, status)

        # anything else -> static file from viewer/static
        return super().do_GET()

    def log_message(self, fmt, *args):  # quieter console
        return


def main():
    if not DATA_DIR.exists():
        print(f"! No data directory at {DATA_DIR}. Fetch some candles first.")
    print(f"Datasets found: {[d['file'] for d in list_datasets()] or 'none'}")
    print(f"\n  Candle viewer running at  http://{HOST}:{PORT}\n  (Ctrl+C to stop)\n")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
