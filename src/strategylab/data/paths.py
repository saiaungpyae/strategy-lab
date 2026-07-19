"""Canonical data-directory layout and back-compat path resolution.

The data root scales by pair: each pair gets one folder for its candle tapes
and one for its derivatives metrics, while file NAMES keep the flat
self-describing convention (``binance_BTC-USDT_5m.csv``) that the symbol
parsers, pinned snapshots and old run artifacts rely on:

    data/
      ohlcv/BTC-USDT/binance_BTC-USDT_5m.csv       (+ .parquet sidecar caches)
      metrics/BTC-USDT/BTCUSDT_metrics.csv
      metrics/BTC-USDT/BTC-USDT-USDT_funding.csv
      snapshots/pin-YYYYMMDD/...                   (frozen copies, any layout)

``locate()`` maps a stored legacy path (frozen track configs, evolution.json
provenance, chart bookmarks) to wherever the file actually lives today, so
artifacts written before the layout change keep replaying.
"""

from __future__ import annotations

import re
from pathlib import Path

DATA_ROOT = Path("data")

CANDLE_RE = re.compile(
    r"^(?P<exchange>.+?)_(?P<pair>.+?)_(?P<tf>[0-9]+[smhdwM])\.(?P<ext>csv|parquet)$")
_METRICS_RE = re.compile(r"^(?P<base>[A-Z0-9]+)USDT_metrics\.csv$")
_FUNDING_RE = re.compile(r"^(?P<base>[A-Z0-9]+)-USDT-USDT_funding\.csv$")


def pair_dir(symbol: str) -> str:
    """Per-pair folder name from any symbol spelling:
    'BTC', 'BTCUSDT', 'BTC-USDT', 'BTC/USDT', 'BTC/USDT:USDT' -> 'BTC-USDT'."""
    base = re.split(r"[/:]", symbol.upper())[0]
    for suffix in ("-USDT", "USDT"):
        if base.endswith(suffix) and len(base) > len(suffix):
            base = base[: -len(suffix)]
            break
    return f"{base}-USDT"


def candles(pair: str, tf: str, exchange: str = "binance",
            root: Path | str = DATA_ROOT) -> Path:
    """Canonical candle path: <root>/ohlcv/<PAIR>/<exchange>_<PAIR>_<tf>.csv"""
    pair = pair_dir(pair)
    return Path(root) / "ohlcv" / pair / f"{exchange}_{pair}_{tf}.csv"


def metrics(pair: str, root: Path | str = DATA_ROOT) -> Path:
    """Canonical Binance Vision metrics path: <root>/metrics/<PAIR>/<SYM>USDT_metrics.csv"""
    pair = pair_dir(pair)
    return Path(root) / "metrics" / pair / f"{pair.split('-')[0]}USDT_metrics.csv"


def funding(pair: str, root: Path | str = DATA_ROOT) -> Path:
    """Canonical perp funding path: <root>/metrics/<PAIR>/<PAIR>-USDT_funding.csv"""
    pair = pair_dir(pair)
    return Path(root) / "metrics" / pair / f"{pair}-USDT_funding.csv"


def _default(canonical: Path, legacy: Path) -> str:
    """Canonical path unless only the legacy location exists (pre-migration)."""
    return str(canonical if canonical.exists() or not legacy.exists() else legacy)


def default_candles(tf: str, pair: str = "BTC-USDT", exchange: str = "binance") -> str:
    """Argparse default for a candle file, tolerant of an unmigrated data dir."""
    canon = candles(pair, tf, exchange)
    return _default(canon, DATA_ROOT / canon.name)


def default_metrics(pair: str = "BTC-USDT") -> str:
    canon = metrics(pair)
    return _default(canon, DATA_ROOT / "metrics" / canon.name)


def default_funding(pair: str = "BTC-USDT") -> str:
    canon = funding(pair)
    return _default(canon, DATA_ROOT / "metrics" / canon.name)


def locate(path: str | Path) -> Path:
    """Resolve `path` to where the file actually lives.

    Stored paths outlive layout changes. When `path` is missing, try its
    file name at the other known location — per-pair canonical dir vs legacy
    flat dir — under the same data root. Returns `path` untouched when it
    exists or no candidate is found (callers keep their own error handling).
    """
    p = Path(path)
    if p.exists():
        return p
    name, parent = p.name, p.parent
    cands: list[Path] = []
    m = CANDLE_RE.match(name)
    if m:
        pair = m["pair"]
        if parent.name == pair and parent.parent.name == "ohlcv":
            cands.append(parent.parent.parent / name)       # canonical -> legacy flat
        else:
            cands.append(parent / "ohlcv" / pair / name)    # legacy flat -> canonical
    else:
        mm = _METRICS_RE.match(name) or _FUNDING_RE.match(name)
        if mm:
            pair = f"{mm['base']}-USDT"
            if parent.name == pair:                          # canonical -> legacy
                cands.append(parent.parent / name)
            elif parent.name == "metrics":                   # legacy metrics/ -> canonical
                cands.append(parent / pair / name)
            else:                                            # bare root (snapshot-style)
                cands += [parent / "metrics" / pair / name,
                          parent / "metrics" / name]
    return next((c for c in cands if c.exists()), p)
