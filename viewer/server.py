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
import mimetypes
import os
import re
import sys
import threading
import time
from functools import lru_cache
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
STATIC_DIR = HERE / "static"
REPORTS_DIR = HERE.parent / "reports"


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines from a .env file into os.environ (shell env wins)."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv(HERE.parent / ".env")
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))

# Reuse the exact indicator + signal code the backtests use, so overlaid
# samples match what the backtests in strategylab.backtest actually traded.
from strategylab.core import Indicators, supertrend_dir  # noqa: E402
from strategylab.backtest.fvg import FVGParams, run_fvg_study  # noqa: E402
from strategylab.data.fetch import update_all  # noqa: E402


# ----------------------------------------------------------------------------
# Background data refresh — incremental update of every dataset in data/.
# Runs once on startup and on demand via POST /api/refresh. The server keeps
# serving whatever is on disk while a refresh is in flight; the mtime-keyed
# caches below pick up rewritten files automatically.
# ----------------------------------------------------------------------------
REFRESH = {"state": "idle", "started": None, "finished": None, "results": []}
_refresh_lock = threading.Lock()


def _refresh_worker() -> None:
    try:
        results = update_all(DATA_DIR)
    except Exception as e:  # never let a refresh failure kill the thread noisily
        results = [{"file": "*", "error": f"{type(e).__name__}: {e}"}]
    with _refresh_lock:
        REFRESH.update(state="done", finished=time.time(), results=results)
    added = sum(r.get("added", 0) for r in results)
    errors = [r for r in results if "error" in r]
    suffix = f", {len(errors)} error(s)" if errors else ""
    print(f"  refresh done: +{added:,} candles across {len(results)} file(s){suffix}")
    for r in errors:
        print(f"    ! {r['file']}: {r['error']}", file=sys.stderr)


def start_refresh() -> bool:
    """Kick off a background refresh; returns False if one is already running."""
    with _refresh_lock:
        if REFRESH["state"] == "running":
            return False
        REFRESH.update(state="running", started=time.time(), finished=None)
    threading.Thread(target=_refresh_worker, daemon=True).start()
    return True


def refresh_status() -> dict:
    with _refresh_lock:
        return dict(REFRESH)

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


# ----------------------------------------------------------------------------
# US equity session shading — 09:30–16:00 America/New_York, weekdays.
# zoneinfo handles the EST/EDT daylight-saving switch, so the UTC candle
# timestamps land on the right wall-clock hours year-round.
# ----------------------------------------------------------------------------
NY_TZ = ZoneInfo("America/New_York")
SESSION_RTH_COLOR = "rgba(59,130,246,0.14)"    # regular hours 09:30–16:00 ET
SESSION_OPEN_COLOR = "rgba(240,180,41,0.22)"   # opening hour 09:30–10:30 ET


def us_session_bars(times: pd.Series) -> list[dict]:
    """Histogram items (full-height background bands) for bars inside US RTH."""
    et = pd.to_datetime(times, unit="s", utc=True).dt.tz_convert(NY_TZ)
    minutes = et.dt.hour * 60 + et.dt.minute
    weekday = et.dt.weekday < 5           # NYSE holidays are not modeled
    rth = weekday & (minutes >= 9 * 60 + 30) & (minutes < 16 * 60)
    open_hour = rth & (minutes < 10 * 60 + 30)
    return [
        {"time": int(t), "value": 1,
         "color": SESSION_OPEN_COLOR if o else SESSION_RTH_COLOR}
        for t, r, o in zip(times, rth, open_hour) if r
    ]


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
        "session": us_session_bars(df["time"]),
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


# ----------------------------------------------------------------------------
# FVG event study — zones + per-trade outcomes + summary vs the random control.
# Cached per (file, mtime, rr) because the study walks every gap in Python.
# ----------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _fvg_study(path_str: str, mtime: float, rr: float):
    df = pd.read_csv(path_str)
    return run_fvg_study(df, FVGParams(rr=rr))


def fvg_payload(filename: str, bars: int, rr: float) -> dict:
    path = safe_data_path(filename)
    if path is None:
        return {"error": f"file not found: {filename}"}
    events, summary = _fvg_study(str(path), path.stat().st_mtime, rr)

    df = load_df(path)
    times = df["time"].to_numpy()
    n = len(times)
    lo = n - bars if 0 < bars < n else 0

    zones, markers = [], []
    for ev in events:
        # include any event still active or resolving inside the visible window
        end_idx = ev.resolve_idx if ev.resolve_idx is not None else min(ev.form_idx + 50, n - 1)
        if end_idx < lo:
            continue
        zones.append({
            "from": int(times[ev.form_idx]),
            "to": int(times[end_idx]),
            "top": ev.top,
            "bottom": ev.bottom,
            "dir": ev.direction,
            "outcome": ev.outcome,
        })
        if ev.touch_idx is not None and ev.touch_idx >= lo:
            tag = {"win": "W", "loss": "L", "timeout": "T", "open": "?"}[ev.outcome]
            color = {"win": "#26a69a", "loss": "#ef5350",
                     "timeout": "#f0b429", "open": "#8b949e"}[ev.outcome]
            markers.append({
                "time": int(times[ev.touch_idx]),
                "position": "belowBar" if ev.direction > 0 else "aboveBar",
                "color": color,
                "shape": "arrowUp" if ev.direction > 0 else "arrowDown",
                "text": tag,
            })
    markers.sort(key=lambda m: m["time"])
    return {"file": filename, "zones": zones, "markers": markers, "summary": summary}


# ----------------------------------------------------------------------------
# Dashboard payloads — data health, latest signal states, reports listing.
# ----------------------------------------------------------------------------
_TF_UNIT_S = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800, "M": 2592000}


def tf_seconds(tf: str) -> int:
    m = re.match(r"^([0-9]+)([smhdwM])$", tf)
    return int(m.group(1)) * _TF_UNIT_S[m.group(2)] if m else 0


def health_payload() -> dict:
    now = time.time()
    items = []
    for d in list_datasets():
        path = DATA_DIR / d["file"]
        df = load_df(path)
        step = tf_seconds(d["timeframe"])
        ts = df["time"]
        gaps = int((ts.diff().dropna() > step).sum()) if step else 0
        last = int(ts.iloc[-1])
        age = now - (last + step)  # measured from when the last candle *closed*
        items.append({
            **d,
            "rows": len(df),
            "first": int(ts.iloc[0]),
            "last": last,
            "age_seconds": max(0, int(age)),
            "bars_behind": int(age // step) if step else None,
            "gaps": gaps,
            "size_bytes": path.stat().st_size,
        })
    return {"datasets": items, "refresh": refresh_status()}


def _regime_state(regime: np.ndarray) -> dict:
    """Latest direction and how many bars ago it flipped."""
    n = len(regime)
    state = bool(regime[-1])
    flip = 0
    for i in range(n - 1, 0, -1):
        if regime[i] != regime[i - 1]:
            flip = n - i
            break
    else:
        flip = n  # never flipped inside the window
    return {"state": "long" if state else "flat", "bars_since_flip": flip}


@lru_cache(maxsize=16)
def _snapshot_one(path_str: str, mtime: float) -> dict:
    # Tail window: enough for SMA200 warmup + a meaningful flip lookback.
    df = pd.read_csv(path_str).tail(1500).reset_index(drop=True)
    df["time"] = (df["timestamp"] // 1000).astype("int64")
    dfi = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                             "close": "Close", "volume": "Volume"})
    ind = Indicators(dfi)
    close = df["close"].to_numpy()

    out = {}
    for key, regime in (
        ("ema_cross", ind.ema(12) > ind.ema(26)),
        ("sma_cross", ind.sma(50) > ind.sma(200)),
        ("supertrend", supertrend_dir(ind, 10, 3.0) > 0),
    ):
        out[key] = _regime_state(np.asarray(regime))

    # FVG: count zones formed in the tail window that are still unresolved.
    events, _ = run_fvg_study(df, FVGParams())
    out["fvg_open"] = sum(1 for ev in events if ev.outcome == "open")

    return {
        "last_close": float(close[-1]),
        "last_time": int(df["time"].iloc[-1]),
        "signals": out,
    }


def snapshot_payload() -> dict:
    items = []
    for d in list_datasets():
        path = DATA_DIR / d["file"]
        try:
            snap = _snapshot_one(str(path), path.stat().st_mtime)
        except Exception as e:
            items.append({**d, "error": f"{type(e).__name__}: {e}"})
            continue
        # 24h % change from the candle ~24h before the last one
        step = tf_seconds(d["timeframe"])
        df = load_df(path)
        back = int(86400 // step) if step else 0
        change = None
        if back and len(df) > back:
            prev = float(df["close"].iloc[-1 - back])
            change = (snap["last_close"] - prev) / prev * 100
        items.append({**d, **snap, "change_24h_pct": change})
    return {"datasets": items}


def reports_payload() -> dict:
    items = []
    if REPORTS_DIR.is_dir():
        for p in sorted(REPORTS_DIR.iterdir()):
            if p.name.startswith(".") or not p.is_file():
                continue
            items.append({
                "file": p.name,
                "kind": p.suffix.lstrip("."),
                "size_bytes": p.stat().st_size,
                "mtime": int(p.stat().st_mtime),
            })
    return {"reports": items}


_MD_CSS = """
:root{--bg:#0e1117;--panel:#161b22;--border:#2a2f38;--text:#e6edf3;--muted:#8b949e;
--accent:#26a69a;--warn:#f0b429}
body{margin:0;background:var(--bg);color:var(--text);
font:14px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:920px;margin:0 auto;padding:28px 20px 80px}
h1,h2,h3{line-height:1.3}h1{font-size:24px}h2{font-size:18px;margin-top:32px;
border-bottom:1px solid var(--border);padding-bottom:6px}h3{font-size:15px}
a{color:var(--accent);text-decoration:none}
code{background:#21262d;border:1px solid var(--border);border-radius:4px;
padding:1px 5px;font-size:12.5px}
pre{background:#21262d;border:1px solid var(--border);border-radius:8px;
padding:12px;overflow-x:auto}pre code{border:none;background:none;padding:0}
table{border-collapse:collapse;margin:14px 0;font-size:13px;display:block;
overflow-x:auto;max-width:100%}
th,td{border:1px solid var(--border);padding:6px 10px;text-align:right;
white-space:nowrap}th:first-child,td:first-child{text-align:left}
th{background:var(--panel);color:var(--muted)}
blockquote{border-left:3px solid var(--warn);margin:14px 0;padding:4px 14px;
color:var(--muted);background:rgba(240,180,41,.06)}
hr{border:none;border-top:1px solid var(--border);margin:24px 0}
.top{padding:10px 20px;background:var(--panel);border-bottom:1px solid var(--border)}
"""


def _md_inline(s: str) -> str:
    import html as _h
    s = _h.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"\[([^\]]+)\]\((https?://[^)]+|/[^)]*|[\w./-]+)\)", r'<a href="\2">\1</a>', s)
    return s


def render_markdown(text: str, title: str) -> bytes:
    """Small dependency-free renderer for the subset of markdown the lab's
    reports use: headers, tables, lists, quotes, fences, bold, code, links."""
    out, para, in_code, table = [], [], False, []

    def flush_para():
        if para:
            out.append("<p>" + _md_inline(" ".join(para)) + "</p>")
            para.clear()

    def flush_table():
        nonlocal table
        if not table:
            return
        head, *body = table
        if body and set(body[0].replace("|", "").strip()) <= set("-: "):
            body = body[1:]
        cells = lambda row: [c.strip() for c in row.strip().strip("|").split("|")]
        html = ["<table><thead><tr>"]
        html += [f"<th>{_md_inline(c)}</th>" for c in cells(head)]
        html.append("</tr></thead><tbody>")
        for row in body:
            html.append("<tr>" + "".join(f"<td>{_md_inline(c)}</td>"
                                         for c in cells(row)) + "</tr>")
        html.append("</tbody></table>")
        out.append("".join(html))
        table = []

    list_tag = None
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            flush_para(); flush_table()
            out.append("<pre><code>" if not in_code else "</code></pre>")
            in_code = not in_code
            continue
        if in_code:
            import html as _h
            out.append(_h.escape(line))
            continue
        if line.lstrip().startswith("|"):
            flush_para()
            table.append(line)
            continue
        flush_table()
        stripped = line.strip()
        m = re.match(r"^(#{1,4})\s+(.*)", stripped)
        is_li = re.match(r"^(-|\d+\.)\s+(.*)", stripped)
        if list_tag and not is_li:
            out.append(f"</{list_tag}>"); list_tag = None
        if not stripped:
            flush_para()
        elif m:
            flush_para()
            out.append(f"<h{len(m.group(1))}>{_md_inline(m.group(2))}</h{len(m.group(1))}>")
        elif stripped.startswith(">"):
            flush_para()
            out.append(f"<blockquote>{_md_inline(stripped.lstrip('> '))}</blockquote>")
        elif re.fullmatch(r"-{3,}", stripped):
            flush_para(); out.append("<hr>")
        elif is_li:
            flush_para()
            tag = "ul" if is_li.group(1) == "-" else "ol"
            if list_tag != tag:
                if list_tag:
                    out.append(f"</{list_tag}>")
                out.append(f"<{tag}>"); list_tag = tag
            out.append(f"<li>{_md_inline(is_li.group(2))}</li>")
        else:
            para.append(stripped)
    flush_para(); flush_table()
    if list_tag:
        out.append(f"</{list_tag}>")
    page = (f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{title}</title><style>{_MD_CSS}</style></head><body>"
            f"<div class='top'><a href='/'>← dashboard</a> · <a href='/swarm'>bot swarm</a>"
            f" · <a href='?raw=1'>raw</a></div><main>" + "\n".join(out) +
            "</main></body></html>")
    return page.encode()


def safe_report_path(filename: str) -> Path | None:
    if not filename:
        return None
    candidate = (REPORTS_DIR / filename).resolve()
    if candidate.parent != REPORTS_DIR.resolve() or not candidate.is_file():
        return None
    return candidate


# ---------------------------------------------------------------------------
# Bot swarm (strategylab.swarm) — the server only reads run artifacts from
# reports/swarm/<run_id>/ and can spawn `sl-swarm run` as a separate process.
# It never simulates anything in-request.

SWARM_DIR = REPORTS_DIR / "swarm"
_swarm_proc = None
_swarm_lock = threading.Lock()


def safe_swarm_dir(run_id: str) -> Path | None:
    if not run_id or not re.fullmatch(r"[A-Za-z0-9_\-]+", run_id):
        return None
    d = (SWARM_DIR / run_id).resolve()
    if d.parent != SWARM_DIR.resolve() or not d.is_dir():
        return None
    return d


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def swarm_runs_payload() -> dict:
    runs = []
    if SWARM_DIR.is_dir():
        for d in sorted(SWARM_DIR.iterdir(), reverse=True):
            if not d.is_dir() or not (d / "config.json").is_file():
                continue  # skip evolution artifact dirs etc.
            cfg = _read_json(d / "config.json") or {}
            prog = _read_json(d / "progress.json") or {}
            runs.append({
                "run_id": d.name,
                "bots": cfg.get("bots"),
                "span": cfg.get("span"),
                "split_date": cfg.get("split_date"),
                "created": cfg.get("created"),
                "stage": prog.get("stage"),
                "frac": prog.get("frac"),
                "has_recap": (d / "recap.json").is_file(),
            })
    running = _swarm_proc is not None and _swarm_proc.poll() is None
    return {"runs": runs, "running": running}


@lru_cache(maxsize=4)
def _swarm_tables(run_dir: str, mtime: float):
    import numpy as _np
    import pandas as _pd
    d = Path(run_dir)
    z = _np.load(d / "daily_equity.npz", allow_pickle=False)
    return (_pd.read_csv(d / "genomes.csv"), _pd.read_csv(d / "results.csv"),
            z["daily"], [str(x) for x in z["days"]], int(z["split_day"]))


def swarm_evos_payload() -> dict:
    """Evolution runs: evolution.json + top hall-of-fame rows per run."""
    evos = []
    if SWARM_DIR.is_dir():
        for d in sorted(SWARM_DIR.iterdir(), reverse=True):
            if not d.is_dir() or not (d / "evolution.json").is_file():
                continue
            e = _read_json(d / "evolution.json") or {}
            hof_csv = d / "hof_test.csv"
            if hof_csv.is_file():
                import csv
                with hof_csv.open() as fh:
                    e["hof_top"] = list(csv.DictReader(fh))[:10]
            else:
                e["hof_top"] = []
            evos.append(e)
    return {"evos": evos}


def swarm_bots_payload(run_id: str) -> dict:
    """Compact per-bot table for the Bots tab — every bot, final $ included."""
    d = safe_swarm_dir(run_id)
    if d is None:
        return {"error": "run not found"}
    gdf, res, daily, days, split_day = _swarm_tables(
        str(d), (d / "results.csv").stat().st_mtime)
    cfg = _read_json(d / "config.json") or {}
    start_cap = float(cfg.get("start_capital", 10_000.0))
    import numpy as _np
    out = {
        "bot_id": res["bot_id"].tolist(),
        "control": res["is_control"].astype(bool).tolist(),
        "tf": gdf["tf"].tolist(),
        "session": gdf["session"].tolist(),
        "rules": gdf["rules"].tolist(),
        "final_usd": _np.round(res["final_mult"].to_numpy(float) * start_cap, 2).tolist(),
        "ret_a": _np.round(res["ret_a"].to_numpy(float) * 100, 2).tolist(),
        "ret_b": _np.round(res["ret_b"].to_numpy(float) * 100, 2).tolist(),
        "sharpe_b": [None if v != v else round(v, 3)
                     for v in res["sharpe_b"].to_numpy(float)],
        "trades": (res["trades_a"] + res["trades_b"]).tolist(),
        "dead": res["dead"].astype(bool).tolist(),
    }
    return {"start_capital": start_cap, "n": len(gdf), "bots": out}


def swarm_bot_payload(run_id: str, bot_id: int) -> dict:
    d = safe_swarm_dir(run_id)
    if d is None:
        return {"error": "run not found"}
    gdf, res, daily, days, split_day = _swarm_tables(
        str(d), (d / "results.csv").stat().st_mtime)
    if not (0 <= bot_id < len(gdf)):
        return {"error": "bot not found"}
    row = daily[bot_id].astype(float)
    # per-calendar-year performance from the full-resolution daily equity
    yearly = []
    yr = [str(x)[:4] for x in days]
    i = 0
    while i < len(yr):
        j = i
        while j + 1 < len(yr) and yr[j + 1] == yr[i]:
            j += 1
        start_val = row[i - 1] if i > 0 else row[i]
        if start_val and start_val == start_val:
            yearly.append({"year": yr[i],
                           "ret_pct": round((row[j] / start_val - 1.0) * 100, 2),
                           "end_usd": round(float(row[j]), 2)})
        i = j + 1
    step = max(1, len(days) // 400)
    sel = list(range(0, len(days), step))
    if sel[-1] != len(days) - 1:
        sel.append(len(days) - 1)

    def _clean(rec):
        return {k: (None if (isinstance(v, float) and v != v) else
                    (v.item() if hasattr(v, "item") else v))
                for k, v in rec.items()}

    return {
        "genome": _clean(gdf.iloc[bot_id].to_dict()),
        "result": _clean(res.iloc[bot_id].to_dict()),
        "days": [days[i] for i in sel],
        "equity": [round(row[i], 2) for i in sel],
        "split_day": days[split_day],
        "yearly": yearly,
    }


def start_swarm(params: dict) -> dict:
    global _swarm_proc
    import subprocess
    with _swarm_lock:
        if _swarm_proc is not None and _swarm_proc.poll() is None:
            return {"started": False, "error": "a swarm is already running"}
        try:
            bots = max(100, min(int(params.get("bots", 2000)), 20000))
            split = min(max(float(params.get("split", 0.7)), 0.5), 0.9)
            seed = int(params.get("seed", 42))
            ctrl = min(max(float(params.get("control_frac", 0.1)), 0.0), 0.5)
        except (TypeError, ValueError):
            return {"started": False, "error": "bad parameters"}
        since = str(params.get("since") or "")
        if since and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", since):
            return {"started": False, "error": "bad since date"}
        cmd = [sys.executable, "-m", "strategylab.swarm.run", "run",
               "--bots", str(bots), "--split", str(split), "--seed", str(seed),
               "--control-frac", str(ctrl)]
        if since:
            cmd += ["--since", since]
        metrics = str(params.get("metrics") or "")
        if metrics:
            p = Path(metrics)
            if not (p.is_file() and p.resolve().is_relative_to(HERE.parent.resolve())):
                return {"started": False, "error": "metrics file not found"}
            cmd += ["--metrics", str(p)]
        SWARM_DIR.mkdir(parents=True, exist_ok=True)
        log = open(SWARM_DIR / "last_start.log", "w")
        _swarm_proc = subprocess.Popen(cmd, cwd=str(HERE.parent),
                                       stdout=log, stderr=subprocess.STDOUT)
        return {"started": True, "cmd": " ".join(cmd)}


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

        if route == "/" or route == "/dashboard.html":
            self.path = "/dashboard.html"
            return super().do_GET()

        if route == "/chart" or route == "/index.html":
            self.path = "/index.html"
            return super().do_GET()

        if route == "/swarm" or route == "/swarm.html":
            self.path = "/swarm.html"
            return super().do_GET()

        if route == "/api/swarm/runs":
            return self._send_json(swarm_runs_payload())

        if route == "/api/swarm/evos":
            return self._send_json(swarm_evos_payload())

        if route == "/api/swarm/run":
            q = parse_qs(parsed.query)
            d = safe_swarm_dir((q.get("id") or [""])[0])
            if d is None:
                return self._send_json({"error": "run not found"}, 404)
            recap = _read_json(d / "recap.json")
            if recap is None:
                return self._send_json(
                    {"error": "recap not ready",
                     "progress": _read_json(d / "progress.json")}, 202)
            return self._send_json(recap)

        if route == "/api/swarm/bots":
            q = parse_qs(parsed.query)
            payload = swarm_bots_payload((q.get("id") or [""])[0])
            return self._send_json(payload, 404 if "error" in payload else 200)

        if route == "/api/swarm/bot":
            q = parse_qs(parsed.query)
            try:
                bot = int((q.get("bot") or ["-1"])[0])
            except ValueError:
                bot = -1
            payload = swarm_bot_payload((q.get("id") or [""])[0], bot)
            return self._send_json(payload, 404 if "error" in payload else 200)

        if route == "/api/files":
            return self._send_json(list_datasets())

        if route == "/api/health":
            return self._send_json(health_payload())

        if route == "/api/snapshot":
            return self._send_json(snapshot_payload())

        if route == "/api/reports":
            return self._send_json(reports_payload())

        if route == "/api/refresh":
            return self._send_json(refresh_status())

        if route.startswith("/reports/"):
            path = safe_report_path(route[len("/reports/"):])
            if path is None:
                return self._send_json({"error": "report not found"}, 404)
            q = parse_qs(parsed.query)
            if path.suffix == ".md" and not q.get("raw"):
                body = render_markdown(path.read_text(), path.stem)
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            body = path.read_bytes()
            ctype = mimetypes.guess_type(path.name)[0] or "text/plain"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

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

        if route == "/api/fvg":
            q = parse_qs(parsed.query)
            filename = (q.get("file") or [""])[0]
            try:
                bars = int((q.get("bars") or ["5000"])[0])
            except ValueError:
                bars = 5000
            try:
                rr = float((q.get("rr") or ["2.0"])[0])
            except ValueError:
                rr = 2.0
            rr = min(max(rr, 0.5), 10.0)
            payload = fvg_payload(filename, bars, rr)
            status = 404 if "error" in payload else 200
            return self._send_json(payload, status)

        # anything else -> static file from viewer/static
        return super().do_GET()

    def do_POST(self):
        route = urlparse(self.path).path
        if route == "/api/refresh":
            triggered = start_refresh()  # False -> one was already running
            return self._send_json({"triggered": triggered, **refresh_status()})
        if route == "/api/swarm/start":
            try:
                length = int(self.headers.get("Content-Length") or 0)
                params = json.loads(self.rfile.read(length) or b"{}")
            except Exception:
                return self._send_json({"started": False, "error": "bad body"}, 400)
            result = start_swarm(params)
            return self._send_json(result, 200 if result.get("started") else 409)
        return self._send_json({"error": "not found"}, 404)

    def handle_one_request(self):
        # A browser closing a tab or aborting a poll mid-response shows up as a
        # broken pipe / reset while we're writing the body. That's expected for
        # a polling dashboard, not a server error, so swallow it quietly.
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def log_message(self, fmt, *args):  # quieter console
        return


def main():
    if not DATA_DIR.exists():
        print(f"! No data directory at {DATA_DIR}. Fetch some candles first.")
    print(f"Datasets found: {[d['file'] for d in list_datasets()] or 'none'}")
    if list_datasets():
        print("  refreshing datasets in the background...")
        start_refresh()
    print(f"\n  Candle viewer running at  http://{HOST}:{PORT}\n  (Ctrl+C to stop)\n")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
