"""Live paper-trading daemon.

Every cycle (default 60s):
  1. incrementally update each pair's 15m candle tape (closed bars only),
  2. every 5 min, append fresh derivatives metrics + funding from Binance's
     public futures endpoints into per-pair live stores (seeded from the
     archival Vision files, same schema),
  3. replay every roster bot from the paper epoch over the live tape with
     the engine-faithful tracer (frozen thresholds from the roster) — the
     open position / pending order / equity fall out deterministically,
  4. fetch spot marks and publish reports/paper/state.json (atomic write).

Stateless by design: there is no incremental position state to corrupt;
restart at any time. Decisions change only on closed 15m bars; between bars
only marks (and thus unrealized PnL) move.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import paths as datapaths
from ..data.fetch import make_exchange, update_file
from ..swarm import engine, features, trace

FAPI = "https://fapi.binance.com"
SPOT = "https://api.binance.com"
MAINT_MARGIN = 0.005      # maintenance-margin approximation for the liq point
FEATURE_PAD = 1200        # bars of pre-warmup padding so EMAs converge
DERIV_EVERY = 300         # seconds between derivatives refreshes
LIVE_KEEP_DAYS = 45       # tail of archival data seeded into the live stores

# live endpoint -> (path, value field, metrics column)
DERIV_ENDPOINTS = [
    ("/futures/data/openInterestHist", "sumOpenInterest", "oi"),
    ("/futures/data/topLongShortPositionRatio", "longShortRatio", "top_ls_pos"),
    ("/futures/data/topLongShortAccountRatio", "longShortRatio", "top_ls_acct"),
    ("/futures/data/globalLongShortAccountRatio", "longShortRatio", "global_ls"),
    ("/futures/data/takerlongshortRatio", "buySellRatio", "taker_ls"),
]


def _get_json(base: str, path: str, params: dict) -> list | dict:
    url = f"{base}{path}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=15) as r:
        return json.loads(r.read())


def _append_store(store: Path, new: pd.DataFrame) -> None:
    if store.exists():
        old = pd.read_csv(store)
        new = pd.concat([old, new], ignore_index=True)
    new = (new.drop_duplicates("timestamp", keep="last")
              .sort_values("timestamp").reset_index(drop=True))
    _write_atomicish(store, lambda p: new.to_csv(p, index=False))


def _write_atomicish(dest: Path, write) -> None:
    """Atomic tmp+rename when the environment allows it; plain overwrite when
    rename is denied (some sandboxes). Readers poll and retry, so a rare torn
    read is acceptable; a dead publisher is not."""
    tmp = dest.with_suffix(".tmp")
    try:
        write(tmp)
        os.replace(tmp, dest)
    except PermissionError:
        write(dest)


class PairFeed:
    """Live data for one pair: candles, deriv metrics, funding."""

    def __init__(self, pair: str, live_dir: Path, exchange, tape: str = "spot"):
        self.pair = pair
        self.sym = f"{pair}USDT"
        self.exchange = exchange
        self.candles = (datapaths.candles(f"{pair}/USDT:USDT", "15m", "binanceusdm")
                        if tape == "perp"
                        else Path(datapaths.default_candles("15m", f"{pair}-USDT")))
        self.metrics_store = live_dir / f"{pair}_metrics.csv"
        self.funding_store = live_dir / f"{pair}_funding.csv"
        self._seed_stores()
        self._last_deriv = 0.0

    def _seed_stores(self) -> None:
        """Merge the archival tail into the live stores (append-dedup, every
        startup) — so a Vision backfill (sl-swarm fetch-metrics) followed by a
        daemon restart heals any deriv gap left by downtime beyond the live
        endpoints' lookback."""
        cutoff = int(time.time() * 1000) - LIVE_KEEP_DAYS * 86_400_000
        pairs_dir = f"{self.pair}-USDT"
        for store, src in (
                (self.metrics_store, Path(datapaths.default_metrics(pairs_dir))),
                (self.funding_store, Path(datapaths.default_funding(pairs_dir)))):
            if not src.exists():
                continue
            df = pd.read_csv(src)
            _append_store(store, df[df["timestamp"] >= cutoff])

    def refresh_candles(self) -> dict:
        return update_file(self.candles, self.exchange)

    def refresh_deriv(self) -> None:
        now = time.time()
        if now - self._last_deriv < DERIV_EVERY:
            return
        frames = []
        for path, field, col in DERIV_ENDPOINTS:
            rows = _get_json(FAPI, path,
                             {"symbol": self.sym, "period": "5m", "limit": 500})
            frames.append(pd.DataFrame(
                {"timestamp": [int(r["timestamp"]) for r in rows],
                 col: [float(r[field]) for r in rows]}).set_index("timestamp"))
            time.sleep(0.1)
        merged = pd.concat(frames, axis=1).reset_index().dropna()
        _append_store(self.metrics_store, merged)

        fr = _get_json(FAPI, "/fapi/v1/fundingRate",
                       {"symbol": self.sym, "limit": 100})
        _append_store(self.funding_store, pd.DataFrame(
            {"timestamp": [int(r["fundingTime"]) for r in fr],
             "funding": [float(r["fundingRate"]) for r in fr]}))
        self._last_deriv = now      # only marked fresh after a full success

    def frame(self, epoch_ms: int) -> tuple[pd.DataFrame, int]:
        """Tail dataframe with features-ready columns; returns (df, epoch_idx
        within df). Sliced so epoch - WARMUP - FEATURE_PAD bars are present."""
        df = pd.read_csv(self.candles)
        ts = df["timestamp"].to_numpy(np.int64)
        epoch_idx = int(np.searchsorted(ts, epoch_ms))
        s0 = max(0, epoch_idx - engine.WARMUP - FEATURE_PAD)
        df = df.iloc[s0:].reset_index(drop=True)
        df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = features.merge_metrics(df, self.metrics_store)
        df = features.merge_funding(df, self.funding_store)
        return df, epoch_idx - s0


QS = np.linspace(0.02, 0.98, 49)  # quantile grid the rule thresholds live on


def replay_bot(bot: dict, w: dict, ts_ms: np.ndarray, names: list[str]) -> dict:
    p = trace.parse_record(bot["rec"], names)
    thr = np.array(bot["thresholds"], dtype=np.float64)
    cfg = bot["cfg"]
    state: dict = {}
    trades = trace._trace_window(
        w, ts_ms, p, thr, cfg["taker_bps"] / 1e4, cfg["maker_bps"] / 1e4,
        cfg["start_capital"], cfg["ruin_frac"] * cfg["start_capital"],
        "paper", state_out=state)
    return {"trades": trades, "state": state,
            "triggers": rule_triggers(bot, w["F"][-1], names)}


def rule_triggers(bot: dict, F_last: np.ndarray, names: list[str]) -> list[dict]:
    """Where the LIVE feature value sits vs each rule's firing quantile —
    'how far is this bot from being triggered', on the threshold's own scale
    (the quantile ladder frozen into the roster at selection)."""
    out = []
    for m in bot.get("trigger_meta", []):
        if m["feature"] not in names:
            continue
        v = float(F_last[names.index(m["feature"])])
        row = {"feature": m["feature"], "op": m["op"], "dir": m["dir"],
               "need_q": m["q"], "thr": m["thr"],
               "value": None, "cur_q": None, "fired": False}
        if np.isfinite(v):
            cur_q = float(np.interp(v, np.array(m["ladder"]), QS))
            row.update({
                "value": round(v, 8), "cur_q": round(cur_q, 4),
                "fired": bool(v > m["thr"] if m["op"] == ">" else v < m["thr"]),
            })
        out.append(row)
    return out


def bot_payload(bot: dict, res: dict, mark: float | None) -> dict:
    st = res["state"]
    cfg = bot["cfg"]
    start = cfg["start_capital"]
    closed_pnl = sum(t["pnl"] + t.get("fund", 0.0) for t in res["trades"])
    out = {
        "pair": bot["pair"], "label": bot["label"], "bot_id": bot["bot_id"],
        "run_id": bot["run_id"], "rules": bot["rec"].get("rules"),
        "tf": bot["rec"].get("tf"), "test_sharpe": bot.get("test_sharpe"),
        # maker_off > 0: entries rest as maker limits offset by maker_off x ATR
        # (order_ttl bars, then cancelled); maker_off == 0: taker market chase
        "maker_off_atr": bot["rec"].get("maker_off_atr"),
        "order_ttl": bot["rec"].get("order_ttl"),
        "status": "dead" if st["dead"] else
                  "position" if st["pos"] != 0 else
                  "pending" if st["pending"] else "idle",
        "triggers": res.get("triggers", []),
        "equity": round(st["eq"], 2), "n_trades": len(res["trades"]),
        "realized_pnl": round(closed_pnl, 2),
        "unrealized_pnl": None, "mark": mark,
    }
    if st["pending"]:
        out["pending_order"] = {
            "side": "LONG" if st["pdir"] > 0 else "SHORT",
            "type": "market" if st["pending"] == 1 else "limit",
            "px": round(st["ppx"], 6) if st["pending"] == 2 else None,
            "ttl_bars": st["pttl"],
        }
    if st["pos"] != 0:
        side, qty = st["pos"], st["qty"]
        entry_raw, entry_eff, eq = st["entry_raw"], st["entry_eff"], st["eq"]
        entry_cost = -qty * (entry_eff - entry_raw) * side
        lev = qty * entry_raw / eq if eq > 0 else float("inf")
        bank = entry_raw - side * eq / qty if qty > 0 else None
        liq = (bank + side * MAINT_MARGIN * entry_raw) if bank else None
        if liq is not None and liq <= 0:
            liq = None                       # sub-1x effective leverage: no liq
        upnl = qty * (mark - entry_raw) * side if mark else None
        out.update({
            "position": {
                "side": "LONG" if side > 0 else "SHORT",
                "entry": round(entry_raw, 6), "entry_eff": round(entry_eff, 6),
                "qty": round(qty, 6), "opened_sec": st["entry_sec"],
                "held_bars": st["held"],
                "stop_px": round(st["stop_px"], 6),
                "tp_px": round(st["tp_px"], 6) if st["tp_px"] else None,
                "liq_px": round(liq, 6) if liq else None,
                "leverage": round(lev, 2),
                "entry_cost": round(entry_cost, 4),
                "funding": round(st["trade_fund"], 4),
            },
            "unrealized_pnl": round(upnl, 2) if upnl is not None else None,
            # entry exec cost + funding are realized the moment they happen
            "realized_pnl": round(closed_pnl + entry_cost + st["trade_fund"], 2),
        })
    return out


def run_cycle(pairs: dict[str, PairFeed], bots_by_pair: dict[str, list[dict]],
              roster: dict, prev: dict, out_dir: Path, interval: int) -> dict:
    epoch = int(roster["paper_start_ms"])
    tape = roster.get("tape", "spot")
    prev_bots = {b["label"]: b for b in prev.get("bots", [])}
    marks, errors, bots_out, all_trades = {}, [], [], []

    try:
        if tape == "perp":   # perp roster marks to the perp last price
            tick = _get_json(FAPI, "/fapi/v1/ticker/price", {})
            want = {f"{p}USDT" for p in pairs}
            marks = {t["symbol"][:-4]: float(t["price"]) for t in tick
                     if t["symbol"] in want}
        else:
            tick = _get_json(SPOT, "/api/v3/ticker/price", {"symbols": json.dumps(
                [f"{p}USDT" for p in pairs], separators=(",", ":"))})
            marks = {t["symbol"][:-4]: float(t["price"]) for t in tick}
    except Exception as e:
        errors.append(f"marks: {type(e).__name__}: {e}")

    for pair, feed in pairs.items():
        try:
            r = feed.refresh_candles()
            if "error" in r:
                errors.append(f"{pair} candles: {r['error']}")
            try:
                feed.refresh_deriv()
            except Exception as e:  # stale deriv data degrades, not disables:
                # deriv rules simply can't fire on bars past the store's tail
                errors.append(f"{pair} deriv: {type(e).__name__}: {e}")
            df, epoch_idx = feed.frame(epoch)
            F, names = features.compute_features(df, 96)
            expect = roster["feature_names"].get(pair)
            if expect and names != expect:
                raise RuntimeError(f"feature schema drift: {names} != {expect}")
            m0 = max(0, epoch_idx - engine.WARMUP)
            w = {"o": df["open"].to_numpy(np.float64)[m0:],
                 "h": df["high"].to_numpy(np.float64)[m0:],
                 "l": df["low"].to_numpy(np.float64)[m0:],
                 "c": df["close"].to_numpy(np.float64)[m0:],
                 "atr": features.atr(df)[m0:],
                 "hour": df["dt"].dt.hour.to_numpy(np.int64)[m0:],
                 "F": F[m0:],
                 "fund": (df["fund_pay"].to_numpy(np.float64)[m0:]
                          if "fund_pay" in df.columns else
                          np.zeros(len(df) - m0))}
            ts_ms = df["timestamp"].to_numpy(np.int64)[m0:]
            for bot in bots_by_pair.get(pair, []):
                res = replay_bot(bot, w, ts_ms, names)
                payload = bot_payload(bot, res, marks.get(pair))
                payload["last_bar_sec"] = int(ts_ms[-1] // 1000)
                bots_out.append(payload)
                for t in res["trades"]:
                    all_trades.append({**t, "bot": bot["label"],
                                       "pair": pair})
        except Exception as e:
            errors.append(f"{pair}: {type(e).__name__}: {e}")
            # carry the pair's previous bot states so the page doesn't blank
            for bot in bots_by_pair.get(pair, []):
                if bot["label"] in prev_bots:
                    bots_out.append(prev_bots[bot["label"]])

    all_trades.sort(key=lambda t: t["xt"], reverse=True)
    open_pos = [b for b in bots_out if b.get("status") == "position"]
    state = {
        "generated_ms": int(time.time() * 1000),
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "paper_start_ms": epoch, "interval_s": interval, "tape": tape,
        "marks": marks, "errors": errors,
        "totals": {
            "bots": len(bots_out),
            "open_positions": len(open_pos),
            "equity": round(sum(b["equity"] for b in bots_out), 2),
            "realized_pnl": round(sum(b["realized_pnl"] for b in bots_out), 2),
            "unrealized_pnl": round(sum(b["unrealized_pnl"] or 0.0
                                        for b in bots_out), 2),
        },
        "bots": bots_out,
        "trades": all_trades[:200],
    }
    _write_atomicish(out_dir / "state.json",
                     lambda p: p.write_text(json.dumps(state)))
    return state


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interval", type=int, default=60, help="cycle seconds")
    ap.add_argument("--once", action="store_true", help="single cycle and exit")
    ap.add_argument("--dir", default="reports/paper")
    args = ap.parse_args(argv)

    out_dir = Path(args.dir)
    roster = json.loads((out_dir / "roster.json").read_text())
    live_dir = out_dir / "live"
    live_dir.mkdir(parents=True, exist_ok=True)

    # roster-level capital wins over the battery's (sizing is %-based, so
    # this is pure display scale — behavior is identical at any capital)
    cap = float(roster.get("start_capital", 10_000.0))
    bots_by_pair: dict[str, list[dict]] = {}
    for b in roster["bots"]:
        b["cfg"]["start_capital"] = cap
        bots_by_pair.setdefault(b["pair"], []).append(b)
    tape = roster.get("tape", "spot")
    exchange = make_exchange("binanceusdm" if tape == "perp" else "binance")
    pairs = {p: PairFeed(p, live_dir, exchange, tape)
             for p in sorted(bots_by_pair)}
    print(f"paper daemon: {sum(map(len, bots_by_pair.values()))} bots / "
          f"{len(pairs)} pairs, tape {tape}, epoch "
          f"{datetime.fromtimestamp(roster['paper_start_ms']/1000, tz=timezone.utc)}", flush=True)

    prev: dict = {}
    if (out_dir / "state.json").exists():
        prev = json.loads((out_dir / "state.json").read_text())
    while True:
        t0 = time.time()
        try:
            prev = run_cycle(pairs, bots_by_pair, roster, prev, out_dir,
                             args.interval)
            t = prev["totals"]
            msg = (f"cycle ok: {t['open_positions']} open, "
                   f"uPnL {t['unrealized_pnl']:+.2f}, "
                   f"rPnL {t['realized_pnl']:+.2f}")
            if prev["errors"]:
                msg += f" ({len(prev['errors'])} errors: {prev['errors'][0]})"
        except Exception as e:  # never die mid-loop; state.json keeps last good
            msg = f"cycle FAILED: {type(e).__name__}: {e}"
        print(f"[{datetime.now():%H:%M:%S}] {msg} ({time.time()-t0:.1f}s)", flush=True)
        if args.once:
            break
        time.sleep(max(5.0, args.interval - (time.time() - t0)))


if __name__ == "__main__":
    main()
