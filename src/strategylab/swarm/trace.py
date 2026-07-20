"""Single-bot trade tracer — re-simulates one stored genome bar-by-bar and
records every fill, so the viewer can overlay a bot's positions on the
candle chart.

The swarm engine is vectorized across thousands of bots and only keeps
aggregate counters; this module mirrors its execution semantics op-for-op
for a single non-control bot (HOF bots never carry the control flag, so the
control-RNG stream is never consumed) and keeps the trade log instead.

Genome values come from hof_history.json, where they were rounded for
display (rule thresholds to 2dp, sizes to 2-3dp), so a re-simulated trade
can occasionally diverge from the original run at the margin. Each fitness
window restarts from clean state, so divergence never compounds across
windows.
"""

from __future__ import annotations

import re

import numpy as np

from . import engine
from .evolve import _window
from .genome import DIR_BIAS, LOSS_REACT, SESSION_HOURS, SESSIONS

_RULE = re.compile(r"(\w+) ([<>]) q([0-9.]+) → (LONG|SHORT)")
_SESS = {v: k for k, v in SESSIONS.items()}
_LOSS = {v: k for k, v in LOSS_REACT.items()}
_BIAS = {v: k for k, v in DIR_BIAS.items()}


def parse_record(rec: dict, feature_names: list[str]) -> dict:
    """Scalar genome params from a hof_history bot record. Rule features are
    resolved by NAME against the rebuilt market's feature list, so a run and
    a re-simulation with different feature sets fail loudly instead of
    silently reading the wrong column."""
    rules = _RULE.findall(rec.get("rules") or "")
    if not rules:
        raise ValueError("no parseable rules on this bot")
    feats, ops_gt, qvals, dirs = [], [], [], []
    for name, op, q, d in rules:
        if name not in feature_names:
            raise ValueError(
                f"feature '{name}' missing from rebuilt market — the run's "
                "metrics/funding data files are probably gone")
        feats.append(feature_names.index(name))
        ops_gt.append(op == ">")
        qvals.append(float(q))
        dirs.append(1 if d == "LONG" else -1)
    tp = rec.get("tp_rr")
    sess = SESSION_HOURS[_SESS[rec.get("session") or "any"]]
    sess_ok = np.zeros(24, dtype=bool)
    sess_ok[sess[0]:sess[1]] = True
    return {
        "rule_feat": np.array([feats], dtype=np.int64),
        "rule_q": np.array([qvals], dtype=np.float64),
        "rule_op_gt": ops_gt,
        "rule_dir": dirs,
        "dir_bias": _BIAS[rec.get("dir_bias") or "both"],
        "risk": float(rec["risk_pct"]) / 100.0,
        "stop_atr": float(rec["stop_atr"]),
        "tp_rr": float(tp) if tp is not None else np.nan,
        "max_hold": int(rec["max_hold_bars"]),
        "maker_off": float(rec["maker_off_atr"]),
        "order_ttl": int(rec["order_ttl"]),
        "loss_react": _LOSS[rec.get("loss_react") or "neutral"],
        "cooldown": int(rec.get("cooldown_bars") or 0),
        "revenge": float(rec.get("revenge_mult") or 1.0),
        "reentry_gap": int(rec.get("reentry_gap") or 0),
        "sess_ok": sess_ok,
    }


def trace_bot(rec: dict, feature_names: list[str], mkt: dict,
              ts: np.ndarray, segments: list[tuple], qs: np.ndarray,
              cfg: dict) -> list[dict]:
    """Re-simulate one bot over `segments` [(label, t0_ms, t1_ms), ...] —
    the run's fitness windows plus the reserved test span, each from clean
    state exactly as the evolution did. Returns the concatenated trade log."""
    p = parse_record(rec, feature_names)
    taker = cfg["taker_bps"] / 1e4
    edge = cfg["maker_bps"] / 1e4
    start_cap = float(cfg["start_capital"])
    ruin = cfg["ruin_frac"] * start_cap
    trades: list[dict] = []
    for label, t0, t1 in segments:
        w, i0, i1 = _window(mkt, ts, int(t0), int(t1), qs)
        m0 = max(0, i0 - engine.WARMUP)
        thr = engine._interp_thresholds(
            p["rule_feat"], p["rule_q"], w["Q"], w["qs"])[0]
        trades += _trace_window(w, ts[m0:i1], p, thr, taker, edge,
                                start_cap, ruin, label)
    return trades


def _trace_window(w, ts_ms, p, thr, taker, edge, start_cap, ruin, label,
                  state_out: dict | None = None):
    """One window from clean state. Mirrors engine._bar_kernel branch-for-
    branch for a single non-control bot; python lists keep the scalar loop
    fast enough (~1s for the full 5m tape) and the whole thing is cached
    server-side per (run, bot).

    With `state_out`, any position still open at the last bar is NOT closed;
    the in-flight state (position, pending order, equity, reaction counters)
    is written into the dict instead — the live paper trader's contract."""
    o = w["o"].tolist()
    h = w["h"].tolist()
    l = w["l"].tolist()  # noqa: E741 — mirrors the engine's naming
    c = w["c"].tolist()
    A = w["atr"].tolist()
    fund = w["fund"].tolist()
    n_bars = len(c)

    # desired direction per bar, vectorized (rules are stateless), with the
    # session filter and ideology direction filter folded in
    sig = np.zeros(n_bars)
    F = w["F"]
    for r in range(len(thr)):
        v = F[:, p["rule_feat"][0, r]].astype(np.float64)
        cond = (v > thr[r]) if p["rule_op_gt"][r] else (v < thr[r])
        sig += np.where(cond & np.isfinite(v), float(p["rule_dir"][r]), 0.0)
    des = np.sign(sig).astype(np.int64)
    if p["dir_bias"] < 0:
        des[des > 0] = 0
    elif p["dir_bias"] > 0:
        des[des < 0] = 0
    want = np.where(p["sess_ok"][w["hour"]], des, 0).tolist()
    sec = (ts_ms // 1000).tolist()

    risk, stop_atr, tp_rr = p["risk"], p["stop_atr"], p["tp_rr"]
    max_hold, maker_off, order_ttl = p["max_hold"], p["maker_off"], p["order_ttl"]
    loss_react, cooldown_len = p["loss_react"], p["cooldown"]
    revenge, reentry_gap = p["revenge"], p["reentry_gap"]

    pos = 0
    entry = qty = stop_px = tp_px = 0.0
    held = 0
    pk = 0            # pending: 0 none / 1 market / 2 limit
    pdir = 0
    ppx = 0.0
    pttl = 0
    cd = 0
    rmult = 1.0
    gap_until = 0
    eq = start_cap
    dead = False
    entry_t = 0
    entry_raw = 0.0
    trade_fund = 0.0   # net funding on the currently open trade (+ = received)
    trades: list[dict] = []

    def open_pos(t, raw, eff, at):
        nonlocal pos, entry, qty, stop_px, tp_px, held, pk, entry_t, entry_raw, \
            trade_fund
        d = float(pdir)
        dist = stop_atr * at
        q = eq * risk * rmult / dist if dist > 0.0 else float("inf")
        cap = engine.MAX_LEV * eq / raw
        if q > cap:
            q = cap
        pos = pdir
        entry = eff
        qty = q
        stop_px = raw - d * dist
        tp_px = raw + d * tp_rr * dist if np.isfinite(tp_rr) else d * float("inf")
        held = 0
        pk = 0
        entry_t = t
        entry_raw = raw
        trade_fund = 0.0

    def close_pos(t, exit_raw, exit_eff, why):
        nonlocal pos, qty, held, cd, rmult, gap_until, eq, dead, pk
        pnl = qty * (exit_eff - entry) * pos
        eq += pnl
        tr = {"seg": label, "side": pos, "et": sec[entry_t],
              "ep": round(entry_raw, 6), "xt": sec[t],
              "xp": round(exit_raw, 6), "qty": round(qty, 8),
              "pnl": round(pnl, 4), "fund": round(trade_fund, 4),
              "why": why, "hold": held}
        if pnl > 0:
            rmult = 1.0
        else:
            if loss_react == 1:
                cd = cooldown_len
            rmult = revenge if loss_react == 2 else 1.0
        gap_until = t + reentry_gap
        pos = 0
        qty = 0.0
        held = 0
        if eq < ruin:  # ruin line
            dead = True
            pk = 0
            tr["ruin"] = True
        trades.append(tr)

    for t in range(engine.WARMUP, n_bars):
        ot, ht, lt, ct, at = o[t], h[t], l[t], c[t], A[t]

        # -- 1. fill pending orders placed on earlier bars ----------------
        if pk > 0 and not dead and pos == 0:
            if pk == 1:                          # market @ open, taker
                open_pos(t, ot, ot * (1.0 + pdir * taker), at)
            elif (lt <= ppx) if pdir > 0 else (ht >= ppx):
                open_pos(t, ppx, ppx * (1.0 + pdir * edge), at)
            else:                                # resting limit ages
                pttl -= 1
                if pttl <= 0:
                    pk = 0

        # -- 2. exits (stop wins over TP within one bar) ------------------
        if pos != 0 and held >= 1:
            long = pos > 0
            if (lt <= stop_px) if long else (ht >= stop_px):
                close_pos(t, stop_px, stop_px * (1.0 - pos * taker), "stop")
            elif (ht >= tp_px) if long else (lt <= tp_px):
                close_pos(t, tp_px, tp_px * (1.0 - pos * edge), "tp")
            elif held >= max_hold:
                close_pos(t, ct, ct * (1.0 - pos * taker), "time")

        # -- 3. new entries (decided at close, fill from next bar) --------
        if (not dead and pos == 0 and pk == 0 and cd == 0
                and t >= gap_until and want[t] != 0):
            pdir = want[t]
            if maker_off == 0.0:                 # taker chase
                pk = 1
                pttl = 1
            else:                                # resting maker limit
                pk = 2
                ppx = ct - pdir * maker_off * at
                pttl = order_ttl

        # -- 4. bookkeeping -----------------------------------------------
        if cd > 0:
            cd -= 1
        if pos != 0:
            held += 1
            if fund[t] != 0.0:                   # funding settles this bar
                pay = fund[t] * pos * qty * ct
                eq -= pay
                trade_fund -= pay

    if state_out is not None:
        state_out.update({
            "pos": pos, "entry_eff": entry, "entry_raw": entry_raw,
            "entry_sec": sec[entry_t] if pos != 0 else None,
            "qty": qty, "stop_px": stop_px,
            "tp_px": tp_px if np.isfinite(tp_px) else None,
            "held": held, "trade_fund": round(trade_fund, 4),
            "pending": pk, "pdir": pdir, "ppx": ppx, "pttl": pttl,
            "cooldown": cd, "rmult": rmult,
            "gap_left": max(0, gap_until - (n_bars - 1)),
            "eq": eq, "dead": dead,
        })
        return trades

    # mark-to-market close of anything still open (taker at last close)
    if pos != 0:
        close_pos(n_bars - 1, c[-1], c[-1] * (1.0 - pos * taker), "eof")
    return trades
