"""Forward-tracking of the pre-registered top_ls_pos capitulation-fade configs.

Spec frozen 2026-07-19 — see top-ls-forward-preregistration.md. Everything here
is deliberately hardcoded: the three configs, the crash gate, the threshold
protocol, and the tracking start date. This command recomputes the full record
from the frozen spec on whatever data is on disk and rewrites
reports/tracking/ — stateless, so re-runs and live data refreshes are safe and
the forward record accumulates with no human (or model) fingerprints on it.

    sl-swarm track            # rescore, rewrite reports/tracking/
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import engine, features, genome
from .evolve import _alloc, _fill_eq

# ---- FROZEN SPEC (2026-07-19) — do not edit; supersede via a new file ------
TRACK_START = np.datetime64("2026-07-19")  # data at/after this date is forward OOS
SMA_DAYS = 200          # crash gate: no NEW entries while daily close < SMA200
FIRST_YEAR = 2026       # thresholds: quantiles of all history before each Jan 1
CONFIGS = [             # (entry_q, stop_atr, tp_rr, hold_bars, maker_off)
    (0.10, 4.0, 3.0, 480, 0.15),
    (0.10, 4.0, 3.0, 480, 0.25),
    (0.20, 3.0, float("nan"), 480, 0.50),
]
RISK_PCT = 0.005
QS = np.linspace(0.02, 0.98, 49)


def _grid(names: list[str], fi: int) -> genome.Genomes:
    n = len(CONFIGS)
    g = genome.Genomes(seed=0, feature_names=names)
    g.is_control = np.zeros(n, bool)
    g.n_rules = np.ones(n, np.int8)
    g.rule_feat = np.full((n, genome.MAX_RULES), fi, np.int16)
    g.rule_op = np.full((n, genome.MAX_RULES), -1, np.int8)   # feature BELOW q
    g.rule_q = np.full((n, genome.MAX_RULES), 0.5, np.float32)
    g.rule_dir = np.ones((n, genome.MAX_RULES), np.int8)      # -> LONG
    g.rule_q[:, 0] = [c[0] for c in CONFIGS]
    g.ctrl_rate = np.full(n, 0.01, np.float32)
    g.tf = np.ones(n, np.int8)
    g.dir_bias = np.zeros(n, np.int8)
    g.risk_pct = np.full(n, RISK_PCT, np.float32)
    g.stop_atr = np.array([c[1] for c in CONFIGS], np.float32)
    g.tp_rr = np.array([c[2] for c in CONFIGS], np.float32)
    g.max_hold = np.array([c[3] for c in CONFIGS], np.int32)
    g.maker_off = np.array([c[4] for c in CONFIGS], np.float32)
    g.order_ttl = np.full(n, 4, np.int16)
    g.session = np.zeros(n, np.int8)
    g.loss_react = np.zeros(n, np.int8)
    g.cooldown = np.zeros(n, np.int16)
    g.revenge = np.ones(n, np.float32)
    g.reentry_gap = np.full(n, 4, np.int16)
    return g


def cmd_track(args) -> None:
    from .run import _load

    df = _load(args.file15, "2021-01-01", args.metrics, args.funding)
    F, names = features.compute_features(df, 96)
    fi = names.index("top_ls_pos")
    ts = df["timestamp"].to_numpy(np.int64)
    c = df["close"].to_numpy(np.float64)
    bar_days = df["dt"].dt.tz_convert(None).dt.floor("D").to_numpy().astype("datetime64[D]")
    all_days = np.unique(bar_days)
    day_pos = np.searchsorted(all_days, bar_days).astype(np.int64)

    # crash gate per day: yesterday's close below SMA200 of closes through
    # yesterday -> no new entries today (open positions run their course)
    day_close = np.full(len(all_days), np.nan)
    day_close[day_pos] = c
    sma = pd.Series(day_close).rolling(SMA_DAYS, min_periods=SMA_DAYS).mean().to_numpy()
    below = np.zeros(len(all_days), bool)
    below[1:] = day_close[:-1] < sma[:-1]
    gate_bar = below[day_pos]           # True -> mask signal on this bar

    base = {"o": df["open"].to_numpy(np.float64), "h": df["high"].to_numpy(np.float64),
            "l": df["low"].to_numpy(np.float64), "c": c,
            "atr": features.atr(df), "hour": df["dt"].dt.hour.to_numpy(np.int64),
            "fund": df["fund_pay"].to_numpy(np.float64) if "fund_pay" in df.columns
            else np.zeros(len(df))}
    F_gated = F.copy()
    F_gated[gate_bar, fi] = np.nan

    g = _grid(names, fi)
    cfg = {"taker_bps": args.taker_bps, "maker_bps": args.maker_bps,
           "start_capital": 10_000.0, "ruin_frac": 0.30, "seed": 0}
    last_year = int(str(all_days[-1])[:4])

    # per-year sims (thresholds from pre-year history), stitched afterwards
    eqs = {"open": [], "gated": []}
    stats = {"open": np.zeros((len(CONFIGS), 3)), "gated": np.zeros((len(CONFIGS), 3))}
    got_days = []
    for Y in range(FIRST_YEAR, last_year + 1):
        i0 = int(np.searchsorted(ts, np.datetime64(f"{Y}-01-01", "ms").astype(np.int64)))
        i1 = int(np.searchsorted(ts, np.datetime64(f"{Y + 1}-01-01", "ms").astype(np.int64)))
        m0 = max(0, i0 - engine.WARMUP)
        Q = features.train_quantiles(F, i0, QS)
        d0, d1 = int(day_pos[i0]), int(day_pos[i1 - 1]) + 1
        got_days.append((d0, d1))
        for key, Fv in (("open", F), ("gated", F_gated)):
            w = {k: v[m0:i1] for k, v in base.items()}
            w.update({"F": Fv[m0:i1], "Q": Q, "qs": QS, "tf_code": 1,
                      "seg_b": np.ones(i1 - m0, bool), "day_pos": day_pos[m0:i1]})
            out = _alloc(g.n, len(all_days))
            engine.run_cohort(w, g, np.arange(g.n), cfg, out)
            eq = _fill_eq(out, d0, d1, cfg["start_capital"]) / cfg["start_capital"]
            eqs[key].append(eq)
            stats[key][:, 0] += out["trades_a"] + out["trades_b"]
            stats[key][:, 1] += out["fees"]
            stats[key][:, 2] += out["fund_paid"]

    # stitch years into one normalized curve per config, then cut at TRACK_START
    days = np.concatenate([all_days[d0:d1] for d0, d1 in got_days])
    curves = {}
    for k, v in eqs.items():
        mult = np.ones((g.n, 1))
        parts = []
        for eq in v:
            eqn = eq / eq[:, :1]
            parts.append(eqn * mult)
            mult = parts[-1][:, -1:]
        curves[k] = np.concatenate(parts, axis=1)

    fwd = days >= TRACK_START
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if fwd.sum() < 2:
        msg = (f"# top_ls_pos forward tracking\n\nSpec frozen 2026-07-19. "
               f"Data through {days[-1]} — fewer than 2 forward days yet; the "
               f"record begins accumulating from the next data refresh.")
        (out_dir / "top_ls_forward.md").write_text(msg)
        print(msg)
        return
    rows = []
    months = days[fwd].astype("datetime64[M]")
    for ci, (q, stop, tp, hold, moff) in enumerate(CONFIGS):
        rec = {"config": f"q{q} stop{stop} tp{tp} hold{hold} moff{moff}"}
        for k in ("open", "gated"):
            cv = curves[k][ci][fwd]
            cv = cv / cv[0]
            r = np.diff(cv) / cv[:-1]
            rec[f"{k}_ret_pct"] = round(float(cv[-1] - 1) * 100, 2)
            rec[f"{k}_sharpe"] = round(float(np.mean(r) / np.std(r) * np.sqrt(365)), 2) \
                if np.std(r) > 0 else None
            rec[f"{k}_maxdd_pct"] = round(float((cv / np.maximum.accumulate(cv) - 1).min()) * 100, 2)
            rec[f"{k}_monthly"] = {str(mo): round(float(cv[months == mo][-1] / cv[months == mo][0] - 1) * 100, 2)
                                   for mo in np.unique(months)}
        rec["trades"] = int(stats["open"][ci, 0])
        rec["fees_usd"] = round(float(stats["open"][ci, 1]), 2)
        rec["funding_usd"] = round(float(stats["open"][ci, 2]), 2)
        rows.append(rec)

    payload = {"generated": datetime.now().isoformat(timespec="seconds"),
               "spec_frozen": "2026-07-19", "track_start": str(TRACK_START),
               "data_through": str(days[-1]), "forward_days": int(fwd.sum()),
               "gate": f"no new entries while daily close < SMA{SMA_DAYS}",
               "configs": rows}
    (out_dir / "top_ls_forward.json").write_text(json.dumps(payload, indent=2))

    lines = [f"# top_ls_pos forward tracking — generated {payload['generated']}",
             "",
             f"Spec frozen 2026-07-19 · tracking from {TRACK_START} · data through "
             f"{payload['data_through']} · **{payload['forward_days']} forward days**",
             "", "| config | ret | S | maxDD | gated ret | gated S | gated DD | trades | fees | funding |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['config']} | {r['open_ret_pct']}% | {r['open_sharpe']} | "
                     f"{r['open_maxdd_pct']}% | {r['gated_ret_pct']}% | {r['gated_sharpe']} | "
                     f"{r['gated_maxdd_pct']}% | {r['trades']} | ${r['fees_usd']} | "
                     f"${r['funding_usd']} |")
    lines += ["", "Monthly returns (ungated):", ""]
    mos = sorted({m for r in rows for m in r["open_monthly"]})
    lines.append("| month | " + " | ".join(r["config"][:18] for r in rows) + " |")
    lines.append("|---|" + "---|" * len(rows))
    for mo in mos:
        lines.append(f"| {mo} | " + " | ".join(
            f"{r['open_monthly'].get(mo, '—')}%" for r in rows) + " |")
    (out_dir / "top_ls_forward.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nartifacts -> {out_dir}/")


def add_parser(sub) -> None:
    t = sub.add_parser("track", help="rescore the pre-registered top_ls configs (forward OOS)")
    t.add_argument("--file15", default="data/binance_BTC-USDT_15m.csv")
    t.add_argument("--metrics", default="data/metrics/BTCUSDT_metrics.csv")
    t.add_argument("--funding", default="data/metrics/BTC-USDT-USDT_funding.csv")
    t.add_argument("--taker-bps", type=float, default=5.0)
    t.add_argument("--maker-bps", type=float, default=1.0)
    t.add_argument("--out", default="reports/tracking")
    t.set_defaults(func=cmd_track)
