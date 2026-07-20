"""Forward-tracking of the pre-registered funding-fade SHORT configs (ADA, XRP).

Spec frozen 2026-07-19 — see funding-fade-forward-preregistration.md. Everything
here is deliberately hardcoded: the pairs, the 18 configs, the melt-up gate, the
threshold protocol, and the tracking start date. This command recomputes the
full record from the frozen spec on whatever data is on disk and rewrites
reports/tracking/funding_fade_forward.{md,json} — stateless, so re-runs and
live data refreshes are safe and the forward record accumulates with no human
(or model) fingerprints on it.

    sl-swarm track-funding      # rescore, rewrite reports/tracking/
"""

from __future__ import annotations

import json
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from . import engine, features, genome
from .evolve import _alloc, _fill_eq

# ---- FROZEN SPEC (2026-07-19) — do not edit; supersede via a new file ------
TRACK_START = np.datetime64("2026-07-19")  # data at/after this date is forward OOS
PAIRS = ("ADA", "XRP")  # the two pairs where the probe replicated (100% breadth)
ENTRY_Q = 0.70          # representative of the collapsed q0.55–0.90 band: funding
                        # sits at the default rate most of the time, so these
                        # quantiles resolve to one threshold (registered as such)
SMA_DAYS = 200          # melt-up gate: no NEW shorts while daily close > SMA200
FIRST_YEAR = 2026       # thresholds: quantiles of all history before each Jan 1
STOPS = (2.0, 3.0, 4.0)
TPS = (1.5, 3.0, float("nan"))
HOLDS = (96, 288)       # 1 and 3 days at 15m
CONFIGS = [(s, t, h) for s, t, h in product(STOPS, TPS, HOLDS)]  # 18, fixed order
MAKER_OFF = 0.25
RISK_PCT = 0.005
QS = np.linspace(0.02, 0.98, 49)


def _grid(names: list[str], fi: int) -> genome.Genomes:
    n = len(CONFIGS)
    g = genome.Genomes(seed=0, feature_names=names)
    g.is_control = np.zeros(n, bool)
    g.n_rules = np.ones(n, np.int8)
    g.rule_feat = np.full((n, genome.MAX_RULES), fi, np.int16)
    g.rule_op = np.full((n, genome.MAX_RULES), 1, np.int8)     # feature ABOVE q
    g.rule_q = np.full((n, genome.MAX_RULES), ENTRY_Q, np.float32)
    g.rule_dir = np.full((n, genome.MAX_RULES), -1, np.int8)   # -> SHORT
    g.ctrl_rate = np.full(n, 0.01, np.float32)
    g.tf = np.ones(n, np.int8)
    g.dir_bias = np.zeros(n, np.int8)
    g.risk_pct = np.full(n, RISK_PCT, np.float32)
    g.stop_atr = np.array([c[0] for c in CONFIGS], np.float32)
    g.tp_rr = np.array([c[1] for c in CONFIGS], np.float32)
    g.max_hold = np.array([c[2] for c in CONFIGS], np.int32)
    g.maker_off = np.full(n, MAKER_OFF, np.float32)
    g.order_ttl = np.full(n, 4, np.int16)
    g.session = np.zeros(n, np.int8)
    g.loss_react = np.zeros(n, np.int8)
    g.cooldown = np.zeros(n, np.int16)
    g.revenge = np.ones(n, np.float32)
    g.reentry_gap = np.full(n, 4, np.int16)
    return g


def _track_pair(pair: str, data_root: str, taker_bps: float, maker_bps: float) -> dict:
    from .run import _load, resolve_pair

    _f5, f15, metrics, funding = resolve_pair(pair, root=data_root, derivs=True)
    df = _load(f15, "2021-01-01", metrics, funding)
    F, names = features.compute_features(df, 96)
    if "funding" not in names:
        raise SystemExit(f"{pair}: funding feature missing — metrics/funding not merged")
    fi = names.index("funding")
    ts = df["timestamp"].to_numpy(np.int64)
    c = df["close"].to_numpy(np.float64)
    bar_days = df["dt"].dt.tz_convert(None).dt.floor("D").to_numpy().astype("datetime64[D]")
    all_days = np.unique(bar_days)
    day_pos = np.searchsorted(all_days, bar_days).astype(np.int64)

    # melt-up gate per day: yesterday's close above SMA200 of closes through
    # yesterday -> no new shorts today (open positions run their course)
    day_close = np.full(len(all_days), np.nan)
    day_close[day_pos] = c
    sma = pd.Series(day_close).rolling(SMA_DAYS, min_periods=SMA_DAYS).mean().to_numpy()
    above = np.zeros(len(all_days), bool)
    above[1:] = day_close[:-1] > sma[:-1]
    gate_bar = above[day_pos]           # True -> mask signal on this bar

    base = {"o": df["open"].to_numpy(np.float64), "h": df["high"].to_numpy(np.float64),
            "l": df["low"].to_numpy(np.float64), "c": c,
            "atr": features.atr(df), "hour": df["dt"].dt.hour.to_numpy(np.int64),
            "fund": df["fund_pay"].to_numpy(np.float64) if "fund_pay" in df.columns
            else np.zeros(len(df))}
    F_gated = F.copy()
    F_gated[gate_bar, fi] = np.nan

    g = _grid(names, fi)
    cfg = {"taker_bps": taker_bps, "maker_bps": maker_bps,
           "start_capital": 10_000.0, "ruin_frac": 0.30, "seed": 0}
    last_year = int(str(all_days[-1])[:4])

    # per-year sims (thresholds from pre-year history), stitched afterwards
    eqs = {"open": [], "gated": []}
    stats = {"open": np.zeros((g.n, 3)), "gated": np.zeros((g.n, 3))}
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

    return {"pair": pair, "days": days, "curves": curves, "stats": stats,
            "gated_today": bool(above[-1]), "data_through": str(days[-1])}


def _fwd_metrics(cv: np.ndarray, months: np.ndarray) -> dict:
    cv = cv / cv[0]
    r = np.diff(cv) / cv[:-1]
    return {
        "ret_pct": round(float(cv[-1] - 1) * 100, 2),
        "sharpe": round(float(np.mean(r) / np.std(r) * np.sqrt(365)), 2)
                  if np.std(r) > 0 else None,
        "maxdd_pct": round(float((cv / np.maximum.accumulate(cv) - 1).min()) * 100, 2),
        "monthly": {str(mo): round(float(cv[months == mo][-1] / cv[months == mo][0] - 1) * 100, 2)
                    for mo in np.unique(months)},
    }


def cmd_track_funding(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = [_track_pair(p, args.data_root, args.taker_bps, args.maker_bps)
               for p in PAIRS]

    fwd_days = min(int((r["days"] >= TRACK_START).sum()) for r in results)
    if fwd_days < 2:
        msg = (f"# funding-fade forward tracking\n\nSpec frozen 2026-07-19. Data "
               f"through {results[0]['data_through']} — fewer than 2 forward days "
               f"yet; the record begins accumulating from the next data refresh.\n\n"
               + "\n".join(f"- {r['pair']}: melt-up gate currently "
                           f"{'ON (above SMA200 — gated book flat)' if r['gated_today'] else 'off'}"
                           for r in results))
        (out_dir / "funding_fade_forward.md").write_text(msg)
        print(msg)
        return

    payload = {"generated": datetime.now().isoformat(timespec="seconds"),
               "spec_frozen": "2026-07-19", "track_start": str(TRACK_START),
               "gate": f"no new shorts while daily close > SMA{SMA_DAYS}",
               "entry": f"funding > q{ENTRY_Q} (pre-year-history quantiles)",
               "pairs": []}
    lines = [f"# funding-fade forward tracking — generated {payload['generated']}", ""]
    for res in results:
        days = res["days"]
        fwd = days >= TRACK_START
        months = days[fwd].astype("datetime64[M]")
        prec = {"pair": res["pair"], "data_through": res["data_through"],
                "forward_days": int(fwd.sum()), "configs": [], "ensemble": {}}
        for ci, (stop, tp, hold) in enumerate(CONFIGS):
            rec = {"config": f"stop{stop} tp{tp} hold{hold}"}
            for k in ("open", "gated"):
                rec[k] = _fwd_metrics(res["curves"][k][ci][fwd], months)
            rec["trades"] = int(res["stats"]["open"][ci, 0])
            rec["fees_usd"] = round(float(res["stats"]["open"][ci, 1]), 2)
            rec["funding_usd"] = round(float(res["stats"]["open"][ci, 2]), 2)
            prec["configs"].append(rec)
        # headline: equal-initial-weight ensemble = mean of normalized curves
        for k in ("open", "gated"):
            cvs = res["curves"][k][:, fwd]
            ens = (cvs / cvs[:, :1]).mean(axis=0)
            prec["ensemble"][k] = _fwd_metrics(ens, months)
            prec["ensemble"][f"{k}_pos_configs"] = int(sum(
                c[k]["ret_pct"] > 0 for c in prec["configs"]))
        payload["pairs"].append(prec)

        e = prec["ensemble"]
        lines += [f"## {res['pair']} — tracking from {TRACK_START}, data through "
                  f"{prec['data_through']}, **{prec['forward_days']} forward days**", "",
                  f"**Ensemble (18 configs, equal initial weight):** "
                  f"ret {e['open']['ret_pct']}% | S {e['open']['sharpe']} | "
                  f"maxDD {e['open']['maxdd_pct']}% | {e['open_pos_configs']}/18 configs "
                  f"positive — gated: ret {e['gated']['ret_pct']}% | S "
                  f"{e['gated']['sharpe']} | maxDD {e['gated']['maxdd_pct']}%", "",
                  "| config | ret | S | maxDD | gated ret | gated S | trades | fees | funding |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in prec["configs"]:
            lines.append(f"| {r['config']} | {r['open']['ret_pct']}% | {r['open']['sharpe']} | "
                         f"{r['open']['maxdd_pct']}% | {r['gated']['ret_pct']}% | "
                         f"{r['gated']['sharpe']} | {r['trades']} | ${r['fees_usd']} | "
                         f"${r['funding_usd']} |")
        mos = sorted(e["open"]["monthly"])
        lines += ["", "| month | ensemble | gated |", "|---|---|---|"]
        for mo in mos:
            lines.append(f"| {mo} | {e['open']['monthly'][mo]}% | "
                         f"{e['gated']['monthly'].get(mo, '—')}% |")
        lines.append("")

    (out_dir / "funding_fade_forward.json").write_text(json.dumps(payload, indent=2))
    (out_dir / "funding_fade_forward.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"artifacts -> {out_dir}/")


def add_parser(sub) -> None:
    t = sub.add_parser("track-funding",
                       help="rescore the pre-registered funding-fade configs (forward OOS)")
    t.add_argument("--data-root", default="data")
    t.add_argument("--taker-bps", type=float, default=5.0)
    t.add_argument("--maker-bps", type=float, default=1.0)
    t.add_argument("--out", default="reports/tracking")
    t.set_defaults(func=cmd_track_funding)
