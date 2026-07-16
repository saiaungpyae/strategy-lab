"""Recap — turn raw swarm output into the honest statistics the dashboard shows.

Everything selection-flavored is computed on the TEST segment; the train
segment exists to demonstrate how much of it was luck. The random control
group is the placebo: its numbers define "what luck looks like here."
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_TRADES = 8          # per segment, to be rankable
MIN_TRADES_TRAIT = 5    # test-segment floor for trait analysis

NUMERIC_TRAITS = ["risk_pct", "stop_atr", "tp_rr", "max_hold_bars", "maker_off_atr",
                  "order_ttl", "cooldown_bars", "revenge_mult", "reentry_gap", "n_rules"]
CAT_TRAITS = ["tf", "session", "dir_bias", "loss_react"]


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.arange(len(x))
    return r / max(len(x) - 1, 1)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 20:
        return float("nan")
    ra, rb = _rank(a[ok]), _rank(b[ok])
    sa, sb = ra.std(), rb.std()
    if sa == 0 or sb == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def seg_metrics(daily: np.ndarray, split_day: int, start_cap: float) -> dict:
    """Per-bot metrics from the daily mark-to-market equity matrix."""
    d = daily.astype(np.float64).copy()
    # warmup / sparse days: forward-fill, seed with starting capital
    for j in range(1, d.shape[1]):
        col = d[:, j]
        prev = d[:, j - 1]
        col[~np.isfinite(col)] = prev[~np.isfinite(col)]
    d[:, 0][~np.isfinite(d[:, 0])] = start_cap
    for j in range(1, d.shape[1]):  # second pass now that col0 is seeded
        col = d[:, j]
        col[~np.isfinite(col)] = d[:, j - 1][~np.isfinite(col)]

    eq_a, eq_b = d[:, :split_day + 1], d[:, split_day:]

    def sharpe(eq):
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.diff(eq, axis=1) / np.where(eq[:, :-1] == 0, np.nan, eq[:, :-1])
            mu = np.nanmean(r, axis=1)
            sd = np.nanstd(r, axis=1)
            return np.where(sd > 0, mu / sd * np.sqrt(365.0), np.nan)

    def maxdd(eq):
        peak = np.maximum.accumulate(eq, axis=1)
        dd = eq / np.where(peak == 0, np.nan, peak) - 1.0
        return np.nanmin(dd, axis=1)

    return {
        "daily_filled": d.astype(np.float32),
        "ret_a": eq_a[:, -1] / start_cap - 1.0,
        "ret_b": eq_b[:, -1] / np.where(eq_b[:, 0] == 0, np.nan, eq_b[:, 0]) - 1.0,
        "sharpe_a": sharpe(eq_a),
        "sharpe_b": sharpe(eq_b),
        "maxdd_b": maxdd(eq_b),
        "maxdd_all": maxdd(d),
        "final_mult": d[:, -1] / start_cap,
    }


def _downsample(xs: np.ndarray, k: int = 220) -> list[int]:
    n = len(xs)
    if n <= k:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, k).astype(int).tolist()))


def regime_slice(daily: np.ndarray, day_close: np.ndarray, split_day: int,
                 pat: np.ndarray, ctl: np.ndarray) -> dict:
    """Median bot daily return (bps/day) per market regime, TEST period only.

    Regimes from trailing 30d B&H return: trend-up > +8%, trend-down < -8%,
    chop in between."""
    px = pd.Series(day_close).ffill().to_numpy()
    ret30 = np.full(len(px), np.nan)
    ret30[30:] = px[30:] / px[:-30] - 1.0
    regime = np.where(ret30 > 0.08, "trend-up",
                      np.where(ret30 < -0.08, "trend-down", "chop"))
    regime[:30] = "chop"
    d = daily.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.diff(d, axis=1) / np.where(d[:, :-1] == 0, np.nan, d[:, :-1])
        bh = np.diff(px) / px[:-1]
    rows = []
    for name in ("trend-up", "chop", "trend-down"):
        m = (regime[1:] == name) & (np.arange(1, d.shape[1]) > split_day)
        if m.sum() < 5:
            continue
        rows.append({
            "regime": name, "days": int(m.sum()),
            "bnh_bps_day": round(float(np.nanmean(bh[m]) * 1e4), 2),
            "pattern_bps_day": round(float(np.nanmedian(
                np.nanmean(r[pat][:, m], axis=1)) * 1e4), 2),
            "control_bps_day": round(float(np.nanmedian(
                np.nanmean(r[ctl][:, m], axis=1)) * 1e4), 2),
        })
    return {"basis": "test period only; regimes = trailing 30d B&H return, ±8%",
            "rows": rows}


def build_recap(res: pd.DataFrame, gdf: pd.DataFrame, daily: np.ndarray,
                days: list[str], split_day: int, cfg: dict, bnh_mult: float,
                day_close: np.ndarray | None = None) -> dict:
    pat = ~res["is_control"].to_numpy()
    ctl = res["is_control"].to_numpy()
    sh_a = res["sharpe_a"].to_numpy(dtype=np.float64)
    sh_b = res["sharpe_b"].to_numpy(dtype=np.float64)
    fm = res["final_mult"].to_numpy(dtype=np.float64)
    ok = ((res["trades_a"] >= MIN_TRADES) & (res["trades_b"] >= MIN_TRADES)).to_numpy()

    # --- tiles ----------------------------------------------------------
    ctl_ok = ctl & ok & np.isfinite(sh_b)
    yard95, yard99 = (float(np.percentile(sh_b[ctl_ok], 95)),
                      float(np.percentile(sh_b[ctl_ok], 99))) if ctl_ok.sum() >= 20 else (None, None)
    rho_pat = spearman(sh_a[pat & ok], sh_b[pat & ok])
    rho_ctl = spearman(sh_a[ctl & ok], sh_b[ctl & ok])
    tiles = {
        "n_bots": int(len(res)),
        "n_control": int(ctl.sum()),
        "alive_pct": float(100.0 * (~res["dead"]).mean()),
        "above_water_pct": float(100.0 * (fm > 1.0).mean()),
        "median_final_mult": float(np.median(fm)),
        "bnh_mult": float(bnh_mult),
        "yardstick_sharpe_p95": yard95,
        "yardstick_sharpe_p99": yard99,
        "rank_corr_pattern": None if np.isnan(rho_pat) else round(rho_pat, 4),
        "rank_corr_control": None if np.isnan(rho_ctl) else round(rho_ctl, 4),
        # Control persistence is NOT ~0: heterogeneous cost drag persists.
        # Skill evidence is the pattern-minus-control GAP, not raw pattern rho.
        "rank_corr_gap": None if (np.isnan(rho_pat) or np.isnan(rho_ctl))
                         else round(rho_pat - rho_ctl, 4),
        "rankable_bots": int(ok.sum()),
    }

    # --- outcome histogram ------------------------------------------------
    hi = float(np.nanpercentile(fm, 99.5))
    lo = float(max(0.0, np.nanpercentile(fm, 0.2)))
    hi = max(hi, 1.05, bnh_mult * 1.02)
    bins = np.linspace(lo, hi, 46)
    hp, _ = np.histogram(np.clip(fm[pat], lo, hi), bins=bins)
    hc, _ = np.histogram(np.clip(fm[ctl], lo, hi), bins=bins)
    ctl_fm = fm[ctl]
    histogram = {
        "bins": [round(float(b), 4) for b in bins],
        "pattern": hp.tolist(), "control": hc.tolist(),
        "refs": {"start": 1.0, "bnh": round(float(bnh_mult), 4),
                 "ctl_p95": round(float(np.percentile(ctl_fm, 95)), 4) if len(ctl_fm) else None,
                 "ctl_p99": round(float(np.percentile(ctl_fm, 99)), 4) if len(ctl_fm) else None},
    }

    # --- survival curves --------------------------------------------------
    sel = _downsample(np.arange(daily.shape[1]))
    death = res["death_day"].to_numpy()
    start_cap = float(cfg["start_capital"])
    surv = {"days": [days[j] for j in sel], "split_day": days[split_day]}
    for name, mask in (("pattern", pat), ("control", ctl)):
        dd = daily[mask][:, sel]
        alive = np.array([(np.where(death[mask] < 0, 10**9, death[mask]) > j).mean() for j in sel])
        water = np.nanmean(dd > start_cap, axis=0)
        q25, q50, q75 = (np.nanpercentile(dd / start_cap, q, axis=0) for q in (25, 50, 75))
        surv[name] = {"alive": np.round(alive * 100, 2).tolist(),
                      "above_water": np.round(water * 100.0, 2).tolist(),
                      "q25": np.round(q25, 4).tolist(),
                      "q50": np.round(q50, 4).tolist(),
                      "q75": np.round(q75, 4).tolist()}

    # --- trait analysis (test segment only, pattern bots only) -----------
    tsel = pat & (res["trades_b"] >= MIN_TRADES_TRAIT).to_numpy() & np.isfinite(sh_b)
    traits = {"n_used": int(tsel.sum()), "numeric": [], "categorical": [], "features": []}
    for tr in NUMERIC_TRAITS:
        v = pd.to_numeric(gdf[tr], errors="coerce").to_numpy(dtype=np.float64)
        rho = spearman(v[tsel], sh_b[tsel])
        if np.isfinite(rho):
            traits["numeric"].append({"trait": tr, "rho": round(rho, 4)})
    traits["numeric"].sort(key=lambda x: -abs(x["rho"]))
    for tr in CAT_TRAITS:
        groups = []
        for val, sub in pd.DataFrame({"v": gdf[tr][tsel], "s": sh_b[tsel]}).groupby("v"):
            if len(sub) >= 15:
                groups.append({"value": str(val), "n": int(len(sub)),
                               "median_sharpe_b": round(float(sub["s"].median()), 4)})
        groups.sort(key=lambda x: -x["median_sharpe_b"])
        traits["categorical"].append({"trait": tr, "groups": groups})
    # feature usage: does watching feature X help, out-of-sample?
    all_sh = sh_b[tsel]
    med_all = float(np.median(all_sh)) if len(all_sh) else float("nan")
    rules = gdf["rules"].astype(str)
    for fname in cfg["feature_names"]:
        uses = rules.str.contains(fname + " ", regex=False).to_numpy() & tsel
        if uses.sum() >= 25:
            med = float(np.median(sh_b[uses]))
            traits["features"].append({"feature": fname, "n": int(uses.sum()),
                                       "median_sharpe_b": round(med, 4),
                                       "delta_vs_all": round(med - med_all, 4)})
    traits["features"].sort(key=lambda x: -x["delta_vs_all"])

    # --- persistence -------------------------------------------------------
    persistence = {"rho_pattern": tiles["rank_corr_pattern"],
                   "rho_control": tiles["rank_corr_control"], "scatter": [],
                   "top_decile_dest": None}
    okp = np.flatnonzero(ok & np.isfinite(sh_a) & np.isfinite(sh_b))
    if len(okp) >= 30:
        ra, rb = _rank(sh_a[okp]), _rank(sh_b[okp])
        keep = okp if len(okp) <= 4000 else np.random.default_rng(0).choice(
            len(okp), 4000, replace=False)
        rows = np.arange(len(okp)) if len(okp) <= 4000 else keep
        persistence["scatter"] = [
            [round(float(ra[i]), 4), round(float(rb[i]), 4), bool(ctl[okp[i]])]
            for i in rows]
        top = ra >= 0.9
        if top.sum() >= 10:
            dest = np.floor(rb[top] * 10).clip(0, 9).astype(int)
            persistence["top_decile_dest"] = np.bincount(dest, minlength=10).tolist()

    # --- leaderboard (entertainment only) ----------------------------------
    lb_ok = np.flatnonzero((res["trades_b"] >= MIN_TRADES).to_numpy() & np.isfinite(sh_b))
    order = lb_ok[np.argsort(-sh_b[lb_ok])][:100]
    leaderboard = []
    for i in order:
        r, gr = res.iloc[i], gdf.iloc[i]
        leaderboard.append({
            "bot_id": int(r["bot_id"]), "control": bool(r["is_control"]),
            "rules": str(gr["rules"]), "tf": str(gr["tf"]), "session": str(gr["session"]),
            "risk_pct": float(gr["risk_pct"]),
            "sharpe_a": None if not np.isfinite(sh_a[i]) else round(float(sh_a[i]), 3),
            "sharpe_b": round(float(sh_b[i]), 3),
            "ret_a": round(float(r["ret_a"]) * 100, 2), "ret_b": round(float(r["ret_b"]) * 100, 2),
            "maxdd_b": round(float(r["maxdd_b"]) * 100, 2),
            "trades": int(r["trades_a"] + r["trades_b"]),
            "expo_b_pct": round(100.0 * float(r["expo_b"]) / max(int(r["bars_b"]), 1), 2),
            "dead": bool(r["dead"]),
        })

    regimes = None
    if day_close is not None:
        regimes = regime_slice(daily, day_close, split_day, pat, ctl)

    return {"tiles": tiles, "histogram": histogram, "survival": surv,
            "traits": traits, "persistence": persistence, "leaderboard": leaderboard,
            "regimes": regimes, "config": cfg}
