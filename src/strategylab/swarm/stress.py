"""Hostile-future stress forecast for a single hall-of-fame bot.

Generates a battery of synthetic 90-day price paths engineered to hurt
(flash crash, melt-up, bear grind, stop-hunt whipsaw, vol collapse, V-shape,
jammed derivative gates), replays the bot's exact genome through the real
engine on each, and returns the daily equity curves. The last 90 real days
are included as the "current season" anchor.

Synthetic tapes keep the run's exact feature schema: OHLC is engineered,
while volume / taker flow / funding / positioning columns are replayed from
the most recent 90 real days, so the bot's derivative gates see the same
regime they trade today. Rule thresholds come from the run's full real
history — exactly what the bot would use live. The `gate_flip` scenario then
overwrites the derivative features the bot's own rules read, pinning them at
the value that force-fires every rule while price whipsaws — the level-drift
failure mode, weaponized.

Everything is seeded by (bot_id, scenario), so the forecast is deterministic
and cacheable.
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from . import engine, features, genome
from .evolve import _alloc, _fill_eq
from .genome import DIR_BIAS, LOSS_REACT, MAX_RULES, SESSIONS
from .trace import parse_record

DAYS = 90
DERIV_FEATS = ("funding", "oi_chg", "top_ls_pos", "global_ls", "taker_ls")

# (key, label, description, phases | None) — phases are (days, mu/day, vol×)
# on log-price; `sine` adds a deterministic oscillation (amplitude, period d).
SCENARIOS: list[dict] = [
    {"key": "replay", "label": "last 90 days (anchor)",
     "desc": "the most recent 90 real days, replayed — the season the bot looks good in"},
    {"key": "crash", "label": "flash crash −45%",
     "desc": "10 quiet days, −45% in 8 days at 3.5× vol, then 2× vol aftershock chop",
     "phases": [(10, 0.0, 1.0), (8, -0.072, 3.5), (72, 0.002, 2.0)]},
    {"key": "meltup", "label": "melt-up +80% → purge",
     "desc": "parabolic +80% grind for 8 weeks, then a −25% purge — kills shorts and fades",
     "phases": [(55, 0.0108, 1.2), (10, -0.028, 2.5), (25, 0.0, 1.5)]},
    {"key": "grind", "label": "bear grind −35%",
     "desc": "−0.5%/day for 3 months at half vol — no capitulation to buy, no bounce to sell",
     "phases": [(90, -0.0048, 0.55)]},
    {"key": "whipsaw", "label": "stop-hunt whipsaw",
     "desc": "±6% oscillation every ~2.5 days, zero net drift, 1.6× vol — pure stop harvesting",
     "phases": [(90, 0.0, 1.6)], "sine": (0.06, 2.5)},
    {"key": "flatline", "label": "vol collapse",
     "desc": "volatility dies to 15% of normal — nothing trends, costs still tick",
     "phases": [(90, 0.0, 0.15)]},
    {"key": "vshape", "label": "V-crash & recover",
     "desc": "−35% in 8 days, full recovery over the next month, then calm",
     "phases": [(8, -0.055, 3.0), (30, 0.015, 1.5), (52, 0.0, 1.0)]},
    {"key": "gate_flip", "label": "gates jammed open",
     "desc": "mild whipsaw price, but every derivative gate in the bot's rules is pinned "
             "to force-fire — max activity in a hostile chop (the level-drift failure mode)",
     "phases": [(90, 0.0, 1.2)], "sine": (0.04, 3.0)},
]

_SESS_INV = {v: k for k, v in SESSIONS.items()}
_LOSS_INV = {v: k for k, v in LOSS_REACT.items()}
_BIAS_INV = {v: k for k, v in DIR_BIAS.items()}


def _genome_from_record(rec: dict, names: list[str]) -> genome.Genomes:
    """1-row Genomes built from a hof_history record (via trace.parse_record,
    so the rules string is validated against the rebuilt feature list)."""
    p = parse_record(rec, names)
    nr = p["rule_feat"].shape[1]
    g = genome.Genomes(seed=0, feature_names=names)
    g.is_control = np.zeros(1, bool)
    g.n_rules = np.array([nr], np.int8)
    g.rule_feat = np.zeros((1, MAX_RULES), np.int16)
    g.rule_op = np.ones((1, MAX_RULES), np.int8)
    g.rule_q = np.full((1, MAX_RULES), 0.5, np.float32)
    g.rule_dir = np.ones((1, MAX_RULES), np.int8)
    for r in range(nr):
        g.rule_feat[0, r] = p["rule_feat"][0, r]
        g.rule_op[0, r] = 1 if p["rule_op_gt"][r] else -1
        g.rule_q[0, r] = p["rule_q"][0, r]
        g.rule_dir[0, r] = p["rule_dir"][r]
    g.ctrl_rate = np.full(1, 0.01, np.float32)
    g.tf = np.array([0 if rec.get("tf") == "5m" else 1], np.int8)
    g.dir_bias = np.array([p["dir_bias"]], np.int8)
    g.risk_pct = np.array([p["risk"]], np.float32)
    g.stop_atr = np.array([p["stop_atr"]], np.float32)
    g.tp_rr = np.array([p["tp_rr"]], np.float32)
    g.max_hold = np.array([p["max_hold"]], np.int32)
    g.maker_off = np.array([p["maker_off"]], np.float32)
    g.order_ttl = np.array([p["order_ttl"]], np.int16)
    g.session = np.array([_SESS_INV[rec.get("session") or "any"]], np.int8)
    g.loss_react = np.array([p["loss_react"]], np.int8)
    g.cooldown = np.array([p["cooldown"]], np.int16)
    g.revenge = np.array([p["revenge"]], np.float32)
    g.reentry_gap = np.array([p["reentry_gap"]], np.int16)
    return g


def _synthetic_ohlc(last_close: float, sig_day: float, bpd: int, sc: dict,
                    rng: np.random.Generator) -> pd.DataFrame:
    """Engineered log-price path -> per-bar OHLC. Vol is calibrated to the
    asset's recent daily return std so 1.0× means 'normal for this market'."""
    n = DAYS * bpd
    mu = np.zeros(n)
    sig = np.zeros(n)
    i = 0
    for days, mu_d, vmult in sc["phases"]:
        j = min(n, i + days * bpd)
        mu[i:j] = mu_d / bpd
        sig[i:j] = max(sig_day, 1e-4) * vmult / np.sqrt(bpd)
        i = j
    if i < n:
        mu[i:] = 0.0
        sig[i:] = max(sig_day, 1e-4) / np.sqrt(bpd)
    r = mu + sig * rng.standard_normal(n)
    if "sine" in sc:
        amp, period_d = sc["sine"]
        t = np.arange(n)
        # derivative of amp*sin(...) so the *price* oscillates with ±amp
        r = r + np.diff(amp * np.sin(2 * np.pi * t / (period_d * bpd)),
                        prepend=0.0)
    c = last_close * np.exp(np.cumsum(r))
    o = np.empty(n)
    o[0] = last_close
    o[1:] = c[:-1]
    wick = (np.abs(r) + 0.6 * sig) * rng.uniform(0.3, 1.0, n)
    h = np.maximum(o, c) * np.exp(wick * 0.5)
    l = np.minimum(o, c) * np.exp(-wick * 0.5)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c})


def stress_bot(rec: dict, names: list[str], df: pd.DataFrame, F_full: np.ndarray,
               qs: np.ndarray, cfg: dict, bars_per_day: int, tf_code: int) -> list[dict]:
    """Run the full scenario battery for one bot. `df` is the bot's-timeframe
    real dataframe (as loaded for the run), `F_full` its feature matrix."""
    g = _genome_from_record(rec, names)
    Q = features.train_quantiles(F_full, len(F_full), qs)
    thr = engine._interp_thresholds(
        g.rule_feat[:, : int(g.n_rules[0])].astype(np.int64),
        g.rule_q[:, : int(g.n_rules[0])].astype(np.float64), Q, qs)[0]

    pad = engine.WARMUP
    n_syn = DAYS * bars_per_day
    tail = df.tail(pad).reset_index(drop=True)
    recent = df.tail(n_syn).reset_index(drop=True)
    last_close = float(df["close"].iloc[-1])
    step_ms = int(np.median(np.diff(df["timestamp"].tail(50).to_numpy(np.int64))))
    # recent daily vol for calibration
    day_close = df["close"].groupby(
        df["dt"].dt.tz_convert(None).dt.floor("D").values).last()
    sig_day = float(day_close.tail(91).pct_change().std())

    replay_cols = [c for c in
                   ("volume", "taker_buy_volume", "funding", "fund_pay", "oi",
                    "top_ls_pos", "global_ls", "taker_ls")
                   if c in df.columns]
    results = []
    for sc in SCENARIOS:
        if sc["key"] == "replay":
            df_syn = df.tail(pad + n_syn).reset_index(drop=True)
        else:
            rng = np.random.default_rng(
                (int(rec["bot_id"]) * 1_000_003 + zlib.crc32(sc["key"].encode())) & 0x7FFFFFFF)
            ohlc = _synthetic_ohlc(last_close, sig_day, bars_per_day, sc, rng)
            syn = ohlc.copy()
            ts0 = int(df["timestamp"].iloc[-1])
            syn["timestamp"] = ts0 + step_ms * (np.arange(n_syn) + 1)
            syn["dt"] = pd.to_datetime(syn["timestamp"], unit="ms", utc=True)
            for c in replay_cols:  # regime columns: replay the current season
                syn[c] = recent[c].to_numpy()
            keep = ["timestamp", "dt", "open", "high", "low", "close"] + replay_cols
            df_syn = pd.concat([tail[keep], syn[keep]], ignore_index=True)

        F_syn, syn_names = features.compute_features(df_syn, bars_per_day)
        if syn_names != names:  # schema drift would silently misread rules
            raise ValueError("synthetic feature schema mismatch")
        if sc["key"] == "gate_flip":
            nr = int(g.n_rules[0])
            for r in range(nr):
                fname = names[int(g.rule_feat[0, r])]
                if fname not in DERIV_FEATS:
                    continue
                # pin just past the threshold on the side that fires the rule
                margin = 0.05 * (np.nanstd(F_full[:, int(g.rule_feat[0, r])]) or 1.0)
                pin = thr[r] + (margin if g.rule_op[0, r] > 0 else -margin)
                F_syn[pad:, int(g.rule_feat[0, r])] = pin

        bar_days = df_syn["dt"].dt.tz_convert(None).dt.floor("D") \
                                .to_numpy().astype("datetime64[D]")
        all_days = np.unique(bar_days)
        w = {"tf_code": tf_code,
             "o": df_syn["open"].to_numpy(np.float64),
             "h": df_syn["high"].to_numpy(np.float64),
             "l": df_syn["low"].to_numpy(np.float64),
             "c": df_syn["close"].to_numpy(np.float64),
             "atr": features.atr(df_syn),
             "hour": df_syn["dt"].dt.hour.to_numpy(np.int64),
             "day_pos": np.searchsorted(all_days, bar_days).astype(np.int64),
             "seg_b": np.ones(len(df_syn), bool), "F": F_syn, "Q": Q, "qs": qs,
             "fund": (df_syn["fund_pay"].to_numpy(np.float64)
                      if "fund_pay" in df_syn.columns else np.zeros(len(df_syn)))}
        out = _alloc(1, len(all_days))
        engine.run_cohort(w, g, np.arange(1), cfg, out)
        d0 = int(w["day_pos"][pad])
        eq = _fill_eq(out, d0, len(all_days), cfg["start_capital"])[0]
        eq = eq / eq[0]
        dd = float((eq / np.maximum.accumulate(eq) - 1.0).min()) * 100.0
        px = df_syn["close"].to_numpy()[pad:]
        # daily close of the scenario price, normalized, on the same day axis
        dp = w["day_pos"][pad:]
        px_day = np.array([px[dp == d][-1] for d in range(d0, len(all_days))]) / px[0]
        results.append({
            "key": sc["key"], "label": sc["label"], "desc": sc["desc"],
            "eq": [round(float(x), 4) for x in eq],
            "px": [round(float(x), 4) for x in px_day],
            "ret_pct": round(float(eq[-1] - 1.0) * 100.0, 1),
            "maxdd_pct": round(dd, 1),
            "trades": int(out["trades_a"][0] + out["trades_b"][0]),
            "dead": bool(out["dead"][0]),
            "fees": round(float(out["fees"][0]), 0),
        })
    return results
