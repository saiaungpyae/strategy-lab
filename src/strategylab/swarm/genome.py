"""Genome sampling — each bot is a tiny hypothesis plus a behavioral personality.

Perception: 1-3 rules, each (feature, > or <, threshold-as-train-quantile, ±dir).
Behavior:   sizing, stops, take-profit, hold horizon, entry patience
            (maker limit vs taker chase), session filter, post-loss reaction,
            re-entry gap. Nothing is hand-written; everything is sampled from
            a per-run seed, and every bot is reproducible from (seed, bot_id).

A slice of the population is the random placebo group ("control"): identical
behavioral genome, but its entry signals are coin flips at a sampled frequency.
It exists so the recap can prove pattern bots beat luck out-of-sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MAX_RULES = 3

SESSIONS = {0: "any", 1: "US", 2: "EU", 3: "Asia"}
# UTC hours [start, end)
SESSION_HOURS = {0: (0, 24), 1: (13, 21), 2: (7, 15), 3: (0, 8)}
LOSS_REACT = {0: "neutral", 1: "cooldown", 2: "revenge"}
DIR_BIAS = {-1: "short-only", 0: "both", 1: "long-only"}


@dataclass
class Genomes:
    """Column-oriented genome table for the whole swarm (n = n_bots)."""

    seed: int
    feature_names: list[str]
    # perception
    is_control: np.ndarray = field(default=None)  # bool
    n_rules: np.ndarray = field(default=None)     # int8 1..3
    rule_feat: np.ndarray = field(default=None)   # int16 [n, MAX_RULES]
    rule_op: np.ndarray = field(default=None)     # int8 +1 '>' / -1 '<'
    rule_q: np.ndarray = field(default=None)      # float32 threshold quantile
    rule_dir: np.ndarray = field(default=None)    # int8 ±1
    ctrl_rate: np.ndarray = field(default=None)   # float32 entry prob/bar (control)
    # behavior
    tf: np.ndarray = field(default=None)          # int8 0=5m 1=15m
    dir_bias: np.ndarray = field(default=None)    # int8 -1/0/+1
    risk_pct: np.ndarray = field(default=None)    # float32 fraction of equity
    stop_atr: np.ndarray = field(default=None)    # float32 ATR multiples
    tp_rr: np.ndarray = field(default=None)       # float32 R:R, nan = time exit
    max_hold: np.ndarray = field(default=None)    # int32 bars (per own tf)
    maker_off: np.ndarray = field(default=None)   # float32 ATR offset, 0 = taker
    order_ttl: np.ndarray = field(default=None)   # int16 bars a limit lives
    session: np.ndarray = field(default=None)     # int8 key of SESSIONS
    loss_react: np.ndarray = field(default=None)  # int8 key of LOSS_REACT
    cooldown: np.ndarray = field(default=None)    # int16 bars (loss_react==1)
    revenge: np.ndarray = field(default=None)     # float32 mult (loss_react==2)
    reentry_gap: np.ndarray = field(default=None) # int16 bars between trades

    @property
    def n(self) -> int:
        return len(self.tf)


def sample(n_bots: int, n_features: int, feature_names: list[str],
           control_frac: float, seed: int, maker_only: bool = False) -> Genomes:
    rng = np.random.default_rng(seed)
    g = Genomes(seed=seed, feature_names=feature_names)

    n_ctrl = int(round(n_bots * control_frac))
    is_control = np.zeros(n_bots, dtype=bool)
    is_control[rng.choice(n_bots, size=n_ctrl, replace=False)] = True
    g.is_control = is_control

    # perception
    g.n_rules = rng.choice([1, 2, 3], size=n_bots, p=[0.5, 0.3, 0.2]).astype(np.int8)
    g.rule_feat = rng.integers(0, n_features, size=(n_bots, MAX_RULES)).astype(np.int16)
    g.rule_op = rng.choice([1, -1], size=(n_bots, MAX_RULES)).astype(np.int8)
    g.rule_q = rng.uniform(0.10, 0.90, size=(n_bots, MAX_RULES)).astype(np.float32)
    g.rule_dir = rng.choice([1, -1], size=(n_bots, MAX_RULES)).astype(np.int8)
    g.ctrl_rate = np.exp(rng.uniform(np.log(0.005), np.log(0.08), n_bots)).astype(np.float32)

    # behavior
    g.tf = rng.choice([0, 1], size=n_bots, p=[0.6, 0.4]).astype(np.int8)
    g.dir_bias = rng.choice([-1, 0, 1], size=n_bots, p=[0.2, 0.6, 0.2]).astype(np.int8)
    g.risk_pct = np.exp(rng.uniform(np.log(0.001), np.log(0.03), n_bots)).astype(np.float32)
    g.stop_atr = rng.uniform(0.5, 5.0, n_bots).astype(np.float32)
    tp = rng.uniform(0.5, 4.0, n_bots).astype(np.float32)
    tp[rng.random(n_bots) < 0.3] = np.nan  # time-exit ideologies
    g.tp_rr = tp
    hold_hours = np.exp(rng.uniform(np.log(0.25), np.log(48.0), n_bots))
    bars_per_hour = np.where(g.tf == 0, 12, 4)
    g.max_hold = np.maximum(1, (hold_hours * bars_per_hour)).astype(np.int32)
    off = rng.uniform(0.05, 1.0, n_bots).astype(np.float32)
    if not maker_only:
        off[rng.random(n_bots) < 0.35] = 0.0  # taker (chase) ideologies
    g.maker_off = off  # maker_only: every entry rests as a limit; stops still exit taker
    g.order_ttl = rng.integers(1, 7, n_bots).astype(np.int16)
    g.session = rng.choice([0, 1, 2, 3], size=n_bots, p=[0.55, 0.15, 0.15, 0.15]).astype(np.int8)
    g.loss_react = rng.choice([0, 1, 2], size=n_bots, p=[0.5, 0.3, 0.2]).astype(np.int8)
    g.cooldown = rng.integers(3, 31, n_bots).astype(np.int16)
    g.cooldown[g.loss_react != 1] = 0
    g.revenge = rng.uniform(1.25, 2.0, n_bots).astype(np.float32)
    g.revenge[g.loss_react != 2] = 1.0
    g.reentry_gap = rng.integers(0, 13, n_bots).astype(np.int16)
    return g


_FIELDS = ["is_control", "n_rules", "rule_feat", "rule_op", "rule_q", "rule_dir",
           "ctrl_rate", "tf", "dir_bias", "risk_pct", "stop_atr", "tp_rr",
           "max_hold", "maker_off", "order_ttl", "session", "loss_react",
           "cooldown", "revenge", "reentry_gap"]


def subset(g: Genomes, idx: np.ndarray) -> Genomes:
    out = Genomes(seed=g.seed, feature_names=g.feature_names)
    for f in _FIELDS:
        setattr(out, f, getattr(g, f)[idx].copy())
    return out


def concat(parts: list[Genomes]) -> Genomes:
    out = Genomes(seed=parts[0].seed, feature_names=parts[0].feature_names)
    for f in _FIELDS:
        setattr(out, f, np.concatenate([getattr(p, f) for p in parts]))
    return out


def breed(g: Genomes, parents: np.ndarray, n_children: int, n_features: int,
          rng: np.random.Generator, maker_only: bool = False) -> Genomes:
    """Uniform crossover of two random parents per child, then mutation.

    Numeric traits get multiplicative/additive noise clipped to the sampler's
    ranges; categoricals occasionally resample; rules mix per-slot and can be
    fully resampled — the same capacity caps as birth, so evolution can't
    inflate genome complexity.
    """
    pa = parents[rng.integers(0, len(parents), n_children)]
    pb = parents[rng.integers(0, len(parents), n_children)]
    child = Genomes(seed=g.seed, feature_names=g.feature_names)

    def mix(field):
        take_a = rng.random(n_children) < 0.5
        return np.where(take_a, getattr(g, field)[pa], getattr(g, field)[pb])

    for f in ("tf", "dir_bias", "session", "loss_react"):
        setattr(child, f, mix(f).astype(getattr(g, f).dtype))
    for f in ("risk_pct", "stop_atr", "tp_rr", "maker_off", "revenge", "ctrl_rate"):
        setattr(child, f, mix(f).astype(np.float32))
    for f in ("max_hold", "order_ttl", "cooldown", "reentry_gap", "n_rules"):
        setattr(child, f, mix(f).astype(getattr(g, f).dtype))
    child.is_control = np.zeros(n_children, dtype=bool)

    # rules: per-slot uniform crossover (all four arrays share the slot mask)
    slot_a = rng.random((n_children, MAX_RULES)) < 0.5
    for f in ("rule_feat", "rule_op", "rule_q", "rule_dir"):
        arr = np.where(slot_a, getattr(g, f)[pa], getattr(g, f)[pb])
        setattr(child, f, arr.astype(getattr(g, f).dtype))

    # --- mutation ---------------------------------------------------------
    child.risk_pct = np.clip(child.risk_pct * np.exp(rng.normal(0, 0.2, n_children)),
                             0.001, 0.03).astype(np.float32)
    child.stop_atr = np.clip(child.stop_atr + rng.normal(0, 0.4, n_children),
                             0.5, 5.0).astype(np.float32)
    with np.errstate(invalid="ignore"):
        child.tp_rr = np.clip(child.tp_rr + rng.normal(0, 0.3, n_children),
                              0.5, 4.0).astype(np.float32)  # nan stays nan
    child.max_hold = np.maximum(1, (child.max_hold *
                     np.exp(rng.normal(0, 0.25, n_children))).astype(np.int32))
    lo_off = 0.05 if maker_only else 0.0
    child.maker_off = np.clip(child.maker_off + rng.normal(0, 0.1, n_children),
                              lo_off, 1.0).astype(np.float32)
    child.order_ttl = np.clip(child.order_ttl + rng.integers(-1, 2, n_children),
                              1, 6).astype(np.int16)
    child.cooldown = np.clip(child.cooldown + rng.integers(-3, 4, n_children),
                             0, 30).astype(np.int16)
    child.revenge = np.clip(child.revenge * np.exp(rng.normal(0, 0.1, n_children)),
                            1.0, 2.0).astype(np.float32)
    child.reentry_gap = np.clip(child.reentry_gap + rng.integers(-2, 3, n_children),
                                0, 12).astype(np.int16)
    for f, p in (("tf", 0.05), ("dir_bias", 0.08), ("session", 0.08), ("loss_react", 0.08)):
        m = rng.random(n_children) < p
        vals = {"tf": [0, 1], "dir_bias": [-1, 0, 1], "session": [0, 1, 2, 3],
                "loss_react": [0, 1, 2]}[f]
        arr = getattr(child, f)
        arr[m] = rng.choice(vals, m.sum())
    # rule mutations
    m = rng.random(n_children) < 0.25          # resample one whole rule slot
    who = np.flatnonzero(m)
    slot = rng.integers(0, MAX_RULES, len(who))
    child.rule_feat[who, slot] = rng.integers(0, n_features, len(who))
    child.rule_op[who, slot] = rng.choice([1, -1], len(who))
    child.rule_q[who, slot] = rng.uniform(0.10, 0.90, len(who)).astype(np.float32)
    child.rule_dir[who, slot] = rng.choice([1, -1], len(who))
    m = rng.random(n_children) < 0.15          # nudge one threshold
    who = np.flatnonzero(m)
    slot = rng.integers(0, MAX_RULES, len(who))
    child.rule_q[who, slot] = np.clip(child.rule_q[who, slot]
                                      + rng.normal(0, 0.1, len(who)), 0.05, 0.95)
    m = rng.random(n_children) < 0.10          # grow/shrink rule count
    child.n_rules[m] = np.clip(child.n_rules[m] + rng.choice([-1, 1], m.sum()),
                               1, MAX_RULES).astype(np.int8)
    return child


def rule_text(g: Genomes, i: int) -> str:
    """Human-readable ideology, e.g. 'rsi < q0.23 → LONG & oi_chg > q0.80 → SHORT'."""
    if g.is_control[i]:
        return f"RANDOM (p={g.ctrl_rate[i]:.3f}/bar)"
    parts = []
    for r in range(g.n_rules[i]):
        f = g.feature_names[g.rule_feat[i, r]]
        op = ">" if g.rule_op[i, r] > 0 else "<"
        d = "LONG" if g.rule_dir[i, r] > 0 else "SHORT"
        parts.append(f"{f} {op} q{g.rule_q[i, r]:.2f} → {d}")
    return "  &  ".join(parts)


def to_frame(g: Genomes) -> pd.DataFrame:
    """Flat per-bot table for the run artifact (genomes.csv)."""
    n = g.n
    rows = {
        "bot_id": np.arange(n),
        "is_control": g.is_control,
        "tf": np.where(g.tf == 0, "5m", "15m"),
        "rules": [rule_text(g, i) for i in range(n)],
        "n_rules": g.n_rules,
        "dir_bias": [DIR_BIAS[int(x)] for x in g.dir_bias],
        "risk_pct": np.round(g.risk_pct * 100, 3),
        "stop_atr": np.round(g.stop_atr, 2),
        "tp_rr": np.round(g.tp_rr, 2),
        "max_hold_bars": g.max_hold,
        "maker_off_atr": np.round(g.maker_off, 2),
        "order_ttl": g.order_ttl,
        "session": [SESSIONS[int(x)] for x in g.session],
        "loss_react": [LOSS_REACT[int(x)] for x in g.loss_react],
        "cooldown_bars": g.cooldown,
        "revenge_mult": np.round(g.revenge, 2),
        "reentry_gap": g.reentry_gap,
    }
    return pd.DataFrame(rows)
