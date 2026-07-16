"""Swarm engine — one pass over the tape, vectorized across bots.

Execution semantics (documented simplifications are v1 choices):
- Decisions happen at bar close t; nothing fills before t+1 (no lookahead).
- Taker (chase) entries fill at next bar's open and pay taker bps.
- Maker entries rest a limit at close ∓ offset*ATR for `order_ttl` bars; they
  fill only if price trades through, and pay an adverse-selection edge in bps
  (you get filled precisely when it's slightly bad for you).
- Stops exit as takers at the stop price; take-profits exit as maker limits;
  time exits (max hold) exit as takers at the close.
- If stop and TP are both touched within one bar, the stop wins (conservative).
- Exit checks start the bar AFTER entry (same-bar entry+stop not modeled).
- Sizing: fixed fractional — qty = equity * risk% * revenge_mult / stop_dist,
  capped at 3x notional leverage. Equity below ruin_frac * start => dead.
"""

from __future__ import annotations

import numpy as np

MAX_LEV = 3.0
WARMUP = 300


def _interp_thresholds(rule_feat, rule_q, Q, qs):
    """Resolve per-rule quantile -> value against the train-only quantile grid."""
    Qsel = Q[rule_feat]                              # [n, R, len(qs)]
    pos = (rule_q - qs[0]) / (qs[-1] - qs[0]) * (len(qs) - 1)
    pos = np.clip(pos, 0, len(qs) - 1 - 1e-9)
    lo = pos.astype(np.int64)
    w = pos - lo
    n, R = rule_feat.shape
    ii = np.arange(n)[:, None]
    jj = np.arange(R)[None, :]
    v_lo = Qsel[ii, jj, lo]
    v_hi = Qsel[ii, jj, np.minimum(lo + 1, len(qs) - 1)]
    return (v_lo * (1 - w) + v_hi * w).astype(np.float64)


def run_cohort(mkt, g, idx, cfg, out, progress=None):
    """Simulate the cohort of bots `idx` (indices into genome arrays) on one
    timeframe's market dict, writing per-bot results into `out`.

    mkt keys: o,h,l,c,atr (float64), hour (int 0-23), day_pos (int index into
    the global day grid), seg_b (bool: bar in test segment), F (float32
    [bars x feats]), Q (train quantile grid [feats x len(qs)]), qs, tf_code.
    """
    m = len(idx)
    if m == 0:
        return
    o, h, l, c = mkt["o"], mkt["h"], mkt["l"], mkt["c"]
    A, hour, day_pos, seg_b, F = mkt["atr"], mkt["hour"], mkt["day_pos"], mkt["seg_b"], mkt["F"]
    n_bars = len(c)

    taker = cfg["taker_bps"] / 1e4
    edge = cfg["maker_bps"] / 1e4
    start_cap = float(cfg["start_capital"])
    ruin = cfg["ruin_frac"] * start_cap

    # --- cohort genome slices -------------------------------------------
    is_ctrl = g.is_control[idx]
    ctrl_rate = g.ctrl_rate[idx].astype(np.float64)
    rule_feat = g.rule_feat[idx].astype(np.int64)
    rule_dir = g.rule_dir[idx].astype(np.int64)
    rule_op_gt = g.rule_op[idx] > 0
    rule_act = np.arange(g.rule_feat.shape[1])[None, :] < g.n_rules[idx][:, None]
    thr = _interp_thresholds(rule_feat, g.rule_q[idx].astype(np.float64), mkt["Q"], mkt["qs"])
    dir_bias = g.dir_bias[idx].astype(np.int64)
    risk = g.risk_pct[idx].astype(np.float64)
    stop_atr = g.stop_atr[idx].astype(np.float64)
    tp_rr = g.tp_rr[idx].astype(np.float64)
    max_hold = g.max_hold[idx].astype(np.int64)
    maker_off = g.maker_off[idx].astype(np.float64)
    order_ttl = g.order_ttl[idx].astype(np.int64)
    loss_react = g.loss_react[idx]
    cooldown_len = g.cooldown[idx].astype(np.int64)
    revenge = g.revenge[idx].astype(np.float64)
    reentry_gap = g.reentry_gap[idx].astype(np.int64)

    from .genome import SESSION_HOURS
    sess_tab = np.zeros((4, 24), dtype=bool)
    for k, (s, e) in SESSION_HOURS.items():
        sess_tab[k, s:e] = True
    sess_ok_bot = sess_tab[g.session[idx].astype(np.int64)]  # [m, 24]

    rng = np.random.default_rng(cfg["seed"] * 7919 + int(mkt["tf_code"]))

    # --- state -----------------------------------------------------------
    pos = np.zeros(m, dtype=np.int64)         # -1 / 0 / +1
    entry = np.zeros(m)                       # cost-adjusted entry price
    qty = np.zeros(m)
    stop_px = np.zeros(m)
    tp_px = np.zeros(m)
    held = np.zeros(m, dtype=np.int64)
    pk = np.zeros(m, dtype=np.int64)          # pending: 0 none / 1 market / 2 limit
    pdir = np.zeros(m, dtype=np.int64)
    ppx = np.zeros(m)
    pttl = np.zeros(m, dtype=np.int64)
    cd = np.zeros(m, dtype=np.int64)
    rmult = np.ones(m)
    gap_until = np.zeros(m, dtype=np.int64)
    eq = np.full(m, start_cap, dtype=np.float64)
    dead = np.zeros(m, dtype=bool)

    trades = np.zeros((m, 2), dtype=np.int64)   # col 0=train, 1=test
    wins = np.zeros((m, 2), dtype=np.int64)
    expo = np.zeros((m, 2), dtype=np.int64)
    death_day = np.full(m, -1, dtype=np.int64)

    def open_position(who, raw_px, eff_px, at):
        """who: cohort indices. raw_px: intended price (stop math), eff_px:
        cost-adjusted fill. Direction comes from the pending order."""
        d = pdir[who].astype(np.float64)
        dist = stop_atr[who] * at
        q = eq[who] * risk[who] * rmult[who] / dist
        q = np.minimum(q, MAX_LEV * eq[who] / raw_px)
        pos[who] = pdir[who]
        entry[who] = eff_px
        qty[who] = q
        stop_px[who] = raw_px - d * dist
        rr = tp_rr[who]
        tp_px[who] = np.where(np.isfinite(rr), raw_px + d * rr * dist, d * np.inf)
        held[who] = 0

    def close_positions(who, exit_eff, t):
        seg = 1 if seg_b[t] else 0
        pnl = qty[who] * (exit_eff - entry[who]) * pos[who]
        eq[who] += pnl
        trades[who, seg] += 1
        won = pnl > 0
        wins[who, seg] += won
        w_idx, l_idx = who[won], who[~won]
        rmult[w_idx] = 1.0
        cd[l_idx] = np.where(loss_react[l_idx] == 1, cooldown_len[l_idx], cd[l_idx])
        rmult[l_idx] = np.where(loss_react[l_idx] == 2, revenge[l_idx], 1.0)
        gap_until[who] = t + reentry_gap[who]
        pos[who] = 0
        qty[who] = 0.0
        held[who] = 0
        d = who[eq[who] < ruin]  # ruin line
        dead[d] = True
        death_day[d] = day_pos[t]
        pk[d] = 0

    for t in range(WARMUP, n_bars):
        ot, ht, lt, ct, at = o[t], h[t], l[t], c[t], A[t]

        # -- 1. fill pending orders placed on earlier bars ----------------
        live = (pk > 0) & ~dead & (pos == 0)
        if live.any():
            mk = np.flatnonzero(live & (pk == 1))     # market @ open, taker
            if len(mk):
                open_position(mk, np.full(len(mk), ot),
                              ot * (1 + pdir[mk] * taker), at)
                pk[mk] = 0
            lm = np.flatnonzero(live & (pk == 2))     # resting limits
            if len(lm):
                fill = np.where(pdir[lm] > 0, lt <= ppx[lm], ht >= ppx[lm])
                fl = lm[fill]
                if len(fl):
                    open_position(fl, ppx[fl], ppx[fl] * (1 + pdir[fl] * edge), at)
                    pk[fl] = 0
                rest = lm[~fill]
                pttl[rest] -= 1
                pk[rest[pttl[rest] <= 0]] = 0

        # -- 2. exits ------------------------------------------------------
        act = np.flatnonzero((pos != 0) & (held >= 1))
        if len(act):
            long = pos[act] > 0
            stop_hit = np.where(long, lt <= stop_px[act], ht >= stop_px[act])
            tp_hit = np.where(long, ht >= tp_px[act], lt <= tp_px[act]) & ~stop_hit
            time_hit = (held[act] >= max_hold[act]) & ~stop_hit & ~tp_hit

            s_ = act[stop_hit]
            if len(s_):
                close_positions(s_, stop_px[s_] * (1 - pos[s_] * taker), t)
            p_ = act[tp_hit]
            if len(p_):
                close_positions(p_, tp_px[p_] * (1 - pos[p_] * edge), t)
            x_ = act[time_hit]
            if len(x_):
                close_positions(x_, ct * (1 - pos[x_] * taker), t)

        # -- 3. new entries (decided at close, fill from next bar) --------
        elig = (~dead & (pos == 0) & (pk == 0) & (cd == 0)
                & (t >= gap_until) & sess_ok_bot[:, hour[t]])
        if elig.any():
            des = np.zeros(m, dtype=np.int64)
            pat = elig & ~is_ctrl
            if pat.any():
                vals = F[t][rule_feat]                       # [m, R] float32
                cond = (np.where(rule_op_gt, vals > thr, vals < thr)
                        & np.isfinite(vals) & rule_act)
                sig = (cond * rule_dir).sum(axis=1)
                des[pat] = np.sign(sig[pat]).astype(np.int64)
            ctl = elig & is_ctrl
            if ctl.any():
                u = rng.random(m)
                d2 = np.where(rng.random(m) < 0.5, 1, -1)
                des[ctl] = np.where(u[ctl] < ctrl_rate[ctl], d2[ctl], 0)
            des[(des > 0) & (dir_bias < 0)] = 0   # ideology direction filter
            des[(des < 0) & (dir_bias > 0)] = 0
            go = np.flatnonzero(des != 0)
            if len(go):
                tk = go[maker_off[go] == 0]
                pk[tk], pdir[tk], pttl[tk] = 1, des[tk], 1
                mkr = go[maker_off[go] > 0]
                pk[mkr], pdir[mkr] = 2, des[mkr]
                ppx[mkr] = ct - des[mkr] * maker_off[mkr] * at
                pttl[mkr] = order_ttl[mkr]

        # -- 4. bookkeeping ------------------------------------------------
        np.maximum(cd - 1, 0, out=cd)
        in_pos = pos != 0
        held[in_pos] += 1
        expo[in_pos, 1 if seg_b[t] else 0] += 1

        if t + 1 >= n_bars or day_pos[t + 1] != day_pos[t]:
            out["daily"][idx, day_pos[t]] = eq + qty * (ct - entry) * pos
        if progress is not None and t % 4000 == 0:
            progress(t / n_bars)

    # mark-to-market close of anything still open (taker at last close)
    open_ = np.flatnonzero(pos != 0)
    if len(open_):
        eq[open_] += qty[open_] * (c[-1] * (1 - pos[open_] * taker) - entry[open_]) * pos[open_]

    out["final_eq"][idx] = eq
    out["trades_a"][idx], out["trades_b"][idx] = trades[:, 0], trades[:, 1]
    out["wins_a"][idx], out["wins_b"][idx] = wins[:, 0], wins[:, 1]
    out["expo_a"][idx], out["expo_b"][idx] = expo[:, 0], expo[:, 1]
    out["bars_a"][idx] = int((~seg_b[WARMUP:]).sum())
    out["bars_b"][idx] = int(seg_b[WARMUP:].sum())
    out["death_day"][idx] = death_day
    out["dead"][idx] = dead
