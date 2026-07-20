"""binance_account.py — read-only Binance account snapshot for the viewer.

Signs requests with BINANCE_API_KEY / BINANCE_API_SECRET from the environment
(loaded from <repo>/.env by server.py). Strictly GET-style account data: key
restrictions, spot/funding balances, USDⓈ-M futures wallet + positions, open
orders. There is deliberately no order/transfer/withdrawal code in this module.

The snapshot is cached for CACHE_TTL_S so a polling dashboard doesn't burn
Binance request weight; all sections degrade independently (a key without
futures permission still gets spot balances, etc.).
"""

from __future__ import annotations

import os
import threading
import time

try:
    import ccxt
except ImportError:  # viewer still works without ccxt, page reports it
    ccxt = None

CACHE_TTL_S = 10
STABLES = {"USDT", "USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDP"}
DUST_USD = 0.01  # hide balances worth less than this (counted, not listed)

_lock = threading.Lock()
_cache: dict = {"at": 0.0, "payload": None}
_ip_cache: dict = {"at": 0.0, "ip": None}
_spot = None
_futures = None


def _public_ip() -> str | None:
    """This machine's public IP as Binance sees it (cached 5 min)."""
    import urllib.request
    if time.time() - _ip_cache["at"] < 300:
        return _ip_cache["ip"]
    ip = None
    for url in ("https://checkip.amazonaws.com", "https://api.ipify.org"):
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                ip = r.read().decode().strip()
                break
        except Exception:
            continue
    _ip_cache.update(at=time.time(), ip=ip)
    return ip


def _keys() -> tuple[str, str]:
    return (os.environ.get("BINANCE_API_KEY", "").strip(),
            os.environ.get("BINANCE_API_SECRET", "").strip())


def _clients():
    """Lazily build one spot and one USDⓈ-M futures ccxt client."""
    global _spot, _futures
    if _spot is None:
        key, secret = _keys()
        common = {
            "apiKey": key,
            "secret": secret,
            "enableRateLimit": True,
            "timeout": 10_000,
        }
        _spot = ccxt.binance({
            **common,
            "options": {"warnOnFetchOpenOrdersWithoutSymbol": False},
        })
        _futures = ccxt.binance({**common, "options": {"defaultType": "future"}})
    return _spot, _futures


def _f(x, default=0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _usd_prices(spot) -> dict:
    """asset -> USD price from the public ticker/price endpoint (one call)."""
    px = {}
    for t in spot.public_get_ticker_price():
        px[t["symbol"]] = _f(t.get("price"))
    btc_usd = px.get("BTCUSDT", 0.0)

    def price(asset: str) -> float | None:
        # LD-prefixed assets are Earn (flexible) positions in the same coin
        a = asset[2:] if asset.startswith("LD") and len(asset) > 3 else asset
        if a in STABLES:
            return px.get(a + "USDT") or 1.0
        if a + "USDT" in px:
            return px[a + "USDT"]
        if a + "BTC" in px and btc_usd:
            return px[a + "BTC"] * btc_usd
        return None

    return {"fn": price}


def _balance_rows(entries, price_of) -> tuple[list, float, int]:
    """entries: [(asset, free, locked)] -> (rows sorted by value, total, dust)."""
    rows, total, dust = [], 0.0, 0
    for asset, free, locked in entries:
        qty = free + locked
        if qty <= 0:
            continue
        p = price_of(asset)
        value = qty * p if p is not None else None
        if value is not None and value < DUST_USD:
            dust += 1
            continue
        if value is not None:
            total += value
        rows.append({
            "asset": asset,
            "earn": asset.startswith("LD") and len(asset) > 3,
            "free": free,
            "locked": locked,
            "total": qty,
            "price_usd": p,
            "value_usd": value,
        })
    rows.sort(key=lambda r: -(r["value_usd"] or 0))
    return rows, total, dust


def _restrictions(spot) -> dict:
    r = spot.sapi_get_account_apirestrictions()
    return {
        "ip_restrict": bool(r.get("ipRestrict")),
        "enable_reading": bool(r.get("enableReading")),
        "enable_spot_trading": bool(r.get("enableSpotAndMarginTrading")),
        "enable_margin": bool(r.get("enableMargin")),
        "enable_futures": bool(r.get("enableFutures")),
        "enable_portfolio_margin": bool(r.get("enablePortfolioMarginTrading")),
        "enable_withdrawals": bool(r.get("enableWithdrawals")),
        "enable_internal_transfer": bool(r.get("enableInternalTransfer")),
        "permits_universal_transfer": bool(r.get("permitsUniversalTransfer")),
        "created_ms": int(_f(r.get("createTime"))),
    }


def _spot_section(spot, price_of) -> dict:
    bal = spot.fetch_balance()
    entries = [(b["asset"], _f(b.get("free")), _f(b.get("locked")))
               for b in bal["info"].get("balances", [])]
    rows, total, dust = _balance_rows(entries, price_of)
    return {"assets": rows, "value_usd": total, "dust_hidden": dust}


def _funding_section(spot, price_of) -> dict:
    rows_raw = spot.sapi_post_asset_get_funding_asset({})
    entries = [(b["asset"], _f(b.get("free")),
                _f(b.get("locked")) + _f(b.get("freeze")))
               for b in rows_raw]
    rows, total, dust = _balance_rows(entries, price_of)
    return {"assets": rows, "value_usd": total, "dust_hidden": dust}


def _futures_section(futures) -> dict:
    bal = futures.fetch_balance()
    info = bal["info"]
    assets = [{
        "asset": a["asset"],
        "wallet": _f(a.get("walletBalance")),
        "unrealized_pnl": _f(a.get("unrealizedProfit")),
        "margin_balance": _f(a.get("marginBalance")),
    } for a in info.get("assets", []) if _f(a.get("walletBalance")) != 0
                                      or _f(a.get("unrealizedProfit")) != 0]
    positions = []
    for p in futures.fetch_positions():
        qty = _f(p.get("contracts"))
        if qty == 0:
            continue
        positions.append({
            "symbol": p.get("symbol"),
            "side": p.get("side"),
            "contracts": qty,
            "notional": _f(p.get("notional")),
            "leverage": _f(p.get("leverage")),
            "entry_price": _f(p.get("entryPrice")),
            "mark_price": _f(p.get("markPrice")),
            "liq_price": _f(p.get("liquidationPrice")) or None,
            "unrealized_pnl": _f(p.get("unrealizedPnl")),
            "margin_mode": p.get("marginMode"),
        })
    return {
        "wallet_usd": _f(info.get("totalWalletBalance")),
        "unrealized_pnl": _f(info.get("totalUnrealizedProfit")),
        "margin_balance_usd": _f(info.get("totalMarginBalance")),
        "available_usd": _f(info.get("availableBalance")),
        "assets": assets,
        "positions": positions,
    }


def _pm_position(p, market: str) -> dict:
    amt = _f(p.get("positionAmt"))
    return {
        "symbol": p.get("symbol"),
        "market": market,
        "side": "long" if amt > 0 else "short",
        "contracts": abs(amt),
        "notional": abs(_f(p.get("notional"))),
        "leverage": _f(p.get("leverage")),
        "entry_price": _f(p.get("entryPrice")),
        "mark_price": _f(p.get("markPrice")),
        "liq_price": _f(p.get("liquidationPrice")) or None,
        "unrealized_pnl": _f(p.get("unRealizedProfit")),
        "margin_mode": "cross",  # PM is cross-margin by construction
    }


def _pm_section(spot, price_of) -> dict:
    """Portfolio Margin account (papi) — replaces fapi on PM-upgraded accounts."""
    assets, value = [], 0.0
    for b in spot.papi_get_balance():
        wallet = _f(b.get("totalWalletBalance"))
        upnl = _f(b.get("umUnrealizedPNL")) + _f(b.get("cmUnrealizedPNL"))
        if wallet == 0 and upnl == 0:
            continue
        p = price_of(b["asset"])
        row_value = (wallet + upnl) * p if p is not None else None
        if row_value is not None:
            value += row_value
        assets.append({
            "asset": b["asset"],
            "wallet": wallet,
            "um_upnl": _f(b.get("umUnrealizedPNL")),
            "cm_upnl": _f(b.get("cmUnrealizedPNL")),
            "value_usd": row_value,
        })
    assets.sort(key=lambda r: -(r["value_usd"] or 0))

    acct = spot.papi_get_account()
    positions = [_pm_position(p, "UM")
                 for p in spot.papi_get_um_positionrisk()
                 if _f(p.get("positionAmt")) != 0]
    try:
        positions += [_pm_position(p, "CM")
                      for p in spot.papi_get_cm_positionrisk()
                      if _f(p.get("positionAmt")) != 0]
    except Exception:
        pass  # coin-margined side may be unopened; UM data is still good

    orders = []
    for fetch, market in ((spot.papi_get_um_openorders, "UM"),
                          (spot.papi_get_cm_openorders, "CM"),
                          (spot.papi_get_margin_openorders, "MARGIN")):
        try:
            for o in fetch():
                orders.append({
                    "symbol": o.get("symbol"),
                    "market": market,
                    "side": (o.get("side") or "").lower(),
                    "type": (o.get("type") or "").lower(),
                    "price": _f(o.get("price")) or None,
                    "amount": _f(o.get("origQty")),
                    "filled": _f(o.get("executedQty")),
                    "reduce_only": bool(o.get("reduceOnly")),
                    "created_ms": int(_f(o.get("time"))) or None,
                })
        except Exception:
            pass  # unopened market side (e.g. CM) — same rationale as positions

    def um_algo_orders():
        r = spot.request("um/algo/openOrders", "papi", "GET", {})
        return (r.get("orders") or r.get("data") or []) if isinstance(r, dict) else r

    algo_query_down = False
    for fetch, market in ((um_algo_orders, "UM"),
                          (spot.papi_get_cm_conditional_openorders, "CM")):
        try:
            for o in fetch():
                orders.append({
                    "symbol": o.get("symbol"),
                    "market": market,
                    "side": (o.get("side") or "").lower(),
                    "type": (o.get("type") or o.get("strategyType") or "").lower(),
                    "price": _f(o.get("price")) or None,
                    "trigger": _f(o.get("triggerPrice")) or _f(o.get("stopPrice")) or None,
                    "amount": _f(o.get("origQty")),
                    "filled": 0.0,
                    "reduce_only": bool(o.get("reduceOnly")),
                    "created_ms": int(_f(o.get("bookTime"))) or None,
                })
        except Exception:
            if market == "UM":
                algo_query_down = True

    if algo_query_down:
        # fall back to the placement log scripts/pm_order.py keeps — live
        # status can't be verified until Binance ships the algo query API
        import json as _json
        from pathlib import Path as _Path
        log = _Path(__file__).resolve().parent.parent / "reports" / "pm" / "algo_orders.json"
        try:
            for r in _json.loads(log.read_text()):
                orders.append({
                    "symbol": r.get("symbol"),
                    "market": "UM",
                    "side": (r.get("side") or "").lower(),
                    "type": (r.get("type") or "").lower(),
                    "price": None,
                    "trigger": _f(r.get("trigger")) or None,
                    "amount": _f(r.get("qty")),
                    "filled": 0.0,
                    "reduce_only": True,
                    "local": True,
                    "created_ms": r.get("placed_ms"),
                })
        except Exception:
            pass

    return {
        "open_orders": orders,
        "algo_query_down": algo_query_down,
        "assets": assets,
        "value_usd": value,
        "equity_usd": _f(acct.get("accountEquity")),
        "actual_equity_usd": _f(acct.get("actualEquity")),
        "uni_mmr": _f(acct.get("uniMMR"), default=None),
        "status": acct.get("accountStatus"),
        "positions": positions,
    }


def _open_orders(spot) -> list:
    return [{
        "symbol": o.get("symbol"),
        "side": o.get("side"),
        "type": o.get("type"),
        "price": _f(o.get("price")) or None,
        "amount": _f(o.get("amount")),
        "filled": _f(o.get("filled")),
        "created_ms": o.get("timestamp"),
    } for o in spot.fetch_open_orders()]


def _build_payload() -> dict:
    key, secret = _keys()
    if ccxt is None:
        return {"configured": False,
                "error": "ccxt not installed — pip install ccxt"}
    if not key or not secret:
        return {"configured": False,
                "error": "BINANCE_API_KEY / BINANCE_API_SECRET not set — "
                         "add them to .env and restart the viewer"}

    spot, futures = _clients()
    payload = {"configured": True, "generated_ms": int(time.time() * 1000),
               "errors": []}

    def section(name, fn):
        try:
            payload[name] = fn()
        except Exception as e:
            payload[name] = None
            payload["errors"].append(f"{name}: {type(e).__name__}: {e}")

    try:
        price_of = _usd_prices(spot)["fn"]
    except Exception as e:
        return {"configured": True, "generated_ms": payload["generated_ms"],
                "error": f"cannot reach Binance: {type(e).__name__}: {e}"}

    section("restrictions", lambda: _restrictions(spot))
    section("spot", lambda: _spot_section(spot, price_of))
    section("funding", lambda: _funding_section(spot, price_of))
    section("futures", lambda: _futures_section(futures))
    section("portfolio_margin", lambda: _pm_section(spot, price_of))
    section("open_orders", lambda: _open_orders(spot))

    # A key without the matching permission gets -2015 on fapi/papi even for
    # reads — expected (e.g. PM-upgraded accounts have no classic futures,
    # classic accounts have no PM), not a problem worth a warning.
    r = payload.get("restrictions")
    for name, flag in (("futures", "enable_futures"),
                       ("portfolio_margin", "enable_portfolio_margin")):
        if (r and not r[flag]
                and any(e.startswith(name + ":") and "-2015" in e
                        for e in payload["errors"])):
            payload["errors"] = [e for e in payload["errors"]
                                 if not e.startswith(name + ":")]
            if name == "futures":
                payload["futures_off"] = True

    # Any other -2015 = key rejected (bad key, missing permission, or — most
    # commonly with a whitelisted residential IP — the IP rotated). Surface
    # the current public IP so re-whitelisting is copy-paste.
    if any("-2015" in e for e in payload["errors"]):
        payload["public_ip"] = _public_ip()

    total = 0.0
    for wallet in ("spot", "funding"):
        if payload.get(wallet):
            total += payload[wallet]["value_usd"]
    if payload.get("futures"):
        total += payload["futures"]["margin_balance_usd"]
    if payload.get("portfolio_margin"):
        total += payload["portfolio_margin"]["value_usd"]
    payload["total_value_usd"] = total
    return payload


def binance_payload() -> dict:
    """Cached account snapshot; safe to poll from the dashboard."""
    with _lock:
        if _cache["payload"] is not None and time.time() - _cache["at"] < CACHE_TTL_S:
            return _cache["payload"]
        payload = _build_payload()
        # don't cache "not configured" so adding keys + restart shows up fast,
        # and a transient network error retries on the next poll
        if payload.get("configured") and "error" not in payload:
            _cache.update(at=time.time(), payload=payload)
        return payload
