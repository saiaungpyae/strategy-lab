#!/usr/bin/env python3
"""pm_order.py — manual PM (papi) order helper for the strategy-lab account.

Safety model: DRY-RUN BY DEFAULT. Nothing is sent to Binance unless you pass
--live. The dry run prints the exact request payload plus current market
context so you can eyeball it first.

Examples:
    # preview a deep resting limit buy (nothing is sent)
    .venv/bin/python scripts/pm_order.py place --symbol BTCUSDT --side buy \
        --price 63000 --qty 0.002

    # actually place it
    .venv/bin/python scripts/pm_order.py place --symbol BTCUSDT --side buy \
        --price 63000 --qty 0.002 --live

    # list open UM orders / cancel one / cancel all for a symbol
    .venv/bin/python scripts/pm_order.py list
    .venv/bin/python scripts/pm_order.py cancel --symbol BTCUSDT --id 123456 --live
    .venv/bin/python scripts/pm_order.py cancel-all --symbol BTCUSDT --live

Uses the UM (USDⓈ-M) side of the Portfolio Margin account via papi.
Keys come from <repo>/.env (BINANCE_API_KEY / BINANCE_API_SECRET).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "viewer"))


def load_env() -> None:
    for line in (REPO / ".env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def client():
    import ccxt
    key = os.environ.get("BINANCE_API_KEY", "")
    secret = os.environ.get("BINANCE_API_SECRET", "")
    if not key or not secret:
        sys.exit("BINANCE_API_KEY / BINANCE_API_SECRET not set in .env")
    return ccxt.binance({"apiKey": key, "secret": secret,
                         "enableRateLimit": True, "timeout": 10_000})


def mark_price(ex, symbol: str) -> float:
    return float(ex.public_get_ticker_price({"symbol": symbol})["price"])


def um_filters(ex, symbol: str) -> tuple[float, float, float]:
    """(qty stepSize, price tickSize, min notional) for a UM contract."""
    for s in ex.fapipublic_get_exchangeinfo()["symbols"]:
        if s["symbol"] == symbol:
            f = {x["filterType"]: x for x in s["filters"]}
            return (float(f["LOT_SIZE"]["stepSize"]),
                    float(f["PRICE_FILTER"]["tickSize"]),
                    float(f.get("MIN_NOTIONAL", {}).get("notional", 0)))
    sys.exit(f"symbol {symbol} not found in UM exchange info")


def snap(value: float, step: float) -> float:
    """Round down to the contract's step so Binance accepts the precision."""
    from decimal import Decimal
    d_step = Decimal(str(step))
    return float((Decimal(str(value)) // d_step) * d_step)


def cmd_place(args) -> None:
    ex = client()
    px = mark_price(ex, args.symbol)
    step, tick, min_notional = um_filters(ex, args.symbol)

    qty = snap(args.qty, step)
    price = snap(args.price, tick)
    if qty != args.qty:
        print(f"  (qty {args.qty} snapped to contract step {step} → {qty})")
    if price != args.price:
        print(f"  (price {args.price} snapped to tick {tick} → {price})")
    if qty <= 0:
        sys.exit(f"qty below contract minimum step {step}")

    notional = price * qty
    if notional < min_notional:
        need = snap(min_notional / price, step) + step
        sys.exit(f"notional ${notional:,.2f} is under the contract minimum "
                 f"${min_notional:,.0f} — at price {price:,.0f} use --qty ≥ {need}")

    dist = (price - px) / px * 100
    payload = {
        "symbol": args.symbol,
        "side": args.side.upper(),
        "type": "LIMIT",
        "timeInForce": "GTC",
        "quantity": f"{qty:.10f}".rstrip("0").rstrip("."),
        "price": f"{price:.10f}".rstrip("0").rstrip("."),
    }

    # Hedge-mode accounts (dualSidePosition) reject orders without a
    # positionSide (-4061); one-way accounts reject orders WITH one.
    dual = str(ex.papi_get_um_positionside_dual().get("dualSidePosition")).lower() == "true"
    if dual:
        payload["positionSide"] = "LONG" if args.side == "buy" else "SHORT"

    print(f"\n  symbol     {args.symbol}")
    print(f"  last price {px:,.2f}")
    print(f"  order      {payload['side']} LIMIT {payload['quantity']} @ {payload['price']}")
    print(f"  pos mode   {'hedge → positionSide=' + payload['positionSide'] if dual else 'one-way'}")
    print(f"  distance   {dist:+.2f}% from last")
    print(f"  notional   ~${notional:,.2f}  (contract min ${min_notional:,.0f})")
    print(f"  endpoint   POST /papi/v1/um/order\n")

    if abs(dist) < 1:
        print("  ⚠ order is within 1% of market — this is NOT a far-away resting order")
    legs = []
    long_side = args.side == "buy"
    close_side = "SELL" if long_side else "BUY"
    pos_side = ("LONG" if long_side else "SHORT") if dual else None

    def leg(kind: str, trigger: float) -> dict:
        l = {
            "symbol": args.symbol,
            "side": close_side,
            "strategyType": kind,
            "stopPrice": f"{snap(trigger, tick):.10f}".rstrip("0").rstrip("."),
            "quantity": payload["quantity"],
            "workingType": "MARK_PRICE",
        }
        if pos_side:
            l["positionSide"] = pos_side  # hedge mode: closing pair, no reduceOnly
        else:
            l["reduceOnly"] = "true"
        return l

    if args.sl is not None:
        # for a long: SL triggers on the way DOWN, below entry — safe to
        # pre-place, it cannot fire before the entry zone is reached
        bad = args.sl >= price if long_side else args.sl <= price
        if bad:
            sys.exit(f"--sl {args.sl} is on the wrong side of entry {price}")
        legs.append(("SL", leg("STOP_MARKET", args.sl)))
    if args.tp is not None:
        # for a long: TP triggers on the way UP — if the trigger is below the
        # CURRENT price it counts as already-triggered and misfires instantly
        misfire = args.tp <= px if long_side else args.tp >= px
        if misfire:
            sys.exit(f"--tp {args.tp} is already beyond the current price {px:,.2f} — "
                     "it would trigger immediately with no position. "
                     "Place the TP after the entry fills:  pm_order.py protect ...")
        legs.append(("TP", leg("TAKE_PROFIT_MARKET", args.tp)))

    for name, l in legs:
        print(f"  {name} leg     {l['side']} {l['strategyType']} trigger {l['stopPrice']}"
              f" (mark-price){' ' + pos_side if pos_side else ' reduce-only'}")
    if legs:
        print()

    if not args.live:
        print("  DRY RUN — nothing sent. Re-run with --live to place it.")
        return
    resp = ex.papi_post_um_order(payload)
    print(f"  ✔ entry placed — orderId {resp.get('orderId')}, status {resp.get('status')}")
    for name, l in legs:
        r = ex.papi_post_um_conditional_order(l)
        print(f"  ✔ {name} conditional placed — strategyId {r.get('strategyId')}")


def cmd_protect(args) -> None:
    """Attach SL/TP conditional legs to an existing UM position."""
    if args.sl is None and args.tp is None:
        sys.exit("nothing to do — pass --sl and/or --tp")
    ex = client()
    _, tick, _ = um_filters(ex, args.symbol)
    dual = str(ex.papi_get_um_positionside_dual().get("dualSidePosition")).lower() == "true"

    pos = [p for p in ex.papi_get_um_positionrisk({"symbol": args.symbol})
           if float(p.get("positionAmt", 0)) != 0]
    if not pos:
        sys.exit(f"no open {args.symbol} position to protect")
    p = pos[0]
    amt = float(p["positionAmt"])
    long_side = amt > 0
    close_side = "SELL" if long_side else "BUY"
    qty = f"{abs(amt):.10f}".rstrip("0").rstrip(".")
    print(f"\n  position   {'LONG' if long_side else 'SHORT'} {qty} @ entry {float(p['entryPrice']):,.2f}")

    legs = []
    for name, kind, trig in (("SL", "STOP_MARKET", args.sl),
                             ("TP", "TAKE_PROFIT_MARKET", args.tp)):
        if trig is None:
            continue
        l = {"symbol": args.symbol, "side": close_side, "strategyType": kind,
             "stopPrice": f"{snap(trig, tick):.10f}".rstrip("0").rstrip("."),
             "quantity": qty, "workingType": "MARK_PRICE"}
        if dual:
            l["positionSide"] = "LONG" if long_side else "SHORT"
        else:
            l["reduceOnly"] = "true"
        legs.append((name, l))
        print(f"  {name} leg     {l['side']} {kind} trigger {l['stopPrice']} (mark-price)")

    if not args.live:
        print("\n  DRY RUN — nothing sent. Re-run with --live to place the legs.")
        return
    for name, l in legs:
        r = ex.papi_post_um_conditional_order(l)
        print(f"  ✔ {name} conditional placed — strategyId {r.get('strategyId')}")


def cmd_list(args) -> None:
    ex = client()
    orders = ex.papi_get_um_openorders()
    if not orders:
        print("no open UM orders")
    for o in orders:
        print(f"  {o['orderId']}  {o['symbol']}  {o['side']} {o['type']} "
              f"{o['origQty']} @ {o['price']}  ({o['status']})")


def cmd_cancel(args) -> None:
    ex = client()
    if not args.live:
        print(f"DRY RUN — would cancel order {args.id} on {args.symbol}")
        return
    r = ex.papi_delete_um_order({"symbol": args.symbol, "orderId": args.id})
    print(f"✔ cancelled {r.get('orderId')} ({r.get('status')})")


def cmd_cancel_all(args) -> None:
    ex = client()
    if not args.live:
        print(f"DRY RUN — would cancel ALL open UM orders on {args.symbol}")
        return
    r = ex.papi_delete_um_allopenorders({"symbol": args.symbol})
    print(f"✔ cancel-all on {args.symbol}: {r}")


def main() -> None:
    load_env()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("place", help="place a UM limit order (dry-run unless --live)")
    pl.add_argument("--symbol", default="BTCUSDT")
    pl.add_argument("--side", choices=["buy", "sell"], required=True)
    pl.add_argument("--price", type=float, required=True)
    pl.add_argument("--qty", type=float, required=True)
    pl.add_argument("--sl", type=float, help="stop-loss trigger — pre-placed with the entry")
    pl.add_argument("--tp", type=float, help="take-profit trigger — only if beyond current price")
    pl.add_argument("--live", action="store_true", help="actually send the order")
    pl.set_defaults(fn=cmd_place)

    pr = sub.add_parser("protect", help="add SL/TP conditionals to an EXISTING position")
    pr.add_argument("--symbol", default="BTCUSDT")
    pr.add_argument("--sl", type=float)
    pr.add_argument("--tp", type=float)
    pr.add_argument("--live", action="store_true")
    pr.set_defaults(fn=cmd_protect)

    ls = sub.add_parser("list", help="list open UM orders")
    ls.set_defaults(fn=cmd_list)

    ca = sub.add_parser("cancel", help="cancel one order by id")
    ca.add_argument("--symbol", default="BTCUSDT")
    ca.add_argument("--id", required=True)
    ca.add_argument("--live", action="store_true")
    ca.set_defaults(fn=cmd_cancel)

    cl = sub.add_parser("cancel-all", help="cancel all open UM orders for a symbol")
    cl.add_argument("--symbol", default="BTCUSDT")
    cl.add_argument("--live", action="store_true")
    cl.set_defaults(fn=cmd_cancel_all)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
