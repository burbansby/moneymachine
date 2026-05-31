"""
Strategy engine: scans open Kalshi markets and scores them for edge.
Works purely off fields returned by the /markets endpoint.

Strategies
──────────
Risk-free (on by default):
1. ARBS      — single-market two-leg arbitrage: yes_ask + no_ask + fees < 100¢,
               so buying one YES and one NO locks in profit at settlement.
2. EVENT_ARB — multi-market arbitrage across a mutually-exclusive event: if the
               YES prices of the outcomes sum above 100¢, buying NO on every leg
               is risk-free (at most one outcome resolves YES). Detected and
               alerted by the bot; see bot.py (not auto-executed by default).

Directional / speculative (off by default — enable deliberately):
3. FAVORITE — favorite–longshot bias: heavy favorites are mildly underpriced.
4. MOMENTUM — recent trade move tends to continue.
5. SPREAD   — wide bid/ask spread with last_price anchoring fair value.
6. DRIFT    — last_price diverges from the current book mid.

Only ARBS / EVENT_ARB (and the model-based longshot engine) carry a defensible
edge. The directional strategies are heuristics with no guaranteed edge.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional

log = logging.getLogger("kalshi.strategy")

# Kalshi's general trading-fee coefficient. The fee on an order is
#   ceil(FEE_COEFF * contracts * P * (1 - P))
# where P is the per-contract price in DOLLARS (0..1). The fee is charged on
# entry only; settlement is free. Fee peaks at P=0.50 (~1.75¢/contract).
FEE_COEFF = 0.07


def kalshi_fee(price_cents: float, contracts: int = 1) -> float:
    """Kalshi trading fee in cents for `contracts` at `price_cents`, rounded up."""
    p = price_cents / 100.0
    raw = FEE_COEFF * contracts * p * (1.0 - p) * 100.0  # dollars*100 = cents
    return math.ceil(raw)


def fee_adjusted_ev(prob: float, price_cents: float) -> float:
    """
    Expected value in cents of buying one contract at `price_cents` when the
    true win probability is `prob`. Pays 100¢ on a win, 0 on a loss; the entry
    fee is paid regardless of outcome.
    """
    fee = kalshi_fee(price_cents, 1)
    return prob * 100.0 - price_cents - fee


def kelly_fraction(prob: float, price_cents: float, kelly_scale: float = 0.25) -> float:
    """
    Fractional-Kelly stake (as a fraction of bankroll) for buying a contract at
    `price_cents` with true win probability `prob`. Uses fee-adjusted net odds.
    """
    p = price_cents / 100.0
    if p <= 0 or p >= 1:
        return 0.0
    fee = kalshi_fee(price_cents, 1)
    cost = price_cents + fee          # amount staked per contract (cents)
    win_profit = 100.0 - cost         # net profit if it wins (cents)
    if win_profit <= 0 or cost <= 0:
        return 0.0
    b = win_profit / cost             # net odds received on the stake
    k = (prob * b - (1.0 - prob)) / b
    return max(0.0, k * kelly_scale)


@dataclass
class Signal:
    ticker: str
    side: str
    strategy: str
    fair_value: float
    market_price: float
    edge: float
    confidence: float
    target_exit_cents: float
    market_title: str = ""
    event_ticker: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.edge > 0 and 0 < self.market_price < 99

    def __str__(self):
        return (f"[{self.strategy}] {self.ticker} BUY {self.side.upper()} "
                f"@ {self.market_price:.0f}¢  fair={self.fair_value*100:.1f}¢  "
                f"edge={self.edge*100:.1f}%  conf={self.confidence*100:.0f}%")


# ── Filters ───────────────────────────────────────────────────────────────────

MIN_VOLUME_24H = 1    # accept any market with at least 1 trade today


def is_liquid(market: dict) -> bool:
    """Only hard requirement: yes_ask exists and at least 1 trade today."""
    yes_ask = market.get("yes_ask", 0)
    vol = market.get("volume_24h", 0)
    return yes_ask > 0 and vol >= MIN_VOLUME_24H


def mid_price(market: dict) -> Optional[float]:
    """
    Best estimate of mid price. Falls back gracefully when yes_bid=0.
    Priority: (yes_bid+yes_ask)/2 → last_price → yes_ask alone
    """
    yes_ask = market.get("yes_ask", 0)
    yes_bid = market.get("yes_bid", 0)
    last = market.get("last_price", 0)

    if yes_bid > 0 and yes_ask > 0:
        return (yes_bid + yes_ask) / 2.0
    if last > 0:
        return float(last)
    if yes_ask > 0:
        return float(yes_ask)
    return None


# ── Strategy 1: True two-leg arbitrage ────────────────────────────────────────
# A Kalshi market pays exactly 100¢ to either the YES or the NO holder at
# settlement. If we can buy one YES and one NO for a combined cost (including
# both entry fees) below 100¢, the position pays 100¢ no matter what — a locked,
# market-neutral profit. We hold both legs to settlement; no exit is needed.

MIN_ARB_NET_CENTS = 1   # require at least 1¢ guaranteed profit per pair


def check_arb(market: dict) -> Optional[Signal]:
    yes_ask = market.get("yes_ask", 0)
    no_ask = market.get("no_ask", 0)
    if yes_ask <= 0 or no_ask <= 0:
        return None

    total_cost = yes_ask + no_ask
    fee = kalshi_fee(yes_ask, 1) + kalshi_fee(no_ask, 1)
    net = 100 - total_cost - fee          # guaranteed cents of profit per pair

    if net < MIN_ARB_NET_CENTS:
        return None

    edge = net / total_cost               # return on capital deployed

    # market_price is the combined cost of one YES+NO pair; the bot buys both
    # legs and holds to settlement. side is nominal ("yes") for bookkeeping.
    return Signal(
        ticker=market["ticker"],
        side="yes",
        strategy="ARBS",
        fair_value=1.0,                   # guaranteed payout
        market_price=float(total_cost),
        edge=edge,
        confidence=0.99,
        target_exit_cents=100.0,          # settles at 100¢; held, not exited
        market_title=market.get("title", ""),
        event_ticker=market.get("event_ticker", ""),
        extra={"yes_ask": yes_ask, "no_ask": no_ask,
               "net_profit": net, "fee": fee},
    )


# ── Strategy 2: Spread Mispricing ─────────────────────────────────────────────

MIN_SPREAD_CENTS = 4    # lowered from 8 to 4
MIN_EDGE_SPREAD = 0.02  # lowered from 0.03


def check_spread(market: dict) -> Optional[Signal]:
    yes_ask = market.get("yes_ask", 0)
    yes_bid = market.get("yes_bid", 0)
    last = market.get("last_price", 0)

    if yes_ask <= 0 or last <= 0:
        return None

    # Works even when yes_bid=0 — use last_price as the lower anchor
    effective_bid = yes_bid if yes_bid > 0 else max(1, last - 2)
    spread = yes_ask - effective_bid

    if spread < MIN_SPREAD_CENTS:
        return None

    dist_to_bid = abs(last - effective_bid)
    dist_to_ask = abs(last - yes_ask)

    if dist_to_bid < dist_to_ask:
        # last is near bid → YES is underpriced → buy YES
        side = "yes"
        price = float(yes_ask)
        fair_prob = (last * 0.65 + effective_bid * 0.35) / 100.0
    else:
        # last is near ask → YES is overpriced → buy NO
        no_ask = market.get("no_ask", 100 - effective_bid)
        if no_ask <= 0:
            return None
        side = "no"
        price = float(no_ask)
        fair_prob = 1.0 - (last * 0.65 + yes_ask * 0.35) / 100.0

    fair_prob = max(0.01, min(0.99, fair_prob))
    if price <= 0 or price >= 100:
        return None

    ev = fee_adjusted_ev(fair_prob, price)
    edge = ev / price

    if edge < MIN_EDGE_SPREAD:
        return None

    return Signal(
        ticker=market["ticker"],
        side=side,
        strategy="SPREAD",
        fair_value=fair_prob,
        market_price=price,
        edge=edge,
        confidence=0.30,
        target_exit_cents=price + spread * 0.4,
        market_title=market.get("title", ""),
        event_ticker=market.get("event_ticker", ""),
        extra={"spread": spread, "last": last, "effective_bid": effective_bid},
    )


# ── Strategy 3: Price Drift ───────────────────────────────────────────────────

MIN_DRIFT_CENTS = 3     # lowered from 6 to 3
MIN_EDGE_DRIFT = 0.02   # lowered from 0.03


def check_drift(market: dict) -> Optional[Signal]:
    yes_ask = market.get("yes_ask", 0)
    last = market.get("last_price", 0)
    prev = market.get("previous_price", 0)

    if yes_ask <= 0 or last <= 0:
        return None

    # Use yes_bid if available, otherwise estimate from last
    yes_bid = market.get("yes_bid", 0)
    effective_bid = yes_bid if yes_bid > 0 else max(1, last - 1)
    mid = (yes_ask + effective_bid) / 2.0
    gap = last - mid

    if abs(gap) < MIN_DRIFT_CENTS:
        return None

    # Only skip on conflicting prev_price if it's meaningfully different
    if prev > 0 and abs(last - prev) > 2:
        trend = last - prev
        if (gap > 0 and trend < 0) or (gap < 0 and trend > 0):
            return None

    if gap > 0:
        # last above mid → YES likely drifting up → buy YES
        side = "yes"
        price = float(yes_ask)
        fair_prob = min(0.97, (last * 0.70 + mid * 0.30) / 100.0)
    else:
        # last below mid → buy NO
        no_ask = market.get("no_ask", 100 - effective_bid)
        if no_ask <= 0:
            return None
        side = "no"
        price = float(no_ask)
        fair_prob = min(0.97, 1.0 - (last * 0.70 + mid * 0.30) / 100.0)

    fair_prob = max(0.01, min(0.99, fair_prob))
    if price <= 0 or price >= 100:
        return None

    ev = fee_adjusted_ev(fair_prob, price)
    edge = ev / price

    if edge < MIN_EDGE_DRIFT:
        return None

    return Signal(
        ticker=market["ticker"],
        side=side,
        strategy="DRIFT",
        fair_value=fair_prob,
        market_price=price,
        edge=edge,
        confidence=0.30,
        target_exit_cents=price + abs(gap) * 0.5,
        market_title=market.get("title", ""),
        event_ticker=market.get("event_ticker", ""),
        extra={"mid": mid, "last": last, "gap": gap},
    )


# ── Strategy: Favorite–longshot bias (FAVORITE) ───────────────────────────────
# A well-documented prediction-market bias: bettors overpay for longshots and
# underpay for favorites, so heavy favorites tend to be modestly underpriced.
# The edge is small (~1-3pp) and low-variance — to trade it you must lower
# RiskConfig.min_edge (e.g. to 0.005), since the honest edge is below the 2%
# default. Off by default.

FAV_MIN_PRICE = 80      # only mid/high favorites
FAV_MAX_PRICE = 95      # above this there is no room left + settlement/fee risk
FAV_BIAS = 0.02         # assume true prob is ~2pp above the quoted price
MIN_EDGE_FAV = 0.005


def check_favorite(market: dict) -> Optional[Signal]:
    yes_ask = market.get("yes_ask", 0)
    if not (FAV_MIN_PRICE <= yes_ask <= FAV_MAX_PRICE):
        return None

    market_prob = yes_ask / 100.0
    fair_prob = min(0.99, market_prob + FAV_BIAS)
    price = float(yes_ask)

    ev = fee_adjusted_ev(fair_prob, price)
    edge = ev / price
    if edge < MIN_EDGE_FAV:
        return None

    return Signal(
        ticker=market["ticker"],
        side="yes",
        strategy="FAVORITE",
        fair_value=fair_prob,
        market_price=price,
        edge=edge,
        confidence=0.45,
        target_exit_cents=min(99.0, price + 2),
        market_title=market.get("title", ""),
        event_ticker=market.get("event_ticker", ""),
        extra={"market_prob": market_prob, "bias": FAV_BIAS},
    )


# ── Strategy: Momentum (MOMENTUM) ─────────────────────────────────────────────
# Recent trade direction tends to continue short-term. Requires a real move and
# meaningful volume. Speculative, off by default.

MOM_MIN_MOVE = 4        # |last - previous| in cents
MOM_MIN_VOLUME = 20     # require some activity behind the move
MIN_EDGE_MOM = 0.02


def check_momentum(market: dict) -> Optional[Signal]:
    last = market.get("last_price", 0)
    prev = market.get("previous_price", 0)
    yes_ask = market.get("yes_ask", 0)
    no_ask = market.get("no_ask", 0)
    vol = market.get("volume_24h", 0)

    if last <= 0 or prev <= 0 or vol < MOM_MIN_VOLUME:
        return None

    move = last - prev
    if abs(move) < MOM_MIN_MOVE:
        return None

    # Project the move to continue by half its size again.
    if move > 0:
        if yes_ask <= 0:
            return None
        side, price = "yes", float(yes_ask)
        fair_prob = (last + move * 0.5) / 100.0
    else:
        if no_ask <= 0:
            return None
        side, price = "no", float(no_ask)
        fair_prob = 1.0 - (last + move * 0.5) / 100.0

    fair_prob = max(0.01, min(0.97, fair_prob))
    if price <= 0 or price >= 100:
        return None

    ev = fee_adjusted_ev(fair_prob, price)
    edge = ev / price
    if edge < MIN_EDGE_MOM:
        return None

    return Signal(
        ticker=market["ticker"],
        side=side,
        strategy="MOMENTUM",
        fair_value=fair_prob,
        market_price=price,
        edge=edge,
        confidence=0.32,
        target_exit_cents=price + abs(move) * 0.5,
        market_title=market.get("title", ""),
        event_ticker=market.get("event_ticker", ""),
        extra={"last": last, "prev": prev, "move": move},
    )


# ── Strategy: Event-level arbitrage (EVENT_ARB) ───────────────────────────────
# Within a mutually-exclusive event, at most one outcome resolves YES. Buying NO
# on N of those outcomes therefore wins on at least N-1 of them. If the combined
# NO cost (plus fees) is below 100*(N-1), the position is risk-free regardless of
# which outcome wins.
#
# This works on any SUBSET of mutually-exclusive markets (exhaustiveness is not
# required): if the winning outcome isn't among our legs, all N NO legs win.
# It is a pure detector — bot.py verifies the event is mutually exclusive and
# alerts; it does not auto-execute the multi-leg order.

MIN_EVENT_ARB_CENTS = 2


def check_event_arb(markets: List[dict],
                    min_profit_cents: float = MIN_EVENT_ARB_CENTS) -> Optional[dict]:
    legs = []
    for m in markets:
        no_ask = m.get("no_ask", 0)
        # A complete, executable NO book is required on every leg; otherwise we
        # cannot guarantee the fill and must not claim an arb.
        if no_ask <= 0 or no_ask >= 100:
            return None
        legs.append((m.get("ticker", ""), float(no_ask)))

    n = len(legs)
    if n < 2:
        return None

    total_cost = sum(p for _, p in legs)
    fees = sum(kalshi_fee(p, 1) for _, p in legs)
    worst_payout = 100.0 * (n - 1)        # mutual exclusivity ⇒ ≤1 YES
    profit = worst_payout - total_cost - fees
    if profit < min_profit_cents:
        return None

    return {
        "legs": legs,
        "n": n,
        "total_cost": total_cost,
        "fees": fees,
        "guaranteed_profit": profit,
        "profit_per_dollar": profit / total_cost,
    }


# ── Signal ranking ────────────────────────────────────────────────────────────

def analyze_market(market: dict, active_strategies: set) -> Optional[Signal]:
    if not is_liquid(market):
        return None

    checks = {
        "ARBS": check_arb,
        "SPREAD": check_spread,
        "DRIFT": check_drift,
        "FAVORITE": check_favorite,
        "MOMENTUM": check_momentum,
    }

    signals = []
    for name, fn in checks.items():
        if name in active_strategies:
            s = fn(market)
            if s:
                signals.append(s)

    if not signals:
        return None
    return max(signals, key=lambda s: s.edge * s.confidence)


def rank_signals(signals: List[Signal]) -> List[Signal]:
    return sorted([s for s in signals if s.is_valid],
                  key=lambda s: s.edge * s.confidence, reverse=True)