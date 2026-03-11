"""
Strategy engine: scans open Kalshi markets and scores them for edge.
Works purely off fields returned by the /markets endpoint.

Strategies
──────────
1. ARBS   — yes_ask + no_ask < 97 (guaranteed profit after fees)
2. SPREAD — wide spread with last_price anchoring fair value
3. DRIFT  — last_price diverges from current ask (works even with yes_bid=0)
4. VALUE  — yes_ask is far above last_price, suggesting ask is stale/high
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

log = logging.getLogger("kalshi.strategy")

FEE_RATE = 0.07


def fee_adjusted_ev(prob: float, price_cents: float) -> float:
    win_payout = (100 - price_cents) * (1 - FEE_RATE)
    return prob * win_payout - (1 - prob) * price_cents


def kelly_fraction(prob: float, price_cents: float, kelly_scale: float = 0.25) -> float:
    p = price_cents / 100.0
    if p <= 0 or p >= 1:
        return 0.0
    b = (1 - FEE_RATE) * (1 - p) / p
    k = (prob * b - (1 - prob)) / b
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


# ── Strategy 1: Arbitrage ─────────────────────────────────────────────────────

MIN_ARB_NET_CENTS = 1   # lowered to 1¢ to catch more arbs


def check_arb(market: dict) -> Optional[Signal]:
    yes_ask = market.get("yes_ask", 0)
    no_ask = market.get("no_ask", 0)
    if yes_ask <= 0 or no_ask <= 0:
        return None

    total_cost = yes_ask + no_ask
    fee = FEE_RATE * (100 - min(yes_ask, no_ask))
    net = 100 - total_cost - fee

    if net < MIN_ARB_NET_CENTS:
        return None

    side, price = ("yes", yes_ask) if yes_ask <= no_ask else ("no", no_ask)
    edge = net / price

    return Signal(
        ticker=market["ticker"],
        side=side,
        strategy="ARBS",
        fair_value=0.5,
        market_price=float(price),
        edge=edge,
        confidence=0.95,
        target_exit_cents=float(price + net * 0.6),
        market_title=market.get("title", ""),
        event_ticker=market.get("event_ticker", ""),
        extra={"yes_ask": yes_ask, "no_ask": no_ask, "net_profit": net},
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
        confidence=0.40,
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
        confidence=0.38,
        target_exit_cents=price + abs(gap) * 0.5,
        market_title=market.get("title", ""),
        event_ticker=market.get("event_ticker", ""),
        extra={"mid": mid, "last": last, "gap": gap},
    )


# ── Strategy 4: Stale Ask (VALUE) ─────────────────────────────────────────────
# When yes_ask is significantly above last_price with no bid pressure,
# the ask is stale/high. Buy YES expecting the ask to drop toward last.

MIN_VALUE_GAP = 4       # yes_ask must be at least 4¢ above last_price
MIN_EDGE_VALUE = 0.02


def check_value(market: dict) -> Optional[Signal]:
    yes_ask = market.get("yes_ask", 0)
    last = market.get("last_price", 0)
    prev = market.get("previous_price", 0)

    if yes_ask <= 0 or last <= 0:
        return None

    # Only fire if last_price is meaningfully below yes_ask
    gap = yes_ask - last
    if gap < MIN_VALUE_GAP:
        return None

    # Confirm with prev_price if available — last should be trending up
    if prev > 0 and last < prev:
        return None  # price falling, don't buy

    fair_prob = min(0.97, (last * 0.75 + yes_ask * 0.25) / 100.0)
    price = float(yes_ask)

    if price <= 0 or price >= 100:
        return None

    ev = fee_adjusted_ev(fair_prob, price)
    edge = ev / price

    if edge < MIN_EDGE_VALUE:
        return None

    return Signal(
        ticker=market["ticker"],
        side="yes",
        strategy="VALUE",
        fair_value=fair_prob,
        market_price=price,
        edge=edge,
        confidence=0.35,
        target_exit_cents=price - gap * 0.5,  # exit when ask drops halfway to last
        market_title=market.get("title", ""),
        event_ticker=market.get("event_ticker", ""),
        extra={"last": last, "yes_ask": yes_ask, "gap": gap},
    )


# ── Signal ranking ────────────────────────────────────────────────────────────

def analyze_market(market: dict, active_strategies: set) -> Optional[Signal]:
    if not is_liquid(market):
        return None

    signals = []
    if "ARBS" in active_strategies:
        s = check_arb(market)
        if s:
            signals.append(s)
    if "SPREAD" in active_strategies:
        s = check_spread(market)
        if s:
            signals.append(s)
    if "DRIFT" in active_strategies:
        s = check_drift(market)
        if s:
            signals.append(s)
    if "VALUE" in active_strategies:
        s = check_value(market)
        if s:
            signals.append(s)

    if not signals:
        return None
    return max(signals, key=lambda s: s.edge * s.confidence)


def rank_signals(signals: List[Signal]) -> List[Signal]:
    return sorted([s for s in signals if s.is_valid],
                  key=lambda s: s.edge * s.confidence, reverse=True)