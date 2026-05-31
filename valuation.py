"""
Valuation engine — buy markets that are historically priced incorrectly.

The core idea: a Kalshi price *is* the market's probability estimate. We estimate
a better ("fair") probability and buy the side that is cheap relative to it.

Two independent, complementary sources of edge are combined:

1. Systematic calibration bias (favorite–longshot)
   ────────────────────────────────────────────────
   A robust, well-documented effect across prediction/betting markets: favorites
   win *more* often than their price implies, longshots *less*. We correct for it
   by sharpening the implied probability in logit space:

       fair = sigmoid( SHARPEN * logit(p) ),   SHARPEN > 1

   With SHARPEN=1.10: p=0.90 → 0.926 (favorite nudged up), p=0.10 → 0.074
   (longshot nudged down). SHARPEN=1.0 disables it. NOTE: Kalshi is generally
   better calibrated than racetracks, so keep SHARPEN modest and validate it
   against trade_log.jsonl — set it to 1.0 if favorites stop paying.

2. Per-market mean reversion
   ──────────────────────────
   We persist a rolling, down-sampled price history per ticker and take the EMA
   of the historical mid as a stable "anchor". When the live ask drops materially
   below the calibrated anchor, the market has likely overshot and tends to
   revert — so we buy the cheap side.

A signal fires only when the calibrated anchor and the live price disagree by at
least MIN_GAP cents AND the fee-adjusted EV is positive. Position sizing/exit are
handled by risk.py and bot.py as for any other Signal.

Because this needs history, it is silent during a warm-up period (a few
observations per market); the bot logs when warm-up is in progress.
"""

import json
import logging
import math
import os
import time
from typing import Dict, List, Optional

from strategy import Signal, fee_adjusted_ev

log = logging.getLogger("kalshi.valuation")

# ── Tunables (overridable via config.json) ────────────────────────────────────
DEFAULTS = {
    "value_min_obs": 5,          # need this many history points before trading
    "value_ema_alpha": 0.35,     # EMA smoothing for the fair-value anchor
    "value_sharpen": 1.10,       # favorite–longshot correction (1.0 = off)
    "value_min_gap_cents": 3,    # live ask must be this far below fair value
    # A gap LARGER than this is treated as real news / a regime change, NOT a
    # reversion opportunity. Mean reversion only works on modest deviations;
    # a price that has collapsed far below its old level usually moved for a
    # reason and keeps going (a falling knife). This is the key guard against
    # the engine buying into genuine directional crashes.
    "value_max_gap_cents": 12,
    "value_min_edge": 0.03,      # min fee-adjusted edge to act
    "value_max_edge": 0.40,      # reject absurd edges (model/anchor breakdown)
    "value_min_price": 5,        # ignore dust (<5¢) and near-certain (>95¢)
    "value_max_price": 95,
    # Only trade markets with real recent activity — a stale mid on a zero-volume
    # market is not "mispricing", just an untouched quote.
    "value_min_volume": 50,
    # The reference (history) window must be STABLE for a mean to be meaningful.
    # If the historical mids span more than this, the market is trending, not
    # mean-reverting — so there is no stable "fair value" to revert to. This
    # filter removes crypto hourly strikes and other trending markets.
    "value_max_anchor_range_cents": 10,
    "value_confidence": 0.50,
    # Price-history store
    "history_path": "price_history.json",
    "history_max_obs": 288,           # cap points per ticker (~1 day at 5-min)
    "history_min_interval_secs": 300, # down-sample: ≥5 min between recorded points
}


def cfg_get(cfg: dict, key: str):
    return cfg.get(key, DEFAULTS[key])


# ── Price-history tracker ─────────────────────────────────────────────────────

class PriceTracker:
    """Persists a rolling, down-sampled price history per market ticker."""

    def __init__(self, cfg: dict):
        self.path = cfg_get(cfg, "history_path")
        self.max_obs = int(cfg_get(cfg, "history_max_obs"))
        self.min_interval = float(cfg_get(cfg, "history_min_interval_secs"))
        self.data: Dict[str, List[dict]] = {}
        self._dirty = False
        self.load()

    def load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                self.data = json.load(f)
            log.info(f"Loaded price history for {len(self.data)} tickers")
        except Exception as e:
            log.warning(f"Could not load price history ({e}) — starting fresh")
            self.data = {}

    def save(self):
        if not self._dirty:
            return
        try:
            with open(self.path, "w") as f:
                json.dump(self.data, f)
            self._dirty = False
        except Exception as e:
            log.warning(f"Could not save price history: {e}")

    @staticmethod
    def _mid(market: dict) -> Optional[float]:
        yes_bid = market.get("yes_bid", 0)
        yes_ask = market.get("yes_ask", 0)
        last = market.get("last_price", 0)
        if yes_bid > 0 and yes_ask > 0:
            return (yes_bid + yes_ask) / 2.0
        if last > 0:
            return float(last)
        if yes_ask > 0:
            return float(yes_ask)
        return None

    def record(self, market: dict):
        ticker = market.get("ticker", "")
        mid = self._mid(market)
        if not ticker or mid is None:
            return
        now = time.time()
        series = self.data.setdefault(ticker, [])
        if series and (now - series[-1]["t"]) < self.min_interval:
            return  # down-sample
        series.append({
            "t": now,
            "mid": round(mid, 1),
            "last": float(market.get("last_price", 0) or 0),
            "vol": int(market.get("volume_24h", 0) or 0),
        })
        if len(series) > self.max_obs:
            del series[: len(series) - self.max_obs]
        self._dirty = True

    def history(self, ticker: str) -> List[dict]:
        return self.data.get(ticker, [])

    def prune(self, live_tickers: set):
        """Drop history for markets no longer open to keep the file small."""
        stale = [t for t in self.data if t not in live_tickers]
        for t in stale:
            del self.data[t]
        if stale:
            self._dirty = True


# ── Math ──────────────────────────────────────────────────────────────────────

def _ema(values: List[float], alpha: float) -> float:
    e = values[0]
    for v in values[1:]:
        e = alpha * v + (1 - alpha) * e
    return e


def calibrate(prob: float, sharpen: float) -> float:
    """Favorite–longshot correction via logit sharpening."""
    p = min(1 - 1e-4, max(1e-4, prob))
    if sharpen == 1.0:
        return p
    z = sharpen * math.log(p / (1 - p))
    return 1.0 / (1.0 + math.exp(-z))


# ── Strategy ──────────────────────────────────────────────────────────────────

def check_value(market: dict, history: List[dict], cfg: dict) -> Optional[Signal]:
    min_obs = int(cfg_get(cfg, "value_min_obs"))
    if len(history) < min_obs:
        return None

    alpha = float(cfg_get(cfg, "value_ema_alpha"))
    sharpen = float(cfg_get(cfg, "value_sharpen"))
    min_gap = float(cfg_get(cfg, "value_min_gap_cents"))
    max_gap = float(cfg_get(cfg, "value_max_gap_cents"))
    min_edge = float(cfg_get(cfg, "value_min_edge"))
    max_edge = float(cfg_get(cfg, "value_max_edge"))
    lo = float(cfg_get(cfg, "value_min_price"))
    hi = float(cfg_get(cfg, "value_max_price"))
    min_vol = float(cfg_get(cfg, "value_min_volume"))
    max_range = float(cfg_get(cfg, "value_max_anchor_range_cents"))
    base_conf = float(cfg_get(cfg, "value_confidence"))

    # Liquidity gate: mean reversion needs a real, traded market. A stale quote
    # on an untouched market is not a mispricing.
    if (market.get("volume_24h", 0) or 0) < min_vol:
        return None

    # Fair-value anchor: calibrated EMA of the historical mid (in probability).
    #
    # IMPORTANT: build the anchor from history EXCLUDING the most recent point.
    # If "now" is included, the EMA chases the live price and the deviation we
    # want to detect (current price vs. its established level) collapses toward
    # zero — the strategy would essentially never fire. Holding out the latest
    # observation lets a fresh dip below the established level show up as a real,
    # positive gap.
    mids = [h["mid"] / 100.0 for h in history if h.get("mid")]
    if len(mids) < min_obs:
        return None
    reference = mids[:-1] if len(mids) >= min_obs else mids

    # Stability gate: a mean is only meaningful if the market has been ranging
    # around it. If the reference window spans a wide range, the market is
    # TRENDING (e.g. a crypto hourly strike marching toward 0 or 100), so there
    # is no stable level to revert to. Skip it.
    ref_range_cents = (max(reference) - min(reference)) * 100.0
    if ref_range_cents > max_range:
        return None

    anchor = _ema(reference, alpha)
    fair = calibrate(anchor, sharpen)              # P(YES) we believe is true
    fair_yes_cents = fair * 100.0

    yes_ask = float(market.get("yes_ask", 0) or 0)
    no_ask = float(market.get("no_ask", 0) or 0)

    candidates = []

    # YES is cheap relative to fair value
    if lo <= yes_ask <= hi:
        gap = fair_yes_cents - yes_ask
        # Gap must be meaningful (>= min) but not so large it signals news (<= max).
        if min_gap <= gap <= max_gap:
            ev = fee_adjusted_ev(fair, yes_ask)
            edge = ev / yes_ask
            if min_edge <= edge <= max_edge:
                candidates.append(("yes", yes_ask, fair, gap, edge))

    # NO is cheap relative to fair value (i.e. YES is overpriced)
    fair_no = 1.0 - fair
    fair_no_cents = fair_no * 100.0
    if lo <= no_ask <= hi:
        gap = fair_no_cents - no_ask
        if min_gap <= gap <= max_gap:
            ev = fee_adjusted_ev(fair_no, no_ask)
            edge = ev / no_ask
            if min_edge <= edge <= max_edge:
                candidates.append(("no", no_ask, fair_no, gap, edge))

    if not candidates:
        return None

    side, price, side_fair, gap, edge = max(candidates, key=lambda c: c[4])

    # Confidence grows with history depth and deviation size, capped.
    depth_factor = min(1.0, len(history) / 20.0)
    gap_factor = min(1.0, gap / 10.0)
    confidence = round(min(0.65, base_conf * (0.7 + 0.3 * depth_factor)
                           * (0.7 + 0.3 * gap_factor)), 3)

    return Signal(
        ticker=market["ticker"],
        side=side,
        strategy="VALUE",
        fair_value=side_fair,
        market_price=price,
        edge=edge,
        confidence=confidence,
        target_exit_cents=min(99.0, price + gap * 0.6),  # revert toward fair
        market_title=market.get("title", ""),
        event_ticker=market.get("event_ticker", ""),
        extra={"anchor": round(anchor, 4), "fair": round(side_fair, 4),
               "gap_cents": round(gap, 1), "obs": len(history),
               "ref_range_cents": round(ref_range_cents, 1)},
    )
