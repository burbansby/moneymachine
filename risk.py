"""
Risk Manager: controls position sizing and portfolio exposure.

Rules
─────
• Max single-position = max_position_pct of balance
• Max total exposure = max_exposure_pct of balance
• Min edge threshold before any bet
• Drawdown guard: halt if daily P&L < max_daily_loss
• Only trade markets with min_volume contracts / day
• Kelly position sizing with configurable fraction
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, Optional

from strategy import Signal, kelly_fraction

log = logging.getLogger("kalshi.risk")


@dataclass
class RiskConfig:
    # Sizing
    max_position_pct: float = 0.05     # max 5% of balance per position
    max_exposure_pct: float = 0.40     # max 40% of balance deployed at once
    kelly_scale: float = 0.25          # quarter-Kelly
    min_contracts: int = 1
    max_contracts_per_trade: int = 50

    # Filters
    min_edge: float = 0.04             # skip signals with edge < 4%
    min_confidence: float = 0.35       # skip low-confidence signals
    min_orderbook_volume: int = 5      # skip thin markets

    # Drawdown
    max_daily_loss_pct: float = 0.10   # halt if down 10% in a day
    max_single_loss_pct: float = 0.03  # stop-loss per position at 3% of balance

    # Exit
    profit_target_mult: float = 0.50   # exit when 50% of edge captured


@dataclass
class PortfolioState:
    balance_cents: int = 0
    start_of_day_balance: int = 0
    positions: Dict[str, dict] = field(default_factory=dict)  # ticker -> position info
    daily_realized_pnl: int = 0
    halted: bool = False


class RiskManager:
    def __init__(self, config: RiskConfig = None):
        self.cfg = config or RiskConfig()
        self.state = PortfolioState()
        self._today = date.today()

    # ── State update ────────────────────────────────────────────────────────

    def update_balance(self, balance_cents: int):
        if date.today() != self._today:
            # New day — reset daily tracking
            self._today = date.today()
            self.state.start_of_day_balance = balance_cents
            self.state.daily_realized_pnl = 0
            log.info(f"New trading day — starting balance: ${balance_cents/100:.2f}")
        elif self.state.start_of_day_balance == 0:
            self.state.start_of_day_balance = balance_cents
        self.state.balance_cents = balance_cents

    def update_positions(self, positions: list):
        """Sync open positions from API response."""
        active = {}
        for pos in positions:
            ticker = pos.get("market_ticker", "")
            if ticker and (pos.get("position", 0) != 0):
                active[ticker] = pos
        self.state.positions = active

    # ── Guard checks ────────────────────────────────────────────────────────

    def is_halted(self) -> bool:
        if self.state.halted:
            return True
        if self.state.balance_cents == 0 or self.state.start_of_day_balance == 0:
            return False
        daily_loss_pct = (
            (self.state.start_of_day_balance - self.state.balance_cents)
            / self.state.start_of_day_balance
        )
        if daily_loss_pct >= self.cfg.max_daily_loss_pct:
            log.warning(f"HALT: daily loss {daily_loss_pct*100:.1f}% exceeds limit")
            self.state.halted = True
        return self.state.halted

    def current_exposure_cents(self) -> int:
        """Sum of cost basis for all open positions."""
        total = 0
        for pos in self.state.positions.values():
            qty = abs(pos.get("position", 0))
            avg_price = pos.get("market_exposure", 0) / max(1, qty) if qty else 0
            total += qty * avg_price
        return int(total)

    def can_open_position(self, signal: Signal) -> bool:
        if self.is_halted():
            log.warning("Risk halt active — skipping trade")
            return False
        if signal.edge < self.cfg.min_edge:
            log.debug(f"Edge {signal.edge:.3f} < min {self.cfg.min_edge} — skip")
            return False
        if signal.confidence < self.cfg.min_confidence:
            log.debug(f"Confidence too low — skip")
            return False
        exposure = self.current_exposure_cents()
        max_exp = self.state.balance_cents * self.cfg.max_exposure_pct
        if exposure >= max_exp:
            log.info(f"Max exposure reached ({exposure/100:.2f} / {max_exp/100:.2f})")
            return False
        if signal.ticker in self.state.positions:
            log.debug(f"Already have position in {signal.ticker} — skip")
            return False
        return True

    # ── Position sizing ─────────────────────────────────────────────────────

    def size_position(self, signal: Signal) -> int:
        """Returns number of contracts to buy (0 = don't trade)."""
        balance = self.state.balance_cents
        if balance <= 0:
            return 0

        # Kelly-based sizing
        kf = kelly_fraction(signal.fair_value, signal.market_price, self.cfg.kelly_scale)

        # Hard cap by max_position_pct
        max_dollars = balance * self.cfg.max_position_pct / 100.0   # in contracts
        kelly_contracts = int(kf * (balance / 100.0) / (signal.market_price / 100.0))

        # Remaining exposure headroom
        exposure = self.current_exposure_cents()
        max_exp = balance * self.cfg.max_exposure_pct
        remaining_cents = max(0, max_exp - exposure)
        max_from_exposure = int(remaining_cents / signal.market_price)

        contracts = min(kelly_contracts,
                        int(max_dollars),
                        max_from_exposure,
                        self.cfg.max_contracts_per_trade)
        contracts = max(self.cfg.min_contracts, contracts)

        if contracts * signal.market_price > balance * self.cfg.max_position_pct:
            contracts = int(balance * self.cfg.max_position_pct / signal.market_price)

        contracts = max(0, contracts)
        log.debug(f"Size: kelly={kelly_contracts} cap={int(max_dollars)} "
                  f"exp={max_from_exposure} → {contracts} contracts")
        return contracts

    def stop_loss_price(self, entry_price: float) -> float:
        """Price at which we exit to cut losses (as YES price)."""
        loss_per_contract = (self.state.balance_cents * self.cfg.max_single_loss_pct
                             / max(1, len(self.state.positions) + 1))
        # If entry=40¢, stop at entry - loss (floor 1¢)
        return max(1.0, entry_price - loss_per_contract / 100.0)
