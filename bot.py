"""
Kalshi Arbitrage & Edge Bot
───────────────────────────
Indefinitely scans open markets, finds +EV lines, enters positions,
and exits when the price moves toward fair value.

Usage:
  python bot.py                       # start (reads .env or config.json)
  python bot.py --dry-run             # simulate without placing orders
  python bot.py --demo                # use Kalshi demo environment
  python bot.py --scan-only           # print signals, never trade
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from kalshi_client import KalshiClient
from strategy import (Signal, analyze_market, rank_signals, check_event_arb)
from risk import RiskConfig, RiskManager
from longshot import check_longshot
from valuation import PriceTracker, check_value as check_value_signal

# ── Logging ───────────────────────────────────────────────────────────────────

LOG_FILE = "kalshi_bot.log"


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s  %(levelname)-7s  %(name)s  %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE, mode="a"),
        ],
    )


log = logging.getLogger("kalshi.bot")

# ── ANSI colors for terminal ──────────────────────────────────────────────────

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    GREEN = Fore.GREEN
    RED = Fore.RED
    YELLOW = Fore.YELLOW
    CYAN = Fore.CYAN
    BOLD = Style.BRIGHT
    RESET = Style.RESET_ALL
except ImportError:
    GREEN = RED = YELLOW = CYAN = BOLD = RESET = ""

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    # Environment — default to the demo (paper) environment for safety.
    "demo": True,
    "dry_run": False,
    "scan_only": False,

    # Scan settings
    "scan_interval_secs": 60,
    "monitor_interval_secs": 30,
    # scan_all_markets=True paginates the entire open-market universe (~600k
    # markets, slow). False = only the target_series list (fast). Default False.
    "scan_all_markets": False,
    "max_markets_per_scan": 0,        # 0 = no cap

    # Risk parameters
    "max_position_pct": 0.10,
    "max_exposure_pct": 0.60,
    "kelly_scale": 0.50,
    "min_edge": 0.02,
    "min_confidence": 0.30,
    "max_daily_loss_pct": 0.15,

    # PRIMARY strategy: historical-mispricing / mean-reversion value engine
    # (valuation.py). Buys markets cheap relative to their calibrated fair value.
    "value_enabled": True,
    "value_min_obs": 5,
    "value_ema_alpha": 0.35,
    "value_sharpen": 1.10,
    "value_min_gap_cents": 3,
    "value_max_gap_cents": 12,
    "value_min_edge": 0.03,
    "value_max_edge": 0.40,
    "value_min_price": 5,
    "value_max_price": 95,
    "value_min_volume": 50,
    "value_max_anchor_range_cents": 10,
    "value_confidence": 0.50,
    "history_path": "price_history.json",
    "history_max_obs": 288,
    "history_min_interval_secs": 300,

    # Analytic per-market strategies. ARBS is risk-free; the others are
    # directional heuristics. ARBS kept on as a free bonus alongside VALUE.
    "strategies": ["ARBS"],

    # Event-level arbitrage detector (risk-free; alerts only, never auto-trades).
    "event_arb_enabled": True,
    "event_arb_min_cents": 2,

    # Don't let reconcile drop a position younger than this (anti-churn safety).
    "reconcile_grace_secs": 120,

    # Exit — risk-managed: cut losers fast, let winners run via a trailing stop.
    "exit_max_loss_cents": 8,           # hard stop: max loss per contract (¢)
    "exit_trail_activate_cents": 3,     # start trailing once up this many ¢
    "exit_trail_give_back_cents": 3,    # exit if bid falls this far from peak
    "profit_target_mult": 0.50,
    "max_hold_hours": 24,
}

def load_config(path: str = "config.json") -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(path):
        with open(path) as f:
            cfg.update(json.load(f))
    return cfg


def load_credentials(path: str = "credentials.json") -> tuple:
    if not os.path.exists(path):
        print(f"{RED}ERROR: {path} not found. Create it with your key_id and private_key_path.{RESET}")
        sys.exit(1)
    with open(path) as f:
        creds = json.load(f)
    key_id = creds.get("key_id", "")
    pem_path = creds.get("private_key_path", "kalshi_private.pem")
    if not key_id:
        print(f"{RED}ERROR: key_id missing from {path}{RESET}")
        sys.exit(1)
    if not os.path.exists(pem_path):
        print(f"{RED}ERROR: Private key not found at '{pem_path}'{RESET}")
        sys.exit(1)
    with open(pem_path) as f:
        private_key_pem = f.read()
    return key_id, private_key_pem


# ── Position tracker ──────────────────────────────────────────────────────────

POSITIONS_FILE = "open_positions.json"


class OpenPosition:
    def __init__(self, signal: Signal, contracts: int, order_id: str):
        self.signal = signal
        self.contracts = contracts
        self.order_id = order_id
        self.entry_time = datetime.now(timezone.utc)
        self.entry_price = signal.market_price
        self.target_price = signal.target_exit_cents
        # NOTE: the actual stop / trailing-stop is computed in _check_exit from
        # config so risk/reward stays symmetric. stop_price is kept only as a
        # display/persistence field; it is no longer the catastrophic
        # entry×0.4 (60% loss) it used to be.
        self.stop_price = max(1.0, signal.market_price * 0.4)
        # High-water mark of the best bid seen since entry — drives the
        # trailing stop that lets winners run instead of selling for pennies.
        self.peak_price = signal.market_price
        self.unrealized_pnl_cents = 0

    def hours_held(self) -> float:
        delta = datetime.now(timezone.utc) - self.entry_time
        return delta.total_seconds() / 3600

    def to_dict(self) -> dict:
        return {
            "ticker": self.signal.ticker,
            "side": self.signal.side,
            "strategy": self.signal.strategy,
            "fair_value": self.signal.fair_value,
            "market_price": self.signal.market_price,
            "edge": self.signal.edge,
            "confidence": self.signal.confidence,
            "target_exit_cents": self.signal.target_exit_cents,
            "market_title": self.signal.market_title,
            "event_ticker": self.signal.event_ticker,
            "contracts": self.contracts,
            "order_id": self.order_id,
            "entry_time": self.entry_time.isoformat(),
            "entry_price": self.entry_price,
            "target_price": self.target_price,
            "stop_price": self.stop_price,
            "peak_price": self.peak_price,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OpenPosition":
        from strategy import Signal
        sig = Signal(
            ticker=d["ticker"],
            side=d["side"],
            strategy=d["strategy"],
            fair_value=d["fair_value"],
            market_price=d["market_price"],
            edge=d["edge"],
            confidence=d["confidence"],
            target_exit_cents=d["target_exit_cents"],
            market_title=d.get("market_title", ""),
            event_ticker=d.get("event_ticker", ""),
        )
        pos = cls(sig, d["contracts"], d["order_id"])
        pos.entry_time = datetime.fromisoformat(d["entry_time"])
        pos.entry_price = d["entry_price"]
        pos.target_price = d.get("target_price", pos.target_price)
        pos.stop_price = d.get("stop_price", pos.stop_price)
        pos.peak_price = d.get("peak_price", pos.entry_price)
        return pos

    def __str__(self):
        return (f"{self.signal.ticker} {self.signal.side.upper()} "
                f"×{self.contracts} entry={self.entry_price:.0f}¢ "
                f"target={self.target_price:.0f}¢ age={self.hours_held():.1f}h")


# ── Bot ───────────────────────────────────────────────────────────────────────

class KalshiBot:
    def __init__(self, client: KalshiClient, cfg: dict,
                 risk: RiskManager, dry_run: bool, scan_only: bool):
        self.client = client
        self.cfg = cfg
        self.risk = risk
        self.dry_run = dry_run
        self.scan_only = scan_only
        self.positions: Dict[str, OpenPosition] = {}  # ticker -> OpenPosition
        self.trade_log: List[dict] = []
        self.cycle = 0
        self.signals_found = 0
        self.trades_placed = 0
        # Primary engine: per-market price history for the VALUE strategy.
        self.prices = PriceTracker(cfg)

    # ── Market scan ───────────────────────────────────────────────────────

    # Default series used only when scan_all_markets is False.
    DEFAULT_SERIES = [
        "KXBTC", "KXETH", "KXINX", "KXFED", "KXCPI",
        "KXNBA", "KXNHL", "KXMLB", "KXUNEMP", "KXGDP",
        "KXOIL", "KXGOLD", "KXEUR", "KXJPY",
    ]
    DEFAULT_LONGSHOT_SERIES = [
        "KXNBAPTS", "KXNBAREB", "KXNBAAST", "KXNBABLK",
        "KXNHLPTS", "KXMLBPTS", "HIGHNY", "HIGHCHI",
        "HIGHLA", "HIGHMIA", "HIGHBOS",
    ]

    def _fetch_all_open_markets(self, cap: int) -> List[dict]:
        """Paginate the full set of open markets (cursor-based). cap<=0 = no cap."""
        markets: List[dict] = []
        cursor = None
        pages = 0
        while True:
            try:
                resp = self.client.get_markets(limit=1000, cursor=cursor,
                                               status="open")
            except Exception as e:
                log.error(f"get_markets (all) failed on page {pages+1}: {e}")
                break
            batch = resp.get("markets", [])
            markets.extend(batch)
            pages += 1
            cursor = resp.get("cursor") or ""
            if cap > 0 and len(markets) >= cap:
                markets = markets[:cap]
                break
            if not cursor or not batch:
                break
            time.sleep(0.2)
        log.info(f"  Fetched {len(markets)} open markets across {pages} page(s)")
        return markets

    def _fetch_series_markets(self, cap: int) -> List[dict]:
        """Fetch markets only for the configured target/longshot series."""
        target_series = list(self.cfg.get("target_series", self.DEFAULT_SERIES))
        if self.cfg.get("longshot_enabled", False):
            for s in self.cfg.get("longshot_series", self.DEFAULT_LONGSHOT_SERIES):
                if s not in target_series:
                    target_series.append(s)

        markets: List[dict] = []
        for series in target_series:
            try:
                resp = self.client.get_markets(limit=200, status="open",
                                               series_ticker=series)
            except Exception as e:
                log.error(f"get_markets failed for {series}: {e}")
                continue
            markets.extend(resp.get("markets", []))
            if cap > 0 and len(markets) >= cap:
                markets = markets[:cap]
                break
            time.sleep(0.2)
        log.info(f"  Fetched {len(markets)} markets across "
                 f"{len(target_series)} series")
        return markets

    def _fetch_market_universe(self) -> List[dict]:
        cap = int(self.cfg.get("max_markets_per_scan", 0) or 0)
        if self.cfg.get("scan_all_markets", False):
            return self._fetch_all_open_markets(cap)
        return self._fetch_series_markets(cap)

    def scan_markets(self) -> List[Signal]:
        log.info(f"── Scanning markets (cycle {self.cycle}) ──")
        signals = []
        all_markets = []          # liquid markets only, for event-level arb
        scanned = 0
        liquid = 0                # markets with a tradeable book (yes_ask>0)
        has_volume = 0            # markets with any 24h volume
        value_warming = 0         # VALUE markets still in history warm-up
        active_strategies = set(self.cfg["strategies"])
        longshot_on = self.cfg.get("longshot_enabled", False)
        value_on = self.cfg.get("value_enabled", True)
        min_obs = int(self.cfg.get("value_min_obs", 4))
        now = datetime.now(timezone.utc)
        live_tickers = set()
        sample_market = None

        for market in self._fetch_market_universe():
            ticker = market.get("ticker", "")
            if not ticker:
                continue
            scanned += 1
            if sample_market is None:
                sample_market = market

            # Cheap liquidity gate: skip the huge tail of dead micro-markets
            # before doing any work. A market with no ask can't be arbed or
            # bought, so it's useless to us regardless of strategy.
            if (market.get("yes_ask", 0) or 0) <= 0 and (market.get("no_ask", 0) or 0) <= 0:
                continue
            liquid += 1
            live_tickers.add(ticker)
            if market.get("volume_24h", 0) > 0:
                has_volume += 1

            all_markets.append(market)

            # Record price into the rolling history (down-sampled internally).
            if value_on:
                self.prices.record(market)

            # Skip markets closing in < 5 min
            close_ts = market.get("close_time", "")
            if close_ts:
                try:
                    close_dt = datetime.fromisoformat(
                        close_ts.replace("Z", "+00:00"))
                    if (close_dt - now).total_seconds() / 60 < 5:
                        continue
                except Exception:
                    pass

            # PRIMARY strategy: historical-mispricing / mean-reversion value.
            if value_on:
                hist = self.prices.history(ticker)
                if len(hist) < min_obs:
                    value_warming += 1
                else:
                    vs = check_value_signal(market, hist, self.cfg)
                    if vs:
                        signals.append(vs)
                        log.info(f"  Value: {vs}  ({vs.extra})")

            # Per-market analytic strategies (ARBS + optional heuristics).
            sig = analyze_market(market, active_strategies)
            if sig:
                signals.append(sig)

            # Model-based longshots run independently on the same market data.
            if longshot_on:
                ls = check_longshot(market, self.cfg)
                if ls:
                    signals.append(ls)
                    log.info(f"  Longshot: {ls}")

        # Event-level arbitrage runs across the liquid universe.
        self.detect_event_arbs(all_markets)

        # Persist + prune the price history once per scan.
        if value_on:
            self.prices.prune(live_tickers)
            self.prices.save()

        self.signals_found += len(signals)
        warm = f", {value_warming} VALUE warming up" if value_warming else ""
        log.info(f"  Scanned {scanned} markets ({liquid} with a book, "
                 f"{has_volume} with 24h volume{warm}) → {len(signals)} signals")

        # Health check: we fetched markets but none were tradeable. This almost
        # always means the API response shape changed (e.g. price fields renamed)
        # and normalization in kalshi_client needs updating. Warn with a sample
        # so it can't fail silently. Throttled to avoid log spam.
        if scanned > 0 and liquid == 0 and sample_market is not None \
                and self.cycle % 20 == 1:
            sample = {k: sample_market.get(k) for k in
                      ("ticker", "yes_ask", "no_ask", "last_price", "volume_24h")}
            log.warning(f"  No tradeable markets in {scanned} fetched — "
                        f"possible API field change. Sample: {sample}")

        return rank_signals(signals)

    # ── Event-level arbitrage (detector / alerter) ─────────────────────────

    def detect_event_arbs(self, all_markets: List[dict]):
        """
        Group scanned markets by event and flag mutually-exclusive events whose
        YES prices over-round (sum above 100¢) — buying NO on every leg is then
        risk-free. This DETECTS and ALERTS only; it does not place the multi-leg
        order automatically (a partial fill across many legs is unsafe). Acted
        opportunities are written to arb_alerts.jsonl.
        """
        if not self.cfg.get("event_arb_enabled", True):
            return

        from collections import defaultdict
        by_event = defaultdict(list)
        for m in all_markets:
            et = m.get("event_ticker", "")
            if et:
                by_event[et].append(m)

        min_cents = self.cfg.get("event_arb_min_cents", 2)
        for event_ticker, mkts in by_event.items():
            if len(mkts) < 2:
                continue
            opp = check_event_arb(mkts, min_cents)
            if not opp:
                continue
            # The arb only holds if the outcomes are mutually exclusive — verify
            # against Kalshi before trusting it.
            if not self._event_is_exclusive(event_ticker):
                continue
            self._alert_event_arb(event_ticker, opp)

    def _event_is_exclusive(self, event_ticker: str) -> bool:
        try:
            resp = self.client.get_event(event_ticker)
        except Exception as e:
            log.debug(f"  get_event failed {event_ticker}: {e}")
            return False
        ev = resp.get("event", resp)
        return bool(ev.get("mutually_exclusive", False))

    def _alert_event_arb(self, event_ticker: str, opp: dict):
        msg = (f"EVENT ARB {event_ticker}: buy NO on {opp['n']} legs for "
               f"${opp['total_cost']/100:.2f} → guaranteed "
               f"${opp['guaranteed_profit']/100:.2f} "
               f"({opp['profit_per_dollar']*100:.1f}% of capital)")
        log.warning(f"{BOLD}{YELLOW}*** {msg} ***{RESET}")
        for tkr, no_ask in opp["legs"]:
            log.warning(f"      NO {tkr} @ {no_ask:.0f}¢")
        record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "event_ticker": event_ticker,
            **opp,
        }
        try:
            with open("arb_alerts.jsonl", "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            log.debug(f"  could not write arb_alerts.jsonl: {e}")

    # ── Position persistence ──────────────────────────────────────────────

    def save_positions(self):
        """Write open positions to disk after every change."""
        data = {ticker: pos.to_dict() for ticker, pos in self.positions.items()}
        with open(POSITIONS_FILE, "w") as f:
            json.dump(data, f, indent=2)

    def load_positions(self):
        """Reload positions from disk on startup."""
        if not os.path.exists(POSITIONS_FILE):
            return
        try:
            with open(POSITIONS_FILE) as f:
                data = json.load(f)
            for ticker, d in data.items():
                self.positions[ticker] = OpenPosition.from_dict(d)
            log.info(f"Loaded {len(self.positions)} positions from disk: "
                     f"{list(self.positions.keys())}")
        except Exception as e:
            log.error(f"Failed to load positions file: {e}")

    def reconcile_positions(self):
        """
        Compare bot's in-memory positions against Kalshi's actual portfolio.
        - Positions on Kalshi but not in bot → adopt them so they get monitored
        - Positions in bot but not on Kalshi → remove (already closed/expired)
        """
        try:
            resp = self.client.get_positions()
            raw_positions = resp.get("market_positions", [])

            # Kalshi may use 'market_ticker' or 'ticker' — handle both
            kalshi_positions = {}
            for p in raw_positions:
                ticker = p.get("market_ticker") or p.get("ticker", "")
                if ticker and p.get("position", 0) != 0:
                    kalshi_positions[ticker] = p

        except Exception as e:
            log.warning(f"Reconcile failed — could not fetch positions: {e}")
            return

        # Remove positions the bot thinks are open but Kalshi says are closed.
        #
        # SAFETY GRACE PERIOD: never drop a position we opened in the last
        # `reconcile_grace_secs` seconds. A freshly placed order may not yet be
        # reflected by the positions endpoint (fill/settlement lag), and a parse
        # mismatch here previously caused the bot to delete its own brand-new
        # positions, lose the duplicate-guard, and re-buy the same tickers in a
        # fee-burning loop. The grace period makes that impossible.
        grace = self.cfg.get("reconcile_grace_secs", 120)
        for ticker in list(self.positions.keys()):
            if ticker not in kalshi_positions:
                pos = self.positions[ticker]
                age = (datetime.now(timezone.utc) - pos.entry_time).total_seconds()
                if age < grace:
                    log.info(f"RECONCILE: {ticker} not on Kalshi yet but only "
                             f"{age:.0f}s old (< {grace}s grace) — keeping")
                    continue
                log.warning(f"RECONCILE: {ticker} not found on Kalshi — removing from bot")
                del self.positions[ticker]

        # Adopt positions on Kalshi that the bot doesn't know about
        for ticker, kpos in kalshi_positions.items():
            if ticker not in self.positions:
                qty = abs(kpos.get("position", 0))
                side = "yes" if kpos.get("position", 0) > 0 else "no"
                exposure = kpos.get("market_exposure", 0) or kpos.get("total_cost", 0)
                entry = float(exposure / max(qty, 1)) if qty else 50.0
                log.warning(f"RECONCILE: adopting orphaned position {ticker} "
                            f"{side.upper()} ×{qty} entry~{entry:.0f}¢")
                sig = Signal(
                    ticker=ticker,
                    side=side,
                    strategy="ORPHAN",
                    fair_value=0.5,
                    market_price=entry,
                    edge=0.0,
                    confidence=0.0,
                    target_exit_cents=entry * 1.1,
                    market_title=ticker,
                )
                pos = OpenPosition(sig, qty, "orphan")
                pos.entry_price = entry
                pos.target_price = entry * 1.1
                pos.stop_price = entry * 0.5
                self.positions[ticker] = pos

        if kalshi_positions or self.positions:
            log.info(f"Reconcile complete — {len(self.positions)} positions active")
        self.save_positions()

    # ── Order execution ───────────────────────────────────────────────────

    def execute_signal(self, signal: Signal) -> bool:
        if not self.risk.can_open_position(signal):
            return False

        contracts = self.risk.size_position(signal)
        if contracts <= 0:
            log.info(f"  Sizing returned 0 contracts for {signal.ticker} — skip")
            return False

        cost_cents = contracts * signal.market_price
        log.info(f"{BOLD}{GREEN}→ ENTER{RESET}  {signal}  ×{contracts}  "
                 f"cost=${cost_cents/100:.2f}  "
                 f"{'[DRY RUN]' if self.dry_run else ''}")

        if self.dry_run or self.scan_only:
            fake_id = f"dry_{signal.ticker}_{int(time.time())}"
            pos = OpenPosition(signal, contracts, fake_id)
            self.positions[signal.ticker] = pos
            self.trades_placed += 1
            return True

        if signal.strategy == "ARBS":
            return self._execute_arb(signal, contracts)
        return self._execute_single(signal, contracts)

    @staticmethod
    def _order_id(ticker: str, tag: str = "") -> str:
        import hashlib
        raw = f"{ticker}{tag}{time.time_ns()}"
        return f"bot_{hashlib.md5(raw.encode()).hexdigest()[:10]}"

    def _execute_single(self, signal: Signal, contracts: int) -> bool:
        try:
            resp = self.client.place_order(
                ticker=signal.ticker,
                side=signal.side,
                action="buy",
                count=contracts,
                price_cents=int(signal.market_price),
                order_type="limit",
                client_order_id=self._order_id(signal.ticker),
            )
            order_id = resp.get("order", {}).get("order_id", "")
            pos = OpenPosition(signal, contracts, order_id)
            self.positions[signal.ticker] = pos
            self.trades_placed += 1
            self._log_trade("OPEN", signal, contracts, order_id)
            self.save_positions()
            log.info(f"  Order placed: {order_id}")
            return True
        except Exception as e:
            log.error(f"  Order failed for {signal.ticker}: {e}")
            return False

    def _execute_arb(self, signal: Signal, contracts: int) -> bool:
        """
        Place both legs of a single-market arbitrage (buy YES + buy NO). Both
        legs must fill for the position to be risk-free. If the second leg
        fails, immediately try to unwind the first so we never sit on a naked
        directional position.
        """
        yes_ask = int(signal.extra.get("yes_ask", 0))
        no_ask = int(signal.extra.get("no_ask", 0))
        if yes_ask <= 0 or no_ask <= 0:
            log.error(f"  ARB {signal.ticker}: missing leg prices — skip")
            return False

        try:
            r1 = self.client.place_order(
                ticker=signal.ticker, side="yes", action="buy",
                count=contracts, price_cents=yes_ask, order_type="limit",
                client_order_id=self._order_id(signal.ticker, "y"))
        except Exception as e:
            log.error(f"  ARB {signal.ticker}: YES leg failed ({e}) — no position opened")
            return False

        try:
            self.client.place_order(
                ticker=signal.ticker, side="no", action="buy",
                count=contracts, price_cents=no_ask, order_type="limit",
                client_order_id=self._order_id(signal.ticker, "n"))
        except Exception as e:
            log.error(f"  ARB {signal.ticker}: NO leg failed ({e}) — unwinding YES leg")
            try:
                self.client.place_order(
                    ticker=signal.ticker, side="yes", action="sell",
                    count=contracts, price_cents=max(1, yes_ask - 1),
                    order_type="limit")
            except Exception as e2:
                log.critical(f"  ARB {signal.ticker}: UNWIND FAILED ({e2}) — "
                             f"NAKED YES x{contracts} @ {yes_ask}c, MANUAL ACTION NEEDED")
            return False

        order_id = r1.get("order", {}).get("order_id", "")
        pos = OpenPosition(signal, contracts, order_id)
        self.positions[signal.ticker] = pos
        self.trades_placed += 1
        net = signal.extra.get("net_profit", 0)
        log.info(f"  ARB legs placed: YES@{yes_ask}c + NO@{no_ask}c x{contracts} "
                 f"-> locked ~${net*contracts/100:.2f} at settlement")
        self._log_trade("OPEN", signal, contracts, order_id)
        self.save_positions()
        return True

    # ── Position monitoring & exit ─────────────────────────────────────────

    def monitor_positions(self):
        if not self.positions:
            return
        log.info(f"── Monitoring {len(self.positions)} positions ──")

        for ticker, pos in list(self.positions.items()):
            try:
                self._check_exit(ticker, pos)
            except Exception as e:
                log.error(f"  monitor error {ticker}: {e}")

    def _check_exit(self, ticker: str, pos: OpenPosition):
        # Arbitrage positions hold both legs to settlement — never exit early.
        if pos.signal.strategy == "ARBS":
            return
        try:
            market_resp = self.client.get_market(ticker)
        except Exception as e:
            log.debug(f"  market fetch failed {ticker}: {e}")
            return

        market = market_resp.get("market", market_resp)
        yes_bid = market.get("yes_bid", 0)
        yes_ask = market.get("yes_ask", 0)
        no_bid = market.get("no_bid", 0)

        if pos.signal.side == "yes":
            current_bid = float(yes_bid) if yes_bid else None
        else:
            current_bid = float(no_bid) if no_bid else None

        if not current_bid:
            return

        pos.unrealized_pnl_cents = (current_bid - pos.entry_price) * pos.contracts

        hours = pos.hours_held()
        reasons = []

        # ── Risk-managed exit (cut losers fast, let winners run) ──────────────
        # Update the high-water mark of the best bid since entry.
        if current_bid > pos.peak_price:
            pos.peak_price = current_bid

        max_loss = float(self.cfg.get("exit_max_loss_cents", 8))
        trail_act = float(self.cfg.get("exit_trail_activate_cents", 3))
        trail_give = float(self.cfg.get("exit_trail_give_back_cents", 3))

        # Hard stop: never lose more than `exit_max_loss_cents` per contract.
        # Replaces the old entry*0.4 (~60% loss) stop that let losers run.
        hard_stop = max(1.0, pos.entry_price - max_loss)

        # Trailing stop: once up at least `trail_act`, exit if the bid falls
        # `trail_give` from its peak — lets a winner keep running instead of
        # being dumped at the tiny fixed profit target.
        trailing_stop = None
        if pos.peak_price - pos.entry_price >= trail_act:
            trailing_stop = pos.peak_price - trail_give

        effective_stop = hard_stop
        if trailing_stop is not None:
            effective_stop = max(hard_stop, trailing_stop)
        pos.stop_price = effective_stop  # keep display/persistence in sync

        if current_bid <= effective_stop:
            kind = ("TRAIL" if trailing_stop is not None
                    and effective_stop == trailing_stop else "STOP")
            reasons.append(f"{kind}  {current_bid:.0f}c <= {effective_stop:.0f}c "
                           f"(peak {pos.peak_price:.0f}c)")
        elif current_bid >= 97:
            reasons.append(f"TAKE_PROFIT  {current_bid:.0f}c >= 97c")
        if hours >= self.cfg["max_hold_hours"]:
            reasons.append(f"MAX_HOLD  {hours:.1f}h >= {self.cfg['max_hold_hours']}h")

        if reasons:
            reason = " | ".join(reasons)
            pnl = pos.unrealized_pnl_cents
            color = GREEN if pnl >= 0 else RED
            log.info(f"{color}← EXIT{RESET}  {pos}  "
                     f"pnl=${pnl/100:+.2f}  reason={reason}")
            self._exit_position(ticker, pos, current_bid)

    def _exit_position(self, ticker: str, pos: OpenPosition, exit_price: float):
        if self.dry_run or self.scan_only:
            del self.positions[ticker]
            self.save_positions()
            return

        # Place the sell first; only forget the position once it's accepted, so
        # a failed exit order doesn't orphan a live position.
        try:
            self.client.place_order(
                ticker=ticker,
                side=pos.signal.side,
                action="sell",
                count=pos.contracts,
                price_cents=int(exit_price),
                order_type="limit",
            )
        except Exception as e:
            log.error(f"  Exit order failed {ticker}: {e} — keeping position for retry")
            return

        self._log_trade("CLOSE", pos.signal, pos.contracts, "",
                        entry=pos.entry_price, exit_p=exit_price)
        del self.positions[ticker]
        self.save_positions()

    def _log_trade(self, action: str, signal: Signal, contracts: int,
                   order_id: str, entry: float = None, exit_p: float = None):
        record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "ticker": signal.ticker,
            "side": signal.side,
            "strategy": signal.strategy,
            "contracts": contracts,
            "price": signal.market_price,
            "edge": round(signal.edge, 4),
            "order_id": order_id,
        }
        if entry:
            record["entry"] = entry
        if exit_p:
            record["exit"] = exit_p
            record["pnl_cents"] = (exit_p - entry) * contracts if entry else None
        self.trade_log.append(record)
        with open("trade_log.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")

    # ── Status display ────────────────────────────────────────────────────

    def print_status(self):
        try:
            bal = self.client.get_balance()
            balance = bal.get("balance", 0)  # in cents
            self.risk.update_balance(balance)
        except Exception as e:
            balance = self.risk.state.balance_cents
            log.debug(f"Balance fetch failed: {e}")

        print(f"\n{'─'*65}")
        print(f"  {BOLD}Kalshi Edge Bot{RESET}  cycle={self.cycle}  "
              f"{datetime.now().strftime('%H:%M:%S')}")
        print(f"  Balance: {BOLD}${balance/100:.2f}{RESET}  "
              f"Positions: {len(self.positions)}  "
              f"Trades: {self.trades_placed}  "
              f"Signals found: {self.signals_found}")
        if self.positions:
            print(f"\n  Open positions:")
            for ticker, pos in self.positions.items():
                pnl = pos.unrealized_pnl_cents
                col = GREEN if pnl >= 0 else RED
                print(f"    {col}{pos}{RESET}  upnl=${pnl/100:+.2f}")
        if self.risk.is_halted():
            print(f"  {RED}{BOLD}⚠  RISK HALT ACTIVE — no new trades{RESET}")
        print(f"{'─'*65}\n")

    # ── Main loop ─────────────────────────────────────────────────────────

    def run(self):
        log.info(f"{'='*60}")
        log.info(f"  KALSHI EDGE BOT STARTING")
        log.info(f"  Mode: {'DEMO' if self.client.demo else 'LIVE'}")
        log.info(f"  Dry-run: {self.dry_run}  Scan-only: {self.scan_only}")
        log.info(f"  Strategies: {self.cfg['strategies']}")
        log.info(f"{'='*60}")

        # Reload and reconcile positions from previous run
        self.load_positions()
        self.reconcile_positions()

        last_scan = 0
        last_monitor = 0
        scan_interval = self.cfg["scan_interval_secs"]
        monitor_interval = self.cfg["monitor_interval_secs"]

        while True:
            now = time.time()
            try:
                # Refresh balance + positions
                try:
                    bal = self.client.get_balance()
                    self.risk.update_balance(bal.get("balance", 0))
                    pos_resp = self.client.get_positions()
                    self.risk.update_positions(
                        pos_resp.get("market_positions", []))
                except Exception as e:
                    log.warning(f"Portfolio refresh error: {e}")

                # Monitor open positions
                if now - last_monitor >= monitor_interval:
                    self.monitor_positions()
                    last_monitor = now

                # Full market scan
                if now - last_scan >= scan_interval:
                    self.cycle += 1
                    self.reconcile_positions()
                    self.print_status()
                    if not self.risk.is_halted():
                        signals = self.scan_markets()
                        # Attempt top signals
                        for sig in signals[:5]:   # max 5 new trades per cycle
                            if not self.risk.can_open_position(sig):
                                continue
                            self.execute_signal(sig)
                            time.sleep(0.5)
                    last_scan = now

            except KeyboardInterrupt:
                log.info("Interrupted by user — shutting down")
                break
            except Exception as e:
                log.error(f"Unexpected error: {e}\n{traceback.format_exc()}")

            time.sleep(2)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Kalshi Edge Trading Bot")
    parser.add_argument("--config", default="config.json",
                        help="Path to config JSON file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate trades without placing orders")
    parser.add_argument("--demo", action="store_true",
                        help="Use Kalshi demo/paper trading environment")
    parser.add_argument("--scan-only", action="store_true",
                        help="Print signals only, never trade")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose debug logging")
    args = parser.parse_args()

    setup_logging(args.verbose)
    cfg = load_config(args.config)
    key_id, private_key_pem = load_credentials()

    # CLI flags override config
    if args.dry_run:
        cfg["dry_run"] = True
    if args.demo:
        cfg["demo"] = True
    if args.scan_only:
        cfg["scan_only"] = True
        cfg["dry_run"] = True

    client = KalshiClient(
        key_id=key_id,
        private_key_pem=private_key_pem,
        demo=cfg.get("demo", False),
    )

    # Preflight auth check — fail fast with a clear message instead of looping
    # on "Portfolio refresh error: 401".
    env = "DEMO" if cfg.get("demo", False) else "LIVE"
    try:
        client.get_balance()
        log.info(f"Auth OK ({env})")
    except Exception as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status == 401:
            print(f"{RED}{BOLD}ERROR: Kalshi rejected your credentials (401) on the "
                  f"{env} environment.{RESET}")
            print("Kalshi's DEMO and LIVE accounts use SEPARATE API keys.")
            print("  • For --demo: generate a key at https://demo.kalshi.co "
                  "(Account → API Keys)")
            print("  • For live:   generate a key at https://kalshi.com")
            print("Then put the matching key_id in credentials.json and the "
                  "private key in kalshi_private.pem.")
            print("If the key is correct, verify your system clock is accurate "
                  "(requests are signed with a timestamp).")
            sys.exit(1)
        # Non-auth error (network, etc.) — let the run loop's retry logic handle it.
        log.warning(f"Preflight balance check failed ({env}): {e}")

    risk_cfg = RiskConfig(
        max_position_pct=cfg["max_position_pct"],
        max_exposure_pct=cfg["max_exposure_pct"],
        kelly_scale=cfg["kelly_scale"],
        min_edge=cfg["min_edge"],
        min_confidence=cfg["min_confidence"],
        max_daily_loss_pct=cfg["max_daily_loss_pct"],
    )
    risk = RiskManager(risk_cfg)

    bot = KalshiBot(
        client=client,
        cfg=cfg,
        risk=risk,
        dry_run=cfg.get("dry_run", False),
        scan_only=cfg.get("scan_only", False),
    )
    bot.run()


if __name__ == "__main__":
    main()