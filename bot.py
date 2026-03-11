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
from strategy import (Signal, analyze_market, rank_signals)
from risk import RiskConfig, RiskManager
from longshot import check_longshot

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
    # Environment
    "demo": False,
    "dry_run": False,
    "scan_only": False,

    # Scan settings
    "scan_interval_secs": 15,
    "monitor_interval_secs": 5,
    "max_markets_per_scan": 150,

    # Risk parameters
    "max_position_pct": 0.10,
    "max_exposure_pct": 0.60,
    "kelly_scale": 0.50,
    "min_edge": 0.02,
    "min_confidence": 0.30,
    "max_daily_loss_pct": 0.15,

    # Strategies
    "strategies": ["ARBS", "SPREAD", "DRIFT", "VALUE"],

    # Exit
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
        self.stop_price = max(1.0, signal.market_price * 0.4)
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
        pos.target_price = d["target_price"]
        pos.stop_price = d["stop_price"]
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

    # ── Market scan ───────────────────────────────────────────────────────

    def scan_markets(self) -> List[Signal]:
        log.info(f"── Scanning markets (cycle {self.cycle}) ──")
        signals = []
        scanned = 0
        active_strategies = set(self.cfg["strategies"])

        # Target series known to have liquidity
        target_series = self.cfg.get("target_series", [
            "KXBTC", "KXETH", "KXINX", "KXFED", "KXCPI",
            "KXNBA", "KXNHL", "KXMLB", "KXUNEMP", "KXGDP",
            "KXOIL", "KXGOLD", "KXEUR", "KXJPY",
        ])

        # Add longshot-specific series if enabled
        if self.cfg.get("longshot_enabled", False):
            longshot_series = self.cfg.get("longshot_series", [
                "KXNBAPTS", "KXNBAREB", "KXNBAAST", "KXNBABLK",
                "KXNHLPTS", "KXMLBPTS", "HIGHNY", "HIGHCHI",
                "HIGHLA", "HIGHMIA", "HIGHBOS",
            ])
            # Merge without duplicates
            for s in longshot_series:
                if s not in target_series:
                    target_series = target_series + [s]

        for series in target_series:
            try:
                resp = self.client.get_markets(
                    limit=50, status="open", series_ticker=series
                )
            except Exception as e:
                log.error(f"get_markets failed for {series}: {e}")
                continue

            for market in resp.get("markets", []):
                ticker = market.get("ticker", "")
                if not ticker:
                    continue

                # Skip markets closing in < 5 min
                close_ts = market.get("close_time", "")
                if close_ts:
                    try:
                        close_dt = datetime.fromisoformat(
                            close_ts.replace("Z", "+00:00"))
                        mins_left = (close_dt - datetime.now(timezone.utc)
                                     ).total_seconds() / 60
                        if mins_left < 5:
                            continue
                    except Exception:
                        pass

                sig = analyze_market(market, active_strategies)
                if sig:
                    signals.append(sig)

                # Longshot strategy runs independently on same market data
                if self.cfg.get("longshot_enabled", False):
                    ls = check_longshot(market, self.cfg)
                    if ls:
                        signals.append(ls)
                        log.info(f"  Longshot: {ls}")

                scanned += 1

            time.sleep(0.2)

        self.signals_found += len(signals)
        log.info(f"  Scanned {scanned} markets → {len(signals)} signals")
        return rank_signals(signals)

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

        # Remove positions the bot thinks are open but Kalshi says are closed
        for ticker in list(self.positions.keys()):
            if ticker not in kalshi_positions:
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

        try:
            import hashlib
            uid = hashlib.md5(f"{signal.ticker}{int(time.time())}".encode()).hexdigest()[:8]
            resp = self.client.place_order(
                ticker=signal.ticker,
                side=signal.side,
                action="buy",
                count=contracts,
                price_cents=int(signal.market_price),
                order_type="limit",
                client_order_id=f"bot_{uid}",
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

        if current_bid >= pos.target_price:
            reasons.append(f"TARGET  {current_bid:.0f}¢ >= {pos.target_price:.0f}¢")
        if current_bid <= pos.stop_price:
            reasons.append(f"STOP  {current_bid:.0f}¢ <= {pos.stop_price:.0f}¢")
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
        del self.positions[ticker]
        self.save_positions()

        if self.dry_run or self.scan_only:
            return

        try:
            self.client.place_order(
                ticker=ticker,
                side=pos.signal.side,
                action="sell",
                count=pos.contracts,
                price_cents=int(exit_price),
                order_type="limit",
            )
            self._log_trade("CLOSE", pos.signal, pos.contracts, "",
                            entry=pos.entry_price, exit_p=exit_price)
        except Exception as e:
            log.error(f"  Exit order failed {ticker}: {e}")

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