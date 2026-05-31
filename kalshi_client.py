"""
Kalshi API Client — handles auth, requests, and order management.
Auth: RSA-PSS signed headers (Kalshi v2 standard).
"""

import base64
import hashlib
import json
import time
import logging
from datetime import datetime, timezone
from typing import Optional

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

log = logging.getLogger("kalshi.client")

PROD_BASE = "https://api.elections.kalshi.com/trade-api/v2"
DEMO_BASE = "https://demo-api.kalshi.co/trade-api/v2"


class KalshiClient:
    def __init__(self, key_id: str, private_key_pem: str, demo: bool = True):
        self.key_id = key_id
        self.base = DEMO_BASE if demo else PROD_BASE
        self.demo = demo
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

        # Load RSA private key
        self._private_key = serialization.load_pem_private_key(
            private_key_pem.encode() if isinstance(private_key_pem, str) else private_key_pem,
            password=None,
        )
        log.info(f"KalshiClient init — {'DEMO' if demo else 'LIVE'} mode")

    # ── Auth ───────────────────────────────────────────────────────────────

    def _sign(self, method: str, path: str, timestamp_ms: int) -> str:
        """RSA-PSS signature over method+timestamp+path (no body)."""
        msg = f"{timestamp_ms}{method}/trade-api/v2{path}".encode()
        sig = self._private_key.sign(msg, padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ), hashes.SHA256())
        return base64.b64encode(sig).decode()

    def _headers(self, method: str, path: str) -> dict:
        ts = str(int(time.time() * 1000))
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": self._sign(method.upper(), path, int(ts)),
        }

    # ── HTTP helpers ───────────────────────────────────────────────────────

    def _get(self, path: str, params: dict = None) -> dict:
        h = self._headers("GET", path)
        r = self.session.get(f"{self.base}{path}", headers=h, params=params, timeout=10)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: dict) -> dict:
        h = self._headers("POST", path)
        r = self.session.post(f"{self.base}{path}", headers=h,
                              data=json.dumps(body), timeout=10)
        r.raise_for_status()
        return r.json()

    def _delete(self, path: str) -> dict:
        h = self._headers("DELETE", path)
        r = self.session.delete(f"{self.base}{path}", headers=h, timeout=10)
        r.raise_for_status()
        return r.json()

    # ── Market data ────────────────────────────────────────────────────────

    # As of 2025 Kalshi's /markets responses use string, dollar-denominated
    # fields (yes_ask_dollars="0.0100", volume_24h_fp="12.00", …) instead of the
    # older integer cents fields (yes_ask=1, volume_24h=12). The rest of this
    # codebase is written against the cents names, so we normalize every market
    # dict here, in one place, right after it comes off the wire.

    # new dollar field  ->  legacy cents field   (value × 100, rounded)
    _PRICE_FIELDS = {
        "yes_ask_dollars": "yes_ask",
        "yes_bid_dollars": "yes_bid",
        "no_ask_dollars": "no_ask",
        "no_bid_dollars": "no_bid",
        "last_price_dollars": "last_price",
        "previous_price_dollars": "previous_price",
        "previous_yes_ask_dollars": "previous_yes_ask",
        "previous_yes_bid_dollars": "previous_yes_bid",
        "liquidity_dollars": "liquidity",
    }
    # new fp field  ->  legacy field   (value as-is, count not price)
    _COUNT_FIELDS = {
        "volume_fp": "volume",
        "volume_24h_fp": "volume_24h",
        "open_interest_fp": "open_interest",
        "yes_ask_size_fp": "yes_ask_size",
        "yes_bid_size_fp": "yes_bid_size",
    }

    @staticmethod
    def _to_float(v) -> Optional[float]:
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _normalize_market(cls, m: dict) -> dict:
        """Add legacy cents/int fields if only the new dollar/fp fields exist."""
        if not isinstance(m, dict):
            return m
        for new, legacy in cls._PRICE_FIELDS.items():
            if m.get(legacy) in (None, "") and new in m:
                f = cls._to_float(m.get(new))
                if f is not None:
                    m[legacy] = int(round(f * 100))
        for new, legacy in cls._COUNT_FIELDS.items():
            if m.get(legacy) in (None, "") and new in m:
                f = cls._to_float(m.get(new))
                if f is not None:
                    m[legacy] = int(round(f))
        return m

    @classmethod
    def _normalize_resp(cls, resp: dict) -> dict:
        if isinstance(resp, dict):
            if isinstance(resp.get("markets"), list):
                for m in resp["markets"]:
                    cls._normalize_market(m)
            if isinstance(resp.get("market"), dict):
                cls._normalize_market(resp["market"])
        return resp

    def get_markets(self, limit: int = 200, cursor: str = None,
                    status: str = "open", series_ticker: str = None) -> dict:
        params = {"limit": limit, "status": status}
        if cursor:
            params["cursor"] = cursor
        if series_ticker:
            params["series_ticker"] = series_ticker
        return self._normalize_resp(self._get("/markets", params))

    def get_market(self, ticker: str) -> dict:
        return self._normalize_resp(self._get(f"/markets/{ticker}"))

    def get_orderbook(self, ticker: str, depth: int = 10) -> dict:
        return self._get(f"/markets/{ticker}/orderbook", {"depth": depth})

    def get_trades(self, ticker: str, limit: int = 50) -> dict:
        return self._get(f"/markets/{ticker}/trades", {"limit": limit})

    def get_events(self, limit: int = 100, status: str = "open") -> dict:
        return self._get("/events", {"limit": limit, "status": status})

    def get_event(self, event_ticker: str) -> dict:
        return self._get(f"/events/{event_ticker}")

    # ── Account ────────────────────────────────────────────────────────────

    def get_balance(self) -> dict:
        # `balance` is still returned in integer cents; nothing to normalize.
        return self._get("/portfolio/balance")

    # The positions endpoint migrated alongside /markets: each market_position
    # now exposes `ticker`, `position_fp` ("15.00"), `market_exposure_dollars`
    # and `total_traded_dollars` instead of the old `market_ticker`, `position`
    # (signed int), `market_exposure` and `total_cost`. Normalize back to the
    # legacy names so reconcile_positions() and RiskManager work unchanged.
    #
    # NOTE: `position_fp` is an ABSOLUTE share count with no YES/NO sign, so the
    # normalized `position` is always positive. The bot already knows the side of
    # positions it opened; side is only ambiguous when ADOPTING an orphan, where
    # it defaults to "yes".
    @classmethod
    def _normalize_position(cls, p: dict) -> dict:
        if not isinstance(p, dict):
            return p
        if not p.get("market_ticker") and p.get("ticker"):
            p["market_ticker"] = p["ticker"]
        if p.get("position") in (None, "") and "position_fp" in p:
            f = cls._to_float(p.get("position_fp"))
            if f is not None:
                p["position"] = int(round(f))
        if p.get("market_exposure") in (None, "") and "market_exposure_dollars" in p:
            f = cls._to_float(p.get("market_exposure_dollars"))
            if f is not None:
                p["market_exposure"] = int(round(f * 100))
        if p.get("total_cost") in (None, "") and "total_traded_dollars" in p:
            f = cls._to_float(p.get("total_traded_dollars"))
            if f is not None:
                p["total_cost"] = int(round(f * 100))
        return p

    def get_positions(self) -> dict:
        resp = self._get("/portfolio/positions")
        if isinstance(resp, dict) and isinstance(resp.get("market_positions"), list):
            for p in resp["market_positions"]:
                self._normalize_position(p)
        return resp

    def get_orders(self, status: str = "resting") -> dict:
        return self._get("/portfolio/orders", {"status": status})

    # ── Trading ────────────────────────────────────────────────────────────

    def place_order(self, ticker: str, side: str, action: str,
                    count: int, price_cents: int,
                    order_type: str = "limit",
                    client_order_id: str = None) -> dict:
        """
        ticker: market ticker e.g. 'HIGHNY-25MAR10-B72'
        side: 'yes' or 'no'
        action: 'buy' or 'sell'
        price_cents: 1-99 (cents per contract, contracts pay 100 on win)
        count: number of contracts
        """

        
        
        body = {
            "ticker": ticker,
            "action": action,
            "side": side,
            "type": order_type,
            "count": count,
        }
        if side == "yes":
            body["yes_price"] = price_cents
        else:
            body["no_price"] = price_cents
        
        if client_order_id:
            body["client_order_id"] = client_order_id
        return self._post("/portfolio/orders", body)

    def cancel_order(self, order_id: str) -> dict:
        return self._delete(f"/portfolio/orders/{order_id}")

    def get_order(self, order_id: str) -> dict:
        return self._get(f"/portfolio/orders/{order_id}")