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

    def get_markets(self, limit: int = 200, cursor: str = None,
                    status: str = "open", series_ticker: str = None) -> dict:
        params = {"limit": limit, "status": status}
        if cursor:
            params["cursor"] = cursor
        if series_ticker:
            params["series_ticker"] = series_ticker
        return self._get("/markets", params)

    def get_market(self, ticker: str) -> dict:
        return self._get(f"/markets/{ticker}")

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
        return self._get("/portfolio/balance")

    def get_positions(self) -> dict:
        return self._get("/portfolio/positions")

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