"""OANDA v20 REST + streaming price feed adapter.

Requires:
    OANDA_API_KEY   — personal access token from OANDA hub
    OANDA_ACCOUNT   — account ID (e.g. "101-001-12345678-001")
    OANDA_ENV       — "practice" (default) or "live"

Docs: https://developer.oanda.com/rest-live-v20/introduction/
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import httpx

from ..models import FeedStatus, Quote
from .base import FXPriceFeed

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)
UTC = timezone.utc

_BASE_URLS = {
    "practice": "https://api-fxtrade.oanda.com",
    "live":     "https://api-fxtrade.oanda.com",
    "stream_practice": "https://stream-fxtrade.oanda.com",
    "stream_live":     "https://stream-fxtrade.oanda.com",
}

# OANDA uses underscore notation: "EUR_USD"
def _to_oanda(pair: str) -> str:
    return f"{pair[:3]}_{pair[3:]}"

# OANDA instruments have the form "EUR_USD" — normalise back to "EURUSD"
def _from_oanda(instrument: str) -> str:
    return instrument.replace("_", "")


class OANDAFeed(FXPriceFeed):
    """Streams live bid/ask prices from OANDA v20 pricing API."""

    def __init__(
        self,
        api_key: str,
        account_id: str,
        pairs: list[str],
        env: str = "practice",
    ) -> None:
        super().__init__("oanda", pairs)
        self._api_key = api_key
        self._account_id = account_id
        self._env = env
        self._rest_base = _BASE_URLS[env]
        self._stream_base = _BASE_URLS[f"stream_{env}"]
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept-Datetime-Format": "RFC3339",
        }

    # ── Streaming ──────────────────────────────────────────────────────────────

    async def _stream_quotes(self) -> AsyncIterator[Quote]:
        instruments = ",".join(_to_oanda(p) for p in self.pairs)
        url = (
            f"{self._stream_base}/v3/accounts/{self._account_id}"
            f"/pricing/stream?instruments={instruments}&snapshot=True"
        )
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("GET", url, headers=self._headers) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if msg.get("type") == "PRICE":
                        quote = self._parse_price(msg)
                        if quote:
                            yield quote
                    elif msg.get("type") == "HEARTBEAT":
                        pass  # keep-alive — ignore

    def _parse_price(self, msg: dict) -> Quote | None:
        try:
            instrument = msg["instrument"]
            bids = msg.get("bids", [])
            asks = msg.get("asks", [])
            if not bids or not asks:
                return None
            bid = float(bids[0]["price"])
            ask = float(asks[0]["price"])
            bid_size = float(bids[0].get("liquidity", 1_000_000))
            ask_size = float(asks[0].get("liquidity", 1_000_000))
            ts_str = msg.get("time", "")
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except ValueError:
                ts = datetime.now(UTC)
            return Quote(
                provider="oanda",
                pair=_from_oanda(instrument),
                bid=bid,
                ask=ask,
                bid_size=bid_size,
                ask_size=ask_size,
                timestamp=ts,
            )
        except (KeyError, ValueError, IndexError):
            logger.warning("Failed to parse OANDA PRICE msg: %s", msg)
            return None

    # ── REST: snapshot prices ──────────────────────────────────────────────────

    async def get_snapshot(self, pairs: list[str] | None = None) -> list[Quote]:
        """Fetch the current bid/ask for a list of pairs via REST (non-streaming)."""
        target = pairs or self.pairs
        instruments = ",".join(_to_oanda(p) for p in target)
        url = (
            f"{self._rest_base}/v3/accounts/{self._account_id}"
            f"/pricing?instruments={instruments}"
        )
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, headers=self._headers)
            resp.raise_for_status()
        data = resp.json()
        quotes = []
        for price in data.get("prices", []):
            q = self._parse_price({**price, "type": "PRICE"})
            if q:
                quotes.append(q)
        return quotes

    # ── Order placement ────────────────────────────────────────────────────────

    async def place_order(
        self,
        pair: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        limit_price: float | None = None,
    ) -> dict:
        """Submit a market or limit order via OANDA v20 Orders API."""
        units = quantity if side.upper() == "BUY" else -quantity
        order_body: dict = {
            "type": order_type.upper(),
            "instrument": _to_oanda(pair),
            "units": str(int(units)),
            "timeInForce": "FOK",          # Fill-or-Kill for market orders
            "positionFill": "DEFAULT",
        }
        if order_type.upper() == "LIMIT" and limit_price is not None:
            order_body["price"] = str(round(limit_price, 5))
            order_body["timeInForce"] = "GTC"

        url = f"{self._rest_base}/v3/accounts/{self._account_id}/orders"
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                url,
                headers={**self._headers, "Content-Type": "application/json"},
                json={"order": order_body},
            )
        raw = resp.json()
        if resp.status_code not in (200, 201):
            logger.error("OANDA order failed %d: %s", resp.status_code, raw)
        return raw
