"""Dukascopy feed adapter — historical tick data and live streaming stub.

Dukascopy provides two distinct services:

Historical data (free, no account needed)
------------------------------------------
Dukascopy publishes their complete tick database (bid/ask with volumes) back
to ~2003 as LZMA-compressed binary files, one per instrument per hour:

    https://datafeed.dukascopy.com/datafeed/{INSTRUMENT}/{YEAR}/{MM}/{DD}/{HH}h_ticks.bi5

Binary format (20 bytes per tick, big-endian):
    uint32  milliseconds from start of hour
    uint32  ask price × point_value
    uint32  bid price × point_value
    float32 ask volume (millions of base currency)
    float32 bid volume (millions of base currency)

Note: month in the URL is zero-indexed (January = 00).
Note: point_value = 100 000 for non-JPY pairs, 1 000 for JPY pairs.

Use ``DukascopyHistory`` to fetch historical ticks.  No credentials required.

Live streaming (requires Dukascopy brokerage account)
------------------------------------------------------
Dukascopy's Connect 1.1 / JForex platform exposes a WebSocket for live
quotes, but it requires an authenticated Dukascopy trading account and a
Java-centric client library.  ``DukascopyFeed`` is a stub; see the TODO
in ``_stream_quotes`` for the implementation skeleton.

Environment variables
---------------------
DUKASCOPY_USERNAME  — Connect 1.1 username (live feed only)
DUKASCOPY_PASSWORD  — Connect 1.1 password (live feed only)
"""

from __future__ import annotations

import asyncio
import lzma
import logging
import struct
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone

import httpx

from ..models import Quote
from .base import FXPriceFeed

logger = logging.getLogger(__name__)
UTC = timezone.utc

# ── Constants ──────────────────────────────────────────────────────────────────

_HISTORY_BASE = "https://datafeed.dukascopy.com/datafeed"
_TICK_SIZE = 20           # bytes per tick record
_CONCURRENT_REQUESTS = 3  # max parallel hour fetches
_RETRY_STATUSES = {429, 503}
_MAX_RETRIES = 5
_RETRY_BASE_DELAY = 2.0   # seconds; doubles on each retry

# Price scaling: Dukascopy stores prices as integers
_POINT_VALUE: dict[str, int] = {
    "JPY": 1_000,          # 3 decimal places for JPY pairs
    "DEFAULT": 100_000,    # 5 decimal places for all other pairs
}


def _point_value(pair: str) -> int:
    return _POINT_VALUE["JPY"] if "JPY" in pair else _POINT_VALUE["DEFAULT"]


def _history_url(pair: str, dt: datetime) -> str:
    """Build the Dukascopy tick file URL for the given pair and hour (UTC)."""
    # Month is 0-indexed in Dukascopy URLs
    return (
        f"{_HISTORY_BASE}/{pair.upper()}/"
        f"{dt.year}/{dt.month - 1:02d}/{dt.day:02d}/{dt.hour:02d}h_ticks.bi5"
    )


# ── Historical data fetcher ────────────────────────────────────────────────────

class DukascopyHistory:
    """Fetch free historical tick data from Dukascopy's public data feed.

    No account or API key is required.

    Usage::

        hist = DukascopyHistory()
        ticks = await hist.fetch_range(
            "EURUSD",
            start=datetime(2024, 1, 2, 8, 0, tzinfo=UTC),
            end=datetime(2024, 1, 2, 17, 0, tzinfo=UTC),
        )
        df = DukascopyHistory.to_dataframe(ticks)

    The tick data integrates directly with the HFF library::

        from fx_agent.hff import midpoint, log_returns, realized_volatility
        mid = midpoint(df["bid"].values, df["ask"].values)
        r   = log_returns(mid)
        rv  = realized_volatility(r)
    """

    def __init__(self, timeout: float = 30.0, max_concurrent: int = _CONCURRENT_REQUESTS) -> None:
        self._timeout = timeout
        self._semaphore = asyncio.Semaphore(max_concurrent)
        # Single shared client for connection pooling across all requests
        self._client = httpx.AsyncClient(timeout=self._timeout)

    async def fetch_hour(self, pair: str, dt: datetime) -> list[Quote]:
        """Fetch one hour of tick data for ``pair`` starting at ``dt`` (UTC).

        Parameters
        ----------
        pair : str
            Currency pair in normalised form, e.g. "EURUSD".
        dt : datetime
            Any UTC datetime within the desired hour; the minute/second/
            microsecond components are ignored.

        Returns
        -------
        list[Quote]
            All ticks within that UTC hour.  Returns an empty list if
            no data exists (weekends, bank holidays, or pre-2003 dates).
        """
        dt = dt.replace(tzinfo=UTC, minute=0, second=0, microsecond=0)
        url = _history_url(pair, dt)

        async with self._semaphore:
            resp = await self._fetch_with_retry(url)

        if resp is None:
            return []
        if resp.status_code == 404:
            return []         # no data for this hour (weekend / holiday)
        if resp.status_code != 200:
            logger.warning("Dukascopy HTTP %d for %s", resp.status_code, url)
            return []
        if not resp.content:
            return []

        try:
            raw = lzma.decompress(resp.content)
        except lzma.LZMAError as exc:
            logger.warning("Dukascopy LZMA decompression failed %s: %s", url, exc)
            return []

        return _parse_ticks(raw, pair, dt)

    async def _fetch_with_retry(self, url: str):
        """GET ``url`` with exponential backoff retry on 429/503."""
        delay = _RETRY_BASE_DELAY
        for attempt in range(_MAX_RETRIES):
            try:
                resp = await self._client.get(url)
            except httpx.RequestError as exc:
                logger.warning("Dukascopy request error %s: %s", url, exc)
                return None

            if resp.status_code not in _RETRY_STATUSES:
                return resp

            logger.warning(
                "Dukascopy HTTP %d for %s — retry %d/%d in %.1fs",
                resp.status_code, url, attempt + 1, _MAX_RETRIES, delay,
            )
            await asyncio.sleep(delay)
            delay *= 2

        logger.error("Dukascopy: giving up on %s after %d retries", url, _MAX_RETRIES)
        return None

    async def fetch_range(
        self,
        pair: str,
        start: datetime,
        end: datetime,
    ) -> list[Quote]:
        """Fetch all ticks for ``pair`` between ``start`` and ``end`` (UTC).

        Downloads one file per calendar hour in the requested range,
        up to ``max_concurrent`` requests in parallel.

        Parameters
        ----------
        pair : str
            Currency pair, e.g. "EURUSD".
        start, end : datetime
            UTC datetime range (inclusive).  Both must be timezone-aware.

        Returns
        -------
        list[Quote]
            Ticks in chronological order, filtered to [start, end].
        """
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("start and end must be timezone-aware (UTC)")

        start = start.astimezone(UTC)
        end   = end.astimezone(UTC)

        # Build list of hours to fetch
        hour_start = start.replace(minute=0, second=0, microsecond=0)
        hours: list[datetime] = []
        current = hour_start
        while current <= end:
            hours.append(current)
            current += timedelta(hours=1)

        logger.info("Dukascopy: fetching %d hour file(s) for %s", len(hours), pair)

        try:
            results = await asyncio.gather(*[self.fetch_hour(pair, h) for h in hours])
        finally:
            await self._client.aclose()

        all_ticks: list[Quote] = []
        for batch in results:
            all_ticks.extend(batch)

        # Filter to exact requested range and sort
        filtered = [t for t in all_ticks if start <= t.timestamp <= end]
        filtered.sort(key=lambda t: t.timestamp)
        logger.info("Dukascopy: returned %d ticks for %s", len(filtered), pair)
        return filtered

    async def fetch_day(self, pair: str, date: datetime) -> list[Quote]:
        """Convenience wrapper: fetch all ticks for a full UTC calendar day.

        Parameters
        ----------
        pair : str
            Currency pair, e.g. "EURUSD".
        date : datetime
            Any datetime within the desired day (UTC).

        Returns
        -------
        list[Quote]
            All ticks for that calendar day (00:00–23:59 UTC).
        """
        day_start = date.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=UTC)
        day_end   = day_start + timedelta(hours=23, minutes=59, seconds=59)
        return await self.fetch_range(pair, day_start, day_end)

    @staticmethod
    def to_dataframe(ticks: list[Quote]):  # type: ignore[return]
        """Convert a tick list to a pandas DataFrame.

        Returns a DataFrame with columns:
            timestamp, bid, ask, mid, bid_size, ask_size, spread_pips

        Parameters
        ----------
        ticks : list[Quote]
            Ticks as returned by ``fetch_range`` or ``fetch_hour``.

        Returns
        -------
        pandas.DataFrame
            Indexed by UTC timestamp.
        """
        import pandas as pd

        if not ticks:
            return pd.DataFrame(
                columns=["timestamp", "bid", "ask", "mid", "bid_size", "ask_size", "spread_pips"]
            )

        rows = [
            {
                "timestamp": t.timestamp,
                "bid": t.bid,
                "ask": t.ask,
                "mid": t.mid,
                "bid_size": t.bid_size,
                "ask_size": t.ask_size,
                "spread_pips": t.spread_pips,
            }
            for t in ticks
        ]
        df = pd.DataFrame(rows)
        df = df.set_index("timestamp").sort_index()
        return df


# ── Binary tick parser ─────────────────────────────────────────────────────────

def _parse_ticks(data: bytes, pair: str, hour_dt: datetime) -> list[Quote]:
    """Decode raw Dukascopy tick bytes into Quote objects.

    Each record is 20 bytes, big-endian:
        [0:4]   uint32  milliseconds from start of hour
        [4:8]   uint32  ask price × point_value
        [8:12]  uint32  bid price × point_value
        [12:16] float32 ask volume (millions of base currency)
        [16:20] float32 bid volume (millions of base currency)
    """
    n = len(data) // _TICK_SIZE
    if n == 0:
        return []

    point = _point_value(pair)
    ticks: list[Quote] = []

    for i in range(n):
        off = i * _TICK_SIZE
        ms, ask_raw, bid_raw, ask_vol, bid_vol = struct.unpack(
            ">IIIff", data[off : off + _TICK_SIZE]
        )
        ts  = hour_dt + timedelta(milliseconds=int(ms))
        ask = ask_raw / point
        bid = bid_raw / point

        if bid <= 0 or ask <= 0 or bid > ask:
            continue   # malformed record — skip

        ticks.append(Quote(
            provider="dukascopy",
            pair=pair,
            bid=bid,
            ask=ask,
            ask_size=float(ask_vol) * 1_000_000,   # stored in millions of base currency
            bid_size=float(bid_vol) * 1_000_000,
            timestamp=ts,
        ))

    return ticks


# ── Live streaming feed (stub) ─────────────────────────────────────────────────

class DukascopyFeed(FXPriceFeed):
    """Live price feed via Dukascopy Connect 1.1 (stub).

    Dukascopy's live quote stream requires an authenticated Connect 1.1
    session (WebSocket) using a Dukascopy SWFX or JForex account.
    The Java-centric client protocol makes a pure-Python implementation
    non-trivial; this class is a stub documenting the approach.

    For live FX data without institutional requirements, use OANDAFeed.
    For free historical tick data, use DukascopyHistory — no account needed.

    Implementation notes
    --------------------
    Connect 1.1 WebSocket flow:
        1. Open wss://trial.dukascopy.com/websocket (demo) or
           wss://platform.dukascopy.com/websocket (live)
        2. Send authentication frame: {"m": "Login", "p": {"username": ..., "password": ...}}
        3. Subscribe to instruments: {"m": "SubscribeOffer", "p": {"instrument": "EURUSD"}}
        4. Parse incoming "Offer" messages: {"m": "Offer", "p": {"instrument": ..., "ask": ..., "bid": ...}}
        5. Send periodic heartbeat: {"m": "Heartbeat"}

    TODO: Implement using ``websockets`` library (add to pyproject.toml market extras).
    """

    def __init__(
        self,
        username: str,
        password: str,
        pairs: list[str],
        demo: bool = True,
    ) -> None:
        super().__init__("dukascopy", pairs)
        self._username = username
        self._password = password
        self._demo = demo

    async def _stream_quotes(self) -> AsyncIterator[Quote]:
        """Connect to Dukascopy Connect 1.1 WebSocket and stream quotes.

        TODO: implement using websockets library.
        Example skeleton::

            import websockets, json
            url = "wss://trial.dukascopy.com/websocket" if self._demo else \
                  "wss://platform.dukascopy.com/websocket"
            async with websockets.connect(url) as ws:
                await ws.send(json.dumps({
                    "m": "Login",
                    "p": {"username": self._username, "password": self._password}
                }))
                for pair in self.pairs:
                    await ws.send(json.dumps({"m": "SubscribeOffer", "p": {"instrument": pair}}))
                async for msg_str in ws:
                    msg = json.loads(msg_str)
                    if msg.get("m") == "Offer":
                        p = msg["p"]
                        yield Quote(
                            provider="dukascopy",
                            pair=p["instrument"].replace("/", ""),
                            bid=p["bid"],
                            ask=p["ask"],
                        )
        """
        raise NotImplementedError("DukascopyFeed live stream not yet implemented; use DukascopyHistory for tick data")
        yield  # type: ignore[misc]

    async def place_order(
        self,
        pair: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        limit_price: float | None = None,
    ) -> dict:
        """Submit order via Dukascopy Connect 1.1 (not implemented)."""
        raise NotImplementedError("DukascopyFeed.place_order not yet implemented")
