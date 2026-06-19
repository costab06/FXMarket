"""FX market data aggregator.

Maintains the best top-of-book (NBBO-style) across all connected feeds.
Thread-safe for reads; callbacks are dispatched in the asyncio event loop.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from .feeds.base import FXPriceFeed
from .models import AggregatedQuote, FeedStatus, Quote

logger = logging.getLogger(__name__)
UTC = timezone.utc

AggregatedQuoteCallback = Callable[[AggregatedQuote], None]


class MarketAggregator:
    """Aggregates quotes from multiple feeds into a best bid/offer book.

    Usage::

        agg = MarketAggregator(stale_seconds=5.0)
        agg.add_feed(oanda_feed)
        agg.subscribe(lambda aq: print(aq))
        await agg.start()
        ...
        await agg.stop()
    """

    def __init__(self, stale_seconds: float = 5.0) -> None:
        self._stale_seconds = stale_seconds
        self._feeds: list[FXPriceFeed] = []
        # Latest quote per (provider, pair)
        self._book: dict[tuple[str, str], Quote] = {}
        self._callbacks: list[AggregatedQuoteCallback] = []
        self._lock = asyncio.Lock()

    # ── Configuration ──────────────────────────────────────────────────────────

    def add_feed(self, feed: FXPriceFeed) -> None:
        feed.subscribe(self._on_quote)
        self._feeds.append(feed)

    def subscribe(self, callback: AggregatedQuoteCallback) -> None:
        self._callbacks.append(callback)

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Connect all registered feeds."""
        await asyncio.gather(*(f.connect() for f in self._feeds))
        logger.info("MarketAggregator started with %d feed(s)", len(self._feeds))

    async def stop(self) -> None:
        """Disconnect all feeds."""
        await asyncio.gather(*(f.disconnect() for f in self._feeds))
        logger.info("MarketAggregator stopped")

    # ── Book access ────────────────────────────────────────────────────────────

    def best_quote(self, pair: str) -> AggregatedQuote | None:
        """Return current aggregated best bid/ask for *pair*, or None if no data."""
        return self._aggregate(pair)

    def all_quotes(self, pair: str) -> list[Quote]:
        """Return all live (non-stale) quotes for *pair* across providers."""
        return [
            q for (_, p), q in self._book.items()
            if p == pair and q.age_seconds() < self._stale_seconds
        ]

    @property
    def connected_providers(self) -> list[str]:
        return [f.provider_name for f in self._feeds if f.status == FeedStatus.CONNECTED]

    # ── Internal ───────────────────────────────────────────────────────────────

    def _on_quote(self, quote: Quote) -> None:
        """Called by feed adapters on every new tick."""
        self._book[(quote.provider, quote.pair)] = quote
        agg = self._aggregate(quote.pair)
        if agg:
            self._dispatch(agg)

    def _aggregate(self, pair: str) -> AggregatedQuote | None:
        live = [
            q for (_, p), q in self._book.items()
            if p == pair and q.age_seconds() < self._stale_seconds
        ]
        if not live:
            return None

        best_bid_q = max(live, key=lambda q: q.bid)
        best_ask_q = min(live, key=lambda q: q.ask)

        return AggregatedQuote(
            pair=pair,
            best_bid=best_bid_q.bid,
            best_bid_provider=best_bid_q.provider,
            best_ask=best_ask_q.ask,
            best_ask_provider=best_ask_q.provider,
            best_bid_size=best_bid_q.bid_size,
            best_ask_size=best_ask_q.ask_size,
            timestamp=datetime.now(UTC),
            contributing_providers=[q.provider for q in live],
        )

    def _dispatch(self, agg: AggregatedQuote) -> None:
        for cb in self._callbacks:
            try:
                cb(agg)
            except Exception:
                logger.exception("AggregatedQuote callback raised")
