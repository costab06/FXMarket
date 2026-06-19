"""Abstract base class for FX price feed adapters."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Callable

from ..models import FeedStatus, Quote

logger = logging.getLogger(__name__)

QuoteCallback = Callable[[Quote], None]


class FXPriceFeed(ABC):
    """Abstract price feed.  Subclass and implement `_stream_quotes`."""

    def __init__(self, provider_name: str, pairs: list[str]) -> None:
        self.provider_name = provider_name
        self.pairs = [p.upper().replace("/", "").replace("-", "") for p in pairs]
        self.status: FeedStatus = FeedStatus.DISCONNECTED
        self._callbacks: list[QuoteCallback] = []
        self._task: asyncio.Task | None = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def subscribe(self, callback: QuoteCallback) -> None:
        """Register a callback invoked on every new quote."""
        self._callbacks.append(callback)

    async def connect(self) -> None:
        """Start streaming quotes in the background."""
        if self._task and not self._task.done():
            return
        self.status = FeedStatus.CONNECTING
        self._task = asyncio.create_task(self._run(), name=f"feed-{self.provider_name}")

    async def disconnect(self) -> None:
        """Stop streaming and clean up."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.status = FeedStatus.DISCONNECTED

    # ── Internals ──────────────────────────────────────────────────────────────

    async def _run(self) -> None:
        while True:
            try:
                self.status = FeedStatus.CONNECTED
                async for quote in self._stream_quotes():
                    self._dispatch(quote)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("%s feed error — reconnecting in 5s", self.provider_name)
                self.status = FeedStatus.ERROR
                await asyncio.sleep(5)

    def _dispatch(self, quote: Quote) -> None:
        for cb in self._callbacks:
            try:
                cb(quote)
            except Exception:
                logger.exception("Quote callback raised")

    # ── Abstract ───────────────────────────────────────────────────────────────

    @abstractmethod
    async def _stream_quotes(self) -> AsyncIterator[Quote]:
        """Yield Quote objects as they arrive from the provider.

        Implementations should yield indefinitely until cancelled or
        the connection drops (in which case raise an exception to
        trigger automatic reconnection).
        """
        ...  # pragma: no cover

    @abstractmethod
    async def place_order(
        self,
        pair: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        limit_price: float | None = None,
    ) -> dict:
        """Submit an order and return the raw provider response."""
        ...  # pragma: no cover
