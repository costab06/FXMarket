"""Smart Order Router (SOR).

Routes an OrderRequest to the best-priced available provider.  Applies
slippage controls and falls back to the next-best provider on execution
failure.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from .aggregator import MarketAggregator
from .feeds.base import FXPriceFeed
from .models import (
    Fill,
    OrderRequest,
    OrderStatus,
    Quote,
    RoutingDecision,
    Side,
)

logger = logging.getLogger(__name__)
UTC = timezone.utc


class SmartOrderRouter:
    """Routes orders to the best available liquidity provider.

    Usage::

        sor = SmartOrderRouter(aggregator, feeds, stale_seconds=3.0)
        fill = await sor.execute(order_request)
    """

    def __init__(
        self,
        aggregator: MarketAggregator,
        feeds: dict[str, FXPriceFeed],
        stale_seconds: float = 3.0,
    ) -> None:
        """
        Args:
            aggregator:    The aggregator that holds the current best book.
            feeds:         Mapping of provider_name → FXPriceFeed for order execution.
            stale_seconds: Reject quotes older than this.
        """
        self._aggregator = aggregator
        self._feeds = feeds
        self._stale_seconds = stale_seconds

    async def execute(self, order: OrderRequest) -> Fill:
        """Route and execute *order*.  Returns a Fill on success.

        Strategy:
        1. Collect live quotes for the pair from the book.
        2. Select the best-priced quote respecting the order's side.
        3. Check the quote is within ``max_slippage_pips`` of the requested
           limit price (if provided) or simply accept best market price.
        4. Send the order to the selected provider.
        5. On failure, try the next-best provider (up to 2 retries).
        """
        quotes = self._live_quotes(order.pair)
        if not quotes:
            return self._reject(order, "No live quotes available")

        ranked = self._rank_quotes(quotes, order.side)
        decision = self._make_routing_decision(order, ranked)
        logger.info(
            "SOR routing %s %s %s → %s @ %.5f (%s)",
            order.side, order.pair, order.quantity,
            decision.selected_provider, decision.selected_price, decision.reason,
        )

        # Try providers in ranked order
        for quote in ranked:
            if quote.provider not in self._feeds:
                continue
            slippage = self._slippage_pips(order, quote)
            if slippage > order.max_slippage_pips:
                logger.warning(
                    "SOR skipping %s: slippage %.1f pips > max %.1f",
                    quote.provider, slippage, order.max_slippage_pips,
                )
                continue
            try:
                fill = await self._send_order(order, quote)
                return fill
            except Exception:
                logger.exception("SOR execution failed on %s, trying next", quote.provider)

        return self._reject(order, "All providers failed or exceeded slippage limit")

    # ── Private helpers ────────────────────────────────────────────────────────

    def _live_quotes(self, pair: str) -> list[Quote]:
        return [
            q for q in self._aggregator.all_quotes(pair)
            if q.age_seconds() < self._stale_seconds
        ]

    def _rank_quotes(self, quotes: list[Quote], side: Side) -> list[Quote]:
        """Best price first for the given side."""
        if side == Side.BUY:
            return sorted(quotes, key=lambda q: q.ask)   # lowest ask = best for buyer
        return sorted(quotes, key=lambda q: q.bid, reverse=True)  # highest bid = best for seller

    def _make_routing_decision(self, order: OrderRequest, ranked: list[Quote]) -> RoutingDecision:
        best = ranked[0]
        price = best.ask if order.side == Side.BUY else best.bid
        return RoutingDecision(
            order_id=order.order_id,
            pair=order.pair,
            side=order.side,
            selected_provider=best.provider,
            selected_price=price,
            all_quotes=list(ranked),
            reason=f"Best {'ask' if order.side == Side.BUY else 'bid'} across {len(ranked)} provider(s)",
        )

    def _slippage_pips(self, order: OrderRequest, quote: Quote) -> float:
        if order.limit_price is None:
            return 0.0
        execute_price = quote.ask if order.side == Side.BUY else quote.bid
        divisor = 0.01 if "JPY" in order.pair else 0.0001
        if order.side == Side.BUY:
            return max(0.0, (execute_price - order.limit_price) / divisor)
        return max(0.0, (order.limit_price - execute_price) / divisor)

    async def _send_order(self, order: OrderRequest, quote: Quote) -> Fill:
        feed = self._feeds[quote.provider]
        execute_price = quote.ask if order.side == Side.BUY else quote.bid
        raw = await feed.place_order(
            pair=order.pair,
            side=order.side.value,
            quantity=order.quantity,
            order_type=order.order_type.value,
            limit_price=order.limit_price,
        )

        # Parse fill price from raw response; fall back to quoted price
        fill_price = _extract_fill_price(raw, execute_price)
        divisor = 0.01 if "JPY" in order.pair else 0.0001
        slippage = abs(fill_price - execute_price) / divisor

        return Fill(
            order_id=order.order_id,
            provider=quote.provider,
            pair=order.pair,
            side=order.side,
            quantity=order.quantity,
            fill_price=fill_price,
            requested_price=execute_price,
            slippage_pips=round(slippage, 1),
            status=OrderStatus.FILLED,
            raw=raw,
        )

    @staticmethod
    def _reject(order: OrderRequest, reason: str) -> Fill:
        logger.error("SOR rejected order %s: %s", order.order_id, reason)
        execute_price = 0.0
        return Fill(
            order_id=order.order_id,
            provider="none",
            pair=order.pair,
            side=order.side,
            quantity=order.quantity,
            fill_price=0.0,
            requested_price=execute_price,
            status=OrderStatus.REJECTED,
            raw={"reason": reason},
        )


def _extract_fill_price(raw: dict, fallback: float) -> float:
    """Pull the actual fill price from a raw provider response."""
    # OANDA v20: orderFillTransaction.price
    try:
        return float(raw["orderFillTransaction"]["price"])
    except (KeyError, TypeError, ValueError):
        pass
    # Generic fallback
    for key in ("price", "fill_price", "avgFillPrice", "avgPx"):
        try:
            return float(raw[key])
        except (KeyError, TypeError, ValueError):
            continue
    return fallback
