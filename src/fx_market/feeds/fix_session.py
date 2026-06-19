"""FIX protocol feed adapter stub (QuickFIX/N).

Suitable for connecting to FXSpotStream (18-bank aggregator requiring prime
brokerage) or direct bank FIX sessions.

Requires:
    quickfix    — C extension via pip; or quickfixn (Python)
    FIX_CONFIG  — path to QuickFIX session config file (.cfg)

FIX dialect: FX Spot uses FIX 4.2/4.4 MarketDataRequest (V) + MarketDataSnapshotFullRefresh (W)
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from ..models import Quote
from .base import FXPriceFeed

logger = logging.getLogger(__name__)


class FIXFeed(FXPriceFeed):
    """FIX 4.x price feed via QuickFIX (stub).

    This adapter is a placeholder for institutional FIX connectivity.
    Implementing it requires:

    1. A QuickFIX session config file specifying the counterparty details
       (BeginString, SenderCompID, TargetCompID, SocketConnectHost, etc.).
    2. A QuickFIX Application subclass that translates incoming
       MarketDataSnapshotFullRefresh (MsgType=W) messages into Quote objects
       and places them on an asyncio.Queue.
    3. A background thread running `quickfix.SocketInitiator` (QuickFIX is
       synchronous; the adapter bridges to asyncio via a queue).

    Skeleton::

        import quickfix as fix
        import quickfix44 as fix44

        class _FIXApp(fix.Application):
            def __init__(self, queue: asyncio.Queue): ...
            def onMessage(self, message, sessionID):
                # parse MDEntry bid/ask → put Quote on queue
                ...

        initiator = fix.SocketInitiator(app, store, settings, log)
        initiator.start()
    """

    def __init__(
        self,
        pairs: list[str],
        config_path: str = "fix.cfg",
        provider_name: str = "fix",
    ) -> None:
        super().__init__(provider_name, pairs)
        self._config_path = config_path

    async def _stream_quotes(self) -> AsyncIterator[Quote]:
        """Bridge QuickFIX callbacks to async generator.

        TODO: start SocketInitiator in a thread, consume from asyncio.Queue.
        """
        raise NotImplementedError("FIXFeed._stream_quotes not yet implemented")
        yield  # type: ignore[misc]

    async def place_order(
        self,
        pair: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        limit_price: float | None = None,
    ) -> dict:
        """Send a NewOrderSingle (MsgType=D) via FIX session.

        TODO: build fix.Message, set OrdType, Side, OrderQty, Symbol and
        call fix.Session.sendToTarget().
        """
        raise NotImplementedError("FIXFeed.place_order not yet implemented")
