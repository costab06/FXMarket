"""Interactive Brokers TWS/Gateway feed adapter (stub).

Requires the `ib_insync` package (not in default dependencies — install via
the `market` optional extra).

Connection parameters:
    IB_HOST     — TWS/Gateway host (default: 127.0.0.1)
    IB_PORT     — TWS port 7497 (paper) / 7496 (live); Gateway 4002/4001
    IB_CLIENT_ID — integer client ID (default: 1)
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from ..models import Quote
from .base import FXPriceFeed

logger = logging.getLogger(__name__)


class IBFeed(FXPriceFeed):
    """IB price feed via ib_insync (TWS/IB Gateway).

    This is a stub implementation.  Full wiring requires `ib_insync` and a
    running TWS or IB Gateway instance.  The `_stream_quotes` and
    `place_order` methods raise `NotImplementedError` until implemented.
    """

    def __init__(
        self,
        pairs: list[str],
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
    ) -> None:
        super().__init__("ib", pairs)
        self._host = host
        self._port = port
        self._client_id = client_id

    async def _stream_quotes(self) -> AsyncIterator[Quote]:
        """Stream quotes from IB via ib_insync.

        TODO: implement using ib_insync reqMktData + asyncio event loop.
        Example skeleton::

            from ib_insync import IB, Forex
            ib = IB()
            await ib.connectAsync(self._host, self._port, clientId=self._client_id)
            contracts = [Forex(p[:3] + p[3:]) for p in self.pairs]
            for contract in contracts:
                ib.reqMktData(contract, '', False, False)
            while True:
                await asyncio.sleep(0.1)
                for ticker in ib.tickers():
                    if ticker.bid and ticker.ask:
                        yield Quote(
                            provider="ib",
                            pair=ticker.contract.symbol + ticker.contract.currency,
                            bid=ticker.bid,
                            ask=ticker.ask,
                        )
        """
        raise NotImplementedError("IBFeed._stream_quotes not yet implemented")
        # Keep linter happy — unreachable
        yield  # type: ignore[misc]

    async def place_order(
        self,
        pair: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        limit_price: float | None = None,
    ) -> dict:
        """Submit order via IB TWS API.

        TODO: implement using ib_insync MarketOrder / LimitOrder.
        """
        raise NotImplementedError("IBFeed.place_order not yet implemented")
