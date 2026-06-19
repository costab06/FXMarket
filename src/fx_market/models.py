"""Data models for the FX market data aggregator and smart order router."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

UTC = timezone.utc


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    ERROR = "error"


class FeedStatus(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class Quote:
    """A two-sided price from a single liquidity provider."""

    provider: str
    pair: str                    # normalised "EURUSD" (no underscore)
    bid: float
    ask: float
    bid_size: float = 1_000_000.0    # notional in base currency
    ask_size: float = 1_000_000.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread_pips(self) -> float:
        divisor = 0.01 if "JPY" in self.pair else 0.0001
        return round((self.ask - self.bid) / divisor, 1)

    def age_seconds(self) -> float:
        return (datetime.now(UTC) - self.timestamp).total_seconds()


@dataclass
class AggregatedQuote:
    """Best bid and ask assembled across all connected providers."""

    pair: str
    best_bid: float
    best_bid_provider: str
    best_ask: float
    best_ask_provider: str
    best_bid_size: float
    best_ask_size: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    contributing_providers: list[str] = field(default_factory=list)

    @property
    def mid(self) -> float:
        return (self.best_bid + self.best_ask) / 2

    @property
    def spread_pips(self) -> float:
        divisor = 0.01 if "JPY" in self.pair else 0.0001
        return round((self.best_ask - self.best_bid) / divisor, 1)


@dataclass
class OrderRequest:
    """A request to execute an FX trade through the SOR."""

    pair: str
    side: Side
    quantity: float                  # base currency units (e.g. 100_000 = 1 standard lot)
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    max_slippage_pips: float = 3.0
    run_id: str | None = None        # link back to a pipeline run
    order_id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass
class Fill:
    """The result of an executed order."""

    order_id: str
    provider: str
    pair: str
    side: Side
    quantity: float
    fill_price: float
    requested_price: float           # best quote at time of routing decision
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    status: OrderStatus = OrderStatus.FILLED
    slippage_pips: float = 0.0
    commission: float = 0.0          # in quote currency
    raw: dict = field(default_factory=dict)


@dataclass
class RoutingDecision:
    """Audit record of a SOR routing choice."""

    order_id: str
    pair: str
    side: Side
    selected_provider: str
    selected_price: float
    all_quotes: list[Quote]
    reason: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
