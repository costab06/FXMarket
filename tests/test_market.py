"""Tests for the FX market data aggregator and smart order router."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fx_market.aggregator import MarketAggregator
from fx_market.models import (
    AggregatedQuote,
    Fill,
    FeedStatus,
    OrderRequest,
    OrderStatus,
    OrderType,
    Quote,
    Side,
)
from fx_market.sor import SmartOrderRouter

UTC = timezone.utc


# ── Helpers ────────────────────────────────────────────────────────────────────

def _quote(provider: str, pair: str, bid: float, ask: float, age: float = 0.0) -> Quote:
    from datetime import timedelta
    ts = datetime.now(UTC) - timedelta(seconds=age)
    return Quote(provider=provider, pair=pair, bid=bid, ask=ask, timestamp=ts)


# ── Quote model ────────────────────────────────────────────────────────────────

class TestQuote:
    def test_mid(self):
        q = _quote("oanda", "EURUSD", 1.0810, 1.0812)
        assert q.mid == pytest.approx(1.0811)

    def test_spread_pips(self):
        q = _quote("oanda", "EURUSD", 1.0810, 1.0812)
        assert q.spread_pips == pytest.approx(2.0)

    def test_spread_pips_jpy(self):
        q = _quote("oanda", "USDJPY", 149.50, 149.53)
        assert q.spread_pips == pytest.approx(3.0)

    def test_age_seconds(self):
        q = _quote("oanda", "EURUSD", 1.0810, 1.0812, age=2.0)
        assert q.age_seconds() == pytest.approx(2.0, abs=0.1)


# ── MarketAggregator ───────────────────────────────────────────────────────────

class TestMarketAggregator:
    def _make_feed(self, provider: str, pairs: list[str]) -> MagicMock:
        feed = MagicMock()
        feed.provider_name = provider
        feed.pairs = pairs
        feed.status = FeedStatus.CONNECTED
        feed.connect = AsyncMock()
        feed.disconnect = AsyncMock()
        # Capture the callback registered via feed.subscribe(cb)
        feed._cb = None
        def _subscribe(cb):
            feed._cb = cb
        feed.subscribe.side_effect = _subscribe
        return feed

    def test_best_quote_single_provider(self):
        agg = MarketAggregator()
        feed = self._make_feed("oanda", ["EURUSD"])
        agg.add_feed(feed)

        # Simulate a quote arriving
        q = _quote("oanda", "EURUSD", 1.0810, 1.0812)
        feed._cb(q)

        best = agg.best_quote("EURUSD")
        assert best is not None
        assert best.best_bid == pytest.approx(1.0810)
        assert best.best_ask == pytest.approx(1.0812)
        assert best.best_bid_provider == "oanda"

    def test_best_quote_multi_provider(self):
        agg = MarketAggregator()
        feed1 = self._make_feed("oanda", ["EURUSD"])
        feed2 = self._make_feed("ib", ["EURUSD"])
        agg.add_feed(feed1)
        agg.add_feed(feed2)

        # oanda has better bid; ib has better ask
        feed1._cb(_quote("oanda", "EURUSD", 1.0811, 1.0813))
        feed2._cb(_quote("ib",    "EURUSD", 1.0810, 1.0812))

        best = agg.best_quote("EURUSD")
        assert best.best_bid == pytest.approx(1.0811)
        assert best.best_bid_provider == "oanda"
        assert best.best_ask == pytest.approx(1.0812)
        assert best.best_ask_provider == "ib"

    def test_stale_quotes_excluded(self):
        agg = MarketAggregator(stale_seconds=1.0)
        feed = self._make_feed("oanda", ["EURUSD"])
        agg.add_feed(feed)

        stale = _quote("oanda", "EURUSD", 1.0810, 1.0812, age=5.0)
        feed._cb(stale)

        assert agg.best_quote("EURUSD") is None

    def test_no_quote_returns_none(self):
        agg = MarketAggregator()
        assert agg.best_quote("GBPUSD") is None

    def test_callback_fired(self):
        agg = MarketAggregator()
        feed = self._make_feed("oanda", ["EURUSD"])
        agg.add_feed(feed)

        received: list[AggregatedQuote] = []
        agg.subscribe(received.append)

        feed._cb(_quote("oanda", "EURUSD", 1.0810, 1.0812))
        assert len(received) == 1
        assert received[0].pair == "EURUSD"

    def test_start_stop(self):
        agg = MarketAggregator()
        feed = self._make_feed("oanda", ["EURUSD"])
        agg.add_feed(feed)

        asyncio.run(agg.start())
        feed.connect.assert_called_once()

        asyncio.run(agg.stop())
        feed.disconnect.assert_called_once()


# ── SmartOrderRouter ───────────────────────────────────────────────────────────

class TestSmartOrderRouter:
    def _make_agg_with_quotes(self, quotes: list[Quote]) -> MarketAggregator:
        agg = MagicMock(spec=MarketAggregator)
        agg.all_quotes.side_effect = lambda pair: [q for q in quotes if q.pair == pair]
        return agg

    def _make_feed(self, fill_price: float) -> MagicMock:
        feed = MagicMock()
        feed.place_order = AsyncMock(return_value={
            "orderFillTransaction": {"price": str(fill_price)}
        })
        return feed

    def test_buy_routes_to_best_ask(self):
        quotes = [
            _quote("oanda", "EURUSD", 1.0810, 1.0813),
            _quote("ib",    "EURUSD", 1.0811, 1.0812),  # better ask
        ]
        agg = self._make_agg_with_quotes(quotes)
        feeds = {
            "oanda": self._make_feed(1.0813),
            "ib":    self._make_feed(1.0812),
        }
        sor = SmartOrderRouter(agg, feeds, stale_seconds=5.0)

        order = OrderRequest(pair="EURUSD", side=Side.BUY, quantity=100_000)
        fill = asyncio.run(sor.execute(order))

        assert fill.status == OrderStatus.FILLED
        assert fill.provider == "ib"
        assert fill.fill_price == pytest.approx(1.0812)

    def test_sell_routes_to_best_bid(self):
        quotes = [
            _quote("oanda", "EURUSD", 1.0811, 1.0813),  # better bid
            _quote("ib",    "EURUSD", 1.0810, 1.0812),
        ]
        agg = self._make_agg_with_quotes(quotes)
        feeds = {
            "oanda": self._make_feed(1.0811),
            "ib":    self._make_feed(1.0810),
        }
        sor = SmartOrderRouter(agg, feeds, stale_seconds=5.0)

        order = OrderRequest(pair="EURUSD", side=Side.SELL, quantity=100_000)
        fill = asyncio.run(sor.execute(order))

        assert fill.status == OrderStatus.FILLED
        assert fill.provider == "oanda"

    def test_no_quotes_returns_rejected(self):
        agg = self._make_agg_with_quotes([])
        sor = SmartOrderRouter(agg, {}, stale_seconds=5.0)

        order = OrderRequest(pair="EURUSD", side=Side.BUY, quantity=100_000)
        fill = asyncio.run(sor.execute(order))

        assert fill.status == OrderStatus.REJECTED

    def test_slippage_exceeded_skips_provider(self):
        quotes = [_quote("oanda", "EURUSD", 1.0810, 1.0815)]
        agg = self._make_agg_with_quotes(quotes)
        feeds = {"oanda": self._make_feed(1.0815)}
        sor = SmartOrderRouter(agg, feeds, stale_seconds=5.0)

        # limit_price = 1.0810; ask = 1.0815 → 5 pips slippage > max 3
        order = OrderRequest(
            pair="EURUSD",
            side=Side.BUY,
            quantity=100_000,
            limit_price=1.0810,
            max_slippage_pips=3.0,
        )
        fill = asyncio.run(sor.execute(order))
        assert fill.status == OrderStatus.REJECTED

    def test_failover_on_execution_error(self):
        quotes = [
            _quote("oanda", "EURUSD", 1.0810, 1.0812),
            _quote("ib",    "EURUSD", 1.0810, 1.0813),
        ]
        agg = self._make_agg_with_quotes(quotes)
        bad_feed = MagicMock()
        bad_feed.place_order = AsyncMock(side_effect=RuntimeError("connection lost"))
        good_feed = self._make_feed(1.0813)
        feeds = {"oanda": bad_feed, "ib": good_feed}
        sor = SmartOrderRouter(agg, feeds, stale_seconds=5.0)

        order = OrderRequest(pair="EURUSD", side=Side.BUY, quantity=100_000)
        fill = asyncio.run(sor.execute(order))

        assert fill.status == OrderStatus.FILLED
        assert fill.provider == "ib"


# ── OANDA feed (mocked HTTP) ───────────────────────────────────────────────────

class TestOANDAFeed:
    def test_parse_price_message(self):
        from fx_market.feeds.oanda import OANDAFeed
        feed = OANDAFeed(
            api_key="test",
            account_id="123",
            pairs=["EURUSD"],
            env="practice",
        )
        msg = {
            "type": "PRICE",
            "instrument": "EUR_USD",
            "bids": [{"price": "1.08100", "liquidity": 2000000}],
            "asks": [{"price": "1.08120", "liquidity": 1000000}],
            "time": "2025-01-01T12:00:00.000000Z",
        }
        quote = feed._parse_price(msg)
        assert quote is not None
        assert quote.pair == "EURUSD"
        assert quote.bid == pytest.approx(1.0810)
        assert quote.ask == pytest.approx(1.0812)
        assert quote.provider == "oanda"

    def test_parse_price_missing_bids_returns_none(self):
        from fx_market.feeds.oanda import OANDAFeed
        feed = OANDAFeed(api_key="test", account_id="123", pairs=["EURUSD"])
        msg = {"type": "PRICE", "instrument": "EUR_USD", "bids": [], "asks": []}
        assert feed._parse_price(msg) is None
