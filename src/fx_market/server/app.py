"""FXMarket FastAPI server.

Exposes live quotes, order execution, and historical tick data over HTTP,
so that FX Agent (and any other consumer) can call this service without
coupling to the feed adapter implementations.

Endpoints
---------
GET  /health                        Liveness probe
GET  /api/feeds                     Status of all configured feeds
GET  /api/quotes/{pair}             Current best aggregated bid/ask
GET  /api/quotes/{pair}/stream      SSE stream of aggregated quote updates
POST /api/orders                    Route and execute an order via SOR
GET  /api/history/{pair}            Dukascopy historical ticks (no feed required)
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from ..aggregator import MarketAggregator
from ..feeds.dukascopy import DukascopyHistory
from ..feeds.oanda import OANDAFeed
from ..models import AggregatedQuote, Fill, OrderRequest, Side, OrderType
from ..settings import Settings
from ..sor import SmartOrderRouter

logger = logging.getLogger(__name__)
UTC = timezone.utc

settings = Settings()

# ── Build aggregator and SOR at module load ───────────────────────────────────

_agg = MarketAggregator(stale_seconds=settings.stale_seconds)
_feeds: dict = {}   # provider_name → FXPriceFeed

if "oanda" in settings.enabled_feeds and settings.oanda_api_key and settings.oanda_account_id:
    _oanda = OANDAFeed(
        api_key=settings.oanda_api_key,
        account_id=settings.oanda_account_id,
        pairs=settings.default_pairs,
        env=settings.oanda_env,
    )
    _agg.add_feed(_oanda)
    _feeds["oanda"] = _oanda

_sor = SmartOrderRouter(
    aggregator=_agg,
    feeds=_feeds,
    stale_seconds=settings.sor_stale_seconds,
)

# Quote update queue for SSE streams
_quote_queues: dict[str, list[asyncio.Queue]] = {}

def _on_quote(aq: AggregatedQuote) -> None:
    for q in _quote_queues.get(aq.pair, []):
        try:
            q.put_nowait(aq)
        except asyncio.QueueFull:
            pass


_agg.subscribe(_on_quote)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("FXMarket starting — connecting %d feed(s)", len(_feeds))
    await _agg.start()
    yield
    await _agg.stop()
    logger.info("FXMarket stopped")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="FX Market", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}


# ── Feeds ─────────────────────────────────────────────────────────────────────

@app.get("/api/feeds", tags=["feeds"])
def list_feeds():
    return [
        {"provider": name, "status": feed.status, "pairs": feed.pairs}
        for name, feed in _feeds.items()
    ]


# ── Quotes ────────────────────────────────────────────────────────────────────

@app.get("/api/quotes/{pair}", tags=["quotes"])
def get_quote(pair: str):
    pair = pair.upper().replace("/", "").replace("-", "")
    aq = _agg.best_quote(pair)
    if aq is None:
        raise HTTPException(status_code=404, detail=f"No live quote for {pair}")
    return asdict(aq)


@app.get("/api/quotes/{pair}/stream", tags=["quotes"])
async def stream_quote(pair: str):
    """Server-Sent Events stream of AggregatedQuote updates for a currency pair."""
    pair = pair.upper().replace("/", "").replace("-", "")
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)

    if pair not in _quote_queues:
        _quote_queues[pair] = []
    _quote_queues[pair].append(queue)

    async def event_gen():
        try:
            # Send current snapshot immediately
            aq = _agg.best_quote(pair)
            if aq:
                yield f"data: {json.dumps(asdict(aq), default=str)}\n\n"
            while True:
                try:
                    aq = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {json.dumps(asdict(aq), default=str)}\n\n"
                except asyncio.TimeoutError:
                    yield "data: {\"type\": \"heartbeat\"}\n\n"
        finally:
            _quote_queues[pair].remove(queue)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# ── Orders ────────────────────────────────────────────────────────────────────

class _OrderIn:
    """Pydantic-free order request body parsed from JSON."""


from pydantic import BaseModel

class OrderIn(BaseModel):
    pair: str
    side: str           # "buy" | "sell"
    quantity: float
    order_type: str = "market"
    limit_price: float | None = None
    max_slippage_pips: float = 3.0
    run_id: str | None = None


@app.post("/api/orders", tags=["orders"])
async def place_order(req: OrderIn):
    order = OrderRequest(
        pair=req.pair.upper().replace("/", "").replace("-", ""),
        side=Side(req.side.lower()),
        quantity=req.quantity,
        order_type=OrderType(req.order_type.lower()),
        limit_price=req.limit_price,
        max_slippage_pips=req.max_slippage_pips,
        run_id=req.run_id,
    )
    fill: Fill = await _sor.execute(order)
    return asdict(fill)


# ── Historical data ───────────────────────────────────────────────────────────

@app.get("/api/history/{pair}", tags=["history"])
async def get_history(
    pair: str,
    start: datetime = Query(..., description="UTC start datetime (ISO 8601)"),
    end: datetime   = Query(..., description="UTC end datetime (ISO 8601)"),
):
    """Return Dukascopy historical ticks as JSON.  No feed connection required."""
    pair = pair.upper().replace("/", "").replace("-", "")

    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)

    if (end - start).total_seconds() > 7 * 86_400:
        raise HTTPException(status_code=400, detail="Max range is 7 days per request")

    hist = DukascopyHistory()
    ticks = await hist.fetch_range(pair, start, end)
    return [
        {
            "timestamp": t.timestamp.isoformat(),
            "bid": t.bid,
            "ask": t.ask,
            "mid": t.mid,
            "bid_size": t.bid_size,
            "ask_size": t.ask_size,
            "spread_pips": t.spread_pips,
        }
        for t in ticks
    ]
