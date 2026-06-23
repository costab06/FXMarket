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
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse

from ..aggregator import MarketAggregator
from ..feeds.dukascopy import DukascopyHistory
from ..feeds.oanda import OANDAFeed
from ..models import AggregatedQuote, Fill, OrderRequest, Side, OrderType
from ..settings import Settings
from ..sor import SmartOrderRouter
from ..strategy.manager import StrategyManager
from ..strategy.backtest import run_backtest

_UI_HTML = Path(__file__).parent / "index.html"
_STRATEGY_DIR = Path.cwd() / "strategies"

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

# ── Position tracking ─────────────────────────────────────────────────────────

# pair -> {"qty": float, "avg_price": float}
_positions: dict[str, dict] = {}
_fills_log: list[dict] = []


def _fill_callback(fill: Fill) -> None:
    """Shared callback: record fill and update positions."""
    from dataclasses import asdict as _asdict
    _fills_log.append(_asdict(fill))
    _update_position(fill)


def _update_position(fill: Fill) -> None:
    if fill.fill_price == 0.0:
        return
    pair = fill.pair
    sign = 1.0 if fill.side == Side.BUY else -1.0
    delta = fill.quantity * sign

    pos = _positions.setdefault(pair, {"qty": 0.0, "avg_price": 0.0})
    old_qty = pos["qty"]
    new_qty = old_qty + delta

    if new_qty == 0.0:
        pos["qty"] = 0.0
        pos["avg_price"] = 0.0
    elif old_qty == 0.0 or (old_qty > 0) == (delta > 0):
        # New position or adding to existing
        total = abs(old_qty) * pos["avg_price"] + abs(delta) * fill.fill_price
        pos["avg_price"] = total / abs(new_qty)
        pos["qty"] = new_qty
    else:
        # Reducing or reversing
        pos["qty"] = new_qty
        if abs(new_qty) > abs(old_qty):  # reversed direction
            pos["avg_price"] = fill.fill_price


_strategy_mgr = StrategyManager(
    strategy_dir=_STRATEGY_DIR,
    aggregator=_agg,
    sor=_sor,
    fill_callback=_fill_callback,
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
    _strategy_mgr.stop_all()
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
    _fill_callback(fill)
    return asdict(fill)


# ── Positions & fills ─────────────────────────────────────────────────────────

@app.get("/api/positions", tags=["positions"])
def get_positions():
    rows = []
    for pair, pos in _positions.items():
        if pos["qty"] == 0.0:
            continue
        aq = _agg.best_quote(pair)
        mid = aq.mid if aq else None
        pnl = (mid - pos["avg_price"]) * pos["qty"] if mid is not None else None
        rows.append({
            "pair": pair,
            "qty": pos["qty"],
            "avg_price": pos["avg_price"],
            "current_mid": mid,
            "unrealized_pnl": round(pnl, 2) if pnl is not None else None,
        })
    return rows


@app.get("/api/fills", tags=["positions"])
def get_fills(limit: int = 50):
    return list(reversed(_fills_log))[:limit]


@app.get("/api/pairs", tags=["meta"])
def get_pairs():
    return settings.default_pairs


# ── Strategies ────────────────────────────────────────────────────────────────

class StrategySaveIn(BaseModel):
    code: str


@app.get("/api/strategies", tags=["strategies"])
def list_strategies():
    return _strategy_mgr.list_all()


@app.get("/api/strategies/{name}/code", tags=["strategies"])
def get_strategy_code(name: str):
    try:
        return {"name": name, "code": _strategy_mgr.read_code(name)}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Strategy '{name}' not found")


@app.put("/api/strategies/{name}", tags=["strategies"])
def save_strategy(name: str, body: StrategySaveIn):
    try:
        _strategy_mgr.save_code(name, body.code)
        return {"saved": name}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.delete("/api/strategies/{name}", tags=["strategies"])
def delete_strategy(name: str):
    _strategy_mgr.delete(name)
    return {"deleted": name}


class StrategyStartIn(BaseModel):
    pair: str | None = None
    quantity: float | None = None


@app.post("/api/strategies/{name}/start", tags=["strategies"])
def start_strategy(name: str, body: StrategyStartIn = StrategyStartIn()):
    try:
        _strategy_mgr.start(name, pair=body.pair, quantity=body.quantity)
        runner = _strategy_mgr._runners.get(name)
        return {"status": "running", "name": name, "pair": runner.pair if runner else body.pair}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/strategies/{name}/stop", tags=["strategies"])
def stop_strategy(name: str):
    _strategy_mgr.stop(name)
    return {"status": "stopped", "name": name}


@app.get("/api/strategies/{name}/log", tags=["strategies"])
def get_strategy_log(name: str):
    return {"name": name, "log": _strategy_mgr.get_log(name)}


# ── Library docs ─────────────────────────────────────────────────────────────

_LIBRARY_SPEC = [
    {
        "library": "hff",
        "label": "HFF — High-Frequency Finance",
        "book": "Dacorogna et al. (2001) — An Introduction to High-Frequency Finance",
        "modules": [
            {"name": "ewma",           "label": "EWMA",                "path": "fx_market.hff.ewma"},
            {"name": "volatility",     "label": "Volatility",          "path": "fx_market.hff.volatility"},
            {"name": "returns",        "label": "Returns",             "path": "fx_market.hff.returns"},
            {"name": "microstructure", "label": "Microstructure",      "path": "fx_market.hff.microstructure"},
            {"name": "data",           "label": "Data Utilities",      "path": "fx_market.hff.data"},
            {"name": "dc",             "label": "Directional Change",  "path": "fx_market.hff.dc"},
            {"name": "time_transform", "label": "Time Transform",      "path": "fx_market.hff.time_transform"},
        ],
    },
    {
        "library": "afml",
        "label": "AFML — Advances in Financial ML",
        "book": "López de Prado (2018) — Advances in Financial Machine Learning",
        "modules": [
            {"name": "bars",              "label": "Bars",               "path": "fx_market.afml.bars"},
            {"name": "labeling",          "label": "Labeling",           "path": "fx_market.afml.labeling"},
            {"name": "sampling",          "label": "Sampling",           "path": "fx_market.afml.sampling"},
            {"name": "fracdiff",          "label": "Fractional Diff",    "path": "fx_market.afml.fracdiff"},
            {"name": "entropy",           "label": "Entropy",            "path": "fx_market.afml.entropy"},
            {"name": "microstructure",    "label": "Microstructure",     "path": "fx_market.afml.microstructure"},
            {"name": "structural_breaks", "label": "Structural Breaks",  "path": "fx_market.afml.structural_breaks"},
            {"name": "hrp",               "label": "HRP Portfolio",      "path": "fx_market.afml.hrp"},
            {"name": "importance",        "label": "Feature Importance", "path": "fx_market.afml.importance"},
            {"name": "cv",                "label": "Cross-Validation",   "path": "fx_market.afml.cv"},
        ],
    },
]


def _extract_module_docs(module_path: str) -> tuple[str, list[dict]]:
    """Return (module_docstring, list_of_function_dicts) for a module path."""
    import importlib
    import inspect as _inspect
    try:
        mod = importlib.import_module(module_path)
    except ImportError as exc:
        return str(exc), []

    mod_doc = _inspect.getdoc(mod) or ""
    funcs = []
    for name, obj in sorted(_inspect.getmembers(mod, _inspect.isfunction)):
        if name.startswith("_"):
            continue
        if getattr(obj, "__module__", None) != module_path:
            continue   # skip re-exported functions from other modules
        try:
            sig = str(_inspect.signature(obj))
        except (ValueError, TypeError):
            sig = "(...)"
        doc = _inspect.getdoc(obj) or ""
        funcs.append({
            "name": name,
            "signature": f"{name}{sig}",
            "doc": doc,
            "import_line": f"from {module_path.rsplit('.', 1)[0]} import {name}",
        })
    return mod_doc, funcs


@app.get("/api/library", tags=["meta"])
def get_library():
    result = []
    for lib in _LIBRARY_SPEC:
        modules_out = []
        for mod_spec in lib["modules"]:
            mod_doc, funcs = _extract_module_docs(mod_spec["path"])
            modules_out.append({
                "name": mod_spec["name"],
                "label": mod_spec["label"],
                "doc": mod_doc,
                "functions": funcs,
            })
        result.append({
            "library": lib["library"],
            "label": lib["label"],
            "book": lib["book"],
            "modules": modules_out,
        })
    return result


# ── Backtest ──────────────────────────────────────────────────────────────────

_backtest_results: dict[str, dict] = {}   # id → result dict


class BacktestRequest(BaseModel):
    strategy_name: str
    pair: str
    start: datetime
    end: datetime
    quantity: float | None = None
    domestic_currency: str = "USD"
    capital: float = 10_000.0


@app.post("/api/backtest/run", tags=["backtest"])
async def start_backtest(req: BacktestRequest):
    try:
        strategy = _strategy_mgr._load(req.strategy_name)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Apply runtime overrides
    strategy.pair = req.pair.upper().replace("/", "").replace("-", "")
    if req.quantity is not None:
        strategy.quantity = req.quantity

    run_id = uuid.uuid4().hex[:8]
    _backtest_results[run_id] = {"id": run_id, "status": "running", "progress": "Starting…"}

    async def _task():
        result = await run_backtest(
            strategy=strategy,
            start=req.start.astimezone(UTC),
            end=req.end.astimezone(UTC),
            domestic_currency=req.domestic_currency.upper(),
            capital=req.capital,
        )
        result["id"] = run_id
        _backtest_results[run_id] = result

    asyncio.create_task(_task())
    return {"id": run_id}


@app.get("/api/backtest/{run_id}", tags=["backtest"])
def get_backtest(run_id: str):
    result = _backtest_results.get(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Backtest not found")
    return result


# ── UI ────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def ui():
    return _UI_HTML.read_text()


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
