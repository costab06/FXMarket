"""Strategy runner — executes a single strategy against live market data."""

from __future__ import annotations

import asyncio
import collections
import logging
import traceback
from collections.abc import Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np

from ..models import Fill, OrderRequest, OrderType, Quote, Side
from .base import StrategyBase, StrategyContext

if TYPE_CHECKING:
    from ..aggregator import MarketAggregator
    from ..sor import SmartOrderRouter

UTC = timezone.utc
logger = logging.getLogger(__name__)

MAX_LOG = 300


class StrategyRunner:
    """Runs one strategy instance against live aggregated quotes.

    Lifecycle::

        runner = StrategyRunner(strategy, name, aggregator, sor, fill_cb)
        runner.start()
        # ... live trading ...
        runner.stop()
    """

    def __init__(
        self,
        strategy: StrategyBase,
        name: str,
        aggregator: "MarketAggregator",
        sor: "SmartOrderRouter",
        fill_callback: Callable[[Fill], None] | None = None,
    ) -> None:
        self._strategy = strategy
        self._name = name
        self._agg = aggregator
        self._sor = sor
        self._fill_callback = fill_callback

        self._history: collections.deque[Quote] = collections.deque(
            maxlen=strategy.max_history
        )
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=2000)
        self._task: asyncio.Task | None = None
        self._running = False
        self._log: list[str] = []

        # Per-runner position (independent of other strategies)
        self._net_qty: float = 0.0
        self._avg_price: float = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._name

    @property
    def pair(self) -> str:
        return self._strategy.pair

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def net_qty(self) -> float:
        return self._net_qty

    def get_log(self) -> list[str]:
        return list(self._log)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._agg.subscribe(self._on_quote)
        self._task = asyncio.create_task(
            self._run(), name=f"strategy-{self._name}"
        )
        self._log_msg(
            f"Started — pair={self._strategy.pair}  qty={self._strategy.quantity:,.0f}"
        )
        try:
            self._strategy.on_start()
        except Exception:
            self._log_msg(f"on_start error:\n{traceback.format_exc()}")

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self._agg.unsubscribe(self._on_quote)
        if self._task and not self._task.done():
            self._task.cancel()
        self._log_msg(
            f"Stopped — final position: {self._net_qty:+,.0f} {self._strategy.pair}"
        )
        try:
            self._strategy.on_stop()
        except Exception:
            pass

    # ── Internal ──────────────────────────────────────────────────────────────

    def _on_quote(self, aq) -> None:
        """Aggregator callback — filters by pair and enqueues."""
        if not self._running or aq.pair != self._strategy.pair:
            return
        try:
            self._queue.put_nowait(aq)
        except asyncio.QueueFull:
            pass

    async def _warmup(self) -> None:
        """Pre-fill history buffer with Dukascopy historical ticks."""
        hours = self._strategy.warmup_hours
        if hours <= 0:
            return

        from datetime import timedelta
        from ..feeds.dukascopy import DukascopyHistory

        end = datetime.now(UTC)
        start = end - timedelta(hours=hours)
        self._log_msg(
            f"Warming up: fetching {hours}h of {self._strategy.pair} ticks from Dukascopy…"
        )
        try:
            hist = DukascopyHistory(max_concurrent=8)
            ticks = await hist.fetch_range(self._strategy.pair, start, end)
        except Exception:
            self._log_msg(f"Warmup fetch failed:\n{traceback.format_exc()}")
            return

        if not ticks:
            self._log_msg("Warmup: no historical ticks returned (weekend / holiday?)")
            return

        for tick in ticks:
            self._history.append(tick)

        self._log_msg(
            f"Warmup complete: {len(ticks):,} ticks loaded "
            f"({len(self._history):,} in buffer, "
            f"span {ticks[0].timestamp.strftime('%Y-%m-%d %H:%M')} → "
            f"{ticks[-1].timestamp.strftime('%Y-%m-%d %H:%M')} UTC)"
        )

    async def _run(self) -> None:
        await self._warmup()
        while self._running:
            try:
                aq = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                await self._process(aq)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception:
                self._log_msg(f"Runner error:\n{traceback.format_exc()}")

    async def _process(self, aq) -> None:
        """Convert AggregatedQuote → context arrays → call strategy."""
        q = Quote(
            provider=aq.best_bid_provider,
            pair=aq.pair,
            bid=aq.best_bid,
            ask=aq.best_ask,
            bid_size=aq.best_bid_size,
            ask_size=aq.best_ask_size,
            timestamp=aq.timestamp,
        )
        self._history.append(q)

        bids = np.array([h.bid for h in self._history], dtype=float)
        asks = np.array([h.ask for h in self._history], dtype=float)
        mids = (bids + asks) / 2.0
        timestamps = [h.timestamp for h in self._history]

        ctx = StrategyContext(
            pair=self._strategy.pair,
            bid=bids,
            ask=asks,
            mid=mids,
            timestamps=timestamps,
            position=self._net_qty,
            avg_price=self._avg_price,
        )

        try:
            signal = self._strategy.on_tick(ctx)
        except Exception:
            self._log_msg(f"on_tick error:\n{traceback.format_exc()}")
            return

        if signal:
            await self._execute(signal, aq)

    async def _execute(self, signal: str, aq) -> None:
        signal = signal.lower()
        qty = self._strategy.quantity

        if signal == "buy":
            if self._net_qty >= qty:
                return       # already long enough
            side = Side.BUY
        elif signal == "sell":
            if self._net_qty <= -qty:
                return       # already short enough
            side = Side.SELL
        elif signal == "close":
            if self._net_qty == 0.0:
                return
            side = Side.SELL if self._net_qty > 0 else Side.BUY
            qty = abs(self._net_qty)
        else:
            return

        order = OrderRequest(
            pair=self._strategy.pair,
            side=side,
            quantity=qty,
            order_type=OrderType.MARKET,
        )

        dec = 3 if "JPY" in self._strategy.pair else 5
        self._log_msg(
            f"Signal {signal.upper()} {qty:,.0f} {self._strategy.pair} "
            f"bid={aq.best_bid:.{dec}f} ask={aq.best_ask:.{dec}f}"
        )

        try:
            fill = await self._sor.execute(order)
            sign = 1.0 if fill.side == Side.BUY else -1.0
            self._update_position(fill.quantity * sign, fill.fill_price)
            self._log_msg(
                f"Fill {fill.status.value} @ {fill.fill_price:.{dec}f}  "
                f"slippage={fill.slippage_pips:.1f}pip  "
                f"position={self._net_qty:+,.0f}"
            )
            if self._fill_callback:
                self._fill_callback(fill)
            try:
                self._strategy.on_fill(fill)
            except Exception:
                pass
        except Exception:
            self._log_msg(f"Order error:\n{traceback.format_exc()}")

    def _update_position(self, delta: float, price: float) -> None:
        old = self._net_qty
        new = old + delta
        if new == 0.0:
            self._net_qty = 0.0
            self._avg_price = 0.0
        elif old == 0.0 or (old > 0) == (delta > 0):
            total = abs(old) * self._avg_price + abs(delta) * price
            self._avg_price = total / abs(new)
            self._net_qty = new
        else:
            self._net_qty = new
            if abs(new) > abs(old):  # reversed
                self._avg_price = price

    def _log_msg(self, msg: str) -> None:
        ts = datetime.now(UTC).strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self._log.append(line)
        if len(self._log) > MAX_LOG:
            self._log = self._log[-MAX_LOG:]
        logger.info("[%s] %s", self._name, msg)
