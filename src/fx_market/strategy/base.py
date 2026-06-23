"""Strategy base class and context."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np


@dataclass
class StrategyContext:
    """Snapshot of market state passed to on_tick() on every quote.

    Arrays are ordered oldest-first (index 0 = oldest, index -1 = latest).
    """

    pair: str
    bid: np.ndarray        # best bid prices
    ask: np.ndarray        # best ask prices
    mid: np.ndarray        # midpoint prices
    timestamps: list[datetime]

    # This strategy's own position (not shared with other strategies)
    position: float        # positive = long, negative = short
    avg_price: float       # average entry price; 0.0 if flat


class StrategyBase(ABC):
    """Base class for all user strategies.

    Subclass this, set the class attributes, and implement on_tick().

    Signals
    -------
    Return one of these strings from on_tick():
        "buy"   — open or add to a long position
        "sell"  — open or add to a short position
        "close" — close any open position
        None    — do nothing

    History warm-up
    ---------------
    Set ``warmup_hours`` to pre-load that many hours of Dukascopy tick data
    into the history buffer before the first live tick arrives.  This ensures
    that indicators that need a large lookback (EWMA volatility, realized vol,
    structural breaks, CUSUM filter, etc.) have meaningful input from the start.

    ``max_history`` controls how many ticks are kept in the rolling buffer.
    The default (5 000) is enough for several hours of EURUSD tick data.

    Example::

        class VolBreakout(StrategyBase):
            pair         = "EURUSD"
            quantity     = 10_000
            warmup_hours = 48      # 2 days of history pre-loaded
            max_history  = 10_000  # keep up to 10k ticks

            def on_tick(self, ctx):
                if len(ctx.mid) < 500:
                    return None
                r   = log_returns(ctx.mid)
                vol = ewma_volatility(r, tau=100)
                ...
    """

    # ── Runtime parameters (set by the runner/backtester, not hardcoded) ──────
    # ``pair`` and ``quantity`` are defaults only.  When you start a strategy
    # or run a backtest from the UI you choose the pair there — the same
    # strategy class can trade any instrument without code changes.
    pair: str = "EURUSD"        # default pair; overridden at runtime
    quantity: float = 10_000    # units per order; overridden at runtime

    # ── History configuration ─────────────────────────────────────────────────
    warmup_hours: int = 24      # hours of Dukascopy history to pre-load on start
    max_history: int = 5_000    # rolling tick buffer size

    @abstractmethod
    def on_tick(self, ctx: StrategyContext) -> str | None:
        """Called on every new aggregated quote.

        Parameters
        ----------
        ctx : StrategyContext
            Current market snapshot including price history and position.

        Returns
        -------
        str | None
            "buy", "sell", "close", or None.
        """

    # ── Optional lifecycle hooks ──────────────────────────────────────────────

    def on_start(self) -> None:
        """Called once when the strategy is started. Initialize state here."""

    def on_stop(self) -> None:
        """Called once when the strategy is stopped."""

    def on_fill(self, fill) -> None:
        """Called after each order fill. ``fill`` is a Fill dataclass."""
