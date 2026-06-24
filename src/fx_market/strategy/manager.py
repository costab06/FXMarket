"""Strategy manager — file I/O and runner lifecycle."""

from __future__ import annotations

import importlib.util
import inspect
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from ..models import Fill
from .base import StrategyBase
from .runner import StrategyRunner

if TYPE_CHECKING:
    from ..aggregator import MarketAggregator
    from ..sor import SmartOrderRouter

logger = logging.getLogger(__name__)

_EXAMPLE_STRATEGY = '''\
"""
EWMA Crossover Strategy (tick-based)
-------------------------------------
Buys when the fast EWMA crosses above the slow EWMA, sells when it crosses
below.  Uses 24 hours of Dukascopy tick history as warmup so the EWMA is
fully initialised before the first live tick arrives.

NOTE: tau values here are in TICKS, not time.  FX tick arrival rate varies
enormously — from ~5/s during the London/NY overlap to <1/s overnight — so
tau=50 means anything from 10 seconds to over a minute depending on the time
of day.  See ewma_cross_theta.py for a version that corrects for this.

Crossover detection
~~~~~~~~~~~~~~~~~~~
The strategy stores the sign of (fast - slow) from the previous call in
self._prev_sign and fires when it changes.  This is important in backtesting:
the engine skips ticks between evaluation calls, so comparing fast[-1] to
fast[-2] (adjacent buffer elements) would miss nearly every crossover.
Comparing against the value from the last call works correctly whether
on_tick is called on every tick (live) or every N ticks (backtest).

warmup_hours = 24  ->  the runner pre-loads ~24 h of ticks from Dukascopy
                       into the history buffer before starting to trade.
max_history  = 5000 -> keeps the 5 000 most recent ticks in memory.
"""
from fx_market.strategy import StrategyBase, StrategyContext
from fx_market.hff import ewma_from_tau, log_returns, ewma_volatility


class EWMACross(StrategyBase):
    # pair and quantity are set at runtime from the UI — no need to hardcode them here
    warmup_hours = 24       # pre-load 24 h of ticks before going live
    max_history  = 5_000

    # Minimum ticks required before trading
    MIN_TICKS = 500

    def on_start(self):
        self._prev_sign = 0   # sign of (fast - slow) at previous call; 0 = uninitialised

    def on_tick(self, ctx: StrategyContext):
        if len(ctx.mid) < self.MIN_TICKS:
            return None

        # EWMA crossover on mid price
        fast = ewma_from_tau(ctx.mid, tau=50)    # ~50-tick decay
        slow = ewma_from_tau(ctx.mid, tau=500)   # ~500-tick decay

        # Current volatility (annualised) from recent returns
        r   = log_returns(ctx.mid[-200:])
        vol = ewma_volatility(r, tau=30, dt=1.0)
        high_vol = vol[-1] > 0.10   # skip signals in very high volatility

        diff = fast[-1] - slow[-1]
        sign = 1 if diff > 0 else (-1 if diff < 0 else 0)

        prev  = self._prev_sign
        self._prev_sign = sign

        if high_vol or prev == 0:
            return None

        # Crossover: fast rose above slow since last call -> buy
        if prev < 0 and sign > 0:
            return "buy"

        # Crossunder: fast fell below slow since last call -> sell
        if prev > 0 and sign < 0:
            return "sell"

        return None
'''

_EXAMPLE_STRATEGY_THETA = '''\
"""
EWMA Crossover Strategy (theta-time bars)
------------------------------------------
Same EWMA crossover logic as ewma_cross.py, but operating on *business-time
bars* rather than raw ticks.  This is the theoretically correct way to apply
the strategy.

Why raw ticks are problematic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FX tick arrival is highly uneven: EURUSD sees ~5 ticks/s during the
London/NY overlap (08:00-17:00 UTC) but fewer than 1/s overnight.  When
EWMA is applied directly to raw ticks, tau=50 can mean 10 seconds at the
open and 2 minutes at 3 AM — the strategy behaves very differently at
different times of day without any intentional change.

What theta time fixes
~~~~~~~~~~~~~~~~~~~~~~
Dacorogna et al. (2001) introduced "theta time" (business time): a
monotone transformation ψ(t) that stretches quiet periods and compresses
active ones so that each unit of theta time contains equal expected market
activity.  The transformation is driven by a seasonal factor that models
the well-known intraday U-shape (peaks at the London open and the
London/NY overlap) and the day-of-week pattern (quieter Mondays and
Fridays).

After transforming to theta time and resampling to fixed-size bars, the
bar series has approximately stationary volatility regardless of time of
day.  tau values in the EWMA now correspond to a consistent number of
"average-activity-seconds", not a variable number of ticks.

Concretely:
    freq_seconds = 60   -> one bar per 60 avg-activity-seconds
    tau = 20 (fast)     -> 20-bar decay ~ 20 minutes of average activity
    tau = 200 (slow)    -> 200-bar decay ~ 3.3 hours of average activity

warmup_hours = 48  ->  48 h of tick history so the bar series is long
                       enough for the slow EWMA to be well-initialised.
max_history  = 20_000 -> enough ticks to cover ~48 h for most pairs.
"""
import numpy as np

from fx_market.strategy import StrategyBase, StrategyContext
from fx_market.hff import ewma_from_tau, log_returns, ewma_volatility
from fx_market.hff.time_transform import seasonal_factor, theta_time
from fx_market.hff.data import sample_regular

# Bar width in "average-activity-seconds"
BAR_SECONDS = 60
# Minimum bars needed before trading
MIN_BARS = 300


class EWMACrossTheta(StrategyBase):
    warmup_hours = 48
    max_history  = 20_000

    def on_start(self):
        self._prev_sign = 0   # sign of (fast - slow) at previous call; 0 = uninitialised

    def on_tick(self, ctx: StrategyContext):
        if len(ctx.mid) < 500:
            return None

        # Convert datetime timestamps to unix seconds (float array)
        unix = np.array([t.timestamp() for t in ctx.timestamps])

        # Build theta-time axis: each unit = one average-activity-second
        sf    = seasonal_factor(unix)
        theta = theta_time(unix, sf)

        # Resample mid prices onto a uniform theta-time grid
        bars, _ = sample_regular(ctx.mid, theta, freq_seconds=BAR_SECONDS)

        if len(bars) < MIN_BARS:
            return None

        # EWMA crossover on theta-time bars (tau is now in bars = avg-activity-minutes)
        fast = ewma_from_tau(bars, tau=20)    # ~20 min of average activity
        slow = ewma_from_tau(bars, tau=200)   # ~3.3 h of average activity

        # Volatility filter on bar returns
        r   = log_returns(bars[-100:])
        vol = ewma_volatility(r, tau=20, dt=1.0)
        if vol[-1] > 0.10:
            return None

        # Stateful crossover: compare sign now vs sign at previous call
        diff = fast[-1] - slow[-1]
        sign = 1 if diff > 0 else (-1 if diff < 0 else 0)
        prev = self._prev_sign
        self._prev_sign = sign

        if prev == 0:
            return None

        if prev < 0 and sign > 0:
            return "buy"
        if prev > 0 and sign < 0:
            return "sell"

        return None
'''


class StrategyManager:
    """Manages strategy files and running strategy instances.

    Strategies are stored as ``.py`` files in ``strategy_dir``.
    Each file must contain exactly one ``StrategyBase`` subclass.
    """

    def __init__(
        self,
        strategy_dir: Path,
        aggregator: "MarketAggregator",
        sor: "SmartOrderRouter",
        fill_callback: Callable[[Fill], None] | None = None,
    ) -> None:
        self._dir = strategy_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._agg = aggregator
        self._sor = sor
        self._fill_callback = fill_callback
        self._runners: dict[str, StrategyRunner] = {}

        self._seed_example()

    # ── File operations ───────────────────────────────────────────────────────

    def list_names(self) -> list[str]:
        return sorted(p.stem for p in self._dir.glob("*.py"))

    def read_code(self, name: str) -> str:
        path = self._dir / f"{name}.py"
        if not path.exists():
            raise FileNotFoundError(f"Strategy '{name}' not found")
        return path.read_text()

    def save_code(self, name: str, code: str) -> None:
        _validate_name(name)
        (self._dir / f"{name}.py").write_text(code)
        logger.info("Saved strategy: %s", name)

    def delete(self, name: str) -> None:
        if name in self._runners and self._runners[name].is_running:
            self.stop(name)
        path = self._dir / f"{name}.py"
        if path.exists():
            path.unlink()
        self._runners.pop(name, None)
        logger.info("Deleted strategy: %s", name)

    # ── Runner lifecycle ──────────────────────────────────────────────────────

    def start(self, name: str, pair: str | None = None, quantity: float | None = None) -> None:
        if name in self._runners and self._runners[name].is_running:
            return
        strategy = self._load(name)
        if pair:
            strategy.pair = pair
        if quantity is not None:
            strategy.quantity = quantity
        runner = StrategyRunner(
            strategy=strategy,
            name=name,
            aggregator=self._agg,
            sor=self._sor,
            fill_callback=self._fill_callback,
        )
        runner.start()
        self._runners[name] = runner

    def stop(self, name: str) -> None:
        if name in self._runners:
            self._runners[name].stop()

    def stop_all(self) -> None:
        for runner in self._runners.values():
            runner.stop()

    # ── Status & log ─────────────────────────────────────────────────────────

    def status(self, name: str) -> str:
        runner = self._runners.get(name)
        return "running" if runner and runner.is_running else "stopped"

    def get_log(self, name: str) -> list[str]:
        runner = self._runners.get(name)
        return runner.get_log() if runner else []

    def list_all(self) -> list[dict]:
        rows = []
        for name in self.list_names():
            runner = self._runners.get(name)
            rows.append({
                "name": name,
                "status": self.status(name),
                "pair": runner.pair if runner else "—",
                "position": runner.net_qty if runner else 0.0,
            })
        return rows

    # ── Private ───────────────────────────────────────────────────────────────

    def _load(self, name: str) -> StrategyBase:
        """Dynamically import strategy file and return an instance."""
        path = self._dir / f"{name}.py"
        if not path.exists():
            raise FileNotFoundError(f"Strategy file not found: {path}")

        spec = importlib.util.spec_from_file_location(f"_strategy_{name}", path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)  # type: ignore[union-attr]
        except Exception as exc:
            raise RuntimeError(f"Error loading {name}.py: {exc}") from exc

        # Find the first StrategyBase subclass defined in the file
        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if (
                inspect.isclass(obj)
                and issubclass(obj, StrategyBase)
                and obj is not StrategyBase
                and obj.__module__ == module.__name__
            ):
                return obj()

        raise RuntimeError(
            f"{name}.py does not contain a StrategyBase subclass"
        )

    def _peek_pair(self, name: str) -> str:
        """Best-effort: extract pair from class attribute without full load."""
        try:
            strategy = self._load(name)
            return strategy.pair
        except Exception:
            return "—"

    def _seed_example(self) -> None:
        example = self._dir / "ewma_cross.py"
        if not example.exists():
            example.write_text(_EXAMPLE_STRATEGY)
            logger.info("Created example strategy: ewma_cross.py")

        example_theta = self._dir / "ewma_cross_theta.py"
        if not example_theta.exists():
            example_theta.write_text(_EXAMPLE_STRATEGY_THETA)
            logger.info("Created example strategy: ewma_cross_theta.py")


def _validate_name(name: str) -> None:
    if not name.replace("_", "").isalnum():
        raise ValueError(
            f"Strategy name must be alphanumeric (underscores allowed): {name!r}"
        )
