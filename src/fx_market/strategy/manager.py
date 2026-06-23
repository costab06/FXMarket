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
EWMA Crossover Strategy
-----------------------
Buys when the fast EWMA crosses above the slow EWMA, sells when it crosses
below.  Uses 24 hours of Dukascopy tick history as warmup so the EWMA is
fully initialised before the first live tick arrives.

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

        if high_vol:
            return None

        # Crossover: fast rises above slow -> buy
        if fast[-2] <= slow[-2] and fast[-1] > slow[-1]:
            return "buy"

        # Crossunder: fast falls below slow -> sell
        if fast[-2] >= slow[-2] and fast[-1] < slow[-1]:
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


def _validate_name(name: str) -> None:
    if not name.replace("_", "").isalnum():
        raise ValueError(
            f"Strategy name must be alphanumeric (underscores allowed): {name!r}"
        )
