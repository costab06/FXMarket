"""Strategy framework for FXMarket.

Users subclass StrategyBase, implement on_tick(), and return a signal string.

Example::

    from fx_market.strategy import StrategyBase, StrategyContext
    from fx_market.hff import ewma

    class EWMACross(StrategyBase):
        pair = "EURUSD"
        quantity = 10_000

        def on_tick(self, ctx: StrategyContext):
            if len(ctx.mid) < 50:
                return None
            fast = ewma(ctx.mid, tau=5)
            slow = ewma(ctx.mid, tau=30)
            if fast[-1] > slow[-1]:
                return "buy"
            if fast[-1] < slow[-1]:
                return "sell"
            return None
"""

from .base import StrategyBase, StrategyContext

__all__ = ["StrategyBase", "StrategyContext"]
