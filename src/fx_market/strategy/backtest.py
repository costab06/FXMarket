"""Strategy backtester — replays Dukascopy historical ticks through a strategy.

Capital & currency model
------------------------
The backtester maintains actual currency balances rather than abstract P&L.

Initialisation
~~~~~~~~~~~~~~
The user supplies a ``capital`` amount in their ``domestic_currency``.  The
pair is parsed into its ``base`` and ``quote`` currencies (e.g. "EURUSD" →
base=EUR, quote=USD).  Capital is split evenly at the opening tick price so
that the strategy can both buy *and* sell from tick 1:

    domestic == quote  (e.g. USD trading EUR/USD):
        wallet = { USD: capital/2,  EUR: (capital/2) / ask₀ }

    domestic == base  (e.g. EUR trading EUR/USD):
        wallet = { EUR: capital/2,  USD: (capital/2) × bid₀ }

    domestic == neither  (e.g. GBP trading EUR/USD):
        treated as domestic == quote (USD), P&L reported in quote.

Trade execution
~~~~~~~~~~~~~~~
    BUY  qty base  →  wallet[base] += qty;   wallet[quote] -= qty × ask
    SELL qty base  →  wallet[base] -= qty;   wallet[quote] += qty × bid
    CLOSE          →  reverse the open position at current bid or ask

Insufficient balance prevents the trade (logged, strategy continues).

Repatriation
~~~~~~~~~~~~
At period end, all remaining foreign-currency holdings are converted back to
the domestic currency at the final tick's bid or ask:

    domestic == quote:  domestic_value = wallet[quote] + wallet[base] × final_bid
    domestic == base:   domestic_value = wallet[base]  + wallet[quote] / final_ask

P&L = final_domestic_value − initial_capital
"""

from __future__ import annotations

import collections
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np

from ..feeds.dukascopy import DukascopyHistory
from ..models import Quote
from .base import StrategyBase, StrategyContext

UTC = timezone.utc

MAX_EVALUATIONS  = 5_000   # max on_tick calls per backtest
MAX_CHART_POINTS = 1_000   # max points in each chart series


# ── Currency wallet ────────────────────────────────────────────────────────────

@dataclass
class Wallet:
    """Tracks actual currency balances for a single-pair backtest."""
    domestic: str
    base: str
    quote: str
    balances: dict[str, float] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        capital: float,
        domestic: str,
        base: str,
        quote: str,
        initial_ask: float,
        initial_bid: float,
    ) -> "Wallet":
        """Split ``capital`` (in domestic) evenly between base and quote."""
        w = cls(domestic=domestic, base=base, quote=quote,
                balances={base: 0.0, quote: 0.0})
        half = capital / 2.0
        if domestic == quote:
            w.balances[quote] = half
            w.balances[base]  = half / initial_ask
        elif domestic == base:
            w.balances[base]  = half
            w.balances[quote] = half * initial_bid
        else:
            # domestic is neither — treat capital as quote currency
            w.balances[quote] = half
            w.balances[base]  = half / initial_ask
        return w

    def value_in_domestic(self, mid: float) -> float:
        """Total wallet value expressed in domestic currency."""
        if self.domestic == self.quote:
            return self.balances[self.quote] + self.balances[self.base] * mid
        elif self.domestic == self.base:
            return self.balances[self.base] + self.balances[self.quote] / mid
        else:
            return self.balances[self.quote] + self.balances[self.base] * mid

    def can_buy(self, qty: float, ask: float) -> bool:
        return self.balances[self.quote] >= qty * ask

    def can_sell(self, qty: float) -> bool:
        return self.balances[self.base] >= qty

    def buy(self, qty: float, ask: float) -> None:
        self.balances[self.base]  += qty
        self.balances[self.quote] -= qty * ask

    def sell(self, qty: float, bid: float) -> None:
        self.balances[self.base]  -= qty
        self.balances[self.quote] += qty * bid

    def snapshot(self) -> dict:
        return {k: round(v, 4) for k, v in self.balances.items()}


# ── Public API ────────────────────────────────────────────────────────────────

async def run_backtest(
    strategy: StrategyBase,
    start: datetime,
    end: datetime,
    domestic_currency: str = "USD",
    capital: float = 10_000.0,
) -> dict:
    """Run a full backtest and return a result dict."""
    base, quote = strategy.pair[:3].upper(), strategy.pair[3:].upper()
    result: dict = {
        "id": uuid.uuid4().hex[:8],
        "status": "running",
        "pair": strategy.pair,
        "base": base,
        "quote": quote,
        "domestic": domestic_currency.upper(),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "initial_capital": capital,
        "progress": "Starting…",
        "total_ticks": 0,
        "evaluated_ticks": 0,
        "starting_wallet": {},
        "ending_wallet": {},
        "final_value_domestic": 0.0,
        "metrics": {},
        "equity_curve": [],
        "price_series": [],
        "trades": [],
        "error": "",
    }
    try:
        await _execute(strategy, start, end, domestic_currency.upper(), capital, result)
        result["status"] = "done"
    except Exception:
        result["status"] = "error"
        result["error"] = traceback.format_exc()
    return result


# ── Engine ────────────────────────────────────────────────────────────────────

async def _execute(
    strategy: StrategyBase,
    start: datetime,
    end: datetime,
    domestic: str,
    capital: float,
    result: dict,
) -> None:
    base  = result["base"]
    quote = result["quote"]
    hist_client = DukascopyHistory(max_concurrent=8)

    # ── 1. Warmup ─────────────────────────────────────────────────────────────
    history: collections.deque[Quote] = collections.deque(maxlen=strategy.max_history)
    if strategy.warmup_hours > 0:
        warmup_start = start - timedelta(hours=strategy.warmup_hours)
        result["progress"] = f"Fetching {strategy.warmup_hours}h warmup…"
        warmup_ticks = await hist_client.fetch_range(strategy.pair, warmup_start, start)
        for tick in warmup_ticks:
            history.append(tick)
        result["progress"] = f"Warmup: {len(warmup_ticks):,} ticks. Fetching backtest window…"

    # ── 2. Backtest data ──────────────────────────────────────────────────────
    result["progress"] = "Fetching backtest ticks…"
    ticks = await hist_client.fetch_range(strategy.pair, start, end)
    result["total_ticks"] = len(ticks)
    if not ticks:
        result["metrics"] = {"error": "No tick data for this range"}
        result["status"] = "done"
        return

    # ── 3. Initialise wallet ──────────────────────────────────────────────────
    first = ticks[0]
    wallet = Wallet.create(
        capital=capital,
        domestic=domestic,
        base=base,
        quote=quote,
        initial_ask=first.ask,
        initial_bid=first.bid,
    )
    result["starting_wallet"] = wallet.snapshot()
    initial_domestic = wallet.value_in_domestic((first.bid + first.ask) / 2)

    result["progress"] = f"Replaying {len(ticks):,} ticks…"

    # Give the strategy a chance to initialise any instance state (e.g. _prev_sign)
    try:
        strategy.on_start()
    except Exception:
        pass

    # ── 4. Replay ─────────────────────────────────────────────────────────────
    step       = max(1, len(ticks) // MAX_EVALUATIONS)
    chart_step = max(1, len(ticks) // MAX_CHART_POINTS)

    # Position in base currency units (signed)
    position  = 0.0
    avg_price = 0.0          # average entry price (in quote currency)
    qty       = strategy.quantity

    trades:        list[dict] = []
    price_points:  list[dict] = []
    equity_points: list[dict] = []
    trade_pnls:    list[float] = []   # realised P&L per closing trade (in quote)

    for i, tick in enumerate(ticks):
        history.append(tick)
        mid = (tick.bid + tick.ask) / 2.0

        # Sample for charts
        if i % chart_step == 0:
            ts = tick.timestamp.isoformat()
            price_points.append({"t": ts, "v": round(mid, 6)})
            equity_points.append({"t": ts, "v": round(wallet.value_in_domestic(mid), 4)})

        # Evaluate strategy
        if i % step != 0:
            continue

        bids = np.array([h.bid for h in history], dtype=float)
        asks = np.array([h.ask for h in history], dtype=float)
        mids = (bids + asks) / 2.0
        ctx  = StrategyContext(
            pair=strategy.pair, bid=bids, ask=asks, mid=mids,
            timestamps=[h.timestamp for h in history],
            position=position, avg_price=avg_price,
        )

        try:
            signal = strategy.on_tick(ctx)
        except Exception:
            continue

        if not signal:
            continue
        signal = signal.lower()
        ts = tick.timestamp.isoformat()

        if signal == "buy":
            if position >= qty:
                continue
            fill_price = tick.ask

            # Close any short first
            if position < 0:
                if not wallet.can_buy(abs(position), fill_price):
                    continue
                realized = (avg_price - fill_price) * abs(position)
                wallet.buy(abs(position), fill_price)
                trade_pnls.append(realized)
                trades.append(_trade(ts, "close_short", fill_price, abs(position),
                                     realized, quote, wallet))
                position = 0.0; avg_price = 0.0

            if not wallet.can_buy(qty, fill_price):
                _log_insufficient(trades, ts, "buy", wallet, base, quote)
                continue
            wallet.buy(qty, fill_price)
            position, avg_price = _update_pos(position, avg_price, qty, fill_price)
            trades.append(_trade(ts, "buy", fill_price, qty, 0.0, quote, wallet))

        elif signal == "sell":
            if position <= -qty:
                continue
            fill_price = tick.bid

            # Close any long first
            if position > 0:
                if not wallet.can_sell(position):
                    continue
                realized = (fill_price - avg_price) * position
                wallet.sell(position, fill_price)
                trade_pnls.append(realized)
                trades.append(_trade(ts, "close_long", fill_price, position,
                                     realized, quote, wallet))
                position = 0.0; avg_price = 0.0

            if not wallet.can_sell(qty):
                _log_insufficient(trades, ts, "sell", wallet, base, quote)
                continue
            wallet.sell(qty, fill_price)
            position, avg_price = _update_pos(position, avg_price, -qty, fill_price)
            trades.append(_trade(ts, "sell", fill_price, qty, 0.0, quote, wallet))

        elif signal == "close":
            if position == 0.0:
                continue
            if position > 0:
                fill_price = tick.bid
                if not wallet.can_sell(position):
                    continue
                realized = (fill_price - avg_price) * position
                wallet.sell(position, fill_price)
            else:
                fill_price = tick.ask
                if not wallet.can_buy(abs(position), fill_price):
                    continue
                realized = (avg_price - fill_price) * abs(position)
                wallet.buy(abs(position), fill_price)
            trade_pnls.append(realized)
            trades.append(_trade(ts, "close", fill_price, abs(position),
                                 realized, quote, wallet))
            position = 0.0; avg_price = 0.0

        result["evaluated_ticks"] += 1

    # ── 5. Close open position at period end ──────────────────────────────────
    if position != 0.0 and ticks:
        last = ticks[-1]
        ts   = last.timestamp.isoformat()
        if position > 0:
            fill_price = last.bid
            if wallet.can_sell(position):
                realized = (fill_price - avg_price) * position
                wallet.sell(position, fill_price)
                trade_pnls.append(realized)
                trades.append(_trade(ts, "close (eob)", fill_price, position,
                                     realized, quote, wallet))
        else:
            fill_price = last.ask
            if wallet.can_buy(abs(position), fill_price):
                realized = (avg_price - fill_price) * abs(position)
                wallet.buy(abs(position), fill_price)
                trade_pnls.append(realized)
                trades.append(_trade(ts, "close (eob)", fill_price, abs(position),
                                     realized, quote, wallet))

    # ── 6. Repatriation ───────────────────────────────────────────────────────
    last_tick  = ticks[-1]
    final_mid  = (last_tick.bid + last_tick.ask) / 2.0
    final_value = wallet.value_in_domestic(final_mid)

    # Final equity point after repatriation
    equity_points.append({
        "t": last_tick.timestamp.isoformat(),
        "v": round(final_value, 4),
    })

    result["ending_wallet"]        = wallet.snapshot()
    result["final_value_domestic"] = round(final_value, 4)
    result["trades"]               = trades
    result["equity_curve"]         = equity_points
    result["price_series"]         = price_points
    result["metrics"]              = _metrics(
        trade_pnls, capital, initial_domestic, final_value,
        domestic, quote, final_mid,
    )
    result["progress"] = "Done"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _trade(
    ts: str,
    side: str,
    price: float,
    qty: float,
    pnl_quote: float,
    quote: str,
    wallet: Wallet,
) -> dict:
    return {
        "timestamp":  ts,
        "side":       side,
        "price":      round(price, 6),
        "quantity":   qty,
        "pnl_quote":  round(pnl_quote, 4),
        "wallet":     wallet.snapshot(),
    }


def _log_insufficient(trades, ts, side, wallet, base, quote):
    trades.append({
        "timestamp": ts,
        "side": f"{side} (insufficient {quote if side == 'buy' else base})",
        "price": 0.0, "quantity": 0.0, "pnl_quote": 0.0,
        "wallet": wallet.snapshot(),
    })


def _update_pos(pos: float, avg: float, delta: float, price: float) -> tuple[float, float]:
    new_pos = pos + delta
    if new_pos == 0.0:
        return 0.0, 0.0
    if pos == 0.0 or (pos > 0) == (delta > 0):
        new_avg = (abs(pos) * avg + abs(delta) * price) / abs(new_pos)
    else:
        new_avg = avg if abs(new_pos) <= abs(pos) else price
    return new_pos, new_avg


def _metrics(
    trade_pnls: list[float],
    capital: float,
    initial_domestic: float,
    final_value: float,
    domestic: str,
    quote: str,
    final_mid: float,
) -> dict:
    pnl_domestic = final_value - capital

    # Convert trade P&Ls from quote to domestic if needed
    if domestic == quote:
        pnl_arr = np.array(trade_pnls, dtype=float)
    else:
        # Approximate: scale by final mid
        pnl_arr = np.array(trade_pnls, dtype=float) / final_mid

    if len(pnl_arr) == 0:
        return {
            "num_closing_trades": 0,
            "total_pnl_domestic": round(pnl_domestic, 4),
            "total_return_pct": round(pnl_domestic / capital * 100, 3),
        }

    wins   = pnl_arr[pnl_arr > 0]
    losses = pnl_arr[pnl_arr < 0]
    gross_profit = float(wins.sum())  if len(wins)   else 0.0
    gross_loss   = float(-losses.sum()) if len(losses) else 0.0
    pf = round(gross_profit / gross_loss, 2) if gross_loss > 0 else None

    sharpe = 0.0
    if pnl_arr.std() > 0:
        sharpe = round(float(pnl_arr.mean() / pnl_arr.std() * np.sqrt(252)), 3)

    cum = np.cumsum(pnl_arr)
    max_dd_domestic = float((cum - np.maximum.accumulate(cum)).min())

    return {
        "num_closing_trades": len(pnl_arr),
        "total_pnl_domestic":  round(pnl_domestic, 4),
        "total_return_pct":    round(pnl_domestic / capital * 100, 3),
        "win_rate_pct":        round(len(wins) / len(pnl_arr) * 100, 1),
        "profit_factor":       pf,
        "sharpe_ratio":        sharpe,
        "max_drawdown":        round(max_dd_domestic, 4),
        "avg_trade_pnl":       round(float(pnl_arr.mean()), 4),
        "gross_profit":        round(gross_profit, 4),
        "gross_loss":          round(gross_loss, 4),
    }
