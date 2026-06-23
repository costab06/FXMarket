"""Financial data structures: tick, volume, dollar, and imbalance bars.

Reference: López de Prado (2018), Chapter 2: "Financial Data Structures"

The central argument of Chapter 2 is that calendar-time sampling of prices is
statistically inferior because it produces bars with non-IID returns.  More
informative sampling schemes produce bars with better statistical properties
(closer to IID) and carry more information per observation.

Four sampling schemes are implemented here, in order of sophistication:

1. **Tick bars** — one bar per N ticks (trades/quotes).
2. **Volume bars** — one bar per N units of traded volume.
3. **Dollar bars** — one bar per N dollars of notional traded.
   Most stable because dollar volume is approximately stationary.
4. **Tick/Volume imbalance bars** — event-driven: a bar is formed when the
   cumulative signed-tick (or signed-volume) imbalance exceeds an adaptive
   threshold.  These bars synchronise with the arrival of informed order flow.

All functions accept tick-level data as numpy arrays and return a pandas
DataFrame with columns [open, high, low, close, volume, vwap, tick_count]
indexed by the closing timestamp of each bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Utility ───────────────────────────────────────────────────────────────────

def _tick_rule(prices: np.ndarray) -> np.ndarray:
    """Apply the tick rule to infer the direction of each trade.

    b_t = sign(Δp_t) if Δp_t ≠ 0, else b_{t-1}

    Returns an array of {−1, +1}, initialised to +1 at t=0.

    Reference: López de Prado (2018), Section 2.3.
    """
    dp = np.diff(prices, prepend=prices[0])
    b = np.where(dp > 0, 1, np.where(dp < 0, -1, 0))
    # Forward-fill zeros with previous non-zero sign
    last = 1
    for i in range(len(b)):
        if b[i] == 0:
            b[i] = last
        else:
            last = b[i]
    return b.astype(float)


def _form_bar(
    prices: np.ndarray,
    timestamps,
    volumes: np.ndarray | None,
    idx: np.ndarray,
) -> dict:
    """Form a single OHLCV bar from a slice of ticks."""
    p = prices[idx]
    v = volumes[idx] if volumes is not None else None
    bar: dict = {
        "open":       float(p[0]),
        "high":       float(p.max()),
        "low":        float(p.min()),
        "close":      float(p[-1]),
        "volume":     float(v.sum()) if v is not None else 0.0,
        "vwap":       float((p * v).sum() / v.sum()) if v is not None and v.sum() > 0 else float(p.mean()),
        "tick_count": int(len(idx)),
    }
    if timestamps is not None:
        bar["timestamp_open"]  = timestamps[idx[0]]
        bar["timestamp_close"] = timestamps[idx[-1]]
    return bar


def _build_df(bars: list[dict], index_key: str = "timestamp_close") -> pd.DataFrame:
    df = pd.DataFrame(bars)
    if index_key in df.columns:
        df = df.set_index(index_key)
    return df


# ── Tick bars ─────────────────────────────────────────────────────────────────

def tick_bars(
    prices: np.ndarray,
    tick_size: int,
    timestamps=None,
    volumes: np.ndarray | None = None,
) -> pd.DataFrame:
    """Form bars by sampling every ``tick_size`` ticks.

    Tick bars synchronise bar boundaries to trade activity, so they carry
    more information than calendar bars during quiet periods, and fewer
    redundant bars during volatile periods.

    Reference: López de Prado (2018), Section 2.2.1.

    Parameters
    ----------
    prices : np.ndarray, shape (T,)
        Trade or mid prices in tick order.
    tick_size : int
        Number of ticks per bar.
    timestamps : array-like, optional
        Timestamps for each tick.  If provided, used as bar index.
    volumes : np.ndarray, optional
        Trade volumes for each tick.

    Returns
    -------
    pd.DataFrame
        OHLCV bars with columns [open, high, low, close, volume, vwap, tick_count].
        Indexed by closing timestamp if timestamps provided, else by integer.
    """
    prices = np.asarray(prices, dtype=float)
    T = len(prices)
    bars = []
    i = 0
    while i + tick_size <= T:
        idx = np.arange(i, i + tick_size)
        bars.append(_form_bar(prices, timestamps, volumes, idx))
        i += tick_size
    return _build_df(bars)


# ── Volume bars ───────────────────────────────────────────────────────────────

def volume_bars(
    prices: np.ndarray,
    volumes: np.ndarray,
    volume_size: float,
    timestamps=None,
) -> pd.DataFrame:
    """Form bars every time ``volume_size`` units of volume have traded.

    Volume bars normalise bar sizes by traded quantity rather than elapsed
    time.  Under the mixture-of-distributions hypothesis, volume is the
    natural clock for price processes.

    Reference: López de Prado (2018), Section 2.2.2.

    Parameters
    ----------
    prices : np.ndarray, shape (T,)
        Trade prices.
    volumes : np.ndarray, shape (T,)
        Traded volume for each tick.
    volume_size : float
        Target volume per bar.
    timestamps : array-like, optional
        Tick timestamps.

    Returns
    -------
    pd.DataFrame
        OHLCV bars.
    """
    prices  = np.asarray(prices,  dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    bars, cum_vol, start = [], 0.0, 0
    for i, v in enumerate(volumes):
        cum_vol += v
        if cum_vol >= volume_size:
            idx = np.arange(start, i + 1)
            bars.append(_form_bar(prices, timestamps, volumes, idx))
            cum_vol, start = 0.0, i + 1
    return _build_df(bars)


# ── Dollar bars ───────────────────────────────────────────────────────────────

def dollar_bars(
    prices: np.ndarray,
    volumes: np.ndarray,
    dollar_size: float,
    timestamps=None,
) -> pd.DataFrame:
    """Form bars every time ``dollar_size`` of notional value has traded.

    Dollar bars (also called "money bars") are the most stable sampling
    scheme because the number of bars per period is approximately constant
    even as price levels and volatility change.  This is the recommended
    default in AFML for most machine-learning applications.

    Reference: López de Prado (2018), Section 2.2.3.

    Parameters
    ----------
    prices : np.ndarray, shape (T,)
        Trade prices.
    volumes : np.ndarray, shape (T,)
        Traded volume per tick.
    dollar_size : float
        Target notional (price × volume) per bar.
    timestamps : array-like, optional
        Tick timestamps.

    Returns
    -------
    pd.DataFrame
        OHLCV bars indexed by closing timestamp.
    """
    prices  = np.asarray(prices,  dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    notional = prices * volumes
    bars, cum_dollar, start = [], 0.0, 0
    for i, d in enumerate(notional):
        cum_dollar += d
        if cum_dollar >= dollar_size:
            idx = np.arange(start, i + 1)
            bars.append(_form_bar(prices, timestamps, volumes, idx))
            cum_dollar, start = 0.0, i + 1
    return _build_df(bars)


# ── Tick imbalance bars ───────────────────────────────────────────────────────

def tick_imbalance_bars(
    prices: np.ndarray,
    initial_expected_ticks: int = 100,
    ewm_span: int = 10,
    timestamps=None,
    volumes: np.ndarray | None = None,
) -> pd.DataFrame:
    """Form bars when the cumulative tick imbalance exceeds an adaptive threshold.

    The tick imbalance θ_T = Σ_{t=1}^T b_t measures the signed order-flow
    imbalance up to bar boundary T, where b_t ∈ {−1, +1} is the tick direction.

    A new bar is formed when:

        |θ_T| ≥ E₀[T] × |2P₀[b_t = 1] − 1|

    Both E[T] (expected ticks per bar) and P[b_t = 1] (probability of uptick)
    are updated after each bar using an exponential moving average, making the
    threshold adaptive to changing market microstructure.

    Imbalance bars synchronise with the arrival of informed order flow and
    have been shown to produce return series closer to IID than calendar bars.

    Reference: López de Prado (2018), Section 2.3.1, Snippet 2.4.

    Parameters
    ----------
    prices : np.ndarray, shape (T,)
        Prices in tick order.
    initial_expected_ticks : int, default 100
        Seed value for E[T] before any bar is formed.
    ewm_span : int, default 10
        Span parameter for the exponential moving average updates of
        E[T] and P[b_t = 1].
    timestamps : array-like, optional
        Tick timestamps.
    volumes : np.ndarray, optional
        Volumes per tick.

    Returns
    -------
    pd.DataFrame
        OHLCV bars.
    """
    prices = np.asarray(prices, dtype=float)
    b = _tick_rule(prices)
    alpha = 2.0 / (ewm_span + 1)

    # Seed estimates
    exp_ticks = float(initial_expected_ticks)
    exp_p_buy = 0.5          # prior: 50/50 uptick / downtick

    bars, start = [], 0
    cum_b = 0.0
    bar_tick_counts: list[int] = []
    bar_buy_fracs: list[float] = []

    for i, bi in enumerate(b):
        cum_b += bi
        threshold = exp_ticks * abs(2 * exp_p_buy - 1)
        if abs(cum_b) >= max(threshold, 1):
            idx = np.arange(start, i + 1)
            bars.append(_form_bar(prices, timestamps, volumes, idx))
            n = i - start + 1
            bar_tick_counts.append(n)
            bar_buy_fracs.append(float(np.sum(b[start : i + 1] > 0)) / n)
            # Update adaptive estimates
            exp_ticks = alpha * n + (1 - alpha) * exp_ticks
            exp_p_buy = alpha * bar_buy_fracs[-1] + (1 - alpha) * exp_p_buy
            cum_b, start = 0.0, i + 1

    return _build_df(bars)


# ── Volume imbalance bars ─────────────────────────────────────────────────────

def volume_imbalance_bars(
    prices: np.ndarray,
    volumes: np.ndarray,
    initial_expected_ticks: int = 100,
    ewm_span: int = 10,
    timestamps=None,
) -> pd.DataFrame:
    """Form bars when the cumulative signed-volume imbalance exceeds a threshold.

    Extends tick imbalance bars by weighting each tick's contribution by its
    volume:

        θ_T = Σ_{t=1}^T b_t × v_t

    Expected imbalance: E[θ_T] ≈ E[T] × E[b_t × v_t]

    Volume imbalance bars are more sensitive to large institutional trades
    than tick imbalance bars, making them useful for detecting informed
    order flow in FX markets.

    Reference: López de Prado (2018), Section 2.3.2.

    Parameters
    ----------
    prices : np.ndarray, shape (T,)
        Prices in tick order.
    volumes : np.ndarray, shape (T,)
        Volume per tick.
    initial_expected_ticks : int, default 100
        Initial E[T] seed.
    ewm_span : int, default 10
        EMA span for adaptive threshold updates.
    timestamps : array-like, optional
        Tick timestamps.

    Returns
    -------
    pd.DataFrame
        OHLCV bars.
    """
    prices  = np.asarray(prices,  dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    b = _tick_rule(prices)
    bv = b * volumes
    alpha = 2.0 / (ewm_span + 1)

    exp_ticks = float(initial_expected_ticks)
    exp_bv = bv[:initial_expected_ticks].mean() if len(bv) >= initial_expected_ticks else bv.mean()

    bars, start = [], 0
    cum_bv = 0.0

    for i in range(len(prices)):
        cum_bv += bv[i]
        threshold = exp_ticks * abs(exp_bv)
        if abs(cum_bv) >= max(threshold, 1e-9):
            idx = np.arange(start, i + 1)
            bars.append(_form_bar(prices, timestamps, volumes, idx))
            n = i - start + 1
            exp_ticks = alpha * n + (1 - alpha) * exp_ticks
            exp_bv = alpha * float(bv[start : i + 1].mean()) + (1 - alpha) * exp_bv
            cum_bv, start = 0.0, i + 1

    return _build_df(bars)
