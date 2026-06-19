"""Market microstructure measures: bid-ask spread, price impact, Roll model.

Reference: Dacorogna et al. (2001), Chapter 2: "Dealing with Data
Irregularities" and Chapter 3 (Section 3.1)

Market microstructure effects — particularly the bid-ask spread — are central
to interpreting high-frequency data.  Key phenomena documented in the book:

    * The **bid-ask spread** creates a systematic negative autocorrelation in
      trade-price returns (bid-ask bounce), even when the fundamental price
      follows a random walk.  At 1-tick lag this bounce is the dominant feature
      of the return ACF.

    * The **Roll (1984) model** estimates the implicit spread from the
      first-order serial covariance of price changes, without needing quote data.

    * **Price impact** (Amihud illiquidity) quantifies how much a given trade
      size moves the price — important for execution cost analysis.

    * **Tick frequency** (number of quotes per unit time) is a direct proxy for
      market activity and correlates strongly with volatility.
"""

from __future__ import annotations

import numpy as np


def quoted_spread(bid: np.ndarray | float, ask: np.ndarray | float) -> np.ndarray | float:
    """Compute the quoted bid-ask spread.

    The quoted spread is the cost of an immediate round-trip:

        spread = ask − bid

    Reference: Dacorogna et al. (2001), Section 2.1.

    Parameters
    ----------
    bid, ask : array-like or float
        Bid and ask prices.

    Returns
    -------
    np.ndarray or float
        Absolute spread(s): ask − bid.
    """
    return np.asarray(ask, dtype=float) - np.asarray(bid, dtype=float)


def relative_spread(
    bid: np.ndarray | float,
    ask: np.ndarray | float,
) -> np.ndarray | float:
    """Compute the relative (percentage) bid-ask spread.

    Also called the *proportional spread*:

        s_rel = (ask − bid) / mid = 2(ask − mid) / mid

    This is more comparable across currency pairs and price levels than the
    absolute spread.

    Reference: Dacorogna et al. (2001), Section 2.1.

    Parameters
    ----------
    bid, ask : array-like or float
        Bid and ask prices.

    Returns
    -------
    np.ndarray or float
        Relative spread(s): (ask − bid) / ((bid + ask) / 2).
    """
    bid = np.asarray(bid, dtype=float)
    ask = np.asarray(ask, dtype=float)
    mid = (bid + ask) / 2.0
    return (ask - bid) / mid


def effective_spread(
    trade_price: np.ndarray | float,
    midpoint_at_trade: np.ndarray | float,
) -> np.ndarray | float:
    """Compute the effective bid-ask spread for executed trades.

    The effective spread measures the actual execution cost relative to the
    contemporaneous midpoint.  It equals twice the signed distance from the
    midpoint:

        s_eff = 2 · |trade_price − midpoint|

    The effective spread is ≤ the quoted spread for active orders (which may
    be filled at improved prices) but can exceed the quoted spread for large
    orders that walk the book.

    Reference: Dacorogna et al. (2001), Section 2.1.

    Parameters
    ----------
    trade_price : array-like or float
        Actual execution price.
    midpoint_at_trade : array-like or float
        Prevailing bid-ask midpoint at the time of the trade.

    Returns
    -------
    np.ndarray or float
        Effective spread(s).
    """
    tp = np.asarray(trade_price, dtype=float)
    mid = np.asarray(midpoint_at_trade, dtype=float)
    return 2.0 * np.abs(tp - mid)


def roll_spread(prices: np.ndarray) -> float:
    """Estimate the implicit bid-ask spread using the Roll (1984) model.

    Roll (1984) showed that in an efficient market where trades alternate
    randomly between bid and ask, the first-order serial autocovariance of
    price changes is negative and equals −(s/2)², where s is the spread:

        Cov(Δpₜ, Δpₜ₋₁) = −(s/2)²

    Therefore the implied spread is:

        ŝ = 2 · √(−Cov(Δpₜ, Δpₜ₋₁))   if Cov < 0, else 0

    This estimator requires only transaction prices (no quote data) and is
    widely used to estimate transaction costs from historical price series.

    Reference: Dacorogna et al. (2001), Section 2.4;
               Roll, R. (1984). Journal of Finance, 39(4), 1127–1139.

    Parameters
    ----------
    prices : np.ndarray, shape (T,)
        Transaction or mid prices (a contiguous segment without gaps).

    Returns
    -------
    float
        Roll implied spread.  Returns 0 if the estimated autocovariance is
        non-negative (which can occur due to sampling noise).
    """
    prices = np.asarray(prices, dtype=float)
    dp = np.diff(prices)
    cov = np.cov(dp[:-1], dp[1:])[0, 1]
    if cov >= 0:
        return 0.0
    return 2.0 * np.sqrt(-cov)


def amihud_illiquidity(
    returns: np.ndarray,
    volumes: np.ndarray,
    scale: float = 1e6,
) -> np.ndarray:
    """Compute the Amihud (2002) illiquidity ratio.

    The Amihud ratio measures price impact — how much the price moves per
    unit of traded volume:

        ILLIQ_t = |r_t| / Volume_t

    Higher values indicate a less liquid market.  While originally defined for
    equities, the ratio is applicable to FX when tick volume (number of
    transactions) is used as the volume proxy.

    Reference: Dacorogna et al. (2001), Section 2.5;
               Amihud, Y. (2002). Journal of Financial Markets, 5(1), 31–56.

    Parameters
    ----------
    returns : np.ndarray, shape (T,)
        Log returns per bar.
    volumes : np.ndarray, shape (T,)
        Traded volume or tick count per bar (must be > 0).
    scale : float, default 1e6
        Scaling factor for readability (e.g. multiply to express per million
        units of volume).

    Returns
    -------
    np.ndarray, shape (T,)
        Illiquidity ratio per bar, scaled by ``scale``.
    """
    returns = np.asarray(returns, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    volumes = np.where(volumes == 0, np.nan, volumes)
    return np.abs(returns) / volumes * scale


def tick_frequency(
    times: np.ndarray,
    window_seconds: float = 60.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Count the number of ticks per rolling time window.

    Tick frequency (quote arrival rate) is one of the strongest proxies for
    contemporaneous market activity and realized volatility in FX markets.
    Periods of high volatility are almost always accompanied by elevated
    tick frequencies.

    Reference: Dacorogna et al. (2001), Section 2.6.

    Parameters
    ----------
    times : np.ndarray, shape (T,)
        Arrival timestamps of each tick, in seconds.
    window_seconds : float, default 60
        Width of the rolling window for counting ticks (in seconds).

    Returns
    -------
    freq_times : np.ndarray, shape (T,)
        Timestamp for each frequency estimate (centre of window).
    freq : np.ndarray, shape (T,)
        Number of ticks arriving within ±(window_seconds/2) of each timestamp.
    """
    times = np.asarray(times, dtype=float)
    half = window_seconds / 2.0
    freq = np.array([
        np.sum((times >= t - half) & (times <= t + half))
        for t in times
    ], dtype=float)
    return times, freq


def quote_imbalance(
    bid_size: np.ndarray | float,
    ask_size: np.ndarray | float,
) -> np.ndarray | float:
    """Compute the order book quote imbalance.

    The quote imbalance (also called order flow imbalance or depth imbalance)
    measures the relative dominance of buyers or sellers at the top of book:

        QI = (bid_size − ask_size) / (bid_size + ask_size)

    Values close to +1 indicate heavy buy-side depth; values near −1 indicate
    heavy sell-side depth.  Positive QI is predictive of short-term price
    appreciation (Lee & Ready, 1991).

    Reference: Dacorogna et al. (2001), Section 2.6.

    Parameters
    ----------
    bid_size, ask_size : array-like or float
        Available size (depth) at best bid and best ask, in base currency units.

    Returns
    -------
    np.ndarray or float
        Quote imbalance ∈ [−1, +1].
    """
    bid_size = np.asarray(bid_size, dtype=float)
    ask_size = np.asarray(ask_size, dtype=float)
    total = bid_size + ask_size
    total = np.where(total == 0, np.nan, total)
    return (bid_size - ask_size) / total
