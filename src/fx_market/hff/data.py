"""Tick data utilities — cleaning, interpolation, and regular sampling.

Reference: Dacorogna et al. (2001), Chapter 2: "Dealing with Data Irregularities"

Raw FX tick data is irregularly spaced in time and must be cleaned of erroneous
quotes before statistical analysis.  The functions in this module implement the
standard pipeline described in Chapter 2:

    1. Compute bid-ask midpoint prices.
    2. Remove quotes with abnormally large spreads (filter_large_spread).
    3. Remove price level outliers using a rolling robust-mean filter (filter_outliers).
    4. Interpolate the cleaned tick series onto a regular time grid (interpolate_linear).
    5. Optionally resample to a chosen frequency (sample_regular).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def midpoint(bid: np.ndarray | float, ask: np.ndarray | float) -> np.ndarray | float:
    """Compute the bid-ask midpoint price.

    The midpoint is the natural price representative for a two-sided quote
    because it is equidistant from both the best offer to buy (bid) and the
    best offer to sell (ask).

    Reference: Dacorogna et al. (2001), Section 2.1, Eq. (2.1).

    Parameters
    ----------
    bid : array-like or float
        Bid price(s).
    ask : array-like or float
        Ask price(s).

    Returns
    -------
    np.ndarray or float
        Midpoint price(s): (bid + ask) / 2.

    Examples
    --------
    >>> midpoint(1.0810, 1.0812)
    1.0811
    >>> midpoint(np.array([1.0810, 1.0820]), np.array([1.0812, 1.0822]))
    array([1.0811, 1.0821])
    """
    bid = np.asarray(bid, dtype=float)
    ask = np.asarray(ask, dtype=float)
    return (bid + ask) / 2.0


def interpolate_linear(
    prices: np.ndarray,
    times: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate an irregularly-spaced tick series onto a regular grid.

    Because tick data arrives at random times, most statistical tools require a
    uniform time grid.  Dacorogna et al. advocate *linear interpolation* between
    adjacent ticks as the simplest method that preserves the price level
    (as opposed to, e.g., last-value-carried-forward which introduces a
    step-function bias in computed variances).

    Reference: Dacorogna et al. (2001), Section 2.2, "Homogenisation of Data".

    Parameters
    ----------
    prices : np.ndarray, shape (N,)
        Observed prices at the tick times.
    times : np.ndarray, shape (N,)
        Timestamps of ticks (any numeric unit; must be monotonically increasing).
    target_times : np.ndarray, shape (M,)
        Target timestamps for the regular grid (must be within [times[0], times[-1]]).

    Returns
    -------
    np.ndarray, shape (M,)
        Interpolated prices at each target timestamp.

    Notes
    -----
    Extrapolation beyond the observed range is intentionally not supported;
    target times outside [times[0], times[-1]] will raise a ValueError.
    """
    prices = np.asarray(prices, dtype=float)
    times = np.asarray(times, dtype=float)
    target_times = np.asarray(target_times, dtype=float)

    if target_times[0] < times[0] or target_times[-1] > times[-1]:
        raise ValueError("target_times must lie within the observed time range")

    return np.interp(target_times, times, prices)


def filter_large_spread(
    bid: np.ndarray,
    ask: np.ndarray,
    max_relative_spread: float = 0.01,
) -> np.ndarray:
    """Return a boolean mask of quotes whose spread is within an acceptable range.

    Erroneous tick quotes frequently manifest as abnormally large bid-ask
    spreads.  Dacorogna et al. recommend removing any quote where the
    relative spread exceeds a threshold (default 1 % of midpoint).

    Reference: Dacorogna et al. (2001), Section 2.3, "Filtering of Data".

    Parameters
    ----------
    bid : np.ndarray, shape (N,)
        Bid prices.
    ask : np.ndarray, shape (N,)
        Ask prices.
    max_relative_spread : float, default 0.01
        Maximum acceptable spread as a fraction of the midpoint.
        E.g. 0.01 → reject quotes where (ask − bid) / mid > 1 %.

    Returns
    -------
    np.ndarray of bool, shape (N,)
        True where the quote passes the spread filter.
    """
    bid = np.asarray(bid, dtype=float)
    ask = np.asarray(ask, dtype=float)
    mid = midpoint(bid, ask)
    relative_spread = (ask - bid) / mid
    return (relative_spread >= 0) & (relative_spread <= max_relative_spread)


def filter_outliers(
    prices: np.ndarray,
    times: np.ndarray,
    window_seconds: float = 300.0,
    sigma_threshold: float = 3.0,
) -> np.ndarray:
    """Return a boolean mask of prices that lie within a rolling robust band.

    Price level outliers (e.g. fat-finger errors) are detected by comparing
    each tick price to a rolling median and median-absolute-deviation (MAD)
    computed over a local window.  A tick is flagged as an outlier if it
    deviates more than ``sigma_threshold`` MADs from the rolling median.

    MAD is used rather than standard deviation because it is robust to the
    very outliers being filtered.

    Reference: Dacorogna et al. (2001), Section 2.3, "Filtering of Data".

    Parameters
    ----------
    prices : np.ndarray, shape (N,)
        Mid prices (or raw bid/ask prices) to filter.
    times : np.ndarray, shape (N,)
        Timestamps in seconds.
    window_seconds : float, default 300
        Half-width of the rolling time window (5 minutes).
    sigma_threshold : float, default 3.0
        Number of MADs outside of which a price is classified as an outlier.

    Returns
    -------
    np.ndarray of bool, shape (N,)
        True where the price is *not* an outlier.
    """
    prices = np.asarray(prices, dtype=float)
    times = np.asarray(times, dtype=float)
    n = len(prices)
    mask = np.ones(n, dtype=bool)

    for i in range(n):
        lo, hi = times[i] - window_seconds, times[i] + window_seconds
        idx = (times >= lo) & (times <= hi)
        window_prices = prices[idx]
        med = np.median(window_prices)
        mad = np.median(np.abs(window_prices - med))
        if mad == 0:
            # All window values are identical; any deviation is an outlier
            if prices[i] != med:
                mask[i] = False
            continue
        if abs(prices[i] - med) > sigma_threshold * mad * 1.4826:
            # 1.4826 makes MAD a consistent estimator of σ for Gaussian data
            mask[i] = False

    return mask


def sample_regular(
    prices: np.ndarray,
    times: np.ndarray,
    freq_seconds: float,
    method: str = "interpolate",
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a tick series to a fixed time frequency.

    Produces a uniformly-spaced price series starting at times[0] and ending
    at the last multiple of ``freq_seconds`` not exceeding times[-1].

    Reference: Dacorogna et al. (2001), Section 2.2.

    Parameters
    ----------
    prices : np.ndarray, shape (N,)
        Observed prices.
    times : np.ndarray, shape (N,)
        Tick timestamps in seconds (must be monotonically increasing).
    freq_seconds : float
        Target sampling frequency in seconds (e.g. 300 for 5-minute bars).
    method : {"interpolate", "last"}
        "interpolate" — linearly interpolate between ticks (recommended).
        "last"        — use the most recent observed price (previous-tick
                        interpolation; introduces a step-function bias).

    Returns
    -------
    sampled_prices : np.ndarray
        Prices on the regular grid.
    sampled_times : np.ndarray
        Regular timestamps.
    """
    prices = np.asarray(prices, dtype=float)
    times = np.asarray(times, dtype=float)

    t_start = times[0]
    t_end = times[-1]
    sampled_times = np.arange(t_start, t_end + 1e-9, freq_seconds)

    if method == "interpolate":
        sampled_prices = np.interp(sampled_times, times, prices)
    elif method == "last":
        indices = np.searchsorted(times, sampled_times, side="right") - 1
        indices = np.clip(indices, 0, len(prices) - 1)
        sampled_prices = prices[indices]
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'interpolate' or 'last'.")

    return sampled_prices, sampled_times
