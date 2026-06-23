"""Fractional differentiation for stationarity with memory.

Reference: López de Prado (2018), Chapter 5: "Fractional Differentiation"

Standard integer differentiation (d=1: first differences, d=2: second
differences) destroys memory.  Fractional differentiation for d ∈ (0, 1)
produces a stationary series while retaining as much memory as possible.

Two implementations:

1. **Expanding window** (``fracdiff_expanding``): Uses all past observations
   with weights that decay geometrically.  The full convolution formula
   makes it slow and impractical for production, but useful as a reference.

2. **Fixed-width window (FFD)** (``fracdiff_ffd``): Truncates the expanding
   weights below a threshold ε, trading a tiny amount of bias for O(w)
   memory and O(n·w) speed.  This is the practical production algorithm.

The ``min_ffd`` function finds the minimum d that makes the series
stationary (ADF p-value < threshold), preserving maximum memory.

Reference: López de Prado (2018), Sections 5.1–5.5, Snippets 5.1–5.3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Weights ────────────────────────────────────────────────────────────────


def _ffd_weights(d: float, threshold: float) -> np.ndarray:
    """Compute fractional differencing weights with FFD truncation.

    Weights: w_k = product_{j=1}^{k} (d - j + 1) / j  for k ≥ 1, w_0 = 1.
    Truncation: stop when |w_k| < threshold.

    Reference: López de Prado (2018), Snippet 5.1.
    """
    w = [1.0]
    k = 1
    while True:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1])  # oldest weight first


def _expanding_weights(d: float, size: int) -> np.ndarray:
    """Compute all fractional differencing weights up to ``size``."""
    w = np.empty(size)
    w[0] = 1.0
    for k in range(1, size):
        w[k] = -w[k - 1] * (d - k + 1) / k
    return w[::-1]


# ── FFD (Fixed-width window) ────────────────────────────────────────────────


def fracdiff_ffd(
    series: pd.Series,
    d: float,
    threshold: float = 1e-5,
) -> pd.Series:
    """Apply fractional differentiation using a fixed-width window.

    The FFD algorithm truncates the expanding weight vector at threshold ε,
    so the effective window width is finite.  The first len(weights)−1
    observations are NaN (insufficient history).

    This is the recommended algorithm for production use.

    Reference: López de Prado (2018), Section 5.4, Snippet 5.2.

    Parameters
    ----------
    series : pd.Series
        Price (or feature) series.  Log-prices are recommended.
    d : float
        Fractional differentiation order.  d=0 is no transformation,
        d=1 is standard first differences.
    threshold : float, default 1e-5
        Weight cutoff for the fixed-width window.  Smaller values give
        longer windows and better accuracy at the cost of computation.

    Returns
    -------
    pd.Series
        Fractionally differentiated series, same index as ``series``.
        First ``len(weights) - 1`` observations are NaN.
    """
    if d == 0:
        return series.copy()

    weights = _ffd_weights(d, threshold)
    w = len(weights)
    arr = series.values
    n = len(arr)
    out = np.full(n, np.nan)

    for i in range(w - 1, n):
        out[i] = np.dot(weights, arr[i - w + 1 : i + 1])

    return pd.Series(out, index=series.index)


# ── Expanding window ────────────────────────────────────────────────────────


def fracdiff_expanding(
    series: pd.Series,
    d: float,
) -> pd.Series:
    """Apply fractional differentiation using an expanding (full) window.

    At each time step t, uses all past values with weights that decay as
    w_k = Π_{j=1}^k (d - j + 1) / j.  This is exact but O(n²) in time.
    Prefer :func:`fracdiff_ffd` for large series.

    Reference: López de Prado (2018), Section 5.3, Snippet 5.1.

    Parameters
    ----------
    series : pd.Series
        Price (or feature) series.  Log-prices are recommended.
    d : float
        Fractional differentiation order.

    Returns
    -------
    pd.Series
        Fractionally differentiated series.  First observation is always
        equal to the first value of ``series`` (only one weight).
    """
    if d == 0:
        return series.copy()

    n = len(series)
    weights = _expanding_weights(d, n)
    arr = series.values
    out = np.empty(n)

    for i in range(n):
        w = weights[-(i + 1):]  # most recent (i+1) weights
        out[i] = np.dot(w, arr[: i + 1])

    return pd.Series(out, index=series.index)


# ── Minimum d ──────────────────────────────────────────────────────────────


def min_ffd(
    series: pd.Series,
    d_range: tuple[float, float] = (0.0, 1.0),
    step: float = 0.1,
    p_threshold: float = 0.05,
    threshold: float = 1e-5,
) -> float:
    """Find the minimum fractional d that makes a series stationary.

    Searches over a grid of d values and returns the smallest d for which
    the ADF test rejects the unit-root null at the specified significance
    level.  This preserves the maximum amount of memory.

    Requires ``statsmodels`` (part of the [ml] extra).

    Reference: López de Prado (2018), Section 5.5, Snippet 5.3.

    Parameters
    ----------
    series : pd.Series
        Price (or feature) series.  Log-prices are recommended.
    d_range : (float, float), default (0.0, 1.0)
        Range of d values to search (inclusive).
    step : float, default 0.1
        Grid step size.
    p_threshold : float, default 0.05
        ADF p-value significance level.
    threshold : float, default 1e-5
        FFD weight truncation threshold.

    Returns
    -------
    float
        Smallest d in ``d_range`` (on the grid) for which the differenced
        series is stationary.  Returns ``d_range[1]`` if no d achieves
        stationarity.
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError as e:
        raise ImportError(
            "min_ffd requires statsmodels.  Install with: pip install statsmodels"
        ) from e

    d_values = np.arange(d_range[0], d_range[1] + step / 2, step)
    for d in d_values:
        fd = fracdiff_ffd(series, d, threshold=threshold).dropna()
        if len(fd) < 20:
            continue
        result = adfuller(fd, maxlag=1, regression="c", autolag=None)
        p_value = result[1]
        if p_value < p_threshold:
            return float(round(d, 10))

    return float(d_range[1])
