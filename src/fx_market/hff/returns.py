"""Log-return series and distributional statistics.

Reference: Dacorogna et al. (2001), Chapter 3: "Statistical Properties of
Foreign Exchange Prices"

Chapter 3 establishes the key empirical stylised facts of FX return distributions:

    * Returns are approximately uncorrelated at short lags (efficient markets).
    * Absolute and squared returns show strong positive autocorrelation
      (volatility clustering / ARCH effects).
    * Return distributions have fat tails (leptokurtosis); the tail index α is
      typically 3–5 for FX, rejecting the Gaussian hypothesis.
    * These properties hold across all time scales from minutes to days.
"""

from __future__ import annotations

import numpy as np


def log_returns(prices: np.ndarray) -> np.ndarray:
    """Compute the log-return series from a price series.

    For a price series p₀, p₁, …, p_N the log return at bar i is:

        r_i = ln(p_i) − ln(p_{i−1})

    Log returns are preferred over simple returns in high-frequency finance
    because they are additive over time intervals and symmetric in up/down moves.

    Reference: Dacorogna et al. (2001), Section 3.1, Eq. (3.1).

    Parameters
    ----------
    prices : np.ndarray, shape (N+1,)
        Price series (midpoints or close prices).

    Returns
    -------
    np.ndarray, shape (N,)
        Log returns r₁, …, r_N.

    Examples
    --------
    >>> log_returns(np.array([1.0, 1.01, 1.005]))
    array([ 0.00995033, -0.00497512])
    """
    prices = np.asarray(prices, dtype=float)
    return np.diff(np.log(prices))


def aggregate_returns(returns: np.ndarray, n: int) -> np.ndarray:
    """Aggregate fine-resolution returns into coarser-period returns.

    The n-period log return is simply the sum of n consecutive single-period
    log returns (by the additive property of log returns).  This is used to
    study how distributional properties change across time scales.

    Reference: Dacorogna et al. (2001), Section 3.2, "Multi-Scale Analysis".

    Parameters
    ----------
    returns : np.ndarray, shape (T,)
        High-frequency log returns.
    n : int
        Aggregation period (e.g. n=12 aggregates 5-minute returns into hourly).

    Returns
    -------
    np.ndarray, shape (T // n,)
        Aggregated returns.  Tail observations that do not form a complete
        period are discarded.
    """
    returns = np.asarray(returns, dtype=float)
    T = len(returns) - (len(returns) % n)
    return returns[:T].reshape(-1, n).sum(axis=1)


def return_autocorrelation(returns: np.ndarray, max_lag: int = 20) -> np.ndarray:
    """Compute the autocorrelation function (ACF) of raw returns.

    For efficient markets, raw return autocorrelations should be close to zero
    at all positive lags.  Dacorogna et al. confirm this for FX at horizons from
    minutes to days, with the exception of a small negative autocorrelation at
    the 1-tick lag caused by bid-ask bounce.

    Reference: Dacorogna et al. (2001), Section 3.3.

    Parameters
    ----------
    returns : np.ndarray, shape (T,)
        Log return series.
    max_lag : int, default 20
        Maximum lag to compute.

    Returns
    -------
    np.ndarray, shape (max_lag,)
        ACF at lags 1, 2, …, max_lag.  (Lag 0 is omitted as it is always 1.)
    """
    returns = np.asarray(returns, dtype=float)
    r = returns - returns.mean()
    acf = np.correlate(r, r, mode="full")
    acf = acf[len(acf) // 2:]          # keep lags 0, 1, 2, …
    acf /= acf[0]                       # normalise by variance (lag 0)
    return acf[1 : max_lag + 1]


def volatility_clustering(
    returns: np.ndarray,
    max_lag: int = 20,
    power: float = 1.0,
) -> np.ndarray:
    """Compute the ACF of |r|^power to quantify volatility clustering.

    While raw returns are nearly uncorrelated, their absolute values and
    squares (power=1 and power=2 respectively) exhibit strong positive
    autocorrelation decaying slowly with lag — the signature of volatility
    clustering / ARCH effects.  The ACF of |r| is typically stronger and
    more robust to outliers than the ACF of r².

    Reference: Dacorogna et al. (2001), Section 3.3, Figure 3.x.

    Parameters
    ----------
    returns : np.ndarray, shape (T,)
        Log return series.
    max_lag : int, default 20
        Maximum lag to compute.
    power : float, default 1.0
        Exponent applied to |r| before computing ACF.
        - 1.0  → ACF of absolute returns (robust, recommended)
        - 2.0  → ACF of squared returns (equivalent to ARCH autocorrelation)

    Returns
    -------
    np.ndarray, shape (max_lag,)
        ACF of |r|^power at lags 1, …, max_lag.
    """
    returns = np.asarray(returns, dtype=float)
    x = np.abs(returns) ** power
    x = x - x.mean()
    acf = np.correlate(x, x, mode="full")
    acf = acf[len(acf) // 2:]
    acf /= acf[0]
    return acf[1 : max_lag + 1]


def excess_kurtosis(returns: np.ndarray) -> float:
    """Compute the excess kurtosis of the return distribution.

    The (excess) kurtosis measures the weight of the tails relative to a
    Gaussian distribution (which has kurtosis = 3, excess kurtosis = 0).
    FX returns at any time scale exhibit positive excess kurtosis
    (leptokurtosis), indicating fatter tails than Gaussian — a central
    finding of Chapter 3.

    As the time scale increases (aggregation), the kurtosis decreases toward
    zero, demonstrating the aggregational Gaussianity of returns, but even at
    daily horizons it remains significantly positive.

    Reference: Dacorogna et al. (2001), Section 3.4, Table 3.x.

    Parameters
    ----------
    returns : np.ndarray, shape (T,)
        Log return series.

    Returns
    -------
    float
        Excess kurtosis = E[(r − μ)⁴] / σ⁴ − 3.
        Positive values indicate fatter-than-Gaussian tails.
    """
    returns = np.asarray(returns, dtype=float)
    mu = returns.mean()
    sigma = returns.std()
    if sigma == 0:
        return float("nan")
    return float(np.mean(((returns - mu) / sigma) ** 4) - 3.0)


def hill_tail_index(returns: np.ndarray, k: int | None = None) -> float:
    """Estimate the tail index α using the Hill (1975) estimator.

    The tail index α characterises the heaviness of the distribution tails.
    For a power-law tail P(|r| > x) ∝ x^{−α}:

        - α > 4 → finite kurtosis (still fat-tailed)
        - 2 < α ≤ 4 → infinite kurtosis, finite variance
        - α ≤ 2 → infinite variance (not typical for FX at daily+ horizons)

    Dacorogna et al. find tail indices of approximately 3–4 for FX returns at
    intraday horizons, declining slightly at longer horizons as the distribution
    becomes less fat-tailed.

    The Hill estimator is computed on the upper tail of |r| only:

        α̂ = 1 / (1/k · Σᵢ₌₁ᵏ ln(X_(n−i+1) / X_(n−k)))

    where X_(n−k+1) ≥ … ≥ X_(n) are the k largest absolute returns.

    Reference: Dacorogna et al. (2001), Section 3.4.2; Hill (1975).

    Parameters
    ----------
    returns : np.ndarray, shape (T,)
        Log return series.  The estimator is applied to |r|.
    k : int or None
        Number of upper-order statistics to use.  If None, defaults to
        max(10, int(0.1 * T)) — the upper 10 % of the sample, a common
        rule of thumb.

    Returns
    -------
    float
        Hill estimate of the tail index α.  Larger values mean thinner tails.
    """
    returns = np.asarray(returns, dtype=float)
    x = np.sort(np.abs(returns))         # ascending order
    n = len(x)
    if k is None:
        k = max(10, int(0.10 * n))
    k = min(k, n - 1)
    # Upper k observations: x[n-k], x[n-k+1], …, x[n-1]
    tail = x[n - k:]
    log_ratios = np.log(tail / tail[0])  # ln(X_(n-i+1) / X_(n-k))
    mean_log = log_ratios[1:].mean()     # skip the ratio at threshold (= 0)
    return 1.0 / mean_log if mean_log > 0 else float("nan")
