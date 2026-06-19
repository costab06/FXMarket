"""Exponentially Weighted Moving Averages (EWMA) at single and multiple horizons.

Reference: Dacorogna et al. (2001), Chapter 5: "Volatility"

The EWMA is the workhorse smoothing operator in high-frequency finance.
Dacorogna et al. parameterise the EWMA through a *decay time* τ rather
than a raw smoothing coefficient α so that the parameter has a transparent
physical interpretation (e.g. τ = 1 hour, τ = 1 day).

The relationship between α and τ for observations spaced Δt apart is:

    α = 1 − e^{−Δt/τ}

so that:
    - α → 0  when τ ≫ Δt  (very slow decay, long memory)
    - α → 1  when τ = Δt  (no memory, instantaneous estimate)

The EWMA update equation is:

    x̄ₜ = α xₜ + (1 − α) x̄ₜ₋₁

**Multi-horizon EWMA**

A key result in Chapter 5 is that combining EWMAs at different time scales τ₁ < τ₂ < …
produces a better estimate of volatility than any single-scale EWMA.  This is
the empirical precursor to the HAR-RV model of Corsi (2009).
"""

from __future__ import annotations

import numpy as np


def ewma(x: np.ndarray, alpha: float, initial: float | None = None) -> np.ndarray:
    """Compute the exponentially weighted moving average of a series.

    x̄ₜ = α xₜ + (1 − α) x̄ₜ₋₁

    Reference: Dacorogna et al. (2001), Section 5.2, Eq. (5.1).

    Parameters
    ----------
    x : np.ndarray, shape (T,)
        Input series.
    alpha : float, in (0, 1]
        Smoothing coefficient.  α = 1 gives x̄ = x (no smoothing).
        See ``ewma_from_tau`` to convert from a decay time τ.
    initial : float or None
        Initial value x̄₀.  If None, uses x[0].

    Returns
    -------
    np.ndarray, shape (T,)
        EWMA series.
    """
    x = np.asarray(x, dtype=float)
    if not 0 < alpha <= 1:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    out = np.empty_like(x)
    out[0] = x[0] if initial is None else initial
    for t in range(1, len(x)):
        out[t] = alpha * x[t] + (1.0 - alpha) * out[t - 1]
    return out


def ewma_from_tau(
    x: np.ndarray,
    tau: float,
    dt: float = 1.0,
    initial: float | None = None,
) -> np.ndarray:
    """Compute EWMA using a physical decay time τ instead of a raw α.

    The decay time τ has a transparent interpretation: it is the half-life of
    the exponential kernel measured in the same units as dt.  The corresponding
    α is:

        α = 1 − e^{−Δt/τ}

    Reference: Dacorogna et al. (2001), Section 5.2.

    Parameters
    ----------
    x : np.ndarray, shape (T,)
        Input series.
    tau : float
        Decay time in the same units as ``dt`` (e.g. seconds, minutes, days).
    dt : float, default 1.0
        Time step between consecutive observations in the same units as τ.
    initial : float or None
        Initial value.  If None, uses x[0].

    Returns
    -------
    np.ndarray, shape (T,)
        EWMA series with decay time τ.

    Examples
    --------
    Daily returns, 10-day decay time:

    >>> ewma_from_tau(returns, tau=10, dt=1)
    """
    alpha = 1.0 - np.exp(-dt / tau)
    return ewma(x, alpha, initial=initial)


def multi_ewma(
    x: np.ndarray,
    taus: list[float],
    dt: float = 1.0,
) -> np.ndarray:
    """Compute EWMAs of x at multiple decay times and stack them column-wise.

    The multi-horizon EWMA matrix is used as the input feature set for
    heterogeneous volatility models (Section 5.5).  Each column corresponds
    to a different time scale τ, from short-term (reactive, noisy) to
    long-term (smooth, trend-following).

    Reference: Dacorogna et al. (2001), Section 5.5, "Combination of EWMAs".

    Parameters
    ----------
    x : np.ndarray, shape (T,)
        Input series (typically squared returns or absolute returns).
    taus : list of float
        Decay times in the same units as ``dt``.
    dt : float, default 1.0
        Time step between observations.

    Returns
    -------
    np.ndarray, shape (T, len(taus))
        Matrix of EWMA series; column j corresponds to taus[j].
    """
    x = np.asarray(x, dtype=float)
    return np.column_stack([ewma_from_tau(x, tau, dt) for tau in taus])


def ewma_std(x: np.ndarray, alpha: float) -> np.ndarray:
    """Compute the EWMA standard deviation (volatility) of a zero-mean series.

    Equivalent to the square root of the EWMA variance:

        σ̂²ₜ = α x²ₜ + (1 − α) σ̂²ₜ₋₁

    Reference: Dacorogna et al. (2001), Section 5.2.

    Parameters
    ----------
    x : np.ndarray, shape (T,)
        Zero-mean series (e.g. log returns, demeaned prices).
    alpha : float, in (0, 1]
        Smoothing coefficient.

    Returns
    -------
    np.ndarray, shape (T,)
        EWMA standard deviation series σ̂ₜ.
    """
    variance = ewma(x ** 2, alpha)
    return np.sqrt(variance)


def ewma_covariance(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Compute the EWMA covariance between two series.

    Cov_t(x, y) = α xₜ yₜ + (1 − α) Cov_{t−1}(x, y)

    Useful for computing EWMA correlation and beta between FX pairs.

    Reference: Dacorogna et al. (2001), Section 5.3.

    Parameters
    ----------
    x : np.ndarray, shape (T,)
        First series (e.g. log returns of EURUSD).
    y : np.ndarray, shape (T,)
        Second series (e.g. log returns of GBPUSD).
    alpha : float, in (0, 1]
        Smoothing coefficient.

    Returns
    -------
    np.ndarray, shape (T,)
        EWMA covariance series.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return ewma(x * y, alpha)


def ewma_correlation(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Compute the EWMA correlation between two series.

    ρ̂ₜ(x, y) = Ĉov_t(x, y) / (σ̂ₓ(t) · σ̂_y(t))

    Reference: Dacorogna et al. (2001), Section 5.3.

    Parameters
    ----------
    x : np.ndarray, shape (T,)
        First series.
    y : np.ndarray, shape (T,)
        Second series.
    alpha : float, in (0, 1]
        Smoothing coefficient.

    Returns
    -------
    np.ndarray, shape (T,)
        EWMA correlation series, values in [−1, 1].
    """
    cov_xy = ewma_covariance(x, y, alpha)
    var_x = ewma(x ** 2, alpha)
    var_y = ewma(y ** 2, alpha)
    denom = np.sqrt(np.maximum(var_x * var_y, 1e-30))
    return cov_xy / denom


def iterative_ewma(
    x: np.ndarray,
    alpha: float,
    n_passes: int = 2,
) -> np.ndarray:
    """Apply the EWMA filter repeatedly to produce a smoother estimate.

    Applying the EWMA filter n times is equivalent to a higher-order
    exponential smoothing with a narrower effective kernel.  Dacorogna et al.
    use two-pass (n=2) EWMA for certain long-horizon trend estimates.

    Reference: Dacorogna et al. (2001), Section 5.2.

    Parameters
    ----------
    x : np.ndarray, shape (T,)
        Input series.
    alpha : float, in (0, 1]
        Smoothing coefficient per pass.
    n_passes : int, default 2
        Number of times to apply the EWMA filter sequentially.

    Returns
    -------
    np.ndarray, shape (T,)
        Iteratively smoothed series.
    """
    result = np.asarray(x, dtype=float).copy()
    for _ in range(n_passes):
        result = ewma(result, alpha)
    return result
