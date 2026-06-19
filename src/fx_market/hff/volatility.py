"""Volatility estimators: realized volatility, OHLC estimators, and HARCH.

Reference: Dacorogna et al. (2001), Chapters 5–6: "Characterising Volatility"
and "The HARCH Model"

Volatility in FX is not directly observable; it must be estimated from price
movements.  Chapter 5 surveys four families of estimators:

    1. **Realized volatility** (sum of squared intraday returns) — the model-free
       benchmark that exploits the full resolution of tick data.
    2. **EWMA volatility** — a parametric single-exponential smoother.
    3. **OHLC estimators** (Parkinson, Garman-Klass, Rogers-Satchell) — efficient
       estimators that use only bar-level data (open, high, low, close) but
       achieve much lower variance than close-to-close estimates.
    4. **HARCH (Heterogeneous AutoRegressive Conditional Heteroskedasticity)** —
       the central model of Chapter 6, which captures the "heterogeneous market
       hypothesis": different market participants react to volatility at different
       time scales simultaneously.

Annualisation convention
------------------------
Unless ``annualize=False``, realized volatility is expressed as annualized
standard deviation, following market convention.  For FX, the standard
annualisation factor is √(252 trading days × 1440/Δt_minutes daily bars).
"""

from __future__ import annotations

import numpy as np

from .ewma import ewma


# ── Realized Volatility ────────────────────────────────────────────────────────

def realized_volatility(
    returns: np.ndarray,
    periods_per_year: float = 252.0,
    annualize: bool = True,
) -> float:
    """Compute realized volatility as the square root of the sum of squared returns.

    Realized volatility over a window is:

        RV = √(Σᵢ r²ᵢ)          (raw)
        RV = √(Σᵢ r²ᵢ) × √(N_year / N)   (annualized, where N = number of returns)

    When computed from intraday returns, realized volatility is a nearly
    unbiased estimate of the integrated variance over the window, and its
    accuracy increases as the sampling frequency increases (subject to
    microstructure noise which biases it upward at very high frequencies —
    the "volatility signature plot" phenomenon discussed in Section 5.4).

    Reference: Dacorogna et al. (2001), Section 5.4, Eq. (5.x).

    Parameters
    ----------
    returns : np.ndarray, shape (T,)
        Log return series over the measurement window.
    periods_per_year : float, default 252
        Number of return periods in a year.  For 5-min returns on FX:
        252 × 288 = 72,576.  For daily returns: 252.
    annualize : bool, default True
        If True, scale the result to annualized standard deviation units.

    Returns
    -------
    float
        Realized volatility.  If ``annualize=True``, expressed as annual σ.
    """
    returns = np.asarray(returns, dtype=float)
    rv = np.sqrt(np.sum(returns ** 2))
    if annualize:
        n = len(returns)
        rv *= np.sqrt(periods_per_year / n)
    return float(rv)


def rolling_realized_volatility(
    returns: np.ndarray,
    window: int,
    periods_per_year: float = 252.0,
    annualize: bool = True,
) -> np.ndarray:
    """Compute rolling realized volatility over a sliding window.

    Reference: Dacorogna et al. (2001), Section 5.4.

    Parameters
    ----------
    returns : np.ndarray, shape (T,)
        Log return series.
    window : int
        Number of returns in each rolling window.
    periods_per_year : float, default 252
        Periods per year for annualisation.
    annualize : bool, default True
        Annualize the output.

    Returns
    -------
    np.ndarray, shape (T − window + 1,)
        Rolling realized volatility series.
    """
    returns = np.asarray(returns, dtype=float)
    T = len(returns)
    out = np.full(T - window + 1, np.nan)
    for i in range(T - window + 1):
        out[i] = realized_volatility(
            returns[i : i + window],
            periods_per_year=periods_per_year,
            annualize=annualize,
        )
    return out


def ewma_volatility(
    returns: np.ndarray,
    tau: float,
    dt: float = 1.0,
    annualize: bool = True,
    periods_per_year: float = 252.0,
) -> np.ndarray:
    """Compute EWMA-based conditional volatility.

    The EWMA variance estimate is updated as:

        σ²ₜ = α r²ₜ + (1 − α) σ²ₜ₋₁,   α = 1 − e^{−Δt/τ}

    This is the RiskMetrics-style variance estimator.  In the Dacorogna framework
    the decay time τ is expressed in the same physical units as dt (e.g. days).

    Reference: Dacorogna et al. (2001), Section 5.2.

    Parameters
    ----------
    returns : np.ndarray, shape (T,)
        Log returns.
    tau : float
        EWMA decay time in units of dt.
    dt : float, default 1.0
        Sampling interval (same units as τ).
    annualize : bool, default True
        Convert variance to annualized standard deviation.
    periods_per_year : float, default 252
        Periods per year for annualisation.

    Returns
    -------
    np.ndarray, shape (T,)
        Time series of conditional volatility estimates σ̂ₜ.
    """
    returns = np.asarray(returns, dtype=float)
    alpha = 1.0 - np.exp(-dt / tau)
    var = ewma(returns ** 2, alpha)
    sigma = np.sqrt(var)
    if annualize:
        sigma *= np.sqrt(periods_per_year)
    return sigma


# ── OHLC Estimators ───────────────────────────────────────────────────────────

def parkinson(
    high: np.ndarray,
    low: np.ndarray,
    periods_per_year: float = 252.0,
    annualize: bool = True,
) -> np.ndarray:
    """Parkinson (1980) range-based volatility estimator.

    Uses the high-low range of each bar as a more efficient estimator of
    volatility than close-to-close returns.  The Parkinson estimator is
    5× more efficient than close-to-close under Brownian motion.

    σ²_P = 1/(4 ln 2) · [ln(H/L)]²

    Reference: Dacorogna et al. (2001), Section 5.3.1;
               Parkinson, M. (1980). Journal of Business, 53(1), 61–65.

    Parameters
    ----------
    high, low : np.ndarray, shape (T,)
        Bar high and low prices.
    periods_per_year : float, default 252
        For annualisation.
    annualize : bool, default True
        Annualize the output.

    Returns
    -------
    np.ndarray, shape (T,)
        Parkinson volatility estimate per bar.
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    log_hl = np.log(high / low)
    var = log_hl ** 2 / (4.0 * np.log(2.0))
    sigma = np.sqrt(var)
    if annualize:
        sigma *= np.sqrt(periods_per_year)
    return sigma


def garman_klass(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    periods_per_year: float = 252.0,
    annualize: bool = True,
) -> np.ndarray:
    """Garman-Klass (1980) OHLC volatility estimator.

    Combines the high-low range (Parkinson term) with the open-to-close
    drift term to produce an estimator that is roughly 7.4× more efficient
    than close-to-close.

    σ²_GK = 0.5·[ln(H/L)]² − (2·ln2 − 1)·[ln(C/O)]²

    Reference: Dacorogna et al. (2001), Section 5.3.2;
               Garman, M.B. & Klass, M.J. (1980). Journal of Business, 53(1), 67–78.

    Parameters
    ----------
    open_, high, low, close : np.ndarray, shape (T,)
        Bar OHLC prices.
    periods_per_year : float, default 252
    annualize : bool, default True

    Returns
    -------
    np.ndarray, shape (T,)
        Garman-Klass volatility estimate per bar.
    """
    open_ = np.asarray(open_, dtype=float)
    high  = np.asarray(high,  dtype=float)
    low   = np.asarray(low,   dtype=float)
    close = np.asarray(close, dtype=float)

    log_hl = np.log(high / low)
    log_co = np.log(close / open_)

    var = 0.5 * log_hl ** 2 - (2.0 * np.log(2.0) - 1.0) * log_co ** 2
    sigma = np.sqrt(np.maximum(var, 0.0))
    if annualize:
        sigma *= np.sqrt(periods_per_year)
    return sigma


def rogers_satchell(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    periods_per_year: float = 252.0,
    annualize: bool = True,
) -> np.ndarray:
    """Rogers-Satchell (1991) drift-corrected OHLC volatility estimator.

    Unlike the Garman-Klass estimator, Rogers-Satchell is unbiased in the
    presence of non-zero drift and does not require the assumption that the
    opening price equals the previous closing price.

    σ²_RS = ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O)

    Reference: Dacorogna et al. (2001), Section 5.3.3;
               Rogers, L.C.G. & Satchell, S.E. (1991). Annals of Applied
               Probability, 1(4), 504–513.

    Parameters
    ----------
    open_, high, low, close : np.ndarray, shape (T,)
        Bar OHLC prices.
    periods_per_year : float, default 252
    annualize : bool, default True

    Returns
    -------
    np.ndarray, shape (T,)
        Rogers-Satchell volatility estimate per bar.
    """
    open_ = np.asarray(open_, dtype=float)
    high  = np.asarray(high,  dtype=float)
    low   = np.asarray(low,   dtype=float)
    close = np.asarray(close, dtype=float)

    var = (np.log(high / close) * np.log(high / open_)
           + np.log(low  / close) * np.log(low  / open_))
    sigma = np.sqrt(np.maximum(var, 0.0))
    if annualize:
        sigma *= np.sqrt(periods_per_year)
    return sigma


# ── HARCH Model ───────────────────────────────────────────────────────────────

def harch_variance(
    returns: np.ndarray,
    lags: list[int],
    coefficients: list[float],
    constant: float = 0.0,
) -> np.ndarray:
    """Compute the HARCH conditional variance series.

    The Heterogeneous AutoRegressive Conditional Heteroskedasticity (HARCH)
    model captures the empirical finding that FX volatility is driven
    simultaneously by short-term traders (reacting to recent tick volatility),
    medium-term traders (reacting to hourly/daily volatility), and long-term
    investors (reacting to weekly/monthly volatility).

    The HARCH(n₁, n₂, …, n_k) model is:

        σ²ₜ = c₀ + Σⱼ cⱼ · (r̄ₜ,ⱼ)²

    where r̄ₜ,ⱼ = (1/nⱼ) Σᵢ₌₁ⁿʲ r_{t−i} is the j-th scale mean return,
    i.e. the return over the nⱼ-period horizon ending at t.

    Each component (r̄ₜ,ⱼ)² represents the realized variance contribution
    from participants operating at scale nⱼ.

    Reference: Dacorogna et al. (2001), Chapter 6, Eq. (6.1)–(6.3).

    Parameters
    ----------
    returns : np.ndarray, shape (T,)
        Log return series (high-frequency, e.g. 5-minute).
    lags : list of int
        HARCH scales n₁ < n₂ < … < nₖ (in number of return periods).
        Example: [1, 12, 288] for 5-min / 1-hour / 1-day scales.
    coefficients : list of float
        HARCH coefficients c₁, c₂, …, cₖ (one per lag).  Must have
        len(coefficients) == len(lags).  All should be ≥ 0 and sum < 1
        for stationarity.
    constant : float, default 0.0
        Intercept c₀ (long-run variance floor).

    Returns
    -------
    np.ndarray, shape (T,)
        Conditional variance σ²ₜ at each time step.  The first max(lags)
        observations use whatever history is available.

    Examples
    --------
    Fit HARCH(1, 12, 288) with equal weights:

    >>> sigma2 = harch_variance(r5min, lags=[1, 12, 288],
    ...                          coefficients=[0.1, 0.3, 0.4], constant=1e-7)
    """
    returns = np.asarray(returns, dtype=float)
    T = len(returns)
    if len(coefficients) != len(lags):
        raise ValueError("len(coefficients) must equal len(lags)")

    variance = np.full(T, constant, dtype=float)

    for j, (n, c) in enumerate(zip(lags, coefficients)):
        for t in range(T):
            start = max(0, t - n + 1)
            r_bar = returns[start : t + 1].mean()  # mean return over n periods
            variance[t] += c * r_bar ** 2

    return variance


def heterogeneous_variance(
    returns: np.ndarray,
    horizons: list[int],
) -> np.ndarray:
    """Decompose variance into heterogeneous time-scale components.

    Computes the average realized variance contribution from each time scale
    n ∈ horizons.  This is the variance decomposition at the heart of the
    "heterogeneous market hypothesis":

        σ²ₜ,ₙ = (r_{t,n})²   where  r_{t,n} = Σᵢ₌₀ⁿ⁻¹ r_{t−i}

    i.e. the squared n-period return ending at t.

    Reference: Dacorogna et al. (2001), Section 5.5; Müller et al. (1997).

    Parameters
    ----------
    returns : np.ndarray, shape (T,)
        High-frequency log return series.
    horizons : list of int
        Aggregation horizons in number of return periods.
        E.g. [1, 12, 288, 1440] for 5-min / 1-hr / 1-day / 1-week.

    Returns
    -------
    np.ndarray, shape (T, len(horizons))
        Matrix of squared n-period returns.  Column j corresponds to
        horizons[j].  NaN for t < horizons[j] − 1.
    """
    returns = np.asarray(returns, dtype=float)
    T = len(returns)
    result = np.full((T, len(horizons)), np.nan)

    for j, n in enumerate(horizons):
        for t in range(n - 1, T):
            r_n = returns[t - n + 1 : t + 1].sum()
            result[t, j] = r_n ** 2

    return result
