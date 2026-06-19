"""Theta time, intraday seasonality, and weekly activity patterns.

Reference: Dacorogna et al. (2001), Chapter 4: "Seasonal Volatility and
Activity"

A fundamental insight of Chapter 4 is that raw physical clock time is a poor
basis for FX statistical modelling because market activity is highly seasonal:

    * **Intraday pattern** — activity follows a characteristic U-shape within
      the trading day, with peaks at the London open (around 08:00 UTC) and
      the New York open / London–New York overlap (13:00–17:00 UTC), and a
      trough during the Asian/Pacific midday.

    * **Weekly pattern** — Monday morning and Friday afternoon are quiet;
      Tuesday–Thursday are most active.

    * **Theta time (θ-time / business time)** — a monotone time transformation
      ψ: t → θ that stretches quiet periods and compresses active periods so
      that the transformed process has approximately constant activity.  In
      theta time, volatility is stationary and many statistical properties
      are much closer to the theoretical ideal.

The activity function λ(t) is estimated from historical tick frequency or
realized volatility, and theta time is its integral: θ(t) = ∫₀ᵗ λ(s) ds.
"""

from __future__ import annotations

import numpy as np


# ── Intraday seasonality ───────────────────────────────────────────────────────

# Pre-computed average relative activity by hour-of-day (UTC), based on the
# stylised U-shape described in Dacorogna et al.  Values are scaled so that
# the daily average is 1.0.  Users can replace this with empirically estimated
# values from their own data via `intraday_activity`.
_DEFAULT_INTRADAY = np.array([
    0.40, 0.35, 0.35, 0.40,   # 00–03 UTC: quiet Asia-Pacific night
    0.55, 0.70, 0.90, 1.10,   # 04–07 UTC: Tokyo/Sydney active, pre-London
    1.60, 1.70, 1.65, 1.55,   # 08–11 UTC: London open — peak activity
    1.40, 1.50, 1.60, 1.70,   # 12–15 UTC: NY open, London/NY overlap peak
    1.50, 1.30, 1.00, 0.75,   # 16–19 UTC: London close, NY mid-session
    0.60, 0.55, 0.45, 0.40,   # 20–23 UTC: NY close, quiet overnight
])
# Normalise to mean = 1
_DEFAULT_INTRADAY = _DEFAULT_INTRADAY / _DEFAULT_INTRADAY.mean()

# Average relative activity by weekday (0=Monday … 4=Friday).
# Dacorogna et al. observe lower activity on Monday mornings and Friday
# afternoons.
_DEFAULT_WEEKDAY = np.array([0.85, 1.05, 1.10, 1.08, 0.92])
_DEFAULT_WEEKDAY = _DEFAULT_WEEKDAY / _DEFAULT_WEEKDAY.mean()


def intraday_activity(
    times: np.ndarray,
    returns: np.ndarray,
    n_bins: int = 24,
) -> np.ndarray:
    """Estimate the intraday volatility seasonal pattern from data.

    Bins returns by hour-of-day and computes the mean squared return in each
    bin, then normalises so that the daily average is 1.0.  The resulting
    array represents the relative activity level at each hour.

    Reference: Dacorogna et al. (2001), Section 4.2, Figure 4.x.

    Parameters
    ----------
    times : np.ndarray, shape (T,)
        Unix timestamps (seconds) corresponding to each return.
    returns : np.ndarray, shape (T,)
        Log returns at each timestamp.
    n_bins : int, default 24
        Number of intraday bins.  24 → hourly bins.

    Returns
    -------
    np.ndarray, shape (n_bins,)
        Estimated relative activity per bin, normalised to mean = 1.
    """
    times = np.asarray(times, dtype=float)
    returns = np.asarray(returns, dtype=float)

    seconds_per_day = 86_400.0
    bin_width = seconds_per_day / n_bins
    bins = ((times % seconds_per_day) / bin_width).astype(int) % n_bins

    activity = np.zeros(n_bins)
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            activity[b] = np.mean(returns[mask] ** 2)

    activity = np.where(activity == 0, np.nanmean(activity[activity > 0]), activity)
    return activity / activity.mean()


def weekday_activity(
    times: np.ndarray,
    returns: np.ndarray,
) -> np.ndarray:
    """Estimate the day-of-week volatility seasonal pattern from data.

    Reference: Dacorogna et al. (2001), Section 4.3.

    Parameters
    ----------
    times : np.ndarray, shape (T,)
        Unix timestamps (seconds).
    returns : np.ndarray, shape (T,)
        Log returns.

    Returns
    -------
    np.ndarray, shape (5,)
        Relative activity for Monday (0) through Friday (4), mean = 1.
        Weekend entries are not returned (FX markets are closed).
    """
    times = np.asarray(times, dtype=float)
    returns = np.asarray(returns, dtype=float)

    # Unix epoch 0 = Thursday 1 Jan 1970
    # weekday: (floor(t / 86400) + 3) % 7  →  0=Mon … 6=Sun
    weekdays = ((times // 86_400).astype(int) + 3) % 7

    activity = np.zeros(5)
    for d in range(5):
        mask = weekdays == d
        if mask.any():
            activity[d] = np.mean(returns[mask] ** 2)

    activity = np.where(activity == 0, np.nanmean(activity[activity > 0]), activity)
    return activity / activity.mean()


def seasonal_factor(
    times: np.ndarray,
    intraday: np.ndarray | None = None,
    weekday: np.ndarray | None = None,
) -> np.ndarray:
    """Return the combined intraday × weekly seasonal activity factor for each timestamp.

    The seasonal factor σ_s(t) is used to scale returns before volatility
    estimation, and to deseasonalize returns for statistical analysis.

    Reference: Dacorogna et al. (2001), Section 4.4, Eq. (4.x).

    Parameters
    ----------
    times : np.ndarray, shape (T,)
        Unix timestamps (seconds).
    intraday : np.ndarray, shape (24,), optional
        Hourly intraday activity pattern (mean = 1).  If None, uses the
        default stylised U-shape from the book.
    weekday : np.ndarray, shape (5,), optional
        Mon–Fri activity pattern (mean = 1).  If None, uses defaults.

    Returns
    -------
    np.ndarray, shape (T,)
        Seasonal factor at each timestamp.  Values > 1 indicate above-average
        activity; < 1 indicates below-average activity.
    """
    times = np.asarray(times, dtype=float)
    if intraday is None:
        intraday = _DEFAULT_INTRADAY
    if weekday is None:
        weekday = _DEFAULT_WEEKDAY

    intraday = np.asarray(intraday)
    weekday = np.asarray(weekday)

    n_intra = len(intraday)
    seconds_per_day = 86_400.0
    bin_width = seconds_per_day / n_intra
    hour_bins = ((times % seconds_per_day) / bin_width).astype(int) % n_intra

    day_bins = ((times // 86_400).astype(int) + 3) % 7
    day_bins = np.clip(day_bins, 0, 4)  # weekend → treat as Friday

    factor = intraday[hour_bins] * weekday[day_bins]
    return factor


def deseasonalize(
    returns: np.ndarray,
    times: np.ndarray,
    intraday: np.ndarray | None = None,
    weekday: np.ndarray | None = None,
) -> np.ndarray:
    """Remove the seasonal volatility component from a return series.

    Deseasonalized returns are defined as:

        r̃_i = r_i / √f(t_i)

    where f(t_i) is the seasonal factor at time t_i.  The deseasonalized
    series has approximately constant unconditional volatility, making it
    suitable for fitting homogeneous volatility models (GARCH, EWMA, etc.)
    without the seasonal bias that would otherwise contaminate parameter
    estimates.

    Reference: Dacorogna et al. (2001), Section 4.4, Eq. (4.x).

    Parameters
    ----------
    returns : np.ndarray, shape (T,)
        Log returns.
    times : np.ndarray, shape (T,)
        Unix timestamps for each return (seconds).
    intraday : np.ndarray, shape (24,), optional
        Hourly activity pattern.  Defaults to the stylised U-shape.
    weekday : np.ndarray, shape (5,), optional
        Day-of-week activity pattern.  Defaults to the book's estimates.

    Returns
    -------
    np.ndarray, shape (T,)
        Deseasonalized log returns r̃_i = r_i / √f(t_i).
    """
    returns = np.asarray(returns, dtype=float)
    f = seasonal_factor(times, intraday, weekday)
    return returns / np.sqrt(np.maximum(f, 1e-12))


def theta_time(
    times: np.ndarray,
    activity: np.ndarray,
) -> np.ndarray:
    """Transform physical clock time to business time (theta time).

    Theta time θ(t) is defined as the cumulative integral of the activity
    function λ(t):

        θ(t) = ∫₀ᵗ λ(s) ds

    which is approximated here using the trapezoidal rule on the discrete
    time/activity arrays.

    In theta time the transformed return process has approximately constant
    instantaneous volatility (stationarity), and many distributional properties
    improve — in particular, the intraday U-shape disappears.  This makes
    theta time the natural time scale for fitting volatility models.

    Reference: Dacorogna et al. (2001), Section 4.5, Eq. (4.x).

    Parameters
    ----------
    times : np.ndarray, shape (T,)
        Physical timestamps in seconds (monotonically increasing).
    activity : np.ndarray, shape (T,)
        Activity λ(tᵢ) at each timestamp.  Typically the seasonal factor
        returned by ``seasonal_factor``.

    Returns
    -------
    np.ndarray, shape (T,)
        Theta time θ(tᵢ) at each physical timestamp.  Starts at 0 and
        increases monotonically.  Units are "average-activity-seconds".

    Notes
    -----
    The inverse transformation (theta time → physical time) is obtained by
    ``np.interp(theta_values, theta, times)`` on the output arrays.
    """
    times = np.asarray(times, dtype=float)
    activity = np.asarray(activity, dtype=float)
    dt = np.diff(times, prepend=times[0])
    increments = activity * dt
    return np.cumsum(increments)
