"""Structural break tests for detecting regime changes.

Reference: López de Prado (2018), Chapter 17: "Structural Breaks"

Financial markets undergo structural breaks: sudden, persistent changes in
the data-generating process.  Standard ADF unit-root tests assume the null
of a random walk against a stationary alternative over the full sample.
These tests have low power when the alternative is a structurally broken
process.

Three tests are implemented:

1. **CUSUM structural break** (Brown-Durbin-Evans): detects whether the
   cumulative sum of recursive residuals exceeds a boundary, indicating
   a break in the regression coefficients.

2. **Chow-type ADF**: tests for a structural break in the ADF regression
   at each candidate break point, taking the supremum as the test statistic
   (Sup-Chow).

3. **Supremum ADF (SADF)**: runs an expanding ADF test over sub-samples,
   taking the supremum.  When log-prices are explosive (bubble), SADF > 0.
   PSY (2015) extension included.

Reference: López de Prado (2018), Sections 17.1–17.4, Snippets 17.1–17.4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── CUSUM structural break ───────────────────────────────────────────────────


def cusum_structural_break(
    y: pd.Series,
    conf: float = 0.95,
) -> pd.DataFrame:
    """CUSUM test for structural breaks in a time series.

    Fits a recursive OLS regression (y_t = a + b*t) and computes cumulative
    recursive residuals (CUSUM statistic).  A break is detected when the
    CUSUM crosses the confidence boundary.

    Reference: Brown, Durbin, and Evans (1975); López de Prado (2018),
    Section 17.2.

    Parameters
    ----------
    y : pd.Series
        Time series to test.
    conf : float, default 0.95
        Confidence level for the critical boundary (0.90, 0.95, or 0.99).

    Returns
    -------
    pd.DataFrame
        Columns: [cusum, upper_bound, lower_bound, break_detected].
        Indexed by the same timestamps as ``y``.
    """
    # Critical value multipliers for Brown-Durbin-Evans boundary
    # c(α): 0.90→0.850, 0.95→1.143, 0.99→1.628
    c_map = {0.90: 0.850, 0.95: 1.143, 0.99: 1.628}
    c = c_map.get(conf, 1.143)

    n = len(y)
    vals = y.values.astype(float)

    # Recursive residuals via sequential OLS
    cusum = np.zeros(n)
    residuals = np.zeros(n)

    for t in range(2, n):
        x = np.column_stack([np.ones(t), np.arange(t)])
        beta = np.linalg.lstsq(x, vals[:t], rcond=None)[0]
        fitted = np.array([1.0, t]) @ beta
        residuals[t] = vals[t] - fitted if t < n else 0.0

    sigma = np.std(residuals[2:], ddof=1) if n > 2 else 1.0
    if sigma == 0:
        sigma = 1.0

    cum_res = np.cumsum(residuals) / sigma
    cusum = cum_res

    # Boundary: c * sqrt(n) * (2k/n - 1) where k is the time index
    k = np.arange(n)
    upper = c * np.sqrt(n) * (2 * k / max(n - 1, 1) - 1 + 1)
    lower = -upper

    break_detected = (np.abs(cusum) > upper)

    return pd.DataFrame(
        {
            "cusum":           cusum,
            "upper_bound":     upper,
            "lower_bound":     lower,
            "break_detected":  break_detected,
        },
        index=y.index,
    )


# ── Chow-type ADF ────────────────────────────────────────────────────────────


def chow_type_adf(
    y: pd.Series,
    min_length: int = 20,
) -> pd.DataFrame:
    """Chow-type ADF test for structural breaks in the ADF regression.

    For each candidate break point τ, fits the ADF regression on the
    sub-sample [0, τ] and records the t-statistic on the lagged level term.
    The supremum of these statistics is the Sup-Chow-ADF test statistic.

    Reference: López de Prado (2018), Section 17.3.

    Parameters
    ----------
    y : pd.Series
        Log-price series.
    min_length : int, default 20
        Minimum sub-sample length required to fit the ADF regression.

    Returns
    -------
    pd.DataFrame
        Column ``adf_stat``: t-statistic for each sub-sample endpoint τ.
        First ``min_length`` entries are NaN.
    """
    vals = np.log(y.values.astype(float)) if (y > 0).all() else y.values.astype(float)
    n = len(vals)
    stats = np.full(n, np.nan)

    for t in range(min_length, n):
        sub = vals[:t + 1]
        dy = np.diff(sub)
        y_lag = sub[:-1]
        if len(dy) < 3:
            continue
        # OLS: dy = a + b * y_lag
        X = np.column_stack([np.ones(len(y_lag)), y_lag])
        try:
            beta, resid, _, _ = np.linalg.lstsq(X, dy, rcond=None)
        except np.linalg.LinAlgError:
            continue

        fitted = X @ beta
        e = dy - fitted
        s2 = np.dot(e, e) / max(len(e) - 2, 1)
        XtX_inv = np.linalg.pinv(X.T @ X)
        se_b = np.sqrt(s2 * XtX_inv[1, 1])
        stats[t] = beta[1] / se_b if se_b > 0 else np.nan

    return pd.DataFrame({"adf_stat": stats}, index=y.index)


# ── Supremum ADF (SADF) ──────────────────────────────────────────────────────


def sadf(
    y: pd.Series,
    min_length: int = 20,
    log_prices: bool = True,
) -> pd.DataFrame:
    """Supremum Augmented Dickey-Fuller (SADF) test for explosive sub-periods.

    For each endpoint t, runs ADF regressions over all sub-samples [τ, t]
    with τ ∈ {0, ..., t - min_length}.  Records the supremum ADF statistic
    at each t.  A large positive SADF indicates an explosive (bubble) regime.

    Reference: Phillips, Shi, and Yu (2015); López de Prado (2018),
    Section 17.4, Snippet 17.4.

    Parameters
    ----------
    y : pd.Series
        Price series.  Log transformation applied if ``log_prices=True``
        and all values are positive.
    min_length : int, default 20
        Minimum sub-sample length for the ADF regression.
    log_prices : bool, default True
        Whether to log-transform prices before testing.

    Returns
    -------
    pd.DataFrame
        Column ``sadf``: supremum ADF statistic at each time point.
        First ``min_length`` entries are NaN.
    """
    if log_prices and (y > 0).all():
        vals = np.log(y.values.astype(float))
    else:
        vals = y.values.astype(float)

    n = len(vals)
    sadf_series = np.full(n, np.nan)

    def _adf_stat(sub: np.ndarray) -> float:
        dy = np.diff(sub)
        y_lag = sub[:-1]
        if len(dy) < 3:
            return np.nan
        X = np.column_stack([np.ones(len(y_lag)), y_lag])
        try:
            beta, _, _, _ = np.linalg.lstsq(X, dy, rcond=None)
        except np.linalg.LinAlgError:
            return np.nan
        fitted = X @ beta
        e = dy - fitted
        s2 = np.dot(e, e) / max(len(e) - 2, 1)
        XtX_inv = np.linalg.pinv(X.T @ X)
        se_b = np.sqrt(s2 * XtX_inv[1, 1]) if s2 > 0 else 0.0
        return float(beta[1] / se_b) if se_b > 0 else np.nan

    for t in range(min_length, n):
        bsadf = -np.inf
        for tau in range(0, t - min_length + 1):
            sub = vals[tau: t + 1]
            stat = _adf_stat(sub)
            if not np.isnan(stat):
                bsadf = max(bsadf, stat)
        sadf_series[t] = bsadf if bsadf > -np.inf else np.nan

    return pd.DataFrame({"sadf": sadf_series}, index=y.index)
