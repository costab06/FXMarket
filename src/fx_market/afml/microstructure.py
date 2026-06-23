"""Microstructural features from AFML Chapter 19.

Reference: López de Prado (2018), Chapter 19: "Microstructural Features"

Market microstructure theory links observable quantities — prices, volumes,
trade directions — to the unobservable characteristics of the trader population:
informed vs uninformed, patient vs impatient.  These microstructural features
are powerful predictors of short-term price moves because they proxy for the
information content and toxicity of incoming order flow.

Four estimators are implemented here:

1. **Kyle's Lambda** (Kyle, 1985): regresses price changes on signed order
   flow volume.  The slope λ measures permanent price impact — how much the
   price moves per unit of flow.  High λ → illiquid or informed market.

2. **Hasbrouck's Lambda** (Hasbrouck, 2009): same as Kyle's but uses the
   square root of dollar volume as the flow proxy, which better fits the
   theoretical price impact curve from inventory models.

3. **VPIN** (Easley, de Prado & O'Hara, 2012): Volume-synchronized
   Probability of Informed Trading.  Classifies volume into buy/sell in
   equal-volume buckets and estimates P[informed] as the normalised
   one-sided imbalance.  VPIN → 1 means one-sided (toxic) flow.

4. **Corwin-Schultz spread** (Corwin & Schultz, 2012): estimates the
   bid-ask spread from OHLC data without needing quote data.  Uses the
   relationship between one-day and two-day high-low ranges to disentangle
   the spread component from volatility.

Additionally, rolling entropy measures applied to microstructure data
(tick rule signs, volume series) are provided as features for ML models.

Reference: López de Prado (2018), Sections 19.1–19.5, Snippets 19.1–19.6.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd

from .bars import _tick_rule


# ── Kyle's Lambda ─────────────────────────────────────────────────────────────


def kyle_lambda(
    prices: np.ndarray,
    volumes: np.ndarray,
    window: int | None = None,
) -> float | np.ndarray:
    """Estimate Kyle's (1985) lambda: price impact per unit of signed flow.

    Kyle (1985) showed that in equilibrium, informed traders optimally submit
    orders proportional to their private signal, and the market maker sets
    prices according to:

        Δp_t = λ · (b_t · v_t) + ε_t

    where b_t ∈ {−1, +1} is the trade direction (tick rule) and v_t is the
    traded volume.  λ is estimated by OLS and measures the permanent price
    impact per unit of signed volume.

    Reference: López de Prado (2018), Section 19.4, Snippet 19.1;
               Kyle, A.S. (1985). Econometrica, 53(6), 1315–1335.

    Parameters
    ----------
    prices : np.ndarray, shape (T,)
        Transaction prices in tick order.
    volumes : np.ndarray, shape (T,)
        Traded volume per tick.
    window : int, optional
        Rolling window length.  If None, computes a single value over the
        full sample.  If provided, returns an array of length T with NaN
        for the first ``window − 1`` positions.

    Returns
    -------
    float or np.ndarray
        Kyle's lambda (slope of Δp on signed volume).  Higher = more impact.
    """
    prices  = np.asarray(prices,  dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    b = _tick_rule(prices)
    dp = np.diff(prices)
    x  = (b * volumes)[1:]   # align with Δp: signed flow at tick t

    def _ols_lambda(dy: np.ndarray, dx: np.ndarray) -> float:
        if len(dy) < 2:
            return np.nan
        x_mean = dx.mean()
        cov_xy = np.dot(dx - x_mean, dy - dy.mean())
        var_x  = np.dot(dx - x_mean, dx - x_mean)
        if var_x == 0:
            return np.nan
        return float(cov_xy / var_x)

    if window is None:
        return _ols_lambda(dp, x)

    T = len(prices)
    out = np.full(T, np.nan)
    for i in range(window, T):
        out[i] = _ols_lambda(dp[i - window: i], x[i - window: i])
    return out


# ── Hasbrouck's Lambda ────────────────────────────────────────────────────────


def hasbrouck_lambda(
    prices: np.ndarray,
    volumes: np.ndarray,
    window: int | None = None,
) -> float | np.ndarray:
    """Estimate Hasbrouck's (2009) lambda: price impact per root-dollar flow.

    Hasbrouck (2009) proposed using the square root of dollar volume as the
    order-flow proxy, motivated by the theoretical price impact function from
    inventory and information models:

        Δp_t = λ · b_t · √(p_t · v_t) + ε_t

    The square-root specification produces a better empirical fit than Kyle's
    linear specification, especially for large trades.

    Reference: López de Prado (2018), Section 19.4, Snippet 19.1;
               Hasbrouck, J. (2009). Journal of Finance, 64(3), 1445–1477.

    Parameters
    ----------
    prices : np.ndarray, shape (T,)
        Transaction prices in tick order.
    volumes : np.ndarray, shape (T,)
        Traded volume per tick.
    window : int, optional
        Rolling window length.  None → single estimate over full sample.

    Returns
    -------
    float or np.ndarray
        Hasbrouck's lambda.
    """
    prices  = np.asarray(prices,  dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    b = _tick_rule(prices)
    dp    = np.diff(prices)
    # Use prices[:-1] to align: Δp_t is price change, sqrt dollar is at t
    dollar = prices[:-1] * volumes[:-1]
    x  = b[1:] * np.sqrt(np.maximum(dollar, 0))

    def _ols_lambda(dy: np.ndarray, dx: np.ndarray) -> float:
        if len(dy) < 2:
            return np.nan
        x_mean = dx.mean()
        cov_xy = np.dot(dx - x_mean, dy - dy.mean())
        var_x  = np.dot(dx - x_mean, dx - x_mean)
        if var_x == 0:
            return np.nan
        return float(cov_xy / var_x)

    if window is None:
        return _ols_lambda(dp, x)

    T = len(prices)
    out = np.full(T, np.nan)
    for i in range(window, T):
        out[i] = _ols_lambda(dp[i - window: i], x[i - window: i])
    return out


# ── VPIN ──────────────────────────────────────────────────────────────────────


def vpin(
    prices: np.ndarray,
    volumes: np.ndarray,
    bucket_size: float | None = None,
    n_buckets: int = 50,
    window_buckets: int = 50,
) -> pd.DataFrame:
    """Compute the Volume-synchronized Probability of Informed Trading (VPIN).

    VPIN partitions the total volume into equal-size buckets.  Within each
    bucket τ of size V, the volume is classified into buy (V_b) and sell (V_s)
    components using the tick rule.  The flow imbalance |V_b − V_s| / V is
    the bucket-level toxicity.  VPIN is the rolling mean of this imbalance
    over the most recent ``window_buckets`` buckets:

        VPIN_τ = (1 / n) · Σ_{j=τ−n+1}^{τ} |V_b(j) − V_s(j)| / V

    VPIN ∈ [0, 1]; values approaching 1 indicate high-toxicity (informed)
    order flow, which precedes adverse price moves and liquidity crises.

    Reference: López de Prado (2018), Section 19.5, Snippet 19.3;
               Easley, D., de Prado, M.L., O'Hara, M. (2012). Journal of
               Finance, 67(5), 1705–1746.

    Parameters
    ----------
    prices : np.ndarray, shape (T,)
        Transaction prices.
    volumes : np.ndarray, shape (T,)
        Traded volume per tick.
    bucket_size : float, optional
        Target volume per bucket.  Defaults to ``total_volume / n_buckets``.
    n_buckets : int, default 50
        Number of equal-volume buckets to divide total volume into.
        Used only when ``bucket_size`` is None.
    window_buckets : int, default 50
        Number of recent buckets for the rolling VPIN average.

    Returns
    -------
    pd.DataFrame
        Columns: [bucket, buy_volume, sell_volume, imbalance, vpin].
        Indexed by the cumulative tick index at each bucket boundary.
    """
    prices  = np.asarray(prices,  dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    b = _tick_rule(prices)

    total_volume = volumes.sum()
    if bucket_size is None:
        bucket_size = total_volume / n_buckets

    buckets = []
    cum_vol = 0.0
    buy_vol = 0.0
    sell_vol = 0.0
    bucket_num = 0

    for i in range(len(prices)):
        v = volumes[i]
        if b[i] > 0:
            buy_vol  += v
        else:
            sell_vol += v
        cum_vol += v

        if cum_vol >= bucket_size:
            imbalance = abs(buy_vol - sell_vol) / bucket_size
            buckets.append({
                "bucket":      bucket_num,
                "buy_volume":  buy_vol,
                "sell_volume": sell_vol,
                "imbalance":   imbalance,
                "tick_end":    i,
            })
            cum_vol  = 0.0
            buy_vol  = 0.0
            sell_vol = 0.0
            bucket_num += 1

    if not buckets:
        return pd.DataFrame(columns=["bucket", "buy_volume", "sell_volume", "imbalance", "vpin"])

    df = pd.DataFrame(buckets).set_index("tick_end")
    df["vpin"] = df["imbalance"].rolling(window=window_buckets, min_periods=1).mean()
    return df


# ── Corwin-Schultz spread ─────────────────────────────────────────────────────


def corwin_schultz_spread(
    high: np.ndarray,
    low: np.ndarray,
) -> np.ndarray:
    """Estimate the bid-ask spread from OHLC high-low ranges.

    Corwin & Schultz (2012) exploit the fact that the observed high-low range
    reflects two additive components: fundamental price volatility (which
    scales with the square root of time) and the bid-ask spread (which is
    constant).  Over one day vs two adjacent days:

        E[ln(H_t / L_t)] = σ√Δt + s
        E[ln(max(H_t,H_{t+1}) / min(L_t,L_{t+1}))] = σ√(2Δt) + s

    Combining these allows σ and s to be disentangled.  Concretely:

        β = (ln(H_t/L_t))² + (ln(H_{t+1}/L_{t+1}))²
        γ = (ln(max(H_t,H_{t+1}) / min(L_t,L_{t+1})))²

        α = (√(2β) − √β) / (3 − 2√2)  −  √(γ / (3 − 2√2))

        spread_t = 2(e^α − 1) / (1 + e^α)

    Negative spread estimates (due to sampling noise) are clamped to zero.

    Reference: López de Prado (2018), Section 19.3;
               Corwin, S.A. & Schultz, P. (2012). Journal of Finance,
               67(2), 719–760.

    Parameters
    ----------
    high : np.ndarray, shape (T,)
        Bar high prices.
    low : np.ndarray, shape (T,)
        Bar low prices.

    Returns
    -------
    np.ndarray, shape (T,)
        Spread estimate for each bar (pair-wise, so index t uses bars t and
        t−1).  First element is NaN.
    """
    high = np.asarray(high, dtype=float)
    low  = np.asarray(low,  dtype=float)
    T    = len(high)
    spread = np.full(T, np.nan)

    k = 3.0 - 2.0 * math.sqrt(2.0)   # ≈ 0.1716

    for t in range(1, T):
        # Single-day log-ranges
        ln_hl0 = math.log(high[t - 1] / low[t - 1]) if low[t - 1] > 0 else np.nan
        ln_hl1 = math.log(high[t]     / low[t])     if low[t] > 0     else np.nan

        if np.isnan(ln_hl0) or np.isnan(ln_hl1):
            continue

        beta = ln_hl0 ** 2 + ln_hl1 ** 2

        # Two-day range
        h2 = max(high[t - 1], high[t])
        l2 = min(low[t - 1],  low[t])
        if l2 <= 0:
            continue
        gamma = math.log(h2 / l2) ** 2

        alpha = (math.sqrt(2 * beta) - math.sqrt(beta)) / k - math.sqrt(gamma / k)

        # Spread estimate: 2(e^α − 1) / (1 + e^α) = 2*tanh(α/2)
        s = 2 * (math.exp(alpha) - 1) / (1 + math.exp(alpha))
        spread[t] = max(s, 0.0)   # clamp negative estimates

    return spread


# ── Rolling microstructure entropy ───────────────────────────────────────────


def _encode_series(x: np.ndarray, bins: int = 10) -> list[str]:
    """Encode a numeric array into a list of single-character symbols."""
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    q = np.nanquantile(x, np.linspace(0, 1, bins + 1))
    q = np.unique(q)
    idx = np.digitize(x, q[:-1]) - 1
    return [alphabet[min(i, len(alphabet) - 1)] for i in idx]


def _shannon(symbols: list[str]) -> float:
    n = len(symbols)
    if n == 0:
        return 0.0
    counts = Counter(symbols)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _lempel_ziv(s: str) -> float:
    """LZ76 normalised complexity of a string."""
    n = len(s)
    if n <= 1:
        return 0.0
    i, c = 0, 1
    dictionary: set[str] = {s[0]}
    while i < n - 1:
        for j in range(i + 1, n):
            sub = s[i + 1: j + 1]
            if sub not in dictionary:
                dictionary.add(sub)
                c += 1
                i = j
                break
        else:
            break
    alpha_size = len(set(s))
    if alpha_size <= 1:
        return 0.0
    max_c = n / math.log2(n) if n > 1 else 1.0
    return c / max_c


def tick_entropy(
    prices: np.ndarray,
    window: int = 100,
    method: str = "shannon",
) -> np.ndarray:
    """Compute rolling entropy of the tick rule direction series.

    The tick rule encodes each trade as +1 (uptick) or −1 (downtick).
    The entropy of this binary series measures the unpredictability of trade
    direction — low entropy suggests herding/informed flow, high entropy
    suggests noise trading.

    This is the primary entropy feature described in AFML Ch. 18 when
    applied to microstructural data.

    Reference: López de Prado (2018), Section 18.3.

    Parameters
    ----------
    prices : np.ndarray, shape (T,)
        Transaction prices.
    window : int, default 100
        Rolling window length in ticks.
    method : {"shannon", "lempel_ziv"}
        Entropy estimator.

    Returns
    -------
    np.ndarray, shape (T,)
        Rolling entropy.  First ``window − 1`` elements are NaN.
    """
    prices = np.asarray(prices, dtype=float)
    b = _tick_rule(prices)
    # Encode: +1 → "a", -1 → "b"
    syms = ["a" if bi > 0 else "b" for bi in b]

    T = len(prices)
    out = np.full(T, np.nan)
    for i in range(window - 1, T):
        window_syms = syms[i - window + 1: i + 1]
        if method == "lempel_ziv":
            out[i] = _lempel_ziv("".join(window_syms))
        else:
            out[i] = _shannon(window_syms)

    return out


def volume_entropy(
    volumes: np.ndarray,
    window: int = 100,
    n_bins: int = 5,
    method: str = "shannon",
) -> np.ndarray:
    """Compute rolling entropy of the volume series.

    Quantises the volume series into ``n_bins`` equal-frequency bins and
    computes the rolling Shannon entropy.  Low volume entropy suggests
    stereotyped (possibly algorithmic) trading patterns; high entropy
    suggests diverse participation.

    Reference: López de Prado (2018), Section 18.3.

    Parameters
    ----------
    volumes : np.ndarray, shape (T,)
        Traded volume per tick.
    window : int, default 100
        Rolling window in ticks.
    n_bins : int, default 5
        Number of quantile bins for discretisation.
    method : {"shannon", "lempel_ziv"}
        Entropy estimator.

    Returns
    -------
    np.ndarray, shape (T,)
        Rolling entropy of volume distribution.
    """
    volumes = np.asarray(volumes, dtype=float)
    T = len(volumes)
    out = np.full(T, np.nan)

    for i in range(window - 1, T):
        sub = volumes[i - window + 1: i + 1]
        syms = _encode_series(sub, bins=n_bins)
        if method == "lempel_ziv":
            out[i] = _lempel_ziv("".join(syms))
        else:
            out[i] = _shannon(syms)

    return out
