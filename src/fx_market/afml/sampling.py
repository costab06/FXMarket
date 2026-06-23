"""Sample uniqueness, weights, and sequential bootstrap.

Reference: López de Prado (2018), Chapter 4: "Sample Weights"

Standard ML assumes IID samples.  Financial time-series violate this: a
bar labelled at t₁ using returns from t₁ to t₂ overlaps with a bar labelled
at t₁+1 using returns from t₁+1 to t₂+1.  This overlap causes:

1. **Data leakage** in cross-validation: test labels depend on training data.
2. **Sample bias**: high-overlap samples receive too much weight in training.

This module implements:

- **CUSUM filter** (Snippet 2.4): event detector that flags structurally
  significant price moves, suitable as t_events input to triple-barrier.
- **Indicator matrix** and **average uniqueness** (Snippet 4.1–4.3): measure
  how much each observation contributes to each label.
- **Sample weights** (Snippet 4.10): down-weight overlapping samples.
- **Sequential bootstrap** (Snippet 4.5): draw samples that reduce overlap.
- **Time-decay weights** (Snippet 4.11): down-weight older observations.

Reference: López de Prado (2018), Chapters 2 and 4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── CUSUM filter ────────────────────────────────────────────────────────────


def cusum_filter(
    prices: pd.Series,
    threshold: float,
) -> pd.DatetimeIndex:
    """Symmetric CUSUM filter for event detection.

    Accumulates positive (S⁺) and negative (S⁻) deviations from a running
    reference level.  When either accumulator exceeds ``threshold``, an event
    is flagged and the accumulators are reset.

    The filter samples adaptively: more events during trending / volatile
    periods, fewer during quiet periods.  This makes it far better than
    calendar-time sampling for producing IID labels.

    Reference: López de Prado (2018), Snippet 2.4.

    Parameters
    ----------
    prices : pd.Series
        Price series (log-prices recommended) indexed by timestamp.
    threshold : float
        CUSUM barrier level (in the same units as ``prices``).

    Returns
    -------
    pd.DatetimeIndex
        Timestamps of detected events.
    """
    events = []
    s_pos, s_neg = 0.0, 0.0
    diff = prices.diff().dropna()

    for t, dp in diff.items():
        s_pos = max(0.0, s_pos + dp)
        s_neg = min(0.0, s_neg + dp)
        if s_pos >= threshold:
            s_pos = 0.0
            events.append(t)
        elif s_neg <= -threshold:
            s_neg = 0.0
            events.append(t)

    return pd.DatetimeIndex(events)


# ── Indicator matrix ────────────────────────────────────────────────────────


def ind_matrix(
    bar_idx: pd.Index,
    t1: pd.Series,
) -> pd.DataFrame:
    """Build the indicator (concurrency) matrix.

    Element [i, j] = 1 if bar i is used to compute label j (i.e. bar i
    falls within [t_j, t1_j]), else 0.

    Reference: López de Prado (2018), Snippet 4.3.

    Parameters
    ----------
    bar_idx : pd.Index
        Timestamps of all bars in the price series.
    t1 : pd.Series
        Vertical barrier times: maps each event start time to its exit time.
        Index = event entry times, values = event exit times.

    Returns
    -------
    pd.DataFrame
        Binary indicator matrix, shape (len(bar_idx), len(t1)).
        Row index: bar timestamps.  Column index: event entry timestamps.
    """
    ind = pd.DataFrame(0, index=bar_idx, columns=t1.index)
    for t0, t_end in t1.items():
        ind.loc[t0:t_end, t0] = 1
    return ind


# ── Average uniqueness ──────────────────────────────────────────────────────


def average_uniqueness(ind: pd.DataFrame) -> pd.Series:
    """Compute the average uniqueness of each label.

    At each bar t, the concurrency c_t is the number of active labels.
    The uniqueness of bar t for label j is 1/c_t (if label j is active).
    The average uniqueness of label j is the mean uniqueness over its
    active bars.

    Reference: López de Prado (2018), Snippet 4.4.

    Parameters
    ----------
    ind : pd.DataFrame
        Indicator matrix from :func:`ind_matrix`.

    Returns
    -------
    pd.Series
        Average uniqueness per label (column).
    """
    concurrency = ind.sum(axis=1)  # c_t for each bar
    uniqueness = ind.div(concurrency, axis=0).replace([np.inf, -np.inf], 0)
    return uniqueness[ind > 0].mean()


# ── Sample weights ──────────────────────────────────────────────────────────


def sample_weights(
    t1: pd.Series,
    bar_idx: pd.Index,
    ret: pd.Series | None = None,
) -> pd.Series:
    """Compute sample weights based on label uniqueness.

    Labels with high overlap (low uniqueness) receive lower weight, reducing
    the effective duplicate information fed to the model.

    Optionally scales by the absolute log-return of each label: larger
    price moves are more informative and receive higher weight.

    Reference: López de Prado (2018), Snippet 4.10.

    Parameters
    ----------
    t1 : pd.Series
        Maps event start times to exit times (vertical barrier or touch time).
    bar_idx : pd.Index
        Timestamps of all bars in the price series.
    ret : pd.Series, optional
        Absolute log-returns per label.  If provided, weights are scaled by
        |ret| / mean(|ret|).

    Returns
    -------
    pd.Series
        Sample weights, indexed by event start time.  Weights sum to
        approximately ``len(t1)`` before any return scaling.
    """
    ind = ind_matrix(bar_idx, t1)
    avg_u = average_uniqueness(ind)
    w = avg_u.abs()

    if ret is not None:
        ret = ret.reindex(t1.index).abs()
        w *= ret / ret.mean()

    w /= w.sum()
    w *= len(w)
    return w


# ── Sequential bootstrap ────────────────────────────────────────────────────


def seq_bootstrap(
    ind: pd.DataFrame,
    n_samples: int | None = None,
    random_state: int | None = None,
) -> list[int]:
    """Draw samples sequentially to reduce label overlap (IID approximation).

    At each step, selects the sample with the highest average uniqueness
    given the samples already drawn.  This is a greedy approximation to
    the optimal IID sub-sample.

    Reference: López de Prado (2018), Snippet 4.5.

    Parameters
    ----------
    ind : pd.DataFrame
        Indicator matrix from :func:`ind_matrix`.
    n_samples : int, optional
        Number of samples to draw.  Defaults to ``len(ind.columns)``
        (same as a standard bootstrap).
    random_state : int, optional
        Random seed for tie-breaking.

    Returns
    -------
    list[int]
        Column indices (integer positions) of the sampled labels.
    """
    rng = np.random.default_rng(random_state)
    if n_samples is None:
        n_samples = ind.shape[1]

    drawn: list[int] = []
    phi = np.zeros(ind.shape[0])  # cumulative concurrency

    for _ in range(n_samples):
        # Recompute uniqueness given current draw
        scores = []
        for j in range(ind.shape[1]):
            col = ind.iloc[:, j].values
            active = col > 0
            if not active.any():
                scores.append(0.0)
                continue
            c = phi[active] + 1  # concurrency if we add this sample
            u = (1.0 / c).mean()
            scores.append(u)

        scores_arr = np.array(scores)
        max_score = scores_arr.max()
        candidates = np.where(scores_arr == max_score)[0]
        pick = int(rng.choice(candidates))
        drawn.append(pick)
        phi += ind.iloc[:, pick].values

    return drawn


# ── Time-decay weights ──────────────────────────────────────────────────────


def time_decay_weights(
    avg_u: pd.Series,
    decay_rate: float = 1.0,
) -> pd.Series:
    """Apply a piecewise-linear time-decay to uniqueness-based weights.

    Recent observations receive weight 1; older observations receive weight
    down to ``1 - decay_rate`` (or ``decay_rate`` if < 0.5, floor at 0).
    When ``decay_rate = 1.0``, the oldest observation gets weight 0.

    Reference: López de Prado (2018), Snippet 4.11.

    Parameters
    ----------
    avg_u : pd.Series
        Average uniqueness weights (output of :func:`average_uniqueness`
        or :func:`sample_weights`), indexed by event time.
    decay_rate : float, default 1.0
        Rate of decay.  1.0 → linear from 0 to 1 over the sample.
        0.0 → equal weights (no decay).

    Returns
    -------
    pd.Series
        Decayed weights, normalized so the mean is 1.
    """
    # Cumulative uniqueness (time proxy)
    cum_u = avg_u.sort_index().cumsum()
    # Piecewise linear decay
    slope = 1.0 - decay_rate
    decay = slope + decay_rate * cum_u / cum_u.iloc[-1]
    decay = decay.clip(lower=0)
    return decay / decay.mean()
