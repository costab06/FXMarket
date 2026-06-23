"""Triple-barrier labeling, fixed-horizon labels, and meta-labeling.

Reference: López de Prado (2018), Chapter 3: "Labeling"

The central problem in financial ML is that standard cross-sectional or
time-series labels (e.g., "return over next 5 days > 0 → BUY") ignore the
path taken by prices between entry and exit.  The triple-barrier method
solves this by defining a label as the first of three barriers touched:

    - Profit-taking (upper horizontal barrier)
    - Stop-loss    (lower horizontal barrier)
    - Time limit   (vertical barrier — maximum holding period)

This produces labels that reflect realistic trade outcomes.  Meta-labeling
then adds a second layer: given a primary model's side prediction, a
meta-model predicts whether to trade at all (and how large the bet should be).

Reference: López de Prado (2018), Sections 3.1–3.5, Snippets 3.1–3.6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Volatility ─────────────────────────────────────────────────────────────


def daily_vol(close: pd.Series, span: int = 100) -> pd.Series:
    """Estimate daily volatility as an EWMA of daily log-returns.

    Used to set adaptive profit-taking and stop-loss barriers whose width
    scales with current volatility, so barriers are meaningful regardless
    of the volatility regime.

    Reference: López de Prado (2018), Snippet 3.1.

    Parameters
    ----------
    close : pd.Series
        Daily (or intraday) close prices, indexed by timestamp.
    span : int, default 100
        EWMA span (in bars) for the volatility estimate.

    Returns
    -------
    pd.Series
        EWMA volatility of daily log-returns, same index as ``close``.
    """
    ret = np.log(close).diff().dropna()
    vol = ret.ewm(span=span, min_periods=span).std()
    return vol.reindex(close.index).ffill()


# ── Event generation ────────────────────────────────────────────────────────


def get_events(
    prices: pd.Series,
    t_events: pd.DatetimeIndex,
    pt_sl: tuple[float, float],
    trgt: pd.Series,
    t1: pd.Series | None = None,
    side: pd.Series | None = None,
) -> pd.DataFrame:
    """Build an events DataFrame for triple-barrier labeling.

    For each candidate event time in ``t_events``, computes the profit-taking
    and stop-loss price levels based on the target (volatility) width and
    the optional vertical barrier (maximum holding period).

    Reference: López de Prado (2018), Snippet 3.3.

    Parameters
    ----------
    prices : pd.Series
        Price series (e.g. close or mid), indexed by timestamp.
    t_events : pd.DatetimeIndex
        Candidate event timestamps (e.g. from CUSUM filter).
    pt_sl : (float, float)
        (profit_take_multiplier, stop_loss_multiplier).  A value of 0
        disables that barrier.  E.g. (1.0, 1.0) creates symmetric barriers
        at ±1σ from entry.
    trgt : pd.Series
        Target width per event (usually daily_vol * multiplier).
        Indexed by timestamp; must cover all ``t_events``.
    t1 : pd.Series, optional
        Vertical barrier: maps each event time to the latest allowed exit
        time (e.g. entry + 5 trading days).  If None, no vertical barrier.
    side : pd.Series, optional
        Primary model's side prediction (+1 = long, -1 = short) per event.
        Required for meta-labeling.  If None, both barriers are active for
        each event.

    Returns
    -------
    pd.DataFrame
        Columns: [t1, trgt, side].  Indexed by event time.
        ``t1`` is the vertical barrier time (or NaT if none).
        ``trgt`` is the barrier half-width in price terms.
    """
    trgt = trgt.reindex(t_events).dropna()
    if t1 is None:
        t1 = pd.Series(pd.NaT, index=t_events)
    if side is None:
        side = pd.Series(1.0, index=trgt.index)

    events = pd.concat({"t1": t1, "trgt": trgt, "side": side}, axis=1)
    events = events.loc[trgt.index]
    return events


# ── Triple-barrier ───────────────────────────────────────────────────────────


def _apply_triple_barrier(
    prices: pd.Series,
    t0: pd.Timestamp,
    t1: pd.Timestamp | None,
    trgt: float,
    pt_sl: tuple[float, float],
    side: float,
) -> tuple[pd.Timestamp, int]:
    """Find the first barrier touch for a single event.

    Returns (touch_time, label) where label ∈ {-1, 0, +1}.
    Label 0 means the vertical barrier was hit first (no clear signal).
    """
    p0 = prices.loc[t0]
    loc0 = prices.index.searchsorted(t0)

    if t1 is pd.NaT or t1 is None:
        path = prices.iloc[loc0:]
    else:
        path = prices.loc[t0:t1]

    if path.empty:
        return t0, 0

    pt_level = p0 * (1 + side * pt_sl[0] * trgt) if pt_sl[0] > 0 else None
    sl_level = p0 * (1 - side * pt_sl[1] * trgt) if pt_sl[1] > 0 else None

    for t, p in path.items():
        if t == t0:
            continue
        if pt_level is not None and side * (p - pt_level) >= 0:
            return t, 1
        if sl_level is not None and side * (p - sl_level) <= 0:
            return t, -1

    # Vertical barrier hit
    return path.index[-1], 0


def triple_barrier(
    prices: pd.Series,
    events: pd.DataFrame,
    pt_sl: tuple[float, float],
) -> pd.DataFrame:
    """Apply the triple-barrier method to a set of events.

    For each event, scans forward prices to find which barrier (profit-take,
    stop-loss, or vertical) is touched first.

    Reference: López de Prado (2018), Section 3.2, Snippet 3.2.

    Parameters
    ----------
    prices : pd.Series
        Price series indexed by timestamp.
    events : pd.DataFrame
        Output of :func:`get_events`: columns [t1, trgt, side].
    pt_sl : (float, float)
        Barrier multipliers (profit_take, stop_loss).  0 disables a barrier.

    Returns
    -------
    pd.DataFrame
        Columns: [t1_touch, label].  Indexed by event entry time.
        ``t1_touch`` is the actual exit time (first barrier touched).
        ``label`` is −1 (stop-loss), 0 (timeout), or +1 (profit-take).
    """
    results = []
    for t0, row in events.iterrows():
        t1   = row.get("t1", None)
        trgt = row["trgt"]
        side = row.get("side", 1.0)
        touch_time, label = _apply_triple_barrier(prices, t0, t1, trgt, pt_sl, side)
        results.append({"t1_touch": touch_time, "label": label})
    out = pd.DataFrame(results, index=events.index)
    return out


# ── Bin labels ───────────────────────────────────────────────────────────────


def get_bins(
    events: pd.DataFrame,
    prices: pd.Series,
) -> pd.DataFrame:
    """Assign binary or ternary labels based on return at barrier touch.

    Computes the realised return between entry and the exit determined by
    triple-barrier, then maps it to a label:

        +1  if ret > 0  (profit-take hit, or positive return at timeout)
        -1  if ret < 0  (stop-loss hit, or negative return at timeout)
         0  if ret == 0

    When a ``side`` column is present in ``events``, a meta-label is also
    computed: 1 if the trade was profitable given the primary model's side,
    else 0.

    Reference: López de Prado (2018), Snippet 3.5.

    Parameters
    ----------
    events : pd.DataFrame
        Output of :func:`triple_barrier`, with columns [t1_touch, label]
        and optionally [side].
    prices : pd.Series
        Price series indexed by timestamp.

    Returns
    -------
    pd.DataFrame
        Columns: [ret, bin] and, if side present, [meta_bin].
    """
    events = events.dropna(subset=["t1_touch"])
    p_entry = prices.reindex(events.index, method="ffill")
    p_exit  = prices.reindex(events["t1_touch"], method="ffill").values
    ret = pd.Series((p_exit / p_entry.values) - 1, index=events.index)

    out = pd.DataFrame({"ret": ret, "bin": np.sign(ret)})

    if "side" in events.columns:
        # Meta-label: 1 if primary bet was correct, 0 otherwise
        out["meta_bin"] = (np.sign(ret) == events["side"]).astype(int)

    return out


# ── Fixed-horizon labels ─────────────────────────────────────────────────────


def fixed_horizon_labels(
    prices: pd.Series,
    h: int,
    threshold: float = 0.0,
) -> pd.Series:
    """Label each observation by the sign of the h-step-ahead return.

    A simpler alternative to triple-barrier when the holding period is fixed.
    Produces ternary labels: +1 (up), -1 (down), 0 (flat) depending on
    whether the ``threshold`` is exceeded.

    Reference: López de Prado (2018), Section 3.1.

    Parameters
    ----------
    prices : pd.Series
        Price series indexed by timestamp.
    h : int
        Look-ahead horizon in bars.
    threshold : float, default 0.0
        Minimum absolute return to assign a non-zero label.
        Returns between –threshold and +threshold are labelled 0.

    Returns
    -------
    pd.Series
        Integer labels (−1, 0, +1), same index as ``prices`` (last h
        observations are NaN).
    """
    ret = prices.pct_change(h).shift(-h)
    labels = np.where(ret > threshold, 1, np.where(ret < -threshold, -1, 0))
    return pd.Series(labels, index=prices.index, dtype=float).where(ret.notna())


# ── Meta-labeling ────────────────────────────────────────────────────────────


def meta_label(
    events: pd.DataFrame,
    prices: pd.Series,
    primary_forecast: pd.Series,
    pt_sl: tuple[float, float],
) -> pd.DataFrame:
    """Compute meta-labels for a primary model's side predictions.

    The meta-label answers: "Given that the primary model says to go long/short,
    should we trade at all (and implicitly, how much)?"

    The meta-model is trained to predict the binary outcome (was the primary
    model right?), not the direction.  This decouples direction prediction
    (primary model) from bet sizing (meta-model).

    Reference: López de Prado (2018), Section 3.5, Snippet 3.6.

    Parameters
    ----------
    events : pd.DataFrame
        DataFrame with columns [t1, trgt] indexed by event entry time.
    prices : pd.Series
        Price series indexed by timestamp.
    primary_forecast : pd.Series
        Side predictions from the primary model: +1 or -1, same index as events.
    pt_sl : (float, float)
        Barrier multipliers applied in the direction of the primary forecast.

    Returns
    -------
    pd.DataFrame
        Columns: [ret, bin, side].  ``bin`` is 1 if the primary model was
        correct (trade was profitable), 0 otherwise.
    """
    events = events.copy()
    events["side"] = primary_forecast.reindex(events.index)
    tb = triple_barrier(prices, events, pt_sl)
    events["t1_touch"] = tb["t1_touch"]
    return get_bins(events, prices)
