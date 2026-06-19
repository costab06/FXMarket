"""Directional Change (DC) event framework and scaling law.

Reference: Dacorogna et al. (2001), Chapter 3 and the related work of
the Olsen group; see also:

    Guillaume, D.M., Dacorogna, M.M., Davé, R.D., Müller, U.A., Olsen, R.B.,
    and Pictet, O.V. (1997). "From the Bird's Eye to the Microscope: A Survey
    of New Stylised Facts of the Intraday Foreign Exchange Markets." Finance and
    Stochastics, 1(2), 95–129.

The Directional Change (DC) framework provides an *event-based* alternative
to calendar-time sampling.  Instead of observing the price at fixed time
intervals, one observes the price only when a threshold Δ has been crossed
in either direction.

Definitions
-----------
* **Directional Change (DC) event**: a price move of at least Δ% from the
  most recent extreme (local high or low), reversing the prevailing direction.
* **Overshoot (OS)**: the price continuation *after* a DC event, until the
  next DC event is detected.
* **DC run**: the sequence (DC event, overshoot) in a given direction.

Key properties discovered by Dacorogna et al.
----------------------------------------------
1. **Scaling law**: the number of DC events N(Δ) scales as a power law in the
   threshold:  N(Δ) · Δ^β ≈ const, with β ≈ 1 for FX — implying that the
   total "coastline" of the price path is scale-invariant.
2. **DC event statistics**: mean DC duration and mean overshoot obey scaling
   laws in Δ.
3. **Asymmetry**: up-DC and down-DC events may have different statistics,
   reflecting directional biases in the FX market.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class DCEventType(str, Enum):
    UPTURN  = "upturn"   # price reversed upward ≥ Δ from a local low
    DOWNTURN = "downturn"  # price reversed downward ≥ Δ from a local high


@dataclass
class DCEvent:
    """A single directional change event.

    Attributes
    ----------
    event_type : DCEventType
        Whether this is an upturn (bullish reversal) or downturn (bearish).
    event_index : int
        Index in the price series where the DC event was confirmed.
    event_price : float
        Price at the moment of the DC event confirmation.
    extreme_index : int
        Index of the local extreme (low for upturn, high for downturn) that
        triggered this DC event.
    extreme_price : float
        Price at the triggering extreme.
    """
    event_type: DCEventType
    event_index: int
    event_price: float
    extreme_index: int
    extreme_price: float


@dataclass
class DCStatistics:
    """Summary statistics of the DC event decomposition.

    Attributes
    ----------
    n_events : int          Total number of DC events detected.
    n_upturns : int         Number of upward DC events.
    n_downturns : int       Number of downward DC events.
    mean_dc_size : float    Mean price change during DC phases (as fraction).
    mean_os_size : float    Mean price change during overshoot phases (as fraction).
    mean_dc_duration : float  Mean number of bars in DC phases.
    mean_os_duration : float  Mean number of bars in overshoot phases.
    coastline : float       Total absolute price variation summed over all DC+OS moves.
    events : list[DCEvent]  Raw event list.
    """
    n_events: int = 0
    n_upturns: int = 0
    n_downturns: int = 0
    mean_dc_size: float = 0.0
    mean_os_size: float = 0.0
    mean_dc_duration: float = 0.0
    mean_os_duration: float = 0.0
    coastline: float = 0.0
    events: list[DCEvent] = field(default_factory=list)


def dc_events(prices: np.ndarray, threshold: float) -> list[DCEvent]:
    """Identify all directional change (DC) events in a price series.

    Scans the price series with a two-state machine:

        - State BULL: looking for a downward DC (price to drop Δ from the high).
        - State BEAR: looking for an upward DC (price to rise Δ from the low).

    A DC event is recorded when the price has moved by at least ``threshold``
    (as a fraction of the reference extreme) in the direction opposite to the
    current state.

    Reference: Dacorogna et al. (2001), Section 3.5.

    Parameters
    ----------
    prices : np.ndarray, shape (T,)
        Price series (midpoints or close prices).
    threshold : float
        Minimum fractional price change to declare a DC event.
        E.g. 0.001 = 0.1% = 10 pips on EURUSD at ~1.10.

    Returns
    -------
    list[DCEvent]
        Chronologically ordered list of all DC events.

    Examples
    --------
    >>> events = dc_events(mid_prices, threshold=0.001)
    >>> print(f"Found {len(events)} DC events at 0.1% threshold")
    """
    prices = np.asarray(prices, dtype=float)
    if len(prices) < 2:
        return []

    events: list[DCEvent] = []

    # Initialise: determine starting state from first two points
    extreme_idx = 0
    extreme_price = prices[0]
    # Start in BULL state (looking for a downturn)
    in_bull = True

    for i in range(1, len(prices)):
        p = prices[i]
        if in_bull:
            # Update running high
            if p > extreme_price:
                extreme_price = p
                extreme_idx = i
            # Check for DOWNTURN: price dropped Δ from high
            elif (extreme_price - p) / extreme_price >= threshold:
                events.append(DCEvent(
                    event_type=DCEventType.DOWNTURN,
                    event_index=i,
                    event_price=p,
                    extreme_index=extreme_idx,
                    extreme_price=extreme_price,
                ))
                # Reset: now in BEAR state, extreme = current low
                in_bull = False
                extreme_price = p
                extreme_idx = i
        else:
            # Update running low
            if p < extreme_price:
                extreme_price = p
                extreme_idx = i
            # Check for UPTURN: price rose Δ from low
            elif (p - extreme_price) / extreme_price >= threshold:
                events.append(DCEvent(
                    event_type=DCEventType.UPTURN,
                    event_index=i,
                    event_price=p,
                    extreme_index=extreme_idx,
                    extreme_price=extreme_price,
                ))
                in_bull = True
                extreme_price = p
                extreme_idx = i

    return events


def dc_statistics(prices: np.ndarray, threshold: float) -> DCStatistics:
    """Compute summary statistics of the DC event decomposition.

    Reference: Dacorogna et al. (2001), Section 3.5.

    Parameters
    ----------
    prices : np.ndarray, shape (T,)
        Price series.
    threshold : float
        DC threshold (fractional price change).

    Returns
    -------
    DCStatistics
        Populated statistics dataclass.
    """
    prices = np.asarray(prices, dtype=float)
    events = dc_events(prices, threshold)

    if not events:
        return DCStatistics(events=events)

    n_upturns  = sum(1 for e in events if e.event_type == DCEventType.UPTURN)
    n_downturns = sum(1 for e in events if e.event_type == DCEventType.DOWNTURN)

    # DC phase sizes and durations
    dc_sizes: list[float] = []
    dc_durations: list[int] = []
    # Overshoot phase sizes and durations (phase between DC event i and DC event i+1)
    os_sizes: list[float] = []
    os_durations: list[int] = []

    for k, event in enumerate(events):
        # DC phase: from extreme to event confirmation
        dc_size = abs(event.event_price - event.extreme_price) / event.extreme_price
        dc_dur  = event.event_index - event.extreme_index
        dc_sizes.append(dc_size)
        dc_durations.append(dc_dur)

        # Overshoot phase: from DC event to next DC event's extreme
        if k + 1 < len(events):
            next_event = events[k + 1]
            os_size = abs(next_event.extreme_price - event.event_price) / event.event_price
            os_dur  = next_event.extreme_index - event.event_index
            os_sizes.append(os_size)
            os_durations.append(os_dur)

    total_coastline = coastline(prices, threshold)

    return DCStatistics(
        n_events=len(events),
        n_upturns=n_upturns,
        n_downturns=n_downturns,
        mean_dc_size=float(np.mean(dc_sizes)) if dc_sizes else 0.0,
        mean_os_size=float(np.mean(os_sizes)) if os_sizes else 0.0,
        mean_dc_duration=float(np.mean(dc_durations)) if dc_durations else 0.0,
        mean_os_duration=float(np.mean(os_durations)) if os_durations else 0.0,
        coastline=total_coastline,
        events=events,
    )


def coastline(prices: np.ndarray, threshold: float) -> float:
    """Compute the DC coastline — the total absolute price variation.

    The coastline is the sum of all price moves in the DC event decomposition
    (both DC phases and overshoot phases):

        C(Δ) = Σᵢ |Δpᵢ|

    where the sum is over all DC and overshoot segments.

    The key empirical finding of Dacorogna et al. is that the coastline scales
    as a power law of the threshold Δ:

        C(Δ) ≈ K · Δ^{−γ}

    with γ ≈ 1 for FX, indicating scale-invariant (fractal) price paths.

    Reference: Dacorogna et al. (2001), Section 3.5.2.

    Parameters
    ----------
    prices : np.ndarray, shape (T,)
        Price series.
    threshold : float
        DC threshold (fractional).

    Returns
    -------
    float
        Total coastline (sum of absolute fractional DC+OS moves).
    """
    prices = np.asarray(prices, dtype=float)
    events = dc_events(prices, threshold)

    if not events:
        return 0.0

    total = 0.0
    prev_price = prices[0]

    for event in events:
        total += abs(event.extreme_price - prev_price) / prev_price  # overshoot
        total += abs(event.event_price - event.extreme_price) / event.extreme_price  # DC
        prev_price = event.event_price

    return total


def scaling_law(
    prices: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the DC scaling law: N(Δ) · Δ^β ≈ const.

    One of the most important empirical regularities documented in the book is
    that the number of DC events N(Δ) at threshold Δ follows a power law:

        ln N(Δ) ≈ α − β · ln Δ

    Dacorogna et al. estimate β ≈ 1 for major FX pairs, which implies that
    the FX price path is approximately self-similar across time scales.

    Fitting a regression of ln N on ln Δ yields the scaling exponent β and
    can be used to test for self-similarity or detect structural breaks.

    Reference: Dacorogna et al. (2001), Section 3.5.3.

    Parameters
    ----------
    prices : np.ndarray, shape (T,)
        Price series.
    thresholds : np.ndarray, shape (K,)
        Array of DC thresholds to test (e.g. np.geomspace(0.0001, 0.01, 20)).

    Returns
    -------
    thresholds : np.ndarray, shape (K,)
        Input thresholds (returned for convenience).
    n_events : np.ndarray, shape (K,)
        Number of DC events detected at each threshold.

    Notes
    -----
    To estimate the scaling exponent β, fit a linear regression of
    log(n_events) on log(thresholds):

    >>> from numpy.polynomial import polynomial as P
    >>> log_n = np.log(n_events[n_events > 0])
    >>> log_d = np.log(thresholds[n_events > 0])
    >>> beta, alpha = np.polyfit(log_d, log_n, 1)  # slope = −β
    """
    prices = np.asarray(prices, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    n_events = np.array([len(dc_events(prices, d)) for d in thresholds], dtype=float)
    return thresholds, n_events
