"""High-Frequency Finance (HFF) library.

Implements the mathematical functions described in:

    Dacorogna, M.M., Gençay, R., Müller, U.A., Olsen, R.B., and Pictet, O.V. (2001).
    "An Introduction to High-Frequency Finance."
    Academic Press, San Diego.

Modules
-------
data            Tick data utilities: cleaning, interpolation, regular sampling (Ch. 2)
returns         Log-return series and distributional statistics (Ch. 3)
time_transform  Theta time, intraday/weekly seasonality (Ch. 4)
volatility      Realized vol, OHLC estimators, HARCH model (Ch. 5–6)
ewma            Exponentially weighted moving averages at single and multiple horizons (Ch. 5)
microstructure  Bid-ask spread, price impact, Roll model (Ch. 2)
dc              Directional-change event framework and scaling law (Ch. 3)
"""

from .data import filter_large_spread, filter_outliers, interpolate_linear, midpoint, sample_regular
from .dc import coastline, dc_events, dc_statistics, scaling_law
from .ewma import ewma, ewma_covariance, ewma_from_tau, ewma_std, iterative_ewma, multi_ewma
from .microstructure import (
    amihud_illiquidity, effective_spread, quote_imbalance, quoted_spread,
    relative_spread, roll_spread, tick_frequency,
)
from .returns import (
    aggregate_returns, excess_kurtosis, hill_tail_index, log_returns,
    return_autocorrelation, volatility_clustering,
)
from .time_transform import deseasonalize, intraday_activity, seasonal_factor, theta_time, weekday_activity
from .volatility import (
    ewma_volatility, garman_klass, harch_variance, parkinson,
    realized_volatility, rogers_satchell, rolling_realized_volatility,
)

__all__ = [
    "midpoint", "interpolate_linear", "filter_large_spread", "filter_outliers", "sample_regular",
    "log_returns", "aggregate_returns", "return_autocorrelation", "volatility_clustering",
    "excess_kurtosis", "hill_tail_index",
    "intraday_activity", "weekday_activity", "seasonal_factor", "deseasonalize", "theta_time",
    "realized_volatility", "rolling_realized_volatility", "ewma_volatility",
    "parkinson", "garman_klass", "rogers_satchell", "harch_variance",
    "ewma", "ewma_from_tau", "multi_ewma", "ewma_std", "ewma_covariance", "iterative_ewma",
    "quoted_spread", "relative_spread", "effective_spread", "roll_spread",
    "amihud_illiquidity", "tick_frequency", "quote_imbalance",
    "dc_events", "dc_statistics", "coastline", "scaling_law",
]
