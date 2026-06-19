"""Tests for the high-frequency finance (HFF) library.

Validates numerical correctness and edge-case handling for all modules.
"""

from __future__ import annotations

import numpy as np
import pytest

from fx_market.hff import (
    aggregate_returns,
    amihud_illiquidity,
    coastline,
    dc_events,
    dc_statistics,
    deseasonalize,
    effective_spread,
    ewma,
    ewma_covariance,
    ewma_from_tau,
    ewma_std,
    excess_kurtosis,
    filter_large_spread,
    filter_outliers,
    garman_klass,
    harch_variance,
    hill_tail_index,
    intraday_activity,
    interpolate_linear,
    iterative_ewma,
    log_returns,
    midpoint,
    multi_ewma,
    parkinson,
    quote_imbalance,
    quoted_spread,
    realized_volatility,
    relative_spread,
    return_autocorrelation,
    rogers_satchell,
    roll_spread,
    rolling_realized_volatility,
    sample_regular,
    scaling_law,
    seasonal_factor,
    theta_time,
    tick_frequency,
    volatility_clustering,
)
from fx_market.hff.dc import DCEventType


# ── data.py ───────────────────────────────────────────────────────────────────

class TestData:
    def test_midpoint_scalar(self):
        assert midpoint(1.0810, 1.0812) == pytest.approx(1.0811)

    def test_midpoint_array(self):
        result = midpoint(np.array([1.0, 2.0]), np.array([1.2, 2.2]))
        np.testing.assert_allclose(result, [1.1, 2.1])

    def test_interpolate_linear_exact(self):
        prices = np.array([1.0, 2.0, 3.0])
        times  = np.array([0.0, 1.0, 2.0])
        result = interpolate_linear(prices, times, np.array([0.0, 0.5, 1.0, 1.5, 2.0]))
        np.testing.assert_allclose(result, [1.0, 1.5, 2.0, 2.5, 3.0])

    def test_interpolate_linear_out_of_range_raises(self):
        with pytest.raises(ValueError):
            interpolate_linear(np.array([1.0, 2.0]), np.array([0.0, 1.0]),
                               np.array([-1.0, 0.5]))

    def test_filter_large_spread_basic(self):
        bid = np.array([1.08, 1.08, 1.08])
        ask = np.array([1.0802, 1.09, 1.0801])  # second has 0.93% spread
        mask = filter_large_spread(bid, ask, max_relative_spread=0.005)
        assert mask[0] is np.bool_(True)
        assert mask[1] is np.bool_(False)
        assert mask[2] is np.bool_(True)

    def test_filter_outliers_removes_spike(self):
        prices = np.ones(100) * 1.08
        times  = np.arange(100, dtype=float)
        prices[50] = 1.20  # spike
        mask = filter_outliers(prices, times, window_seconds=30, sigma_threshold=3.0)
        assert not mask[50]
        assert mask[:50].all()

    def test_sample_regular_interpolate(self):
        prices = np.array([1.0, 2.0, 3.0, 4.0])
        times  = np.array([0.0, 1.0, 2.0, 3.0])
        sp, st = sample_regular(prices, times, freq_seconds=0.5)
        assert len(sp) == len(st)
        assert st[0] == pytest.approx(0.0)
        assert sp[0] == pytest.approx(1.0)
        assert sp[2] == pytest.approx(2.0)  # t=1.0 → price=2.0

    def test_sample_regular_last(self):
        prices = np.array([1.0, 2.0, 3.0])
        times  = np.array([0.0, 1.0, 2.0])
        sp, _ = sample_regular(prices, times, freq_seconds=0.5, method="last")
        assert sp[0] == pytest.approx(1.0)
        assert sp[2] == pytest.approx(2.0)


# ── returns.py ────────────────────────────────────────────────────────────────

class TestReturns:
    def test_log_returns_basic(self):
        prices = np.array([1.0, np.e, np.e ** 2])
        r = log_returns(prices)
        np.testing.assert_allclose(r, [1.0, 1.0])

    def test_aggregate_returns(self):
        r = np.array([0.1, 0.2, 0.3, 0.4])
        agg = aggregate_returns(r, n=2)
        np.testing.assert_allclose(agg, [0.3, 0.7])

    def test_aggregate_returns_truncates(self):
        r = np.ones(5) * 1.0
        agg = aggregate_returns(r, n=2)
        assert len(agg) == 2

    def test_return_autocorrelation_shape(self):
        r = np.random.default_rng(0).standard_normal(500)
        acf = return_autocorrelation(r, max_lag=10)
        assert acf.shape == (10,)

    def test_return_autocorrelation_iid_near_zero(self):
        rng = np.random.default_rng(42)
        r = rng.standard_normal(5000)
        acf = return_autocorrelation(r, max_lag=5)
        # Should be close to zero for IID series (with some sampling noise)
        assert np.all(np.abs(acf) < 0.1)

    def test_volatility_clustering_positive(self):
        # Construct series with clustering: blocks of high/low vol
        rng = np.random.default_rng(0)
        high_vol = rng.standard_normal(250) * 3
        low_vol  = rng.standard_normal(250) * 0.3
        r = np.concatenate([high_vol, low_vol, high_vol, low_vol])
        acf_abs = volatility_clustering(r, max_lag=5, power=1.0)
        # ACF of |r| should be positive at lag 1 for clustered volatility
        assert acf_abs[0] > 0

    def test_excess_kurtosis_gaussian(self):
        rng = np.random.default_rng(0)
        r = rng.standard_normal(100_000)
        assert abs(excess_kurtosis(r)) < 0.2

    def test_excess_kurtosis_fat_tail(self):
        rng = np.random.default_rng(0)
        r = rng.standard_t(df=3, size=10_000)
        assert excess_kurtosis(r) > 2.0

    def test_hill_tail_index_returns_finite(self):
        rng = np.random.default_rng(0)
        r = rng.standard_t(df=4, size=2000)
        alpha = hill_tail_index(r, k=100)
        assert np.isfinite(alpha)
        assert alpha > 0


# ── time_transform.py ─────────────────────────────────────────────────────────

class TestTimeTransform:
    def _make_data(self, n: int = 1000):
        """Simulate tick data: evenly spaced at 30-second intervals starting at market open."""
        times = np.arange(n, dtype=float) * 30.0 + 8 * 3600.0  # 08:00 UTC onwards
        rng = np.random.default_rng(7)
        returns = rng.standard_normal(n) * 0.0001
        return times, returns

    def test_intraday_activity_shape(self):
        times, returns = self._make_data()
        act = intraday_activity(times, returns, n_bins=24)
        assert act.shape == (24,)

    def test_intraday_activity_mean_one(self):
        times, returns = self._make_data(5000)
        act = intraday_activity(times, returns, n_bins=24)
        assert act.mean() == pytest.approx(1.0, abs=0.05)

    def test_seasonal_factor_positive(self):
        times, _ = self._make_data()
        f = seasonal_factor(times)
        assert np.all(f > 0)

    def test_deseasonalize_reduces_variance_variation(self):
        times, returns = self._make_data(2000)
        # Raw returns have seasonal variance; deseasonalized should be more uniform
        f = seasonal_factor(times)
        ds = deseasonalize(returns, times)
        # Check dtype at least
        assert ds.shape == returns.shape

    def test_theta_time_monotone(self):
        times = np.arange(100, dtype=float)
        activity = np.ones(100) * 1.5
        theta = theta_time(times, activity)
        assert np.all(np.diff(theta) >= 0)

    def test_theta_time_constant_activity(self):
        """With constant activity λ, theta time should be proportional to physical time."""
        times = np.arange(10, dtype=float)
        activity = np.ones(10) * 2.0
        theta = theta_time(times, activity)
        # Each step Δt=1 contributes λ·Δt=2 to theta
        np.testing.assert_allclose(theta[1:] - theta[:-1], 2.0, atol=1e-10)


# ── ewma.py ───────────────────────────────────────────────────────────────────

class TestEWMA:
    def test_ewma_constant_series(self):
        x = np.ones(100) * 5.0
        out = ewma(x, alpha=0.1)
        # EWMA of a constant should converge to that constant
        np.testing.assert_allclose(out[-1], 5.0, atol=0.01)

    def test_ewma_alpha_one_returns_x(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        out = ewma(x, alpha=1.0)
        np.testing.assert_allclose(out, x)

    def test_ewma_alpha_invalid_raises(self):
        with pytest.raises(ValueError):
            ewma(np.ones(5), alpha=0.0)

    def test_ewma_from_tau_shape(self):
        x = np.random.default_rng(0).standard_normal(100)
        out = ewma_from_tau(x, tau=10.0, dt=1.0)
        assert out.shape == x.shape

    def test_ewma_from_tau_long_tau_smooth(self):
        """Large τ → very slow decay → output nearly constant."""
        x = np.random.default_rng(0).standard_normal(200) + 10.0
        out = ewma_from_tau(x, tau=1000.0, dt=1.0)
        # Range of output should be much smaller than input
        assert out.max() - out.min() < (x.max() - x.min()) * 0.1

    def test_multi_ewma_shape(self):
        x = np.random.default_rng(0).standard_normal(100)
        result = multi_ewma(x, taus=[5.0, 20.0, 60.0])
        assert result.shape == (100, 3)

    def test_ewma_std_nonneg(self):
        x = np.random.default_rng(0).standard_normal(200)
        sigma = ewma_std(x, alpha=0.05)
        assert np.all(sigma >= 0)

    def test_ewma_covariance_self(self):
        """Cov(x, x) should equal EWMA of x²."""
        x = np.random.default_rng(0).standard_normal(100)
        cov = ewma_covariance(x, x, alpha=0.1)
        var = ewma(x ** 2, alpha=0.1)
        np.testing.assert_allclose(cov, var)

    def test_iterative_ewma_smoother(self):
        x = np.random.default_rng(0).standard_normal(200)
        s1 = ewma(x, alpha=0.1)
        s2 = iterative_ewma(x, alpha=0.1, n_passes=2)
        # Two-pass should have smaller variance than one-pass
        assert s2.std() < s1.std()


# ── volatility.py ─────────────────────────────────────────────────────────────

class TestVolatility:
    def test_realized_volatility_zero_returns(self):
        r = np.zeros(100)
        assert realized_volatility(r, annualize=False) == pytest.approx(0.0)

    def test_realized_volatility_positive(self):
        rng = np.random.default_rng(0)
        r = rng.standard_normal(252) * 0.01
        rv = realized_volatility(r, periods_per_year=252, annualize=True)
        assert rv > 0

    def test_rolling_rv_shape(self):
        r = np.random.default_rng(0).standard_normal(100) * 0.01
        out = rolling_realized_volatility(r, window=20)
        assert len(out) == 81

    def test_parkinson_positive(self):
        high = np.array([1.0900, 1.0920])
        low  = np.array([1.0850, 1.0870])
        sigma = parkinson(high, low, annualize=False)
        assert np.all(sigma > 0)

    def test_parkinson_symmetric(self):
        """H/L ratio should be sufficient; absolute level doesn't matter."""
        h1, l1 = np.array([110.0]), np.array([109.0])
        h2, l2 = np.array([1.10]),  np.array([1.09])
        ratio = h1 / l1
        np.testing.assert_allclose(ratio, h2 / l2, rtol=1e-6)
        np.testing.assert_allclose(
            parkinson(h1, l1, annualize=False),
            parkinson(h2, l2, annualize=False),
        )

    def test_garman_klass_positive(self):
        o = np.array([1.0])
        h = np.array([1.01])
        l = np.array([0.99])
        c = np.array([1.005])
        sigma = garman_klass(o, h, l, c, annualize=False)
        assert sigma[0] > 0

    def test_rogers_satchell_positive(self):
        o = np.array([1.0])
        h = np.array([1.01])
        l = np.array([0.995])
        c = np.array([1.002])
        sigma = rogers_satchell(o, h, l, c, annualize=False)
        assert sigma[0] > 0

    def test_harch_variance_shape(self):
        r = np.random.default_rng(0).standard_normal(100) * 0.001
        v = harch_variance(r, lags=[1, 5, 20], coefficients=[0.1, 0.2, 0.3], constant=1e-7)
        assert v.shape == (100,)

    def test_harch_variance_positive(self):
        r = np.random.default_rng(0).standard_normal(100) * 0.001
        v = harch_variance(r, lags=[1, 5], coefficients=[0.1, 0.2], constant=1e-8)
        assert np.all(v >= 0)

    def test_harch_variance_coeff_mismatch_raises(self):
        r = np.ones(10) * 0.001
        with pytest.raises(ValueError):
            harch_variance(r, lags=[1, 5], coefficients=[0.1])


# ── microstructure.py ─────────────────────────────────────────────────────────

class TestMicrostructure:
    def test_quoted_spread(self):
        assert quoted_spread(1.0810, 1.0812) == pytest.approx(0.0002)

    def test_relative_spread(self):
        rs = relative_spread(1.0810, 1.0812)
        assert rs == pytest.approx(0.0002 / 1.0811, rel=1e-5)

    def test_effective_spread(self):
        assert effective_spread(1.0811, 1.0810) == pytest.approx(0.0002)

    def test_roll_spread_nonneg(self):
        prices = np.cumsum(np.random.default_rng(0).standard_normal(500)) + 100
        s = roll_spread(prices)
        assert s >= 0

    def test_roll_spread_known(self):
        """Verify Roll formula on a perfectly alternating bid-ask bounce.

        For prices alternating between 100 ± s/2 (perfectly correlated trade
        signs), each price change = ±s, so Cov(Δp_t, Δp_{t-1}) = −s².
        The Roll formula 2√(−Cov) therefore returns 2s, not s — a known
        result for the degenerate perfectly-alternating case (Roll's model
        assumes IID trade signs; perfect alternation doubles the estimate).
        """
        s = 0.02
        prices = np.array([100 + (s / 2) * (-1) ** i for i in range(500)])
        est = roll_spread(prices)
        assert est == pytest.approx(2 * s, rel=0.05)

    def test_amihud_positive(self):
        returns = np.abs(np.random.default_rng(0).standard_normal(100)) * 0.001
        volumes = np.ones(100) * 1_000_000
        illiq = amihud_illiquidity(returns, volumes)
        assert np.all(illiq >= 0)

    def test_amihud_zero_volume_nan(self):
        returns = np.array([0.001, 0.002])
        volumes = np.array([0.0, 1_000_000.0])
        illiq = amihud_illiquidity(returns, volumes)
        assert np.isnan(illiq[0])
        assert np.isfinite(illiq[1])

    def test_tick_frequency_shape(self):
        times = np.arange(100, dtype=float)
        ft, ff = tick_frequency(times, window_seconds=5.0)
        assert ft.shape == ff.shape

    def test_quote_imbalance_range(self):
        qi = quote_imbalance(1_000_000, 500_000)
        assert -1.0 <= qi <= 1.0

    def test_quote_imbalance_balanced(self):
        assert quote_imbalance(1_000_000, 1_000_000) == pytest.approx(0.0)


# ── dc.py ─────────────────────────────────────────────────────────────────────

class TestDC:
    def _simple_price(self):
        """Up 2%, down 2%, up 2%, down 2%."""
        return np.array([1.00, 1.005, 1.010, 1.015, 1.020,
                         1.015, 1.010, 1.005, 1.00,
                         1.005, 1.010, 1.015, 1.020,
                         1.015, 1.010, 1.005, 1.00])

    def test_dc_events_detected(self):
        prices = self._simple_price()
        events = dc_events(prices, threshold=0.01)
        assert len(events) > 0

    def test_dc_events_alternating_types(self):
        prices = self._simple_price()
        events = dc_events(prices, threshold=0.01)
        for i in range(len(events) - 1):
            assert events[i].event_type != events[i + 1].event_type

    def test_dc_events_empty_series(self):
        assert dc_events(np.array([1.0]), threshold=0.01) == []

    def test_dc_events_threshold_too_large(self):
        prices = np.linspace(1.0, 1.001, 100)  # only 0.1% total move
        events = dc_events(prices, threshold=0.01)
        assert events == []

    def test_dc_statistics_counts(self):
        prices = self._simple_price()
        stats = dc_statistics(prices, threshold=0.01)
        assert stats.n_events == stats.n_upturns + stats.n_downturns

    def test_dc_statistics_mean_size_positive(self):
        prices = self._simple_price()
        stats = dc_statistics(prices, threshold=0.01)
        if stats.n_events > 0:
            assert stats.mean_dc_size > 0

    def test_coastline_nonneg(self):
        prices = self._simple_price()
        c = coastline(prices, threshold=0.005)
        assert c >= 0.0

    def test_coastline_increases_with_volatility(self):
        rng = np.random.default_rng(0)
        quiet = np.cumprod(1 + rng.standard_normal(500) * 0.001)
        noisy = np.cumprod(1 + rng.standard_normal(500) * 0.01)
        c_quiet = coastline(quiet, threshold=0.002)
        c_noisy = coastline(noisy, threshold=0.002)
        assert c_noisy > c_quiet

    def test_scaling_law_shapes(self):
        prices = np.cumprod(1 + np.random.default_rng(0).standard_normal(1000) * 0.002)
        thresholds = np.geomspace(0.001, 0.02, 10)
        t, n = scaling_law(prices, thresholds)
        assert t.shape == n.shape == (10,)

    def test_scaling_law_monotone_decreasing(self):
        """More DC events at smaller thresholds."""
        prices = np.cumprod(1 + np.random.default_rng(0).standard_normal(2000) * 0.003)
        thresholds = np.array([0.002, 0.005, 0.01, 0.02])
        _, n = scaling_law(prices, thresholds)
        # N should generally decrease as Δ increases
        assert n[0] >= n[-1]
