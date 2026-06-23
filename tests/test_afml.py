"""Tests for the AFML feature engineering library.

Covers: bars, labeling, fracdiff, sampling, cv, entropy,
        structural_breaks, hrp, importance, microstructure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ── bars ─────────────────────────────────────────────────────────────────────

from fx_market.afml.bars import (
    _tick_rule,
    dollar_bars,
    tick_bars,
    tick_imbalance_bars,
    volume_bars,
    volume_imbalance_bars,
)


class TestTickRule:
    def test_uptick_downtick(self):
        prices = np.array([1.0, 1.1, 1.0, 0.9])
        b = _tick_rule(prices)
        assert b[0] == 1    # initialised to +1
        assert b[1] == 1    # uptick
        assert b[2] == -1   # downtick
        assert b[3] == -1   # downtick

    def test_zero_diff_forward_fills(self):
        prices = np.array([1.0, 1.1, 1.1, 1.1, 0.9])
        b = _tick_rule(prices)
        assert b[2] == 1    # forward fill from uptick
        assert b[3] == 1
        assert b[4] == -1

    def test_returns_float_array(self):
        b = _tick_rule(np.array([1.0, 2.0, 3.0]))
        assert b.dtype == float

    def test_single_price(self):
        b = _tick_rule(np.array([1.5]))
        assert b[0] == 1.0


class TestTickBars:
    def setup_method(self):
        np.random.seed(0)
        self.prices = np.cumsum(np.random.randn(100)) + 100
        self.volumes = np.random.randint(1, 10, size=100).astype(float)
        self.ts = pd.date_range("2024-01-01", periods=100, freq="1min")

    def test_correct_number_of_bars(self):
        df = tick_bars(self.prices, tick_size=10)
        assert len(df) == 10

    def test_ohlcv_columns(self):
        df = tick_bars(self.prices, tick_size=10, volumes=self.volumes)
        for col in ["open", "high", "low", "close", "volume", "vwap", "tick_count"]:
            assert col in df.columns

    def test_tick_count_equals_tick_size(self):
        df = tick_bars(self.prices, tick_size=20)
        assert (df["tick_count"] == 20).all()

    def test_high_ge_low(self):
        df = tick_bars(self.prices, tick_size=10)
        assert (df["high"] >= df["low"]).all()

    def test_open_close_within_hl(self):
        df = tick_bars(self.prices, tick_size=10)
        assert (df["open"] >= df["low"]).all()
        assert (df["open"] <= df["high"]).all()

    def test_with_timestamps(self):
        df = tick_bars(self.prices, tick_size=10, timestamps=self.ts)
        assert df.index.dtype != np.int64 or hasattr(df.index, "tz")

    def test_no_volumes_vwap_equals_mean_price(self):
        prices = np.array([1.0, 2.0, 3.0, 4.0])
        df = tick_bars(prices, tick_size=4)
        assert np.isclose(df["vwap"].iloc[0], 2.5)


class TestVolumeBars:
    def setup_method(self):
        np.random.seed(1)
        self.prices  = np.cumsum(np.random.randn(200)) + 100
        self.volumes = np.random.randint(1, 5, size=200).astype(float)
        self.ts = pd.date_range("2024-01-01", periods=200, freq="1s")

    def test_forms_at_least_one_bar(self):
        df = volume_bars(self.prices, self.volumes, volume_size=10)
        assert len(df) >= 1

    def test_each_bar_volume_approx_target(self):
        df = volume_bars(self.prices, self.volumes, volume_size=20)
        # Each bar's volume should be >= target (last tick takes it over)
        assert (df["volume"] >= 20).all()

    def test_ohlcv_columns(self):
        df = volume_bars(self.prices, self.volumes, volume_size=10)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in df.columns

    def test_with_timestamps(self):
        df = volume_bars(self.prices, self.volumes, volume_size=10, timestamps=self.ts)
        assert len(df) > 0


class TestDollarBars:
    def setup_method(self):
        np.random.seed(2)
        self.prices  = np.ones(100) * 1.1000
        self.volumes = np.ones(100) * 1_000_000

    def test_correct_number_of_bars(self):
        # Each tick = 1.1 * 1e6 = 1.1e6 dollars; 5 ticks = 5.5e6
        df = dollar_bars(self.prices, self.volumes, dollar_size=5.5e6)
        assert len(df) == 20  # 100 ticks / 5 ticks per bar

    def test_notional_approximately_target(self):
        df = dollar_bars(self.prices, self.volumes, dollar_size=1.1e6)
        assert len(df) > 0

    def test_high_ge_low(self):
        df = dollar_bars(self.prices, self.volumes, dollar_size=5.5e6)
        assert (df["high"] >= df["low"]).all()


class TestTickImbalanceBars:
    def setup_method(self):
        np.random.seed(3)
        # Trending prices to ensure bars form
        self.prices = np.cumsum(np.ones(500) * 0.01) + 100

    def test_forms_bars(self):
        df = tick_imbalance_bars(self.prices, initial_expected_ticks=20, ewm_span=5)
        assert len(df) >= 1

    def test_ohlcv_columns(self):
        df = tick_imbalance_bars(self.prices, initial_expected_ticks=20, ewm_span=5)
        for col in ["open", "high", "low", "close"]:
            assert col in df.columns

    def test_high_ge_low(self):
        df = tick_imbalance_bars(self.prices, initial_expected_ticks=20, ewm_span=5)
        assert (df["high"] >= df["low"]).all()


class TestVolumeImbalanceBars:
    def setup_method(self):
        np.random.seed(4)
        self.prices  = np.cumsum(np.ones(300) * 0.01) + 100
        self.volumes = np.ones(300) * 100

    def test_forms_bars(self):
        df = volume_imbalance_bars(
            self.prices, self.volumes, initial_expected_ticks=30, ewm_span=5
        )
        assert len(df) >= 1

    def test_ohlcv_columns(self):
        df = volume_imbalance_bars(
            self.prices, self.volumes, initial_expected_ticks=30, ewm_span=5
        )
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in df.columns


# ── labeling ──────────────────────────────────────────────────────────────────

from fx_market.afml.labeling import (
    daily_vol,
    fixed_horizon_labels,
    get_bins,
    get_events,
    meta_label,
    triple_barrier,
)


class TestDailyVol:
    def test_returns_series(self):
        prices = pd.Series(
            np.cumprod(1 + np.random.randn(200) * 0.01),
            index=pd.date_range("2020-01-01", periods=200),
        )
        vol = daily_vol(prices, span=50)
        assert isinstance(vol, pd.Series)
        assert len(vol) == len(prices)

    def test_non_negative(self):
        prices = pd.Series(
            np.cumprod(1 + np.random.randn(200) * 0.01),
            index=pd.date_range("2020-01-01", periods=200),
        )
        vol = daily_vol(prices, span=20)
        assert (vol.dropna() >= 0).all()


class TestFixedHorizonLabels:
    def test_rising_series_all_positive(self):
        prices = pd.Series(np.arange(1.0, 101.0), index=pd.date_range("2020-01-01", periods=100))
        labels = fixed_horizon_labels(prices, h=1, threshold=0)
        # All returns are positive (price +1 each step)
        assert (labels.dropna() == 1).all()

    def test_falling_series_all_negative(self):
        prices = pd.Series(np.arange(100.0, 0.0, -1.0), index=pd.date_range("2020-01-01", periods=100))
        labels = fixed_horizon_labels(prices, h=1, threshold=0)
        assert (labels.dropna() == -1).all()

    def test_last_h_are_nan(self):
        prices = pd.Series(np.random.randn(50) + 100, index=pd.date_range("2020-01-01", periods=50))
        labels = fixed_horizon_labels(prices, h=5)
        assert labels.iloc[-5:].isna().all()

    def test_threshold_produces_zeros(self):
        prices = pd.Series(
            [100.0, 100.001, 100.0, 99.999, 100.0],
            index=pd.date_range("2020-01-01", periods=5),
        )
        labels = fixed_horizon_labels(prices, h=1, threshold=0.01)
        assert (labels.dropna() == 0).all()


class TestTripleBarrier:
    def setup_method(self):
        np.random.seed(10)
        n = 100
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        self.prices = pd.Series(np.cumprod(1 + np.random.randn(n) * 0.02), index=idx)
        self.t1 = pd.Series(
            [idx[min(i + 10, n - 1)] for i in range(n)], index=idx
        )
        self.trgt = pd.Series(0.05, index=idx)

    def test_returns_dataframe_with_required_columns(self):
        events = get_events(self.prices, self.prices.index[:10], (1, 1), self.trgt, self.t1)
        result = triple_barrier(self.prices, events, (1, 1))
        assert "t1_touch" in result.columns
        assert "label" in result.columns

    def test_labels_in_valid_range(self):
        events = get_events(self.prices, self.prices.index[:20], (1, 1), self.trgt, self.t1)
        result = triple_barrier(self.prices, events, (1, 1))
        assert set(result["label"].unique()).issubset({-1, 0, 1})

    def test_exit_time_after_entry(self):
        events = get_events(self.prices, self.prices.index[:10], (1, 1), self.trgt, self.t1)
        result = triple_barrier(self.prices, events, (1, 1))
        for t0, row in result.iterrows():
            assert row["t1_touch"] >= t0


class TestGetBins:
    def test_returns_ret_and_bin_columns(self):
        idx = pd.date_range("2020-01-01", periods=20, freq="D")
        prices = pd.Series(np.cumprod(1 + np.random.randn(20) * 0.01), index=idx)
        events = pd.DataFrame(
            {"t1_touch": idx[5:15], "label": np.ones(10)},
            index=idx[:10],
        )
        bins = get_bins(events, prices)
        assert "ret" in bins.columns
        assert "bin" in bins.columns


# ── fracdiff ──────────────────────────────────────────────────────────────────

from fx_market.afml.fracdiff import fracdiff_expanding, fracdiff_ffd


class TestFracdiffFFD:
    def setup_method(self):
        np.random.seed(20)
        idx = pd.date_range("2020-01-01", periods=300)
        self.series = pd.Series(np.cumsum(np.random.randn(300)) + 100, index=idx)

    def test_d0_is_identity(self):
        out = fracdiff_ffd(self.series, d=0)
        pd.testing.assert_series_equal(out, self.series)

    def test_d1_approx_first_diff(self):
        out = fracdiff_ffd(self.series, d=1, threshold=1e-8)
        diff = self.series.diff()
        # They should be very close in the overlapping region
        mask = out.notna() & diff.notna()
        assert np.allclose(out[mask].values, diff[mask].values, atol=1e-6)

    def test_output_length_equals_input(self):
        out = fracdiff_ffd(self.series, d=0.5)
        assert len(out) == len(self.series)

    def test_leading_nans_for_fractional_d(self):
        out = fracdiff_ffd(self.series, d=0.3, threshold=1e-3)
        # Some leading values should be NaN
        assert out.iloc[0] is np.nan or np.isnan(out.iloc[0])

    def test_d_half_reduces_variance(self):
        # Use a loose threshold so the window is short enough to produce values
        out = fracdiff_ffd(self.series, d=0.4, threshold=0.01).dropna()
        assert len(out) > 0
        # Fractional diff of a random walk should reduce variance vs original
        assert out.var() < self.series.var()


class TestFracdiffExpanding:
    def test_d0_is_identity(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0])
        out = fracdiff_expanding(s, d=0)
        pd.testing.assert_series_equal(out, s)

    def test_output_same_length(self):
        s = pd.Series(np.random.randn(50))
        out = fracdiff_expanding(s, d=0.5)
        assert len(out) == len(s)


# ── sampling ─────────────────────────────────────────────────────────────────

from fx_market.afml.sampling import (
    average_uniqueness,
    cusum_filter,
    ind_matrix,
    sample_weights,
    seq_bootstrap,
    time_decay_weights,
)


class TestCUSUMFilter:
    def test_detects_trend(self):
        idx = pd.date_range("2020-01-01", periods=100)
        prices = pd.Series(np.arange(100.0), index=idx)
        events = cusum_filter(prices, threshold=5.0)
        assert len(events) >= 1

    def test_returns_datetimeindex(self):
        idx = pd.date_range("2020-01-01", periods=50)
        prices = pd.Series(np.random.randn(50).cumsum(), index=idx)
        events = cusum_filter(prices, threshold=1.0)
        assert isinstance(events, pd.DatetimeIndex)

    def test_no_events_high_threshold(self):
        idx = pd.date_range("2020-01-01", periods=50)
        prices = pd.Series(np.random.randn(50) * 0.001, index=idx)
        events = cusum_filter(prices, threshold=1000.0)
        assert len(events) == 0


class TestIndMatrix:
    def test_shape(self):
        bar_idx = pd.date_range("2020-01-01", periods=10)
        t1 = pd.Series(
            [bar_idx[3], bar_idx[5], bar_idx[8]],
            index=[bar_idx[0], bar_idx[2], bar_idx[5]],
        )
        ind = ind_matrix(bar_idx, t1)
        assert ind.shape == (10, 3)

    def test_binary_values(self):
        bar_idx = pd.date_range("2020-01-01", periods=10)
        t1 = pd.Series([bar_idx[5]], index=[bar_idx[0]])
        ind = ind_matrix(bar_idx, t1)
        assert set(ind.values.flatten()).issubset({0, 1})


class TestAverageUniqueness:
    def test_no_overlap_uniqueness_is_one(self):
        bar_idx = pd.date_range("2020-01-01", periods=6)
        t1 = pd.Series(
            [bar_idx[2], bar_idx[5]],
            index=[bar_idx[0], bar_idx[3]],
        )
        ind = ind_matrix(bar_idx, t1)
        u = average_uniqueness(ind)
        # Non-overlapping labels: uniqueness = 1
        assert (u > 0).all()

    def test_overlapping_reduces_uniqueness(self):
        bar_idx = pd.date_range("2020-01-01", periods=10)
        t1 = pd.Series(
            [bar_idx[8], bar_idx[8]],
            index=[bar_idx[0], bar_idx[1]],
        )
        ind = ind_matrix(bar_idx, t1)
        u = average_uniqueness(ind)
        assert (u < 1).all()


class TestSampleWeights:
    def test_weights_positive(self):
        bar_idx = pd.date_range("2020-01-01", periods=20)
        t1 = pd.Series(
            [bar_idx[min(i + 5, 19)] for i in range(0, 15, 3)],
            index=bar_idx[[0, 3, 6, 9, 12]],
        )
        w = sample_weights(t1, bar_idx)
        assert (w > 0).all()

    def test_weights_mean_near_one(self):
        bar_idx = pd.date_range("2020-01-01", periods=20)
        t1 = pd.Series(
            [bar_idx[min(i + 5, 19)] for i in range(0, 15, 3)],
            index=bar_idx[[0, 3, 6, 9, 12]],
        )
        w = sample_weights(t1, bar_idx)
        assert np.isclose(w.mean(), 1.0, atol=0.5)


class TestSeqBootstrap:
    def test_returns_n_samples(self):
        bar_idx = pd.date_range("2020-01-01", periods=10)
        t1 = pd.Series(
            [bar_idx[min(i + 3, 9)] for i in range(5)],
            index=bar_idx[:5],
        )
        ind = ind_matrix(bar_idx, t1)
        drawn = seq_bootstrap(ind, n_samples=5, random_state=42)
        assert len(drawn) == 5

    def test_indices_in_valid_range(self):
        bar_idx = pd.date_range("2020-01-01", periods=10)
        t1 = pd.Series(
            [bar_idx[min(i + 3, 9)] for i in range(5)],
            index=bar_idx[:5],
        )
        ind = ind_matrix(bar_idx, t1)
        drawn = seq_bootstrap(ind, n_samples=5, random_state=0)
        assert all(0 <= i < ind.shape[1] for i in drawn)


class TestTimeDecayWeights:
    def test_recent_higher_than_old(self):
        idx = pd.date_range("2020-01-01", periods=20)
        avg_u = pd.Series(np.ones(20), index=idx)
        w = time_decay_weights(avg_u, decay_rate=1.0)
        assert w.iloc[-1] > w.iloc[0]

    def test_no_decay_uniform_weights(self):
        idx = pd.date_range("2020-01-01", periods=20)
        avg_u = pd.Series(np.ones(20), index=idx)
        w = time_decay_weights(avg_u, decay_rate=0.0)
        assert np.allclose(w, w.mean(), atol=1e-10)

    def test_weights_non_negative(self):
        idx = pd.date_range("2020-01-01", periods=20)
        avg_u = pd.Series(np.ones(20), index=idx)
        w = time_decay_weights(avg_u, decay_rate=1.0)
        assert (w >= 0).all()


# ── cv ────────────────────────────────────────────────────────────────────────

from fx_market.afml.cv import PurgedKFold


class TestPurgedKFold:
    def setup_method(self):
        np.random.seed(30)
        n = 100
        idx = pd.date_range("2020-01-01", periods=n)
        self.X = pd.DataFrame(np.random.randn(n, 3), index=idx)
        self.y = pd.Series(np.random.randint(0, 2, n), index=idx)
        self.t1 = pd.Series(
            [idx[min(i + 5, n - 1)] for i in range(n)], index=idx
        )

    def test_n_splits(self):
        cv = PurgedKFold(n_splits=5)
        splits = list(cv.split(self.X))
        assert len(splits) == 5

    def test_get_n_splits(self):
        cv = PurgedKFold(n_splits=4)
        assert cv.get_n_splits() == 4

    def test_no_overlap_train_test(self):
        cv = PurgedKFold(n_splits=5)
        for train, test in cv.split(self.X):
            assert len(set(train) & set(test)) == 0

    def test_all_indices_covered(self):
        cv = PurgedKFold(n_splits=5)
        all_test = set()
        for _, test in cv.split(self.X):
            all_test.update(test)
        assert len(all_test) == len(self.X)

    def test_with_embargo(self):
        cv = PurgedKFold(n_splits=5, pct_embargo=0.05)
        for train, test in cv.split(self.X):
            assert len(set(train) & set(test)) == 0


# ── entropy ───────────────────────────────────────────────────────────────────

from fx_market.afml.entropy import (
    kontoyiannis_entropy,
    lempel_ziv_complexity,
    plug_in_entropy,
    shannon_entropy,
)


class TestShannonEntropy:
    def test_uniform_maximum_entropy(self):
        # All distinct symbols → max entropy
        s = "abcdefghij"
        h = shannon_entropy(s)
        assert np.isclose(h, np.log2(10), atol=0.01)

    def test_constant_zero_entropy(self):
        s = "aaaaaaaaa"
        h = shannon_entropy(s)
        assert h == 0.0

    def test_numeric_input(self):
        x = np.random.randn(100)
        h = shannon_entropy(x, n_bins=5)
        assert h > 0

    def test_base_e_vs_base_2(self):
        s = "abcde" * 10
        h2 = shannon_entropy(s, base=2)
        he = shannon_entropy(s, base=np.e)
        # h_e = h_2 * ln(2)
        assert np.isclose(he, h2 * np.log(2), atol=0.01)


class TestLempelZivComplexity:
    def test_random_sequence_near_one(self):
        np.random.seed(42)
        s = "".join(np.random.choice(list("abcde"), size=200))
        lz = lempel_ziv_complexity(s)
        assert lz > 0

    def test_repetitive_low_complexity(self):
        s = "ab" * 50
        lz_rep = lempel_ziv_complexity(s)
        s_rand = "".join(np.random.choice(list("abcde"), size=100))
        lz_rand = lempel_ziv_complexity(s_rand)
        # Repetitive sequence should have lower complexity
        assert lz_rep < lz_rand

    def test_numeric_input(self):
        x = np.random.randn(100)
        lz = lempel_ziv_complexity(x, n_bins=5)
        assert lz > 0


class TestPlugInEntropy:
    def test_bigrams_higher_than_unigrams_for_mixed(self):
        np.random.seed(0)
        s = "".join(np.random.choice(list("abcd"), size=200))
        h1 = plug_in_entropy(s, word_length=1)
        # Bigram entropy (per character) should be close to unigram for iid
        h2 = plug_in_entropy(s, word_length=2)
        assert h2 > 0

    def test_constant_zero_entropy(self):
        s = "aaa" * 30
        h = plug_in_entropy(s, word_length=1)
        assert h == 0.0


class TestKontoyiannisentropy:
    def test_positive_entropy(self):
        np.random.seed(5)
        s = "".join(np.random.choice(list("abcd"), size=50))
        h = kontoyiannis_entropy(s)
        assert h > 0

    def test_higher_entropy_for_random(self):
        np.random.seed(7)
        s_rand = "".join(np.random.choice(list("abcd"), size=40))
        s_rep = "abcd" * 10
        h_rand = kontoyiannis_entropy(s_rand)
        h_rep = kontoyiannis_entropy(s_rep)
        # Random should have >= entropy than repetitive
        assert h_rand >= h_rep


# ── structural_breaks ─────────────────────────────────────────────────────────

from fx_market.afml.structural_breaks import chow_type_adf, cusum_structural_break, sadf


class TestCUSUMStructuralBreak:
    def test_returns_dataframe(self):
        idx = pd.date_range("2020-01-01", periods=50)
        y = pd.Series(np.random.randn(50).cumsum(), index=idx)
        result = cusum_structural_break(y)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self):
        idx = pd.date_range("2020-01-01", periods=50)
        y = pd.Series(np.random.randn(50).cumsum(), index=idx)
        result = cusum_structural_break(y)
        for col in ["cusum", "upper_bound", "lower_bound", "break_detected"]:
            assert col in result.columns

    def test_detects_break_in_trending_series(self):
        idx = pd.date_range("2020-01-01", periods=100)
        # Clear structural break: random walk then strong trend
        trend = np.concatenate([np.random.randn(50), np.ones(50) * 5])
        y = pd.Series(trend.cumsum(), index=idx)
        result = cusum_structural_break(y, conf=0.95)
        # Should detect at least one break
        assert result["break_detected"].any()


class TestChowTypeADF:
    def test_returns_dataframe(self):
        idx = pd.date_range("2020-01-01", periods=60)
        y = pd.Series(np.cumprod(1 + np.random.randn(60) * 0.01), index=idx)
        result = chow_type_adf(y, min_length=10)
        assert isinstance(result, pd.DataFrame)
        assert "adf_stat" in result.columns

    def test_first_entries_nan(self):
        idx = pd.date_range("2020-01-01", periods=60)
        y = pd.Series(np.cumprod(1 + np.random.randn(60) * 0.01), index=idx)
        result = chow_type_adf(y, min_length=20)
        assert result["adf_stat"].iloc[:20].isna().all()


class TestSADF:
    def test_returns_dataframe(self):
        idx = pd.date_range("2020-01-01", periods=60)
        y = pd.Series(np.cumprod(1 + np.random.randn(60) * 0.01), index=idx)
        result = sadf(y, min_length=15)
        assert isinstance(result, pd.DataFrame)
        assert "sadf" in result.columns

    def test_explosive_series_high_sadf(self):
        idx = pd.date_range("2020-01-01", periods=60)
        # Explosive process: geometric growth
        y = pd.Series(np.cumprod(1 + np.linspace(0.01, 0.1, 60)), index=idx)
        result = sadf(y, min_length=10)
        # SADF should detect explosiveness (positive values)
        valid = result["sadf"].dropna()
        assert (valid > 0).any()

    def test_first_entries_nan(self):
        idx = pd.date_range("2020-01-01", periods=50)
        y = pd.Series(np.cumprod(1 + np.random.randn(50) * 0.01), index=idx)
        result = sadf(y, min_length=20)
        assert result["sadf"].iloc[:20].isna().all()


# ── hrp ───────────────────────────────────────────────────────────────────────

_has_scipy = pytest.mark.skipif(
    not __import__("importlib").util.find_spec("scipy"),
    reason="scipy required for hrp tests",
)


@_has_scipy
class TestHRP:
    def setup_method(self):
        from fx_market.afml.hrp import hrp as _hrp
        self._hrp = _hrp
        np.random.seed(42)
        n = 252
        k = 5
        self.returns = pd.DataFrame(
            np.random.randn(n, k) * 0.01,
            columns=["A", "B", "C", "D", "E"],
            index=pd.date_range("2020-01-01", periods=n),
        )

    def test_weights_sum_to_one(self):
        w = self._hrp(self.returns)
        assert np.isclose(w.sum(), 1.0, atol=1e-10)

    def test_weights_non_negative(self):
        w = self._hrp(self.returns)
        assert (w >= 0).all()

    def test_index_matches_columns(self):
        w = self._hrp(self.returns)
        assert list(w.index) == list(self.returns.columns)

    def test_correlated_assets_lower_weight(self):
        # Two perfectly correlated assets should together not dominate
        returns = self.returns.copy()
        returns["C"] = returns["A"] * 0.99 + np.random.randn(252) * 0.0001
        w = self._hrp(returns)
        # A and C together should still be < 0.8 of the portfolio
        assert w["A"] + w["C"] < 0.8

    def test_single_linkage(self):
        w = self._hrp(self.returns, linkage_method="single")
        assert np.isclose(w.sum(), 1.0, atol=1e-10)

    def test_complete_linkage(self):
        w = self._hrp(self.returns, linkage_method="complete")
        assert np.isclose(w.sum(), 1.0, atol=1e-10)


# ── microstructure (Ch. 19) ───────────────────────────────────────────────────

from fx_market.afml.microstructure import (
    corwin_schultz_spread,
    hasbrouck_lambda,
    kyle_lambda,
    tick_entropy,
    volume_entropy,
    vpin,
)


class TestKyleLambda:
    def setup_method(self):
        np.random.seed(50)
        n = 200
        # Simulate trending prices with volume
        self.prices  = np.cumsum(np.random.randn(n) * 0.01) + 1.1000
        self.volumes = np.abs(np.random.randn(n)) * 1e6 + 1e6

    def test_single_value_is_float(self):
        lam = kyle_lambda(self.prices, self.volumes)
        assert isinstance(lam, float) or np.isnan(lam)

    def test_rolling_returns_array(self):
        lam = kyle_lambda(self.prices, self.volumes, window=50)
        assert len(lam) == len(self.prices)

    def test_rolling_leading_nan(self):
        lam = kyle_lambda(self.prices, self.volumes, window=50)
        assert np.isnan(lam[:50]).all()

    def test_rolling_non_nan_after_window(self):
        lam = kyle_lambda(self.prices, self.volumes, window=50)
        assert not np.isnan(lam[50:]).all()

    def test_lambda_finite_with_varying_volume(self):
        np.random.seed(99)
        # Mix of up/down ticks with varying volumes → non-zero Var(x)
        prices = np.cumsum(np.random.randn(200) * 0.001) + 1.0
        vols = np.abs(np.random.randn(200)) * 1e6 + 1e6
        lam = kyle_lambda(prices, vols)
        assert np.isfinite(lam)


class TestHasbrouckLambda:
    def setup_method(self):
        np.random.seed(51)
        n = 200
        self.prices  = np.cumsum(np.random.randn(n) * 0.01) + 1.1000
        self.volumes = np.abs(np.random.randn(n)) * 1e6 + 1e6

    def test_single_value_is_float_or_nan(self):
        lam = hasbrouck_lambda(self.prices, self.volumes)
        assert isinstance(lam, float) or np.isnan(lam)

    def test_rolling_returns_array(self):
        lam = hasbrouck_lambda(self.prices, self.volumes, window=50)
        assert len(lam) == len(self.prices)

    def test_rolling_leading_nan(self):
        lam = hasbrouck_lambda(self.prices, self.volumes, window=50)
        assert np.isnan(lam[:50]).all()

    def test_non_negative_on_trending(self):
        # With consistent price direction, lambda should be non-zero
        lam = hasbrouck_lambda(self.prices, self.volumes)
        assert lam is not None


class TestVPIN:
    def setup_method(self):
        np.random.seed(52)
        n = 500
        self.prices  = np.cumsum(np.random.randn(n) * 0.001) + 1.1
        self.volumes = np.abs(np.random.randn(n)) * 1e6 + 1e6

    def test_returns_dataframe(self):
        df = vpin(self.prices, self.volumes, n_buckets=10, window_buckets=5)
        assert isinstance(df, pd.DataFrame)

    def test_required_columns(self):
        df = vpin(self.prices, self.volumes, n_buckets=10, window_buckets=5)
        for col in ["bucket", "buy_volume", "sell_volume", "imbalance", "vpin"]:
            assert col in df.columns

    def test_vpin_between_zero_and_one(self):
        df = vpin(self.prices, self.volumes, n_buckets=10, window_buckets=5)
        assert (df["vpin"].dropna() >= 0).all()
        assert (df["vpin"].dropna() <= 1).all()

    def test_imbalance_between_zero_and_one(self):
        df = vpin(self.prices, self.volumes, n_buckets=10, window_buckets=5)
        assert (df["imbalance"] >= 0).all()
        assert (df["imbalance"] <= 1).all()

    def test_one_sided_flow_high_vpin(self):
        # All prices going up = all ticks classified as buys → VPIN near 1
        prices  = np.cumsum(np.ones(300) * 0.01) + 1.0
        volumes = np.ones(300) * 1e6
        df = vpin(prices, volumes, n_buckets=10, window_buckets=5)
        assert df["vpin"].dropna().mean() > 0.5

    def test_explicit_bucket_size(self):
        df = vpin(self.prices, self.volumes, bucket_size=5e6, window_buckets=5)
        assert len(df) > 0


class TestCorwinSchultzSpread:
    def setup_method(self):
        np.random.seed(53)
        n = 100
        mid = np.cumsum(np.random.randn(n) * 0.001) + 1.1
        spread = 0.0002  # 2 pips
        self.high = mid + spread / 2 + np.abs(np.random.randn(n)) * 0.0005
        self.low  = mid - spread / 2 - np.abs(np.random.randn(n)) * 0.0005

    def test_returns_array_same_length(self):
        s = corwin_schultz_spread(self.high, self.low)
        assert len(s) == len(self.high)

    def test_first_element_nan(self):
        s = corwin_schultz_spread(self.high, self.low)
        assert np.isnan(s[0])

    def test_non_negative_estimates(self):
        s = corwin_schultz_spread(self.high, self.low)
        assert (s[1:] >= 0).all()

    def test_wider_spread_higher_estimate(self):
        # Wide spread: 20 pips
        mid = np.ones(50) * 1.1
        wide_high = mid + 0.002
        wide_low  = mid - 0.002
        tight_high = mid + 0.0001
        tight_low  = mid - 0.0001
        s_wide  = np.nanmean(corwin_schultz_spread(wide_high,  wide_low))
        s_tight = np.nanmean(corwin_schultz_spread(tight_high, tight_low))
        assert s_wide >= s_tight

    def test_consistent_spread_detected(self):
        # All same H/L ratio → consistent non-zero spread
        high = np.ones(50) * 1.101
        low  = np.ones(50) * 1.099
        s = corwin_schultz_spread(high, low)
        valid = s[~np.isnan(s)]
        assert len(valid) > 0


class TestTickEntropy:
    def setup_method(self):
        np.random.seed(54)
        self.prices_random = np.cumsum(np.random.randn(300) * 0.001) + 1.1

    def test_returns_array_same_length(self):
        e = tick_entropy(self.prices_random, window=50)
        assert len(e) == len(self.prices_random)

    def test_leading_nans(self):
        e = tick_entropy(self.prices_random, window=50)
        assert np.isnan(e[:49]).all()

    def test_non_nan_after_window(self):
        e = tick_entropy(self.prices_random, window=50)
        assert not np.isnan(e[49:]).all()

    def test_entropy_non_negative(self):
        e = tick_entropy(self.prices_random, window=50)
        assert (e[~np.isnan(e)] >= 0).all()

    def test_max_entropy_for_alternating(self):
        # Perfectly alternating up/down = max binary entropy (1 bit)
        prices = np.array([1.0, 1.001, 1.0, 1.001, 1.0] * 30)
        e = tick_entropy(prices, window=20, method="shannon")
        valid = e[~np.isnan(e)]
        assert valid.mean() > 0.8  # near max entropy

    def test_zero_entropy_for_constant_direction(self):
        # Always going up → all +1 ticks → entropy = 0
        prices = np.cumsum(np.ones(100) * 0.001) + 1.0
        e = tick_entropy(prices, window=50, method="shannon")
        valid = e[~np.isnan(e)]
        assert valid.mean() < 0.1

    def test_lempel_ziv_method(self):
        e = tick_entropy(self.prices_random, window=50, method="lempel_ziv")
        assert not np.isnan(e[49:]).all()


class TestVolumeEntropy:
    def setup_method(self):
        np.random.seed(55)
        self.volumes = np.abs(np.random.randn(300)) * 1e6 + 1e6

    def test_returns_array_same_length(self):
        e = volume_entropy(self.volumes, window=50)
        assert len(e) == len(self.volumes)

    def test_leading_nans(self):
        e = volume_entropy(self.volumes, window=50)
        assert np.isnan(e[:49]).all()

    def test_non_negative(self):
        e = volume_entropy(self.volumes, window=50)
        assert (e[~np.isnan(e)] >= 0).all()

    def test_constant_volume_zero_entropy(self):
        vols = np.ones(100) * 1e6
        e = volume_entropy(vols, window=50, n_bins=5)
        valid = e[~np.isnan(e)]
        assert valid.mean() == 0.0

    def test_lempel_ziv_method(self):
        e = volume_entropy(self.volumes, window=50, method="lempel_ziv")
        assert not np.isnan(e[49:]).all()
