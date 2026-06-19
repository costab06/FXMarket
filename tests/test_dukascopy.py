"""Tests for the Dukascopy feed adapter."""

from __future__ import annotations

import asyncio
import lzma
import struct
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fx_market.feeds.dukascopy import (
    DukascopyHistory,
    _history_url,
    _parse_ticks,
    _point_value,
)

UTC = timezone.utc


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_tick_bytes(ms: int, ask: float, bid: float, point: int,
                     ask_vol: float = 1.0, bid_vol: float = 1.0) -> bytes:
    """Encode a single tick record in Dukascopy binary format."""
    return struct.pack(
        ">IIIff",
        ms,
        int(round(ask * point)),
        int(round(bid * point)),
        ask_vol,
        bid_vol,
    )


def _make_bi5(ticks: list[bytes]) -> bytes:
    """Compress a list of raw tick records into LZMA (.bi5) format."""
    raw = b"".join(ticks)
    return lzma.compress(raw)


# ── URL construction ───────────────────────────────────────────────────────────

class TestHistoryURL:
    def test_eurusd_format(self):
        dt = datetime(2024, 3, 15, 9, 0, 0, tzinfo=UTC)
        url = _history_url("EURUSD", dt)
        # Month 3 → 02 (zero-indexed)
        assert "EURUSD" in url
        assert "/2024/02/15/09h_ticks.bi5" in url

    def test_january_month_zero(self):
        dt = datetime(2024, 1, 5, 0, 0, 0, tzinfo=UTC)
        url = _history_url("GBPUSD", dt)
        assert "/2024/00/05/00h_ticks.bi5" in url

    def test_december_month_11(self):
        dt = datetime(2024, 12, 31, 23, 0, 0, tzinfo=UTC)
        url = _history_url("USDJPY", dt)
        assert "/2024/11/31/23h_ticks.bi5" in url

    def test_lowercase_pair_uppercased(self):
        dt = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
        url = _history_url("eurusd", dt)
        assert "EURUSD" in url


# ── Point value ────────────────────────────────────────────────────────────────

class TestPointValue:
    def test_non_jpy(self):
        assert _point_value("EURUSD") == 100_000
        assert _point_value("GBPUSD") == 100_000
        assert _point_value("AUDUSD") == 100_000

    def test_jpy_pairs(self):
        assert _point_value("USDJPY") == 1_000
        assert _point_value("EURJPY") == 1_000
        assert _point_value("GBPJPY") == 1_000


# ── Binary parser ──────────────────────────────────────────────────────────────

class TestParseTicks:
    def _hour(self):
        return datetime(2024, 6, 17, 10, 0, 0, tzinfo=UTC)

    def test_single_tick_eurusd(self):
        point = 100_000
        raw = _make_tick_bytes(ms=1500, ask=1.08120, bid=1.08100,
                               point=point, ask_vol=2.5, bid_vol=3.0)
        ticks = _parse_ticks(raw, "EURUSD", self._hour())
        assert len(ticks) == 1
        t = ticks[0]
        assert t.provider == "dukascopy"
        assert t.pair == "EURUSD"
        assert t.ask == pytest.approx(1.08120, abs=1e-5)
        assert t.bid == pytest.approx(1.08100, abs=1e-5)
        assert t.ask_size == pytest.approx(2.5 * 1_000_000)
        assert t.bid_size == pytest.approx(3.0 * 1_000_000)

    def test_timestamp_offset(self):
        point = 100_000
        # ms=3_600_000 - 1  → last millisecond of the hour
        raw = _make_tick_bytes(ms=3_599_000, ask=1.0812, bid=1.0810, point=point)
        ticks = _parse_ticks(raw, "EURUSD", self._hour())
        assert len(ticks) == 1
        expected_minute = 59
        assert ticks[0].timestamp.minute == expected_minute

    def test_multiple_ticks(self):
        point = 100_000
        raw = b"".join(
            _make_tick_bytes(ms=i * 1000, ask=1.0810 + i * 0.0001,
                             bid=1.0808 + i * 0.0001, point=point)
            for i in range(5)
        )
        ticks = _parse_ticks(raw, "EURUSD", self._hour())
        assert len(ticks) == 5

    def test_jpy_pair_scaling(self):
        point = 1_000
        raw = _make_tick_bytes(ms=0, ask=149.530, bid=149.510, point=point)
        ticks = _parse_ticks(raw, "USDJPY", self._hour())
        assert len(ticks) == 1
        assert ticks[0].ask == pytest.approx(149.530, abs=0.001)
        assert ticks[0].bid == pytest.approx(149.510, abs=0.001)

    def test_malformed_tick_skipped(self):
        """Ticks with bid > ask or zero prices should be discarded."""
        point = 100_000
        bad = _make_tick_bytes(ms=0, ask=1.0808, bid=1.0812, point=point)  # bid > ask
        raw = bad
        ticks = _parse_ticks(raw, "EURUSD", self._hour())
        assert len(ticks) == 0

    def test_empty_data_returns_empty(self):
        assert _parse_ticks(b"", "EURUSD", self._hour()) == []

    def test_partial_record_ignored(self):
        """Incomplete record at end of buffer should be silently ignored."""
        point = 100_000
        good = _make_tick_bytes(ms=0, ask=1.0812, bid=1.0810, point=point)
        partial = good[:10]          # half a record
        ticks = _parse_ticks(good + partial, "EURUSD", self._hour())
        assert len(ticks) == 1      # only the complete record parsed


# ── DukascopyHistory (mocked HTTP) ────────────────────────────────────────────

def _mock_response(status: int, content: bytes) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.content = content
    return resp


class TestDukascopyHistory:
    def _bi5(self, pair: str = "EURUSD") -> bytes:
        point = _point_value(pair)
        ticks = [
            _make_tick_bytes(ms=i * 10_000, ask=1.0810 + i * 0.0001,
                             bid=1.0808 + i * 0.0001, point=point)
            for i in range(10)
        ]
        return _make_bi5(ticks)

    def test_fetch_hour_parses_ticks(self):
        hist = DukascopyHistory()
        hour = datetime(2024, 6, 17, 10, 0, 0, tzinfo=UTC)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=_mock_response(200, self._bi5()))
            mock_client_cls.return_value = mock_client

            ticks = asyncio.run(hist.fetch_hour("EURUSD", hour))

        assert len(ticks) == 10
        assert all(t.provider == "dukascopy" for t in ticks)
        assert all(t.pair == "EURUSD" for t in ticks)

    def test_fetch_hour_404_returns_empty(self):
        hist = DukascopyHistory()
        hour = datetime(2024, 6, 15, 10, 0, 0, tzinfo=UTC)  # Saturday

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=_mock_response(404, b""))
            mock_client_cls.return_value = mock_client

            ticks = asyncio.run(hist.fetch_hour("EURUSD", hour))

        assert ticks == []

    def test_fetch_hour_empty_content(self):
        hist = DukascopyHistory()
        hour = datetime(2024, 6, 17, 10, 0, 0, tzinfo=UTC)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=_mock_response(200, b""))
            mock_client_cls.return_value = mock_client

            ticks = asyncio.run(hist.fetch_hour("EURUSD", hour))

        assert ticks == []

    def test_fetch_range_naive_datetimes_raise(self):
        hist = DukascopyHistory()
        with pytest.raises(ValueError, match="timezone-aware"):
            asyncio.run(hist.fetch_range(
                "EURUSD",
                start=datetime(2024, 6, 17, 8, 0),  # naive
                end=datetime(2024, 6, 17, 9, 0),
            ))

    def test_to_dataframe_shape(self):
        point = _point_value("EURUSD")
        raw = b"".join(
            _make_tick_bytes(i * 1000, 1.0812, 1.0810, point) for i in range(5)
        )
        hour = datetime(2024, 6, 17, 10, 0, 0, tzinfo=UTC)
        ticks = _parse_ticks(raw, "EURUSD", hour)
        df = DukascopyHistory.to_dataframe(ticks)
        assert len(df) == 5
        assert set(df.columns) >= {"bid", "ask", "mid", "spread_pips"}

    def test_to_dataframe_empty(self):
        df = DukascopyHistory.to_dataframe([])
        assert len(df) == 0

    def test_to_dataframe_mid_correct(self):
        point = _point_value("EURUSD")
        raw = _make_tick_bytes(0, 1.0812, 1.0810, point)
        hour = datetime(2024, 6, 17, 10, 0, 0, tzinfo=UTC)
        ticks = _parse_ticks(raw, "EURUSD", hour)
        df = DukascopyHistory.to_dataframe(ticks)
        assert df["mid"].iloc[0] == pytest.approx(1.0811, abs=1e-5)

    def test_ticks_sorted_chronologically(self):
        """fetch_range should return ticks in timestamp order."""
        point = _point_value("EURUSD")
        # Reverse order within the hour — parser should still sort
        raw_ticks = [
            _make_tick_bytes(ms=3000, ask=1.0815, bid=1.0813, point=point),
            _make_tick_bytes(ms=1000, ask=1.0810, bid=1.0808, point=point),
            _make_tick_bytes(ms=2000, ask=1.0812, bid=1.0810, point=point),
        ]
        raw = b"".join(raw_ticks)
        hour = datetime(2024, 6, 17, 10, 0, 0, tzinfo=UTC)
        ticks = _parse_ticks(raw, "EURUSD", hour)
        # Raw parse order (not sorted by _parse_ticks, sorted by fetch_range)
        # Check fetch_range sorts them
        from datetime import timedelta
        start = hour
        end   = hour + timedelta(hours=1)

        bi5 = _make_bi5(raw_ticks)
        hist = DukascopyHistory()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=_mock_response(200, bi5))
            mock_client_cls.return_value = mock_client

            result = asyncio.run(hist.fetch_range("EURUSD", start, end))

        timestamps = [t.timestamp for t in result]
        assert timestamps == sorted(timestamps)
