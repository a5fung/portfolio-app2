"""2026-10-15 Apollo Trades dashboard fixes (phone screenshots).

1. Calendar months read in date order on a phone.
2. AVG R measured from the ORIGINAL stop, not the trailed one (Apollo's own rule).
3. Realized-P&L KPI names its period and prints the all-time figure beside it.

Inputs are a FROZEN copy of the trades export (test_fixtures/trades_snapshot_2026-10-15.json,
37 closed live trades incl. PENG's -$81.60 that morning, carrying hard_stop/orb_low). It is NOT
apollo_trades_paper.json, which the nightly export rewrites under a test. The rendered page is
SSO-gated, so the page's own functions are recomputed on that snapshot (portfolio-app2 CLAUDE.md,
"Verifying a dashboard change").
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import date, timedelta

import pandas as pd
import pytest

import apollo_data as ad

FIXTURE = os.path.join(os.path.dirname(__file__), "test_fixtures", "trades_snapshot_2026-10-15.json")
TODAY = date(2026, 10, 15)
CUTOFF_90 = TODAY - timedelta(days=90)  # 2026-07-17 = the page's default "Last 90 days"


def _load(drop_original_stop: bool = False) -> pd.DataFrame:
    """The snapshot through the page's own reader. `drop_original_stop` strips hard_stop/orb_low
    to reproduce TODAY's committed snapshot (which predates those columns)."""
    if not drop_original_stop:
        return ad._load_from_snapshot("live", path=FIXTURE)
    rows = [{k: v for k, v in r.items() if k not in ("hard_stop", "orb_low")}
            for r in json.load(open(FIXTURE))]
    with tempfile.TemporaryDirectory() as td:
        legacy = os.path.join(td, "legacy.json")
        with open(legacy, "w") as f:
            json.dump(rows, f)
        return ad._load_from_snapshot("live", path=legacy)


@pytest.fixture(scope="module")
def df():
    return _load()


@pytest.fixture(scope="module")
def legacy_df():
    return _load(drop_original_stop=True)


def _closed(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["status"].isin(["closed", "stopped"])]


def _period(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["alert_date"] >= CUTOFF_90]


def _r(df: pd.DataFrame, ticker: str) -> float:
    return float(df.loc[df["ticker"] == ticker, "r_multiple"].iloc[0])


# ── 1. Calendar order ───────────────────────────────────────────────────────

class TestCalendarOrder:
    JUL_OCT = [(2026, 7), (2026, 8), (2026, 9), (2026, 10)]

    def test_window_months_are_oldest_first(self):
        assert ad.months_in_window(CUTOFF_90, TODAY) == self.JUL_OCT

    def test_window_crosses_a_year_boundary(self):
        assert ad.months_in_window(date(2025, 11, 20), date(2026, 2, 3)) == [
            (2025, 11), (2025, 12), (2026, 1), (2026, 2)]

    def test_rows_of_three_keep_date_order(self):
        assert ad.chunk_months(self.JUL_OCT, 3) == [
            [(2026, 7), (2026, 8), (2026, 9)], [(2026, 10)]]

    @pytest.mark.parametrize("n", range(1, 14))
    def test_a_phone_reads_every_window_in_date_order(self, n):
        # A phone stacks each COLUMN whole. With one fresh column set per row, the stacking
        # order is row 1 (left to right), then row 2 ... = the flattened chunks.
        months = ad.months_in_window(date(2026, 1, 1), date(2026, 1, 1) + timedelta(days=31 * n))[:n]
        rows = ad.chunk_months(months, 3)
        assert [m for row in rows for m in row] == months
        assert all(1 <= len(row) <= 3 for row in rows)

    def test_the_old_single_column_set_is_what_scrambled_it(self):
        # Regression pin: the old layout put month idx into column idx % 3 of ONE column set;
        # a phone then stacks column 0 whole, then column 1, then column 2.
        cols = [[], [], []]
        for idx, m in enumerate(self.JUL_OCT):
            cols[idx % 3].append(m)
        stacked = [m for c in cols for m in c]
        assert stacked == [(2026, 7), (2026, 10), (2026, 8), (2026, 9)]  # Jul, Oct, Aug, Sep
        assert stacked != self.JUL_OCT


# ── 2. R from the original stop ─────────────────────────────────────────────

class TestRFromOriginalRisk:
    def test_matches_the_slash_trades_screenshot(self, df):
        # His /trades screenshot: KOD +5.9R, VICR +7.3R, HUM -1.0R, CEG -1.0R, PENG -0.8R
        for ticker, want in (("KOD", 5.9), ("VICR", 7.3), ("HUM", -1.0), ("CEG", -1.0), ("PENG", -0.8)):
            assert _r(df, ticker) == pytest.approx(want, abs=0.05), ticker

    def test_no_winner_is_excluded_any_more(self, df):
        closed = _closed(df)
        assert int(closed["r_multiple"].isna().sum()) == 0
        winners = closed[closed["total_pnl"] > 0]
        assert len(winners) == 10 and (winners["r_multiple"] > 0).all()

    def test_it_is_apollos_formula_pnl_over_original_risk_times_shares(self, df):
        # KOD: 148.805 / ((63.15 - 58.0818) x 5)
        kod = df[df["ticker"] == "KOD"].iloc[0]
        assert kod["r_multiple"] == pytest.approx(148.805 / ((63.15 - 58.0818) * 5), abs=1e-6)

    def test_hard_stop_beats_orb_low_and_orb_low_fills_in(self):
        base = pd.DataFrame({"entry_price": [10.0, 10.0, 10.0, 10.0],
                             "hard_stop": [9.0, float("nan"), 11.0, float("nan")],
                             "orb_low": [9.5, 9.5, 9.5, float("nan")]})
        got = ad.original_risk_per_share(base).tolist()
        assert got[0] == pytest.approx(1.0)      # hard_stop wins
        assert got[1] == pytest.approx(0.5)      # no hard_stop -> orb_low
        assert got[2] == pytest.approx(0.5)      # hard_stop >= entry is invalid -> orb_low
        assert pd.isna(got[3])                   # neither -> no R frame, never a zero denominator

    def test_period_average_r_is_no_longer_minus_one(self, df):
        # The KPI's AVG R over the default 90-day window: 32 closed, all with an R.
        closed = _closed(_period(df))
        assert len(closed) == 32 and int(closed["r_multiple"].isna().sum()) == 0
        assert closed["r_multiple"].mean() == pytest.approx(0.135, abs=0.005)

    def test_outcome_counts_do_not_move(self, df):
        # Winners used to classify through the P&L-sign fallback (NaN R); now by R. Same buckets.
        new = pd.Series(ad.classify_outcome(_closed(_period(df)))).value_counts().to_dict()
        old = pd.Series(ad.classify_outcome(_closed(_period(_load(True))))).value_counts().to_dict()
        assert new == old == {"loss": 22, "win": 10}

    def test_every_r_derived_metric_uses_the_same_risk_unit(self, df):
        # best_r / worst_r / capture are priced in the same original risk as r_multiple.
        kod = df[df["ticker"] == "KOD"].iloc[0]
        risk = 63.15 - 58.0818
        assert kod["best_r"] == pytest.approx((kod["highest_price_seen"] - 63.15) / risk, abs=1e-6)
        assert kod["worst_r"] == pytest.approx((kod["lowest_price_seen"] - 63.15) / risk, abs=1e-6)
        assert kod["capture_pct"] == pytest.approx(kod["r_multiple"] / kod["best_r"], abs=1e-6)
        winners = _closed(df)[lambda d: d["total_pnl"] > 0]
        assert winners["capture_pct"].notna().all()
        assert ((winners["capture_pct"] > 0) & (winners["capture_pct"] <= 1.0)).all()

    def test_excursion_stats_see_every_trade(self, df):
        ex = ad.excursion_stats(df)
        assert int(ex.loc[0, "n"]) == 37 and int(ex.loc[0, "n_capture"]) == 10


class TestLegacyFrameUnchanged:
    """No hard_stop/orb_low in the snapshot (today's committed file, the mock) -> old behaviour."""

    def test_trailed_winners_are_still_excluded_as_before(self, legacy_df):
        closed = _closed(_period(legacy_df))
        assert int(closed["r_multiple"].isna().sum()) == 10
        assert closed["r_multiple"].mean() == pytest.approx(-1.03, abs=0.01)

    def test_matches_the_old_formula_row_for_row(self, legacy_df):
        risk = (legacy_df["entry_price"] - legacy_df["stop_price"]).where(lambda s: s > 0)
        want = (legacy_df["exit_price"] - legacy_df["entry_price"]) / risk
        pd.testing.assert_series_equal(legacy_df["r_multiple"], want, check_names=False)
        pd.testing.assert_series_equal(legacy_df["worst_r"], (legacy_df["lowest_price_seen"] - legacy_df["entry_price"]) / risk,
                                       check_names=False)
        pd.testing.assert_series_equal(legacy_df["best_r"], (legacy_df["highest_price_seen"] - legacy_df["entry_price"]) / risk,
                                       check_names=False)

    def test_mock_generator_takes_the_legacy_path(self):
        m = ad.generate_mock_trades(n_trades=40, days_back=60)
        assert "hard_stop" not in m.columns
        risk = (m["entry_price"] - m["stop_price"]).where(lambda s: s > 0)
        want = (m["exit_price"] - m["entry_price"]) / risk
        pd.testing.assert_series_equal(m["r_multiple"], want, check_names=False)


# ── 3. Realized P&L: period AND all-time ────────────────────────────────────

class TestRealizedPnlSubtitle:
    def test_all_time_comes_from_the_whole_book(self, df):
        # /trades says -$36 all-time after PENG; the 90-day KPI says +$61. Both are true.
        assert ad.realized_pnl(df) == pytest.approx(-35.73, abs=0.01)
        assert ad.realized_pnl(_period(df)) == pytest.approx(60.69, abs=0.01)

    def test_subtitle_after_peng_names_period_and_all_time(self, df):
        closed = _closed(_period(df))
        sub = ad.pnl_kpi_subtitle(len(closed), len(_period(df)) - len(closed), "Last 90 days",
                                  ad.realized_pnl(df), covers_all=False)
        assert sub == "32 closed · 0 open · last 90 days · all time $-36"

    def test_subtitle_in_the_screenshot_state_peng_still_open(self, df):
        # Today's committed snapshot: PENG still open -> headline +$142 (31 closed, 1 open),
        # while all-time was +$46. The two numbers he found contradictory, now side by side.
        d = df.copy()
        i = d.index[d["ticker"] == "PENG"][0]
        d.loc[i, ["status", "total_pnl"]] = ["open", 0.0]
        d.loc[i, "exit_price"] = float("nan")
        per = _period(d)
        closed = _closed(per)
        assert ad.realized_pnl(per) == pytest.approx(142.29, abs=0.01)
        sub = ad.pnl_kpi_subtitle(len(closed), len(per) - len(closed), "Last 90 days",
                                  ad.realized_pnl(d), covers_all=False)
        assert sub == "31 closed · 1 open · last 90 days · all time $+46"

    def test_all_time_period_does_not_say_all_time_twice(self):
        assert ad.pnl_kpi_subtitle(37, 0, "All time", -35.73, covers_all=True) == "37 closed · 0 open · all time"

    def test_period_that_truncates_the_book_still_prints_the_figure(self):
        assert ad.pnl_kpi_subtitle(30, 0, "All time", -35.73, covers_all=False).endswith("all time $-36")

    def test_empty_book_is_zero(self):
        assert ad.realized_pnl(pd.DataFrame()) == 0.0
