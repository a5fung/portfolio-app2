"""Pin tests for theme_flow.compute_band_flow / _band_of / _ribbon_color (#640).

#640's diagnosis of Rank Flow: the old "Outside top 30" band merged four
different facts — rank 31, rank 140, never-seen, and a data gap all read
identically — so every ribbon into or out of that band was uninterpretable.
Measured against the live snapshot at the page's own 10-week default: 190 of
1230 cohort-weeks that would have fallen into that merged band carried a
KNOWN rank past the cut (see theme_flow.py's own module docstring for the
full count). These tests pin the fix that makes that distinction real:

  1. A known rank past the cut ("31+") and no rank at all ("No rank") must
     land in different bands and produce different link keys — never merge
     into one (`TestKnownVsNoRank` — the WOULD-FAIL-IF case from the task).
  2. Only the "No rank -> No rank" self-loop is dropped from the plotted
     links; a cohort sitting continuously in "31+" is real signal and must
     stay (`TestSelfLoopDropping`).
  3. Ribbons are colored by DIRECTION (promotion/demotion/flat), not by the
     source band, so "which move matters" is visible at a glance
     (`TestRibbonColor`).

All fixtures are small synthetic DataFrames shaped like
`theme_data.get_canonical_weekly_grid`'s output (canonical_id,
canonical_name, week_start, week_rank) — the same shape
`test_theme_movers.py` uses, no snapshot file, no Streamlit.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from theme_data import get_canonical_weekly_grid
from theme_flow import (
    _band_of,
    _BAND_ORDER,
    _KNOWN_LOW_BAND,
    _NO_RANK_BAND,
    _ribbon_color,
    _to_rgba,
    _usable_weeks,
    compute_band_flow,
)
from theme_grid import _DELTA_DOWN, _DELTA_FLAT, _DELTA_UP

W0 = date(2026, 8, 3)   # a Monday, arbitrary anchor
W1 = W0 + timedelta(days=7)
W2 = W1 + timedelta(days=7)


def _grid(rows: list[tuple[str, str, date, float | None]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["canonical_id", "canonical_name", "week_start", "week_rank"])


class TestBandOf:
    def test_none_and_nan_are_no_rank(self):
        assert _band_of(None) == _NO_RANK_BAND
        assert _band_of(float("nan")) == _NO_RANK_BAND

    def test_known_rank_past_the_cut_is_31_plus(self):
        assert _band_of(31) == _KNOWN_LOW_BAND
        assert _band_of(140) == _KNOWN_LOW_BAND

    def test_band_boundaries(self):
        assert _band_of(5) == "Top 5"
        assert _band_of(6) == "6-15"
        assert _band_of(15) == "6-15"
        assert _band_of(16) == "16-30"
        assert _band_of(30) == "16-30"
        assert _band_of(31) == _KNOWN_LOW_BAND


class TestKnownVsNoRank:
    """The core #640 ask: a known low rank and no rank at all must never
    collapse into the same band or the same link."""

    def test_no_rank_and_known_low_rank_are_distinct_bands_and_links(self):
        # A: no row at all in W0 (absent from that week's snapshot) -> No rank.
        # B: a real row in W0 at rank 45 (past the board cut) -> 31+.
        # Both reach rank 12 (on the board) in W1, so both qualify for the
        # population. WOULD-FAIL-IF: a ribbon still merges a data gap with a
        # weak rank — i.e. both collapse into one link key with count 2.
        grid = _grid([
            ("A", "Absent Cohort", W1, 12),
            ("B", "Weak Rank Cohort", W0, 45), ("B", "Weak Rank Cohort", W1, 12),
        ])
        band_piv, links = compute_band_flow(grid, [W0, W1])

        assert band_piv.at["A", W0] != band_piv.at["B", W0]
        assert band_piv.at["A", W0] == _NO_RANK_BAND
        assert band_piv.at["B", W0] == _KNOWN_LOW_BAND

        assert (0, _NO_RANK_BAND, "6-15") in links
        assert links[(0, _NO_RANK_BAND, "6-15")]["count"] == 1
        assert (0, _KNOWN_LOW_BAND, "6-15") in links
        assert links[(0, _KNOWN_LOW_BAND, "6-15")]["count"] == 1
        # Never one merged "Outside" key carrying both cohorts.
        assert not any(b0 not in _BAND_ORDER for (_, b0, _b1) in links)

    def test_a_data_gap_row_is_also_no_rank(self):
        # A row that EXISTS but has a null week_rank (rs_avg was null that
        # week) must read the same as an absent row — both are "no usable
        # rank," not two different things.
        grid = _grid([
            ("C", "Gap Cohort", W0, None), ("C", "Gap Cohort", W1, 12),
        ])
        band_piv, _links = compute_band_flow(grid, [W0, W1])
        assert band_piv.at["C", W0] == _NO_RANK_BAND


class TestSelfLoopDropping:
    def test_no_rank_self_loop_is_excluded(self):
        # D has no rank in W0 or W1, then reaches the board in W2 (needed to
        # even enter the population — a cohort that never reaches the board
        # is excluded entirely, see TestPopulation below).
        grid = _grid([("D", "Late Arrival", W2, 12)])
        band_piv, links = compute_band_flow(grid, [W0, W1, W2])
        assert band_piv.loc["D", W0] == _NO_RANK_BAND
        assert band_piv.loc["D", W1] == _NO_RANK_BAND
        assert (0, _NO_RANK_BAND, _NO_RANK_BAND) not in links
        # The real transition (no rank -> on the board) stays.
        assert (1, _NO_RANK_BAND, "6-15") in links

    def test_known_low_self_loop_is_kept(self):
        # E sits in "31+" for two straight weeks (a real, if unexciting,
        # fact) then reaches the board — must qualify for the population via
        # that on-board week, and its 31+ -> 31+ self-loop must NOT be
        # dropped (only "No rank" self-loops are).
        grid = _grid([
            ("E", "Stuck Low", W0, 45), ("E", "Stuck Low", W1, 50), ("E", "Stuck Low", W2, 12),
        ])
        band_piv, links = compute_band_flow(grid, [W0, W1, W2])
        assert band_piv.loc["E", W0] == _KNOWN_LOW_BAND
        assert band_piv.loc["E", W1] == _KNOWN_LOW_BAND
        assert (0, _KNOWN_LOW_BAND, _KNOWN_LOW_BAND) in links
        assert links[(0, _KNOWN_LOW_BAND, _KNOWN_LOW_BAND)]["count"] == 1


class TestPopulation:
    def test_cohort_that_never_reaches_the_board_is_excluded_entirely(self):
        # F is never ranked <= 30 in any week in the window — not part of
        # the population at all, NOT folded into "No rank" (that band means
        # "on our radar, no usable rank this week," not "never relevant").
        grid = _grid([
            ("F", "Never Close", W0, 45), ("F", "Never Close", W1, 60),
        ])
        band_piv, _links = compute_band_flow(grid, [W0, W1])
        assert "F" not in band_piv.index


class TestEmptyAndDegenerate:
    def test_empty_grid_returns_empty(self):
        band_piv, links = compute_band_flow(pd.DataFrame(), [])
        assert band_piv.empty and links == {}

    def test_single_week_returns_empty(self):
        grid = _grid([("A", "Alpha", W0, 5)])
        band_piv, links = compute_band_flow(grid, [W0])
        assert band_piv.empty and links == {}

    def test_nobody_on_board_returns_empty(self):
        grid = _grid([("A", "Alpha", W0, 45), ("A", "Alpha", W1, 60)])
        band_piv, links = compute_band_flow(grid, [W0, W1])
        assert band_piv.empty and links == {}


class TestRibbonColor:
    def test_promotion_toward_a_better_row_is_up_colored(self):
        assert _ribbon_color("16-30", "Top 5") == _to_rgba(_DELTA_UP, 0.55)

    def test_demotion_toward_a_worse_row_is_down_colored(self):
        assert _ribbon_color("Top 5", "16-30") == _to_rgba(_DELTA_DOWN, 0.5)

    def test_holding_the_same_band_is_flat_and_low_opacity(self):
        color = _ribbon_color("6-15", "6-15")
        assert color == _to_rgba(_DELTA_FLAT, 0.12)
        # Flat ribbons must be visibly lower-opacity than a real move —
        # the ink-weighting defect (#4) this exists to fix.
        up = _ribbon_color("16-30", "Top 5")
        flat_alpha = float(color.rsplit(",", 1)[1].rstrip(")"))
        up_alpha = float(up.rsplit(",", 1)[1].rstrip(")"))
        assert flat_alpha < up_alpha

    def test_falling_off_the_board_entirely_is_NOT_down_colored(self):
        # REVERSED 2026-09-11, deliberately. The first version of this test pinned
        # 31+ -> No rank as a DEMOTION, reasoning that No rank sits below 31+ in
        # _BAND_RANK. But the No-rank band's own definition is "absent that week OR
        # a null-rs_avg data gap" — so red asserts weakness for a cohort that may
        # simply have no data. That is THIS TASK'S OWN DEFECT (a data fact rendered
        # as a strength fact) relocated from the structure into the colour channel,
        # which is why it is corrected rather than pinned. Neutral at full opacity:
        # disappearing is worth SEEING, it is just not a fall.
        assert _ribbon_color(_KNOWN_LOW_BAND, _NO_RANK_BAND) == _to_rgba(_DELTA_FLAT, 0.45)


class TestAgainstLiveSnapshot:
    """Not a pin on exact numbers (the snapshot changes) — proves the split
    is not a no-op against the real committed data every other test in this
    file uses synthetic fixtures for."""

    def test_split_finds_at_least_one_known_low_and_one_no_rank_week(self):
        grid = get_canonical_weekly_grid(weeks=10)
        if grid.empty:
            pytest.skip("no committed snapshot data available")
        weeks = _usable_weeks(grid)
        if len(weeks) < 2:
            pytest.skip("fewer than 2 usable weeks in the committed snapshot")
        band_piv, _links = compute_band_flow(grid, weeks)
        counts = band_piv.stack().value_counts()
        assert counts.get(_KNOWN_LOW_BAND, 0) > 0, "no known-rank-past-cut cell found in live data"
        assert counts.get(_NO_RANK_BAND, 0) > 0, "no true no-rank cell found in live data"


# ── A move involving "No rank" is not a strength move (2026-09-11) ────────────────────────────
#
# `_BAND_RANK` puts No-rank at the bottom row, so a raw delta paints "we stopped having data" as a
# demotion (red) and "data came back" as a promotion (green). That is exactly this task's defect —
# a DATA fact rendered as a STRENGTH fact — moved into the colour channel. The band split fixes the
# structure; this fixes the signal the structure is drawn in.

def test_falling_out_of_data_is_not_painted_as_a_demotion():
    from theme_flow import _DELTA_DOWN, _ribbon_color, _to_rgba
    assert _ribbon_color("31+", "No rank") != _to_rgba(_DELTA_DOWN, 0.5)


def test_data_coming_back_is_not_painted_as_a_promotion():
    from theme_flow import _DELTA_UP, _ribbon_color, _to_rgba
    assert _ribbon_color("No rank", "16-30") != _to_rgba(_DELTA_UP, 0.55)


def test_a_no_rank_move_is_neutral_but_still_visible():
    """Neutral, NOT hidden — appearing or disappearing is worth seeing; it just is not a climb."""
    from theme_flow import _DELTA_FLAT, _ribbon_color, _to_rgba
    assert _ribbon_color("Top 5", "No rank") == _to_rgba(_DELTA_FLAT, 0.45)
    assert _ribbon_color("No rank", "No rank") == _to_rgba(_DELTA_FLAT, 0.12), (
        "the self-loop must stay at stayer opacity — it is the boring case"
    )


def test_real_strength_moves_are_untouched():
    """The guard must not swallow the moves the colour exists for."""
    from theme_flow import _DELTA_DOWN, _DELTA_UP, _ribbon_color, _to_rgba
    assert _ribbon_color("16-30", "Top 5") == _to_rgba(_DELTA_UP, 0.55)
    assert _ribbon_color("Top 5", "31+") == _to_rgba(_DELTA_DOWN, 0.5)
