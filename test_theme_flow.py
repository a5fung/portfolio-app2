"""Pin tests for theme_flow (#640): compute_band_flow / _band_of / _ribbon_color,
and since the 2026-09-19 rebuild the one-hop page pieces — summarize_transition
(the sentence), band_shares (the permanent node labels), classify_unranked_edges
(the caption's honesty), _node_layout / build_flow_figure (the two-column chart),
movers_lines (the list under it). Every assertion added on 09-19 was RED-proved:
the mutation that fails it is named in the test's docstring.

#640's original diagnosis of Rank Flow: the old "Outside top 30" band merged four
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
    _ALL_HISTORY_WEEKS,
    _band_of,
    _BAND_ORDER,
    _DEFAULT_HOP,
    _KEEP_ALIVE,
    _KNOWN_LOW_BAND,
    _NO_RANK_BAND,
    _node_layout,
    _node_values,
    _ribbon_color,
    _to_rgba,
    _usable_weeks,
    band_label,
    band_shares,
    build_flow_figure,
    classify_unranked_edges,
    compute_band_flow,
    movers_lines,
    summarize_transition,
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

    def test_holding_the_same_band_is_flat_and_lower_opacity(self):
        # 0.12 → 0.35 on 2026-09-19: on a single hop the stayers are 3 of 55,
        # not the bulk, so they no longer need to be nearly invisible — but a
        # hold must still carry less ink than a move.
        color = _ribbon_color("6-15", "6-15")
        assert color == _to_rgba(_DELTA_FLAT, 0.35)
        up = _ribbon_color("16-30", "Top 5")
        flat_alpha = float(color.rsplit(",", 1)[1].rstrip(")"))
        up_alpha = float(up.rsplit(",", 1)[1].rstrip(")"))
        assert flat_alpha < up_alpha

    def test_falling_off_the_board_entirely_IS_down_colored(self):
        # REVERSED TWICE, both times on evidence. 09-11 made 31+ -> No rank
        # NEUTRAL: the band meant "absent OR a data gap", so red asserted
        # weakness about a cohort that might simply have no data. 09-19
        # measured the band (design doc § MOBILE PASS): of 20 cohorts leaving
        # it, 18 were brand-new themes and 0 were returning; and EVERY
        # null-rank row in the snapshot is stage Fading or Retired — a null
        # rank is the engine's verdict, not a hole in the export. So a move
        # into the renamed "New / unranked" band is a real fall, the sentence
        # above the chart counts it as one, and the ribbon must agree with
        # the sentence (EXPECT on the PLAN line: header matches ribbons).
        # RED-proved: restoring the 09-11 neutral special-case fails this.
        assert _ribbon_color(_KNOWN_LOW_BAND, _NO_RANK_BAND) == _to_rgba(_DELTA_DOWN, 0.5)


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


# ── A move into or out of "New / unranked" IS a strength move (2026-09-19) ───────────────────
#
# The 09-11 build painted these neutral (the band then meant "absent OR a data gap"). The 09-19
# measurement settled it — 18 of the 20 cohorts leaving the band were themes being BORN, none were
# returning, and every null-rank row in the snapshot is the engine's own Fading/Retired verdict —
# and the operator's ruling was to RENAME the band and carry the honesty in the caption. The
# sentence above the chart counts these moves as climbs and falls; the ribbons must say the same.

def test_a_theme_being_born_is_painted_as_a_climb():
    """RED-proved: re-adding the 09-11 `if _NO_RANK_BAND in (b0, b1)` neutral branch fails this."""
    assert _ribbon_color(_NO_RANK_BAND, "16-30") == _to_rgba(_DELTA_UP, 0.55)
    assert _ribbon_color(_NO_RANK_BAND, "Top 5") == _to_rgba(_DELTA_UP, 0.55)


def test_dropping_off_the_map_is_painted_as_a_fall():
    assert _ribbon_color("Top 5", _NO_RANK_BAND) == _to_rgba(_DELTA_DOWN, 0.5)


def test_the_no_rank_self_loop_would_be_a_hold():
    """Never drawn (compute_band_flow drops it), but the colour rule must not special-case it."""
    assert _ribbon_color(_NO_RANK_BAND, _NO_RANK_BAND) == _to_rgba(_DELTA_FLAT, 0.35)


def test_real_strength_moves_are_untouched():
    assert _ribbon_color("16-30", "Top 5") == _to_rgba(_DELTA_UP, 0.55)
    assert _ribbon_color("Top 5", _KNOWN_LOW_BAND) == _to_rgba(_DELTA_DOWN, 0.5)


class TestBandRename:
    def test_bottom_band_is_new_unranked_and_stays_distinct_from_31_plus(self):
        """The operator's 09-19 ruling: RENAME, do not split, and do not re-merge
        with 31+ (the 09-11 fix). RED-proved: setting `_NO_RANK_BAND = "31+"`
        fails this, as does the old value "No rank"."""
        assert _NO_RANK_BAND == "New / unranked"
        assert _KNOWN_LOW_BAND == "31+"
        assert _BAND_ORDER[-2:] == [_KNOWN_LOW_BAND, _NO_RANK_BAND]
        assert len(set(_BAND_ORDER)) == 5
        assert _band_of(None) == _NO_RANK_BAND != _band_of(31)


# ── The one-hop page pieces (2026-09-19 rebuild) ─────────────────────────────────────────────

def _hop_links() -> dict:
    """A small hand-built `links` dict shaped like compute_band_flow's output
    for ONE hop: 4 born straight into Top 5, 6 slipping 16-30 → 31+, 3 falling
    off the map from Top 5, 2 holding 6-15, 1 climbing 16-30 → 6-15."""
    return {
        (0, _NO_RANK_BAND, "Top 5"): {"count": 4, "names": ["Cardiac", "AI Diagnostics", "Zeta", "Mid"]},
        (0, "16-30", _KNOWN_LOW_BAND): {"count": 6, "names": list("abcdef")},
        (0, "Top 5", _NO_RANK_BAND): {"count": 3, "names": ["Cyber", "Robotic", "Oil"]},
        (0, "6-15", "6-15"): {"count": 2, "names": ["Genomics", "Enterprise"]},
        (0, "16-30", "6-15"): {"count": 1, "names": ["Server"]},
    }


class TestSummarizeTransition:
    def test_counts_split_by_direction_and_sum_to_the_population(self):
        """The sentence: climbed / fell / held add up to every cohort in the hop.
        RED-proved: flipping `_direction`'s sign swaps climbed and fell (5/9 → 9/5)."""
        s = summarize_transition(_hop_links())
        assert (s["climbed"], s["fell"], s["held"]) == (5, 9, 2)
        assert s["total"] == 16

    def test_biggest_is_the_largest_band_distance_and_a_climb_wins_the_tie(self):
        """New/unranked → Top 5 and Top 5 → New/unranked are both 4 rows; the
        climb wins (his question is which themes are climbing), and inside the
        ribbon the first name alphabetically is shown with the rest counted.
        RED-proved: preferring falls on the tie names "Cyber" instead."""
        big = summarize_transition(_hop_links())["biggest"]
        assert big["direction"] == "climb" and big["distance"] == 4
        assert (big["from"], big["to"]) == (_NO_RANK_BAND, "Top 5")
        assert big["name"] == "AI Diagnostics"
        assert big["count"] == 4 and big["others"] == 3

    def test_a_bigger_fall_beats_a_smaller_climb(self):
        """Distance outranks direction — a 3-row fall beats a 1-row climb.
        RED-proved: ranking by count instead of distance picks the 5-cohort climb."""
        links = {
            (0, "16-30", "6-15"): {"count": 5, "names": list("vwxyz")},
            (0, "Top 5", _KNOWN_LOW_BAND): {"count": 1, "names": ["Marketing"]},
        }
        big = summarize_transition(links)["biggest"]
        assert big["direction"] == "fall" and big["name"] == "Marketing" and big["others"] == 0

    def test_no_moves_means_no_biggest(self):
        s = summarize_transition({(0, "6-15", "6-15"): {"count": 2, "names": ["x", "y"]}})
        assert s["biggest"] is None and s["held"] == 2 and s["total"] == 2

    def test_empty_links(self):
        s = summarize_transition({})
        assert s == {"climbed": 0, "fell": 0, "held": 0, "total": 0, "biggest": None}


class TestBandShares:
    def test_every_band_present_with_count_and_share_of_the_whole_population(self):
        """The permanent node labels: all five bands, zero when empty, shares
        over the SAME denominator on both sides — the whole population.
        RED-proved: dividing by the number of bands present that week (3)
        instead of the population (4) turns the 1/2 into 2/3."""
        grid = _grid([
            ("A", "a", W0, 2), ("A", "a", W1, 2),
            ("D", "d", W0, 3), ("D", "d", W1, 40),
            ("B", "b", W0, 45), ("B", "b", W1, 12),
            ("C", "c", W1, 25),
        ])
        band_piv, _links = compute_band_flow(grid, [W0, W1])
        shares = band_shares(band_piv, W0)
        assert list(shares) == _BAND_ORDER
        assert shares["Top 5"] == (2, pytest.approx(1 / 2))
        assert shares[_KNOWN_LOW_BAND] == (1, pytest.approx(1 / 4))
        assert shares[_NO_RANK_BAND] == (1, pytest.approx(1 / 4))
        assert shares["6-15"] == (0, 0.0)
        assert sum(c for c, _s in shares.values()) == len(band_piv)
        assert sum(c for c, _s in band_shares(band_piv, W1).values()) == len(band_piv)

    def test_label_reads_band_count_share(self):
        """`16-30 · 15 · 27%` — the format the design doc names."""
        assert band_label("16-30", 15, 15 / 55) == "16-30 · 15 · 27%"

    def test_empty_population(self):
        shares = band_shares(pd.DataFrame(columns=[W0]), W0)
        assert all(v == (0, 0.0) for v in shares.values())


class TestClassifyUnrankedEdges:
    def test_new_unscored_and_returning_are_told_apart(self):
        """The caption's honesty. N never had a row before W2 (new); U had a
        row at W1 with no rank (listed but unscored); R was ranked at W0, had
        no row at W1, and is back (returning). RED-proved: dropping the
        `cid in rows_w0` check counts U as new."""
        grid = _grid([
            ("N", "new", W2, 10),
            ("U", "unscored", W1, None), ("U", "unscored", W2, 10),
            ("R", "returning", W0, 8), ("R", "returning", W2, 10),
        ])
        band_piv, _links = compute_band_flow(grid, [W1, W2])
        e = classify_unranked_edges(grid, band_piv, W1, W2)
        assert e["left"] == {"new": 1, "unscored": 1, "returning": 1}
        assert e["entered"] == {"unscored": 0, "absent": 0}
        assert e["history_weeks"] == 1   # W0 is the only ranked week before W1

    def test_entrants_split_into_unscored_and_absent(self):
        """G and H still have a row at W2 with no rank (the engine marked them
        fading / retired); A has no row at W2 at all. Deliberately 2-vs-1 so
        the split is asymmetric. RED-proved: swapping the two branches reads
        1 unscored / 2 absent and fails this (a 1-vs-1 fixture could not tell)."""
        grid = _grid([
            ("G", "gone-unscored", W1, 5), ("G", "gone-unscored", W2, None),
            ("H", "gone-unscored-2", W1, 6), ("H", "gone-unscored-2", W2, None),
            ("A", "gone-absent", W1, 7),
        ])
        band_piv, _links = compute_band_flow(grid, [W1, W2])
        e = classify_unranked_edges(grid, band_piv, W1, W2)
        assert e["entered"] == {"unscored": 2, "absent": 1}
        assert e["left"] == {"new": 0, "unscored": 0, "returning": 0}

    def test_empty_population_is_all_zero(self):
        e = classify_unranked_edges(_grid([]), pd.DataFrame(), W0, W1)
        assert sum(e["left"].values()) == 0 and sum(e["entered"].values()) == 0 and e["history_weeks"] == 0


class TestNodeLayout:
    def test_nodes_stack_in_band_order_with_exactly_pad_between_and_fill_the_height(self):
        """Reproduces d3-sankey's own height rule so the margin labels sit on
        their nodes. RED-proved: forgetting `(n-1)*pad` in ky makes the last
        node overflow the plot height."""
        src, tgt = _node_values([5, 10, 17, 3, 20], [5, 10, 15, 11, 14])
        plot_h, pad = 442.0, 14.0
        ys, yt = _node_layout(src, tgt, plot_h, pad)
        assert ys == sorted(ys) and yt == sorted(yt)
        ky = (plot_h - 4 * pad) / 55
        for vals, centres in ((src, ys), (tgt, yt)):
            tops = [y * plot_h - v * ky / 2 for y, v in zip(centres, vals)]
            bottoms = [y * plot_h + v * ky / 2 for y, v in zip(centres, vals)]
            assert tops[0] == pytest.approx(0.0)
            for i in range(4):
                assert bottoms[i] + pad == pytest.approx(tops[i + 1])
            assert bottoms[-1] == pytest.approx(plot_h)

    def test_empty_band_gets_the_keep_alive_on_both_sides_and_a_nonzero_centre(self):
        """An empty band still draws (as a hairline) and Plotly ignores a zero
        coordinate, so no centre may be 0. `_node_values` already keeps every
        value above 0; the clamp in `_node_layout` is the guard for a direct
        call with a raw 0 count. RED-proved: returning `y + h/2` unclamped
        puts an empty first band at exactly 0.0 in the raw-counts call below."""
        raw_src, raw_tgt = _node_layout([0, 1, 2, 0, 4], [0, 3, 6, 0, 1], 442.0, 14.0)
        assert all(y > 0 for y in raw_src + raw_tgt)
        src, tgt = _node_values([0, 1, 2, 0, 4], [0, 3, 6, 0, 1])
        assert src[3] == tgt[3] == _KEEP_ALIVE                 # 31+ empty on both sides
        assert src[0] == tgt[0] == _KEEP_ALIVE                 # Top 5 empty on both sides
        assert src[1] == 1 and tgt[1] == 3                     # a full band gets nothing added
        src2, tgt2 = _node_values([3, 1, 2, 0, 4], [0, 3, 6, 0, 1])
        assert src2[0] == 3 + _KEEP_ALIVE and tgt2[0] == _KEEP_ALIVE   # empty on ONE side: both get it
        assert sum(src2) == pytest.approx(sum(tgt2))
        ys, yt = _node_layout(src, tgt, 442.0, 14.0)
        assert all(y > 0 for y in ys + yt)


class TestMoversLines:
    def test_one_line_per_moving_ribbon_biggest_first_and_held_collapsed(self):
        """RED-proved: sorting by count instead of distance puts the 6-cohort
        16-30 → 31+ ribbon first."""
        lines = movers_lines(_hop_links(), names_per_line=2)
        assert len(lines) == 4 + 1                              # four moving ribbons + one held line
        assert lines[0].startswith(f":green[▲] **{_NO_RANK_BAND} → Top 5** · 4 — AI Diagnostics, Cardiac")
        assert lines[0].endswith("+2 more")
        assert lines[1].startswith(f":red[▼] **Top 5 → {_NO_RANK_BAND}** · 3 — ")
        assert lines[-1] == ":grey[—] **held their band** · 2 — Enterprise, Genomics"
        assert not any("6-15 → 6-15" in line for line in lines)  # the stayers never get a ribbon line

    def test_markdown_specials_in_a_name_are_escaped_not_html_escaped(self):
        """A name with `*`/`_` must not turn into emphasis, and `&` must stay
        `&` (an `&amp;` shows literally in st.markdown). RED-proved: routing
        names through html.escape fails the `&` assertion."""
        links = {(0, "16-30", "Top 5"): {"count": 1, "names": ["A*B_C & D"]}}
        line = movers_lines(links)[0]
        assert "A\\*B\\_C & D" in line and "&amp;" not in line


class TestBuildFlowFigure:
    def _fig(self):
        grid = _grid([
            ("A", "Alpha", W0, 2), ("A", "Alpha", W1, 8),
            ("B", "Beta", W0, 20), ("B", "Beta", W1, 22),
            ("C", "Gamma", W1, 25),
            ("D", "Delta", W0, 10),
        ])
        band_piv, links = compute_band_flow(grid, [W0, W1])
        return band_piv, links, build_flow_figure(band_piv, links, W0, W1, "4 weeks ago · 3 Aug", "31 Aug · now")

    def test_ten_fixed_nodes_none_at_zero_and_no_plotly_label(self):
        """Two columns of five, `arrangement="fixed"`, every coordinate truthy
        (Plotly skips a falsy one), Plotly's own labels blank so nothing is
        drawn over the ribbons. RED-proved: `_X_SRC = 0.0` fails this."""
        _piv, _links, fig = self._fig()
        sk = fig.data[0]
        assert sk.arrangement == "fixed"
        assert len(sk.node.x) == len(sk.node.y) == 10
        assert all(x for x in sk.node.x) and all(y for y in sk.node.y)
        assert all(label == "" for label in sk.node.label)

    def test_every_band_labelled_twice_with_count_and_share_permanently(self):
        """Ten margin annotations (five a side), each carrying its band name,
        count and a `%` — on the chart, never hover-only. RED-proved: dropping
        the `%` from the label text fails this."""
        _piv, _links, fig = self._fig()
        texts = [a.text for a in fig.layout.annotations]
        for band in _BAND_ORDER:
            mine = [t for t in texts if t.startswith(f"<b>{band}</b><br>")]
            assert len(mine) == 2, band
            assert all("%" in t and "·" in t for t in mine), band
        assert any("4 WEEKS AGO" in t for t in texts) and any("NOW" in t for t in texts)
        # 31+ is empty on both sides here and must still be labelled "0 · 0%".
        assert sum(t.startswith(f"<b>{_KNOWN_LOW_BAND}</b><br>") and "0 · 0%" in t for t in texts) == 2

    def test_visible_ribbons_equal_links_and_sum_to_the_population(self):
        """Every plotted link is one of `links` (its colour is a real ribbon
        colour) except the transparent keep-alives, whose value is exactly
        `_KEEP_ALIVE` and which exist only for a band empty on either side.
        RED-proved: a zero-value keep-alive fails the value assertion (and, as
        rendered on 09-19, Plotly drops the node with it)."""
        band_piv, links, fig = self._fig()
        sk = fig.data[0]
        pairs = list(zip(sk.link.value, sk.link.color))
        visible = [v for v, c in pairs if c != "rgba(0,0,0,0)"]
        keep = [v for v, c in pairs if c == "rgba(0,0,0,0)"]
        assert len(visible) == len(links)
        assert sum(visible) == len(band_piv)
        # Bands empty on at least one side: Top 5 (right), 31+ (both), 6-15? no — D holds 6-15 on the left only.
        src, tgt = band_shares(band_piv, W0), band_shares(band_piv, W1)
        expected_keep = sum(1 for b in _BAND_ORDER if src[b][0] == 0 or tgt[b][0] == 0)
        assert len(keep) == expected_keep >= 2
        assert all(v == _KEEP_ALIVE for v in keep)

    def test_ribbon_colours_follow_direction(self):
        band_piv, links, fig = self._fig()
        sk = fig.data[0]
        colours = {c for c in sk.link.color if c != "rgba(0,0,0,0)"}
        assert _to_rgba(_DELTA_DOWN, 0.5) in colours        # Alpha: Top 5 → 6-15, Delta: 6-15 → New / unranked
        assert _to_rgba(_DELTA_UP, 0.55) in colours         # Gamma: New / unranked → 16-30
        assert _to_rgba(_DELTA_FLAT, 0.35) in colours       # Beta held 16-30


class TestLiveSnapshotDefaultHop:
    """The page's own default (latest week, a four-week hop) against the
    committed snapshot — STRUCTURE, not the figures: the nightly auto-export
    advances the latest week, so `25 climbed` is true today and stale next
    Monday. What must hold every week: the sentence adds up to the cohorts
    drawn, the ribbon count is phone-sized (EXPECT on the PLAN line: 17
    today against the old view's 357), and the caption's split accounts for
    every cohort that crossed the New / unranked line."""

    def test_default_hop_sentence_adds_up_and_ribbons_fit_a_phone(self):
        grid = get_canonical_weekly_grid(weeks=_ALL_HISTORY_WEEKS)
        if grid.empty:
            pytest.skip("no committed snapshot data available")
        weeks = _usable_weeks(grid)
        if len(weeks) <= _DEFAULT_HOP:
            pytest.skip("fewer usable weeks on file than the default hop")
        w0, w1 = weeks[-1 - _DEFAULT_HOP], weeks[-1]
        band_piv, links = compute_band_flow(grid, [w0, w1])
        s = summarize_transition(links)
        assert s["total"] == len(band_piv), "sentence does not add up to the cohorts drawn"
        assert 0 < len(links) <= 25, f"{len(links)} ribbons is not a phone-readable single hop"
        assert s["biggest"] is not None
        for w in (w0, w1):
            assert sum(c for c, _s in band_shares(band_piv, w).values()) == len(band_piv)
        e = classify_unranked_edges(grid, band_piv, w0, w1)
        left = sum(v["count"] for (_i, b0, b1), v in links.items() if b0 == _NO_RANK_BAND and b1 != _NO_RANK_BAND)
        entered = sum(v["count"] for (_i, b0, b1), v in links.items() if b1 == _NO_RANK_BAND and b0 != _NO_RANK_BAND)
        assert sum(e["left"].values()) == left
        assert sum(e["entered"].values()) == entered
        assert e["history_weeks"] == len([w for w in weeks if w < w0])
        fig = build_flow_figure(band_piv, links, w0, w1, "4 weeks ago", "now")
        assert len([c for c in fig.data[0].link.color if c != "rgba(0,0,0,0)"]) == len(links)
