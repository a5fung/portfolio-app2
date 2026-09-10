"""Pin tests for theme_movers.compute_weekly_movers (#561).

#561's diagnosis of the old Rank Flow view: it answers "how many cohorts
changed band" (a count) when the operator asks "which themes are climbing"
(a name), and even its plain-text "band changes" companion sorted the boring
"stayed put" case in with real movers. These tests pin the two behaviors
that make the replacement actually answer the right question:

  1. Ranking is by MAGNITUDE OF CHANGE, not by where a cohort ended up —
     `test_ranks_gainers_by_magnitude_not_by_current_rank` fails under a
     "sort by curr_rank" implementation even though that would still be a
     defensible-looking gainers list.
  2. A first appearance in the top board is called out SEPARATELY from
     gainers, with no synthetic numeric delta invented for "outside".

All fixtures are small synthetic DataFrames shaped like
`theme_data.get_canonical_weekly_grid`'s output (canonical_id,
canonical_name, week_start, week_rank) — no snapshot file, no Streamlit.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from theme_movers import _md_escape, compute_weekly_movers

W0 = date(2026, 8, 3)   # a Monday, arbitrary anchor
W1 = W0 + timedelta(days=7)
W2 = W1 + timedelta(days=7)


def _grid(rows: list[tuple[str, str, date, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["canonical_id", "canonical_name", "week_start", "week_rank"])


class TestMarkdownEscape:
    def test_escapes_commonmark_special_chars(self):
        # A cohort name is engine-generated text, not code we control — an
        # unescaped "_" pair or "*" would break the bold formatting this view
        # exists to make readable. No name in the current snapshot needs
        # this, but the escape has to survive one that does.
        assert _md_escape("Semi_Cap Equipment") == r"Semi\_Cap Equipment"
        assert _md_escape("A*B*C") == r"A\*B\*C"
        assert _md_escape("Plain Name") == "Plain Name"


class TestEmptyAndDegenerate:
    def test_empty_grid_returns_empty(self):
        assert compute_weekly_movers(pd.DataFrame(), []) == []

    def test_single_week_returns_empty(self):
        grid = _grid([("A", "Alpha", W0, 5)])
        assert compute_weekly_movers(grid, [W0]) == []


class TestGainerRanking:
    def test_ranks_gainers_by_magnitude_not_by_current_rank(self):
        # B ends the week at a BETTER absolute rank (3) than A (8), but A
        # moved FARTHER (22 -> 8, delta 14) than B (5 -> 3, delta 2). #561
        # is explicit: rank by what CHANGED. A "sort by curr_rank" bug would
        # put B first — this assertion fails under that bug.
        grid = _grid([
            ("A", "Defense Primes", W0, 22), ("A", "Defense Primes", W1, 8),
            ("B", "Neoclouds",      W0, 5),  ("B", "Neoclouds",      W1, 3),
        ])
        out = compute_weekly_movers(grid, [W0, W1])
        assert len(out) == 1
        gainers = out[0]["gainers"]
        assert [g["name"] for g in gainers] == ["Defense Primes", "Neoclouds"]
        assert gainers[0]["delta"] == 14
        assert gainers[0]["prev_rank"] == 22 and gainers[0]["curr_rank"] == 8

    def test_unchanged_rank_excluded_from_both_lists(self):
        # The exact failure mode being replaced: a cohort holding its spot
        # must not dominate (or even appear on) the surface.
        grid = _grid([("E", "Steady Cohort", W0, 10), ("E", "Steady Cohort", W1, 10)])
        out = compute_weekly_movers(grid, [W0, W1])
        assert out[0]["gainers"] == [] and out[0]["entrants"] == []

    def test_worsening_rank_on_board_is_not_a_gainer(self):
        grid = _grid([("F", "Slipping Cohort", W0, 3), ("F", "Slipping Cohort", W1, 9)])
        out = compute_weekly_movers(grid, [W0, W1])
        assert out[0]["gainers"] == [] and out[0]["entrants"] == []

    def test_exit_from_board_is_not_reported(self):
        # #561 asked for "the biggest rank GAINERS", not a full two-way
        # leaderboard — falling off the board is deliberately silent here.
        grid = _grid([("G", "Fading Cohort", W0, 10), ("G", "Fading Cohort", W1, 45)])
        out = compute_weekly_movers(grid, [W0, W1], board_size=30)
        assert out[0]["gainers"] == [] and out[0]["entrants"] == []


class TestEntrants:
    def test_entrant_from_unranked_and_from_outside_board(self):
        grid = _grid([
            ("C", "Satellite Comms", W1, 29),  # no W0 row at all -> unranked that week
            ("D", "Nylon & Chemicals", W0, 45), ("D", "Nylon & Chemicals", W1, 14),
        ])
        out = compute_weekly_movers(grid, [W0, W1], board_size=30)
        entrants = {e["name"]: e for e in out[0]["entrants"]}
        assert set(entrants) == {"Satellite Comms", "Nylon & Chemicals"}
        assert entrants["Nylon & Chemicals"]["curr_rank"] == 14
        assert entrants["Satellite Comms"]["curr_rank"] == 29
        # entrants carry no numeric delta — "outside" isn't a number to subtract from
        assert "delta" not in entrants["Satellite Comms"]
        assert "delta" not in entrants["Nylon & Chemicals"]
        # neither is double-counted as a "gainer"
        assert out[0]["gainers"] == []

    def test_entrants_sorted_by_current_rank_ascending(self):
        grid = _grid([
            ("X", "Later Entrant", W1, 25),
            ("Y", "Stronger Entrant", W1, 4),
        ])
        out = compute_weekly_movers(grid, [W0, W1])
        assert [e["name"] for e in out[0]["entrants"]] == ["Stronger Entrant", "Later Entrant"]


class TestMultiWeekAndTruncation:
    def test_newest_transition_first(self):
        grid = _grid([
            ("H", "Multi Week", W0, 20), ("H", "Multi Week", W1, 15), ("H", "Multi Week", W2, 5),
        ])
        out = compute_weekly_movers(grid, [W0, W1, W2])
        assert [e["week_start"] for e in out] == [W2, W1]
        assert [e["prev_week_start"] for e in out] == [W1, W0]

    def test_top_n_truncation_keeps_biggest_and_reports_total(self):
        rows = []
        for i in range(12):
            cid = f"T{i}"
            rows.append((cid, f"Theme {i}", W0, 30 - i))          # all start on the board (30..19)
            rows.append((cid, f"Theme {i}", W1, 29 - 2 * i))       # every cohort improves by a different amount
        grid = _grid(rows)
        out = compute_weekly_movers(grid, [W0, W1], top_n=5)
        gainers = out[0]["gainers"]
        assert len(gainers) == 5
        assert out[0]["gainers_total"] == 12
        deltas = [g["delta"] for g in gainers]
        assert deltas == sorted(deltas, reverse=True)
        # the single biggest mover in the whole set must survive truncation
        assert gainers[0]["name"] == "Theme 11"
