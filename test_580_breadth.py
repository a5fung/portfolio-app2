"""Pin tests for #580 — the Ecosystems ranking matches /themes, and each row says
how broad the theme's strength is.

Three things are pinned:

  1. ORDER: the latest row per theme name is taken FIRST, then Retired names
     are dropped (Apollo db.get_active_themes, #214). Dropping Retired first
     resurrects a theme retired on a later date from its older non-Retired
     row (12 ghost themes on 2026-09-30, one of them ranked #7).
  2. TIES: equal scores sort by name, so the order is deterministic and equals
     /themes.
  3. BREADTH: "3 of 5 above 20-day avg" beside the score; nothing when the
     breadth is missing; never a made-up count.

The tiny-frame tests prove the ORDER logic on a fixture that the OLD order
demonstrably fails. The snapshot tests recompute the page's own function
against the committed snapshot (CLAUDE.md "Verifying a dashboard change"),
with "today" pinned to the day after the snapshot so they do not rot as the
calendar moves.
"""
from __future__ import annotations

import json
from datetime import date, timedelta

import pandas as pd
import pytest
import streamlit as st

import theme_data as td
from ecosystem_score import trimmed_mean
from theme_ecosystem_view import ECOSYSTEMS_RANK_LABEL, _breadth_text
from theme_grid import GRID_RANK_LABEL

D0 = date(2026, 9, 22)
D1 = date(2026, 9, 29)
CUTOFF = date(2026, 9, 20)


def _frame(rows: list[tuple[str, date, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["name", "theme_date", "stage"])


class TestLatestThenDropRetired:
    def _fixture(self) -> pd.DataFrame:
        return _frame([
            # retired on a LATER date than an older live row -> must vanish
            ("Ghost", D0, "Mainstream"), ("Ghost", D1, "Retired"),
            # plain live theme
            ("Live", D1, "Nascent"),
            # retired earlier, live again later -> latest row is live, keep
            ("Revived", D0, "Retired"), ("Revived", D1, "Accelerating"),
            # fading is NOT retired; the board still needs it
            ("Fader", D1, "Fading"),
            # only row is outside the recency window
            ("Stale", date(2026, 9, 1), "Mainstream"),
        ])

    def test_theme_retired_on_a_later_date_is_dropped(self):
        out = td._latest_active_rows(self._fixture(), CUTOFF)
        assert "Ghost" not in set(out["name"])

    def test_keeps_live_revived_and_fading_on_their_latest_row(self):
        out = td._latest_active_rows(self._fixture(), CUTOFF).set_index("name")
        assert set(out.index) == {"Live", "Revived", "Fader"}
        assert out.loc["Revived", "stage"] == "Accelerating"
        assert out.loc["Revived", "theme_date"] == D1
        assert out.loc["Fader", "stage"] == "Fading"

    def test_the_old_order_would_have_kept_the_ghost(self):
        # Proves the fixture bites: the pre-#580 order (drop Retired, THEN take
        # the latest row) returns Ghost's older Mainstream row.
        df = self._fixture()
        old = df[(df["theme_date"] >= CUTOFF) & (df["stage"] != "Retired")]
        old = old.sort_values("theme_date").groupby("name").tail(1)
        assert "Ghost" in set(old["name"])

    def test_empty_window_returns_empty(self):
        out = td._latest_active_rows(self._fixture(), date(2026, 12, 1))
        assert out.empty

    def test_all_retired_returns_empty(self):
        df = _frame([("A", D1, "Retired"), ("B", D1, "Retired")])
        assert td._latest_active_rows(df, CUTOFF).empty


class TestBreadthText:
    def test_whole_count_in_plain_words(self):
        assert _breadth_text(60.0, 5) == "3 of 5 above 20-day avg"
        assert _breadth_text(100.0, 4) == "4 of 4 above 20-day avg"
        assert _breadth_text(0.0, 5) == "0 of 5 above 20-day avg"

    def test_survives_the_stored_three_decimal_rounding(self):
        # the engine stores 0.667 for 2 of 3 (and 0.333 for 1 of 3)
        assert _breadth_text(66.7, 3) == "2 of 3 above 20-day avg"
        assert _breadth_text(33.3, 3) == "1 of 3 above 20-day avg"

    def test_missing_breadth_shows_nothing(self):
        assert _breadth_text(None, 5) == ""
        assert _breadth_text(float("nan"), 5) == ""

    def test_never_invents_a_count_that_does_not_reconcile(self):
        # Real row, 2026-09-29: "Bitcoin Balance Sheet Proxies" lists 2 names
        # but stores 0.667 (2 of 3). "1 of 2" would be false, so state the
        # percentage and the member count instead.
        assert _breadth_text(66.7, 2) == "67% above 20-day avg (2 names)"
        # and a member count of zero can only give the percentage
        assert _breadth_text(50.0, 0) == "50% above 20-day avg"
        assert _breadth_text(50.0, None) == "50% above 20-day avg"


# ── Real committed snapshot ──────────────────────────────────────────────────


@pytest.fixture
def snapshot_day(monkeypatch):
    """Pin theme_data's 'today' to the day after the snapshot's newest row, so
    the 7-day window contains the same rows a live viewer sees the morning after
    an export — regardless of when the suite runs. Clears Streamlit's data cache
    around the test so the pinned date never leaks in or out."""
    st.cache_data.clear()
    newest = max(td._load()["themes"]["theme_date"])
    today = newest + timedelta(days=1)

    class _PinnedDate(date):
        @classmethod
        def today(cls):
            return today

    monkeypatch.setattr(td, "date", _PinnedDate)
    st.cache_data.clear()
    yield today
    st.cache_data.clear()


def _flat_scored(board: dict) -> list[dict]:
    flat = [t for group in board["active_by_eco"].values() for t in group]
    return sorted(flat, key=lambda t: board["global_rank"][t["name"]])


class TestRealSnapshotRanking:
    def test_no_ghost_theme_is_ranked(self, snapshot_day):
        board = td.get_ecosystem_board()
        assert board, "board empty against the committed snapshot"
        df = td._load()["themes"]
        window = df[df["theme_date"] >= snapshot_day - timedelta(days=7)]
        latest = window.sort_values(["name", "theme_date"]).drop_duplicates("name", keep="last")
        ghosts = set(latest.loc[latest["stage"] == "Retired", "name"])
        if not ghosts:
            pytest.skip("no theme is Retired in this snapshot's window; the tiny-frame tests still pin the order")
        ranked = set(board["global_rank"])
        fading = {t["name"] for grp in board["fading_by_eco"].values() for t in grp}
        assert not (ghosts & (ranked | fading)), sorted(ghosts & (ranked | fading))

    def test_order_is_score_desc_then_name(self, snapshot_day):
        board = td.get_ecosystem_board()
        flat = _flat_scored(board)
        keys = [(-t["comp"], t["name"]) for t in flat]
        assert keys == sorted(keys)
        assert sorted(board["global_rank"].values()) == list(range(1, len(flat) + 1))

    def test_ranking_equals_an_independent_recompute_from_the_raw_snapshot(self, snapshot_day):
        # The /themes rule, written out from the raw JSON without theme_data:
        # latest row per name in the window, drop Retired, skip Fading, score =
        # trimmed mean of current members' rs_composite, sort (-score, name).
        with open(td._SNAPSHOT_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        cutoff = snapshot_day - timedelta(days=7)
        latest: dict[str, tuple] = {}
        for r in raw["themes"]:
            d = date.fromisoformat(r["theme_date"])
            if d >= cutoff and (r["name"] not in latest or d >= latest[r["name"]][0]):
                latest[r["name"]] = (d, r)
        rs = {s["ticker"]: s["rs_composite"] for s in raw["stock_scores"]
              if s["rs_composite"] is not None}
        expect = []
        for name, (_, r) in latest.items():
            if r["stage"] in ("Retired", "Fading"):
                continue
            comps = [rs[t] for t in (r["tickers"] or []) if t in rs]
            if comps:
                expect.append((-trimmed_mean(comps), name))
        expect_names = [n for _, n in sorted(expect)]
        got = sorted(td.get_ecosystem_board()["global_rank"],
                     key=td.get_ecosystem_board()["global_rank"].get)
        assert got[:20] == expect_names[:20]
        assert got == expect_names


class TestRealSnapshotBreadth:
    def test_every_row_carries_breadth_and_member_count(self, snapshot_day):
        flat = _flat_scored(td.get_ecosystem_board())
        assert flat
        for t in flat:
            assert "breadth" in t and "n_members" in t
            assert t["n_members"] == len(t["tickers"])
            assert t["breadth"] is None or 0.0 <= t["breadth"] <= 100.0
        assert any(t["breadth"] is not None for t in flat), "no theme carried a breadth"

    def test_rendered_text_is_a_true_statement_about_the_row(self, snapshot_day):
        flat = _flat_scored(td.get_ecosystem_board())
        for t in flat:
            text = _breadth_text(t["breadth"], t["n_members"])
            if t["breadth"] is None:
                assert text == ""
                continue
            assert text.endswith("above 20-day avg") or text.endswith("names)")
            if " of " in text:
                k, n = (int(x) for x in text.split(" above")[0].split(" of "))
                assert n == t["n_members"] and 0 <= k <= n
                assert abs(k / n * 100 - t["breadth"]) < 5.0


# ── The pages say which question each ranking answers ────────────────────────

_PAGE = "pages/Apollo_Themes.py"


def _view_text(view: str) -> tuple[str, str]:
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(_PAGE, default_timeout=90)
    at.run()
    at.sidebar.radio[0].set_value(view).run()
    assert not at.exception, f"{view!r} raised: {list(at.exception)}"
    return (" ".join(m.value for m in at.markdown),
            " ".join(c.value for c in at.caption))


class TestPagesLabelTheirRanking:
    def test_ecosystems_says_it_ranks_like_themes_and_shows_breadth(self, snapshot_day):
        md, captions = _view_text("Ecosystems")
        assert ECOSYSTEMS_RANK_LABEL in captions
        assert "above 20-day avg" in md, "no ranking row showed breadth"
        assert "Corporate Digital Asset Treasury Vehicles" not in md, "a retired ghost theme rendered"

    def test_grid_says_it_is_the_stored_weekly_score(self, snapshot_day):
        _, captions = _view_text("Grid")
        assert GRID_RANK_LABEL in captions

    def test_the_two_labels_are_the_operators_words(self):
        assert ECOSYSTEMS_RANK_LABEL == "Ranked the same way as /themes (live RS of current members)"
        assert GRID_RANK_LABEL == "Weekly history: the engine's stored score"
