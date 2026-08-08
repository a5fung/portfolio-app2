"""Pin test for theme_canon.py (#315 R3 — cross-day theme canonicalization).

Each `TestCanonicalizeThemes` case below is a synthetic, hand-built fixture
that reproduces a specific failure mode found by running earlier drafts of
`canonicalize_themes()` against the real `apollo_themes_snapshot.json` (see
that module's docstring for the full story + real theme names). Pinning them
as small fixtures here keeps the regression check fast and independent of the
live snapshot, which grows and drifts daily.

`TestLiveSnapshotSmoke` is the one test that touches the real committed
snapshot — a shape/sanity check (no exceptions, canonical ids collapse SOME
raw names, every row gets an id), not an exact-cohort pin, since the live
data changes every night.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from theme_canon import canonicalize_themes, cohort_aliases, _jaccard


def _df(rows: list[tuple[date, str, list[str]]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["theme_date", "name", "tickers"])


D0 = date(2026, 1, 5)  # a Monday, arbitrary anchor


def _d(offset: int) -> date:
    return D0 + timedelta(days=offset)


class TestJaccardHelper:
    def test_disjoint_is_zero(self):
        assert _jaccard(frozenset({"A"}), frozenset({"B"})) == 0.0

    def test_empty_is_zero(self):
        assert _jaccard(frozenset(), frozenset({"A"})) == 0.0

    def test_known_ratio(self):
        a, b = frozenset({"A", "B", "C"}), frozenset({"B", "C", "D"})
        assert _jaccard(a, b) == pytest.approx(0.5)   # 2 shared / 4 union


class TestCanonicalizeThemes:
    def test_empty_input(self):
        out = canonicalize_themes(_df([]))
        assert out.empty
        assert list(out.columns) >= []  # just must not raise

    def test_simple_rename_same_tickers_merges(self):
        # The core R3 case: identical membership, name churns day to day.
        df = _df([
            (_d(0), "U.S. Government/Defense Spending Surge", ["LMT", "NOC", "RTX"]),
            (_d(1), "U.S. Government/Defense Contract Surge", ["LMT", "NOC", "RTX"]),
        ])
        out = canonicalize_themes(df)
        assert out["canonical_id"].nunique() == 1
        assert out["canonical_name"].nunique() == 1
        # canonical_name is the freshest description, not the oldest.
        assert out["canonical_name"].iloc[0] == "U.S. Government/Defense Contract Surge"

    def test_one_member_drift_still_merges(self):
        # "gains or loses one member" (task spec) on an 8-ticker cohort — a
        # small fractional change, should NOT fragment the lineage.
        base = ["A", "B", "C", "D", "E", "F", "G", "H"]
        drifted = ["A", "B", "C", "D", "E", "F", "G", "I"]   # H -> I
        df = _df([
            (_d(0), "Theme Alpha", base),
            (_d(1), "Theme Alpha Redux", drifted),
        ])
        out = canonicalize_themes(df)
        assert out["canonical_id"].nunique() == 1

    def test_unrelated_themes_do_not_merge(self):
        df = _df([
            (_d(0), "Cybersecurity Endpoint Defense", ["CRWD", "PANW", "ZS"]),
            (_d(1), "Nuclear Power Restart Plays", ["CEG", "VST", "TLN"]),
        ])
        out = canonicalize_themes(df)
        assert out["canonical_id"].nunique() == 2

    def test_tiny_set_fully_inside_huge_set_does_not_merge(self):
        # Regression: containment-style scoring over-merges a small theme
        # into an unrelated large one. Real example: "Oil & Gas" (6 tickers)
        # fully contained 3-ticker sub-themes it had nothing to do with.
        huge = [f"T{i}" for i in range(12)]
        small = huge[:2]   # fully inside `huge`, but only 2/12 of it
        df = _df([
            (_d(0), "Broad Basket Theme", huge),
            (_d(1), "Narrow Sub Theme", small),
        ])
        out = canonicalize_themes(df)
        assert out["canonical_id"].nunique() == 2

    def test_one_ticker_reference_does_not_merge(self):
        # Regression: a 1-ticker theme is not a real signature — any 2-ticker
        # set sharing that ticker clears Jaccard 0.5 trivially. Real example:
        # "Crypto Recovery" ({CRCL}) vs "CLO & Structured Credit Income"
        # ({CRCL, XFLT}).
        df = _df([
            (_d(0), "Solo Ticker Theme", ["CRCL"]),
            (_d(1), "Two Ticker Theme", ["CRCL", "XFLT"]),
        ])
        out = canonicalize_themes(df)
        assert out["canonical_id"].nunique() == 2

    def test_oversized_glitch_row_does_not_bridge_cohorts(self):
        # Regression: an upstream data glitch briefly attaches a near-
        # universe-wide basket to a normally-tiny theme for one day. That day
        # must not let the tiny theme bridge to an unrelated large theme via
        # ticker overlap on the huge sets.
        tiny_before = ["BETR", "WLTH"]
        glitch_day = [f"T{i}" for i in range(57)] + ["BETR", "WLTH"]
        tiny_after = ["BETR", "WLTH"]
        other_huge_theme = [f"T{i}" for i in range(50)]  # overlaps glitch_day heavily
        df = _df([
            (_d(0), "Robo-Advisor Platforms", tiny_before),
            (_d(1), "Robo-Advisor Platforms", glitch_day),
            (_d(2), "Robo-Advisor Platforms", tiny_after),
            (_d(1), "Unrelated Mega Basket", other_huge_theme),
        ])
        out = canonicalize_themes(df)
        robo_cids = set(out.loc[out["name"] == "Robo-Advisor Platforms", "canonical_id"])
        mega_cid = out.loc[out["name"] == "Unrelated Mega Basket", "canonical_id"].iloc[0]
        # The exact-name rows all stay one cohort (name continuation)...
        assert len(robo_cids) == 1
        # ...but that cohort must never equal the unrelated mega-basket's id.
        assert mega_cid not in robo_cids

    def test_exact_name_continues_through_low_overlap_day(self):
        # A theme keeping its EXACT name should stay one cohort even when a
        # single day's ticker overlap with the prior day would, on its own,
        # fall short of the overlap threshold (membership can move a lot
        # under a stable label — the name itself is the stronger signal).
        df = _df([
            (_d(0), "Same Name Theme", ["A", "B", "C", "D", "E"]),
            (_d(1), "Same Name Theme", ["A", "F"]),   # only 1/5 shared -> low Jaccard
        ])
        out = canonicalize_themes(df)
        assert out["canonical_id"].nunique() == 1

    def test_gap_beyond_horizon_does_not_merge(self):
        df = _df([
            (_d(0), "Old Name", ["A", "B", "C"]),
            (_d(30), "New Name", ["A", "B", "C"]),   # far beyond max_gap_days default (10)
        ])
        out = canonicalize_themes(df)
        assert out["canonical_id"].nunique() == 2

    def test_old_name_retiring_hands_off_to_its_own_alias(self):
        # A same-day near-duplicate ("New Name" is a subset of "Old Name")
        # correctly becomes a same-day ALIAS of "Old Name" (the existing
        # dedup_themes mechanism, unchanged) — that is not a hijack, it is
        # the intra-day dedup this module deliberately reuses. The real test
        # of the cross-day structural guard is what happens once "Old Name"
        # stops being emitted: "New Name" (already tied to the cohort via
        # its alias day) should carry the SAME identity forward, not fork
        # into a second cohort.
        df = _df([
            (_d(0), "Old Name", ["A", "B", "C", "D", "E", "F"]),
            (_d(1), "Old Name", ["A", "B", "C", "D", "E", "F"]),
            (_d(1), "New Name", ["A", "B", "C"]),   # subset -> same-day alias of Old Name
            (_d(2), "New Name", ["A", "B", "C"]),   # Old Name retires; New Name carries on
        ])
        out = canonicalize_themes(df)
        assert out["canonical_id"].nunique() == 1

    def test_unrelated_theme_sharing_the_alias_day_is_not_pulled_in(self):
        # A THIRD, genuinely unrelated theme present on the same day as the
        # Old/New handoff must not get swept into that cohort just because
        # it co-occurred that day.
        df = _df([
            (_d(0), "Old Name", ["A", "B", "C", "D", "E", "F"]),
            (_d(1), "Old Name", ["A", "B", "C", "D", "E", "F"]),
            (_d(1), "New Name", ["A", "B", "C"]),
            (_d(1), "Unrelated Theme", ["X", "Y", "Z"]),
        ])
        out = canonicalize_themes(df)
        old_cid = out.loc[out["name"] == "Old Name", "canonical_id"].iloc[0]
        unrelated_cid = out.loc[out["name"] == "Unrelated Theme", "canonical_id"].iloc[0]
        assert old_cid != unrelated_cid


class TestCohortAliases:
    def test_only_multi_name_cohorts_listed(self):
        df = _df([
            (_d(0), "Solo Theme", ["A", "B", "C"]),
            (_d(0), "Renamed A", ["X", "Y", "Z"]),
            (_d(1), "Renamed B", ["X", "Y", "Z"]),
        ])
        canon = canonicalize_themes(df)
        aliases = cohort_aliases(canon)
        assert len(aliases) == 1
        row = aliases.iloc[0]
        assert row["canonical_name"] == "Renamed B"
        assert row["aliases"] == ["Renamed A"]
        assert row["n_names"] == 2

    def test_empty_when_no_merges(self):
        df = _df([(_d(0), "Solo Theme", ["A", "B", "C"])])
        canon = canonicalize_themes(df)
        assert cohort_aliases(canon).empty


class TestLiveSnapshotSmoke:
    """Sanity check against the real committed snapshot — shape only, not an
    exact-cohort pin (the live data grows/drifts every night)."""

    @pytest.fixture
    def raw_themes(self):
        import json
        import os
        path = os.path.join(os.path.dirname(__file__), "apollo_themes_snapshot.json")
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        df = pd.DataFrame(raw["themes"])
        df["theme_date"] = pd.to_datetime(df["theme_date"]).dt.date
        df["tickers"] = df["tickers"].apply(lambda t: list(t) if t else [])
        return df

    def test_runs_without_raising_and_covers_every_row(self, raw_themes):
        out = canonicalize_themes(raw_themes)
        assert len(out) == len(raw_themes)
        assert out["canonical_id"].notna().all()
        assert out["canonical_name"].notna().all()

    def test_some_collapsing_happens(self, raw_themes):
        # The whole point of R3: at least some cohorts wore more than one raw
        # name. NOTE: nunique(canonical_id) is NOT guaranteed to be less than
        # nunique(name) overall — a single raw name can also independently
        # recur as TWO unrelated cohorts months apart (the engine reusing a
        # generic phrase for a different basket later), which is correct
        # fragmentation, not a bug. The real "did canonicalization do
        # anything" check is: some cohort merged >1 distinct name.
        out = canonicalize_themes(raw_themes)
        assert not cohort_aliases(out).empty
