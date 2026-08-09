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


@pytest.fixture
def raw_themes():
    import json
    import os
    path = os.path.join(os.path.dirname(__file__), "apollo_themes_snapshot.json")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    df = pd.DataFrame(raw["themes"])
    df["theme_date"] = pd.to_datetime(df["theme_date"]).dt.date
    df["tickers"] = df["tickers"].apply(lambda t: list(t) if t else [])
    return df


class TestLiveSnapshotSmoke:
    """Sanity check against the real committed snapshot — shape only, not an
    exact-cohort pin (the live data grows/drifts every night)."""

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


class TestRealSnapshotOverMergeRegressions:
    """#553 — cohort matching glued unrelated themes together. Pins the 3
    confirmed-bad merges as separated and the 1 real duplicate as still
    merged, using REAL ticker sets from the committed snapshot (not toy
    fixtures) — found by instrumenting every merge decision against the live
    data, see theme_canon.py's module docstring "#553 fix" section for the
    full mechanism-by-mechanism story."""

    @staticmethod
    def _cids(out: pd.DataFrame, name: str) -> set[str]:
        return set(out.loc[out["name"] == name, "canonical_id"])

    def test_satellite_mobile_does_not_absorb_defense_primes(self, raw_themes):
        # Bug: "Satellite Mobile & IoT Connectivity Services" absorbed "U.S.
        # Defense Primes & Aerospace" ({GD, LMT, NOC, RTX}) via Tier 2 on
        # first contact at Jaccard 0.667 (2026-03-25) — a brand-new name
        # grabbing a cohort it had never touched, on partial overlap alone.
        # The ORIGINAL 4-ticker capture (2026-03-19..03-24, before the
        # hijack) must now stay its own identity for its whole run.
        out = canonicalize_themes(raw_themes)
        dp_original = out[
            (out["name"] == "U.S. Defense Primes & Aerospace")
            & (out["theme_date"] <= date(2026, 3, 24))
        ]
        assert not dp_original.empty
        dp_cids = set(dp_original["canonical_id"])
        sat_cids = self._cids(out, "Satellite Mobile & IoT Connectivity Services")
        assert dp_cids.isdisjoint(sat_cids), (
            f"Defense Primes {dp_cids} and Satellite Mobile {sat_cids} still share an id"
        )

    def test_niche_specialty_chemicals_does_not_fuse_with_ip_licensing(self, raw_themes):
        # Bug: "Niche Specialty Chemicals & Industrial Intermediates" and two
        # IP-licensing/ad-tech names both reduced to the identical 2-ticker
        # {ADEA, RYAM} set at different points and merged via Tier 2's
        # min_shared relaxation (2/2 "full" match on a signature too small to
        # trust — same shape as the already-guarded 1-ticker CRCL/XFLT case).
        out = canonicalize_themes(raw_themes)
        chem_cids = self._cids(out, "Niche Specialty Chemicals & Industrial Intermediates")
        ip1_cids = self._cids(out, "IP Licensing & Ad-Tech Royalty Software")
        ip2_cids = self._cids(
            out, "IP Licensing & Patent Monetization Software Platforms"
        )
        assert chem_cids.isdisjoint(ip1_cids)
        assert chem_cids.isdisjoint(ip2_cids)

    def test_nitrogen_chemicals_nylon_chain_does_not_collapse_to_one_identity(
        self, raw_themes
    ):
        # Bug: "Nitrogen & Specialty Crop Nutrient Producers" + "Niche
        # Specialty Chemicals & Industrial Intermediates" + "Nylon &
        # Engineered Polymer Intermediates" chain-merged onto ONE canonical
        # id — each day's Tier 2 hop looked individually plausible (Jaccard
        # 0.7-1.0 against whatever the cohort currently held) while the
        # cumulative walk drifted from a 5-ticker specialty-chemicals cohort
        # to a 15-ticker nitrogen-fertilizer/agri-business basket sharing
        # only one ticker with where it started. The anchor-set check (Fix D)
        # must stop all three names from EVER sharing a single canonical_id.
        out = canonicalize_themes(raw_themes)
        nitro_cids = self._cids(out, "Nitrogen & Specialty Crop Nutrient Producers")
        chem_cids = self._cids(out, "Niche Specialty Chemicals & Industrial Intermediates")
        nylon_cids = self._cids(out, "Nylon & Engineered Polymer Intermediates")
        assert not (nitro_cids & chem_cids & nylon_cids), (
            "all three raw names still share at least one canonical_id"
        )
        # Strongest, cleanest separation the fix actually achieves: Nitrogen
        # (the fertilizer/agri cluster) never touches the chemicals cluster
        # at all, in either of its two names.
        assert nitro_cids.isdisjoint(chem_cids)

    def test_defense_spending_and_contract_surge_still_merge(self, raw_themes):
        # The REAL duplicate the whole feature exists to catch — must
        # survive every #553 guard. Both rows are the SAME day (2026-08-04),
        # identical 4-ticker set {AMRC, PLTR, TSAT, VOYG} — an intra-day
        # dedup_themes merge (Jaccard 1.0), never touched by any Tier 2
        # guard, so it should be untouched by this fix by construction.
        out = canonicalize_themes(raw_themes)
        d0 = date(2026, 8, 4)
        spending = out[
            (out["name"] == "U.S. Government/Defense Spending Surge")
            & (out["theme_date"] == d0)
        ]
        contract = out[
            (out["name"] == "U.S. Government/Defense Contract Surge")
            & (out["theme_date"] == d0)
        ]
        assert not spending.empty and not contract.empty
        assert spending["canonical_id"].iloc[0] == contract["canonical_id"].iloc[0]

    @staticmethod
    def _dedup_themes_pre_553_oracle(theme_tickers, threshold=0.50, min_shared=3):
        # Verbatim re-derivation of dedup_themes as it existed BEFORE #553
        # (containment-only, Jaccard used only as a tie-break — no floor) —
        # an independent oracle, not a call into the function under test, so
        # this test can't pass by both sides sharing a bug.
        if not theme_tickers:
            return {}
        by_size = sorted(theme_tickers.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        parent_of = {name: name for name, _ in by_size}
        sets = {name: set(tickers) for name, tickers in by_size}
        for i, (s_name, _s_tickers) in enumerate(by_size):
            s_set = sets[s_name]
            if not s_set:
                continue
            candidates = []
            for j in range(i):
                l_name = by_size[j][0]
                if parent_of[l_name] != l_name:
                    continue
                l_set = sets[l_name]
                shared = len(s_set & l_set)
                if shared < min_shared:
                    continue
                if shared / len(s_set) >= threshold:
                    jaccard = shared / len(s_set | l_set)
                    candidates.append((l_name, jaccard))
            if candidates:
                parent_of[s_name] = max(candidates, key=lambda kv: kv[1])[0]
        return parent_of

    def test_grid_output_unchanged(self, raw_themes):
        # THE HARD CONSTRAINT (#553): theme_grid.py's dedup call (threshold
        # 0.50, min_shared from its 0..6 slider, NO jaccard_floor) must
        # produce byte-identical parent_of output to the pre-#553 function,
        # across the FULL slider range Grid exposes — not just the default.
        from theme_data import dedup_themes

        latest = (
            raw_themes.sort_values(["name", "theme_date"])
            .drop_duplicates("name", keep="last")
        )
        theme_tickers = {
            row["name"]: tuple(row["tickers"])
            for _, row in latest.iterrows()
            if row["tickers"]
        }
        for min_shared in range(0, 7):
            got = dedup_themes(theme_tickers, threshold=0.50, min_shared=min_shared)
            want = self._dedup_themes_pre_553_oracle(
                theme_tickers, threshold=0.50, min_shared=min_shared
            )
            assert got == want, f"min_shared={min_shared} diverged from pre-#553 behavior"
