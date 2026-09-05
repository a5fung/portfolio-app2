"""Pin test for theme_canon.py — cohort identity (#315 R3; model rewrite #555/#553).

`TestCanonicalizeThemes` pins the MODEL on small synthetic fixtures. Fixture
names are load-bearing: the model's second axis is the theme DESCRIPTION
(two names must share a subject word for a rename to count), so every
fixture uses realistic theme names, not "Theme A" / "Theme B".

`TestRealSnapshot*` runs against the real committed `apollo_themes_snapshot.json`:
the three false merges the operator evidenced (#553) stay split ACROSS THE
FULL SERIES, the one true duplicate stays merged on every day it appears,
plus the newly found same-day absorb and the false non-merges the old
matcher produced. `test_grid_output_unchanged` is the hard constraint: the
Grid view's own dedup is byte-identical to the pre-#553 original.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from theme_canon import (
    _NON_SUBJECT_WORDS,
    _jaccard,
    _subject_words,
    canonicalize_themes,
    cohort_aliases,
)


def _df(rows: list[tuple[date, str, list[str]]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["theme_date", "name", "tickers"])


D0 = date(2026, 1, 5)  # a Monday, arbitrary anchor


def _d(offset: int) -> date:
    return D0 + timedelta(days=offset)


def _cids(out: pd.DataFrame, name: str) -> set[str]:
    return set(out.loc[out["name"] == name, "canonical_id"])


class TestHelpers:
    def test_jaccard(self):
        assert _jaccard(frozenset({"A"}), frozenset({"B"})) == 0.0
        assert _jaccard(frozenset(), frozenset({"A"})) == 0.0
        a, b = frozenset({"A", "B", "C"}), frozenset({"B", "C", "D"})
        assert _jaccard(a, b) == pytest.approx(0.5)

    def test_subject_words_drop_form_size_and_narrative_words(self):
        # "Services" / "Pure-Play" / "Recovery" describe the form, size and
        # story of a basket, not its subject — they never make two names agree.
        assert _subject_words("Space Launch & Orbital Services") == {"space", "launch", "orbital"}
        assert _subject_words("Pure-Play Hydraulic Fracturing & Completion Services") == {
            "hydraulic", "fracturing", "completion",
        }
        assert _subject_words("Crypto Recovery") == {"crypto"}
        assert "manufacturing" in _NON_SUBJECT_WORDS

    def test_subject_words_singularize_and_split_hyphens(self):
        assert _subject_words("Engineered Polymers") == _subject_words("Engineered Polymer")
        assert _subject_words("Gene-Editing Therapeutics") == _subject_words("Gene Editing Therapeutics")
        assert "gas" in _subject_words("Oil & Gas")   # 3-letter words are never stripped


class TestCanonicalizeThemes:
    def test_empty_input(self):
        out = canonicalize_themes(_df([]))
        assert out.empty
        assert "canonical_id" in out.columns

    def test_simple_rename_same_tickers_merges(self):
        # The core R3 case: identical membership, name churns day to day.
        df = _df([
            (_d(0), "U.S. Government/Defense Spending Surge", ["LMT", "NOC", "RTX"]),
            (_d(1), "U.S. Government/Defense Contract Surge", ["LMT", "NOC", "RTX"]),
        ])
        out = canonicalize_themes(df)
        assert out["canonical_id"].nunique() == 1
        assert out["canonical_name"].nunique() == 1

    def test_one_member_drift_still_merges(self):
        base = ["A", "B", "C", "D", "E", "F", "G", "H"]
        drifted = ["A", "B", "C", "D", "E", "F", "G", "I"]   # H -> I
        df = _df([
            (_d(0), "Optical Networking & Photonics Infrastructure", base),
            (_d(1), "Optical Networking & AI Data Transmission Infrastructure", drifted),
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

    def test_superset_under_unrelated_name_is_a_different_theme(self):
        # THE CRUX (#553 evidence #1, real geometry): an established 4-ticker
        # theme, then a 6-ticker superset under a name that shares no subject
        # word. Membership alone says "same basket, grown" (Jaccard 0.67);
        # the description says the newcomers are the point. Different theme.
        df = _df([
            (_d(0), "U.S. Defense Primes & Aerospace", ["GD", "LMT", "NOC", "RTX"]),
            (_d(1), "U.S. Defense Primes & Aerospace", ["GD", "LMT", "NOC", "RTX"]),
            (_d(2), "Satellite Mobile & IoT Connectivity Services",
             ["GD", "IRDM", "LMT", "NOC", "PL", "RTX"]),
        ])
        out = canonicalize_themes(df)
        assert out["canonical_id"].nunique() == 2

    def test_superset_under_compatible_name_is_a_rename(self):
        # Same geometry as above, but the new name still describes the old
        # members ("...Intermediates" both times) — a rename, one cohort.
        df = _df([
            (_d(0), "Niche Specialty Chemicals & Industrial Intermediates",
             ["ASIX", "CC", "CE", "LXU", "RYAM"]),
            (_d(1), "Niche Specialty Chemicals & Industrial Intermediates",
             ["ASIX", "CC", "CE", "LXU", "RYAM"]),
            (_d(2), "Nylon & Engineered Polymer Intermediates",
             ["ASIX", "CC", "CE", "DOW", "LXU", "LYB", "RYAM"]),
        ])
        out = canonicalize_themes(df)
        assert out["canonical_id"].nunique() == 1

    def test_identical_tiny_basket_under_unrelated_names_stays_split(self):
        # #553 evidence #2, real tickers: {ADEA, RYAM} was emitted as both an
        # ad-tech theme and a chemicals theme. An identical basket is NOT
        # enough when the descriptions contradict — that is the false merge.
        df = _df([
            (_d(0), "IP Licensing & Ad-Tech Royalty Software", ["ADEA", "RYAM"]),
            (_d(1), "Niche Specialty Chemicals & Industrial Intermediates", ["ADEA", "RYAM"]),
        ])
        out = canonicalize_themes(df)
        assert out["canonical_id"].nunique() == 2

    def test_identical_tiny_basket_under_compatible_names_merges(self):
        # ...whereas an identical 2-ticker basket whose names share a subject
        # word is one theme ("Cybersecurity Network Edge" / "Network Security
        # & Zero-Trust Edge" ran side by side for 31 days in the snapshot).
        df = _df([
            (_d(0), "Cybersecurity Network Edge & SD-WAN", ["FTNT", "PANW"]),
            (_d(0), "Network Security & Zero-Trust Edge", ["FTNT", "PANW"]),
            (_d(1), "Cybersecurity Network Edge & SD-WAN", ["FTNT", "PANW"]),
            (_d(1), "Network Security & Zero-Trust Edge", ["FTNT", "PANW"]),
        ])
        out = canonicalize_themes(df)
        assert out["canonical_id"].nunique() == 1

    def test_tiny_set_fully_inside_huge_set_does_not_merge(self):
        huge = [f"T{i}" for i in range(12)]
        small = huge[:2]   # fully inside `huge`, but only 2/12 of it
        df = _df([
            (_d(0), "Broad Energy Basket", huge),
            (_d(1), "Narrow Energy Sub Theme", small),
        ])
        out = canonicalize_themes(df)
        assert out["canonical_id"].nunique() == 2

    def test_one_ticker_reference_does_not_merge(self):
        # "Crypto Recovery" ({CRCL}) vs "CLO & Structured Credit Income"
        # ({CRCL, XFLT}): Jaccard 0.5 on one shared stock, and the names
        # share no subject word — unrelated themes that touch one stock.
        df = _df([
            (_d(0), "Crypto Recovery", ["CRCL"]),
            (_d(1), "CLO & Structured Credit Income", ["CRCL", "XFLT"]),
        ])
        out = canonicalize_themes(df)
        assert out["canonical_id"].nunique() == 2

    def test_chain_of_plausible_hops_cannot_walk_the_core(self):
        # #553 evidence #3 in miniature. Each hop clears 0.5 against the
        # PREVIOUS observation; the second does not against the majority
        # core, so the cohort stops there instead of walking away.
        df = _df([
            (_d(0), "Nitrogen Fertilizer Producers", ["A", "B", "C", "D", "E"]),
            (_d(1), "Nitrogen Fertilizer Producers", ["A", "B", "C", "D", "E"]),
            (_d(2), "Nitrogen & Crop Nutrient Producers", ["A", "B", "C", "D", "E", "F", "G"]),
            (_d(3), "Crop Nutrient & Agri-Chemical Producers", ["C", "D", "E", "F", "G", "H", "I"]),
        ])
        out = canonicalize_themes(df)
        assert _cids(out, "Nitrogen & Crop Nutrient Producers") == _cids(out, "Nitrogen Fertilizer Producers")
        assert _cids(out, "Crop Nutrient & Agri-Chemical Producers").isdisjoint(
            _cids(out, "Nitrogen Fertilizer Producers")
        )

    def test_exact_name_continues_through_low_overlap_day(self):
        # Under its own label a theme may widen or narrow around its core —
        # the engine's continuity claim is honored.
        df = _df([
            (_d(0), "Gold & Silver Miners", ["A", "B", "C", "D", "E"]),
            (_d(1), "Gold & Silver Miners", ["A", "F"]),   # only 1/5 shared
        ])
        out = canonicalize_themes(df)
        assert out["canonical_id"].nunique() == 1

    def test_label_reused_on_disjoint_basket_splits(self):
        # ...but a label on a basket sharing NOTHING with the theme's core is
        # a reused label ("Satellite Mobile" became four biotechs on 04-16).
        df = _df([
            (_d(0), "Satellite Mobile & IoT Connectivity Services", ["ASTS", "IRDM", "GSAT"]),
            (_d(1), "Satellite Mobile & IoT Connectivity Services", ["ASTS", "IRDM", "GSAT"]),
            (_d(2), "Satellite Mobile & IoT Connectivity Services", ["AVBP", "DAWN", "INKT", "RVMD"]),
        ])
        out = canonicalize_themes(df)
        assert out["canonical_id"].nunique() == 2

    def test_short_hijack_does_not_move_the_core(self):
        # The engine attaches a 20-ticker foreign blob to a 2-ticker label for
        # two days (the real "Independent Semiconductor Foundry" case). The
        # blob days ride along under the label, but two of six observations
        # are not a majority — the core stays {PLAB, TSEM}, so when another
        # name carries that blob on a later day it cannot join the foundry
        # cohort (its basket is nothing like the core), and the label's own
        # 2-set continues it. (A basket that persists for MORE than half a
        # cohort's life does become its core: the dashboard follows the
        # engine's slot — see the module docstring. And on a hijack day
        # itself, a second wording of the same emitted blob IS that day's
        # duplicate — see test_same_day_second_wording_of_todays_basket.)
        blob = [f"OPT{i}" for i in range(18)] + ["PLAB", "TSEM"]
        df = _df([
            (_d(0), "Independent Semiconductor Foundry & Specialty IC Manufacturing", ["PLAB", "TSEM"]),
            (_d(1), "Independent Semiconductor Foundry & Specialty IC Manufacturing", ["PLAB", "TSEM"]),
            (_d(2), "Independent Semiconductor Foundry & Specialty IC Manufacturing", ["PLAB", "TSEM"]),
            (_d(3), "Independent Semiconductor Foundry & Specialty IC Manufacturing", blob),
            (_d(4), "Independent Semiconductor Foundry & Specialty IC Manufacturing", blob),
            (_d(5), "Independent Semiconductor Foundry & Specialty IC Manufacturing", ["PLAB", "TSEM"]),
            (_d(5), "Compound Semiconductor & Specialty Photonic Materials", blob),
        ])
        out = canonicalize_themes(df)
        foundry = _cids(out, "Independent Semiconductor Foundry & Specialty IC Manufacturing")
        assert len(foundry) == 1
        assert foundry.isdisjoint(_cids(out, "Compound Semiconductor & Specialty Photonic Materials"))

    def test_oversized_glitch_row_does_not_bridge_cohorts(self):
        # A 57-ticker glitch day on a tiny theme rides along under its label
        # (shares its 2 tickers) but cannot pull in an unrelated mega basket.
        tiny_before = ["BETR", "WLTH"]
        glitch_day = [f"T{i}" for i in range(57)] + ["BETR", "WLTH"]
        tiny_after = ["BETR", "WLTH"]
        other_huge_theme = [f"T{i}" for i in range(50)]
        df = _df([
            (_d(0), "Robo-Advisor & AI-Driven Wealth Management Platforms", tiny_before),
            (_d(1), "Robo-Advisor & AI-Driven Wealth Management Platforms", glitch_day),
            (_d(2), "Robo-Advisor & AI-Driven Wealth Management Platforms", tiny_after),
            (_d(1), "Large-Cap Upstream Oil & Gas E&P", other_huge_theme),
        ])
        out = canonicalize_themes(df)
        robo_cids = _cids(out, "Robo-Advisor & AI-Driven Wealth Management Platforms")
        mega_cid = out.loc[out["name"] == "Large-Cap Upstream Oil & Gas E&P", "canonical_id"].iloc[0]
        assert len(robo_cids) == 1
        assert mega_cid not in robo_cids

    def test_gap_beyond_horizon_does_not_merge(self):
        df = _df([
            (_d(0), "Uranium Miners", ["A", "B", "C"]),
            (_d(30), "Uranium & Nuclear Fuel Miners", ["A", "B", "C"]),   # beyond max_gap_days (10)
        ])
        out = canonicalize_themes(df)
        assert out["canonical_id"].nunique() == 2

    def test_old_name_retiring_hands_off_to_its_own_alias(self):
        # A same-day subset with a compatible name becomes an alias; once the
        # old name stops being emitted the alias carries the SAME identity.
        df = _df([
            (_d(0), "Nitrogen Fertilizer & Ammonia Producers", ["A", "B", "C", "D", "E", "F"]),
            (_d(1), "Nitrogen Fertilizer & Ammonia Producers", ["A", "B", "C", "D", "E", "F"]),
            (_d(1), "Nitrogen Fertilizer Pure-Play Producers", ["A", "B", "C"]),
            (_d(2), "Nitrogen Fertilizer Pure-Play Producers", ["A", "B", "C"]),
        ])
        out = canonicalize_themes(df)
        assert out["canonical_id"].nunique() == 1

    def test_unrelated_theme_sharing_the_alias_day_is_not_pulled_in(self):
        df = _df([
            (_d(0), "Nitrogen Fertilizer & Ammonia Producers", ["A", "B", "C", "D", "E", "F"]),
            (_d(1), "Nitrogen Fertilizer & Ammonia Producers", ["A", "B", "C", "D", "E", "F"]),
            (_d(1), "Nitrogen Fertilizer Pure-Play Producers", ["A", "B", "C"]),
            (_d(1), "Ophthalmology Drug Developers", ["X", "Y", "Z"]),
        ])
        out = canonicalize_themes(df)
        assert _cids(out, "Nitrogen Fertilizer & Ammonia Producers").isdisjoint(
            _cids(out, "Ophthalmology Drug Developers")
        )

    def test_same_day_second_wording_of_todays_basket_is_an_alias(self):
        # A label that narrowed to 2 tickers for weeks (core = the 2) re-emits
        # its old 5-ticker basket under two wordings on one day: the second
        # wording matches today's representative row, not the core — still
        # one theme that day (the real Bitcoin-mining case, 2026-07-17).
        five = ["CIFR", "CORZ", "HUT", "IREN", "WULF"]
        rows = [(_d(0), "Bitcoin Mining & Crypto Infrastructure Operators", five)]
        rows += [(_d(i), "Bitcoin Mining & Crypto Infrastructure Operators", ["CIFR", "CORZ"]) for i in range(1, 6)]
        rows += [
            (_d(6), "Bitcoin Mining & Crypto Infrastructure Operators", five),
            (_d(6), "Bitcoin & Crypto Mining Infrastructure", five),
        ]
        out = canonicalize_themes(_df(rows))
        assert out["canonical_id"].nunique() == 1

    def test_empty_ticker_row_attaches_by_name(self):
        # The engine's Retired marker carries no tickers; it follows its name.
        df = _df([
            (_d(0), "U.S. Government/Defense Contract Surge", ["AMRC", "PLTR", "TSAT", "VOYG"]),
            (_d(0), "U.S. Government/Defense Spending Surge", ["AMRC", "PLTR", "TSAT", "VOYG"]),
            (_d(1), "U.S. Government/Defense Contract Surge", []),
            (_d(1), "U.S. Government/Defense Spending Surge", []),
        ])
        out = canonicalize_themes(df)
        assert out["canonical_id"].nunique() == 1

    def test_canonical_name_ignores_a_one_day_visitor(self):
        # Nine days of one name, then a single Retired-day visitor under a
        # compatible name: the label stays with the name that recurred.
        rows = [(_d(i), "Satellite Mobile & IoT Connectivity Services", ["GD", "IRDM", "LMT", "NOC", "PL", "RTX"]) for i in range(9)]
        rows.append((_d(9), "Satellite & IoT Connectivity Operators", ["IRDM", "LMT", "NOC", "PL", "RTX"]))
        out = canonicalize_themes(_df(rows))
        assert out["canonical_id"].nunique() == 1
        assert out["canonical_name"].iloc[0] == "Satellite Mobile & IoT Connectivity Services"

    def test_canonical_name_follows_a_real_rename(self):
        # ...but a rename that recurs takes over, so the label matches the
        # name the board shows today.
        rows = [(_d(i), "Packaged & Shelf-Stable Food Manufacturers", ["CAG", "CPB", "GIS", "KHC"]) for i in range(7)]
        rows += [(_d(7 + i), "Branded Packaged Food & Consumer Staples Producers", ["CPB", "GIS", "KHC"]) for i in range(3)]
        out = canonicalize_themes(_df(rows))
        assert out["canonical_id"].nunique() == 1
        assert out["canonical_name"].iloc[0] == "Branded Packaged Food & Consumer Staples Producers"


class TestCohortAliases:
    def test_only_multi_name_cohorts_listed(self):
        df = _df([
            (_d(0), "Uranium Miners", ["A", "B", "C"]),
            (_d(0), "Space Launch Services & Orbital Infrastructure", ["X", "Y", "Z"]),
            (_d(1), "Space Launch & Orbital Services", ["X", "Y", "Z"]),
        ])
        canon = canonicalize_themes(df)
        aliases = cohort_aliases(canon)
        assert len(aliases) == 1
        row = aliases.iloc[0]
        assert row["n_names"] == 2
        assert set(row["aliases"]) | {row["canonical_name"]} == {
            "Space Launch Services & Orbital Infrastructure", "Space Launch & Orbital Services",
        }

    def test_empty_when_no_merges(self):
        df = _df([(_d(0), "Uranium Miners", ["A", "B", "C"])])
        canon = canonicalize_themes(df)
        assert cohort_aliases(canon).empty


@pytest.fixture(scope="module")
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


@pytest.fixture(scope="module")
def canon(raw_themes):
    return canonicalize_themes(raw_themes)


class TestRealSnapshotSmoke:
    def test_runs_without_raising_and_covers_every_row(self, raw_themes, canon):
        assert len(canon) == len(raw_themes)
        assert canon["canonical_id"].notna().all()
        assert canon["canonical_name"].notna().all()

    def test_some_collapsing_happens(self, canon):
        assert not cohort_aliases(canon).empty


class TestRealSnapshotIdentity:
    """#553 DoD, measured across the FULL series (the 2026-08-10 verify
    showed the old matcher only fixed each case for part of its run)."""

    def test_satellite_mobile_never_shares_a_cohort_with_defense_primes(self, canon):
        sat = _cids(canon, "Satellite Mobile & IoT Connectivity Services")
        dp = _cids(canon, "U.S. Defense Primes & Aerospace")
        assert sat and dp and sat.isdisjoint(dp)
        # ...and the nine March-April rows are shown under their own name.
        early = canon[(canon["name"] == "Satellite Mobile & IoT Connectivity Services")
                      & (canon["theme_date"] <= date(2026, 4, 8))]
        assert (early["canonical_name"] == "Satellite Mobile & IoT Connectivity Services").all()

    def test_niche_specialty_chemicals_never_shares_a_cohort_with_ip_licensing(self, canon):
        chem = _cids(canon, "Niche Specialty Chemicals & Industrial Intermediates")
        assert chem.isdisjoint(_cids(canon, "IP Licensing & Ad-Tech Royalty Software"))
        assert chem.isdisjoint(_cids(canon, "IP Licensing & Patent Monetization Software Platforms"))

    def test_nylon_rows_never_share_a_cohort_with_agri_or_nitrogen(self, canon):
        nylon = _cids(canon, "Nylon & Engineered Polymer Intermediates")
        assert nylon.isdisjoint(_cids(canon, "Agricultural Commodities & Agri-Business"))
        assert nylon.isdisjoint(_cids(canon, "Nitrogen & Specialty Crop Nutrient Producers"))
        assert nylon.isdisjoint(_cids(canon, "Nitrogen Fertilizer & Ammonia Producers"))

    def test_defense_spending_and_contract_surge_merge_on_every_day(self, canon):
        both = canon[canon["name"].isin([
            "U.S. Government/Defense Spending Surge", "U.S. Government/Defense Contract Surge",
        ])]
        assert both["theme_date"].nunique() >= 2
        assert both["canonical_id"].nunique() == 1

    def test_same_day_blob_does_not_absorb_optical_networking(self, canon):
        # Found during #555: on 2026-04-06 a 17-ticker "Independent
        # Semiconductor Foundry" row swallowed "Optical Networking & AI Data
        # Transmission Infrastructure" by same-day containment.
        d = date(2026, 4, 6)
        opt = canon[(canon["name"] == "Optical Networking & AI Data Transmission Infrastructure") & (canon["theme_date"] == d)]
        fnd = canon[(canon["name"] == "Independent Semiconductor Foundry & Specialty IC Manufacturing") & (canon["theme_date"] == d)]
        assert not opt.empty and not fnd.empty
        assert opt["canonical_id"].iloc[0] != fnd["canonical_id"].iloc[0]

    def test_alternating_names_on_one_basket_are_one_cohort(self, canon):
        # The false NON-merges the old matcher produced (identical baskets,
        # alternating names, separate cohorts for weeks).
        for a, b in [
            ("Cybersecurity Network Edge & SD-WAN", "Network Security & Zero-Trust Edge"),
            ("Edge CDN & Cloud-Native Developer Platforms", "Edge Cloud & Developer CDN Platforms"),
            ("Domestic Steel Producers", "U.S. Domestic Steel Producers"),
        ]:
            assert _cids(canon, a) == _cids(canon, b), (a, b)

    @staticmethod
    def _dedup_themes_pre_553_oracle(theme_tickers, threshold=0.50, min_shared=3):
        # Verbatim re-derivation of dedup_themes as it existed BEFORE #553
        # (containment-only, Jaccard used only as a tie-break) — an
        # independent oracle, not a call into the function under test.
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
        # THE HARD CONSTRAINT: theme_grid.py's dedup call (threshold 0.50,
        # min_shared from its 0..6 slider) must produce byte-identical
        # parent_of output to the pre-#553 function across the FULL slider
        # range Grid exposes.
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
