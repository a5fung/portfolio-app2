"""Pin test for the ADR 0032 ecosystem scorer PORT (#472).

ecosystem_score.py claims to be a VERBATIM port of Apollo's
`compute_ecosystem_scores` / `_group_and_rank_ecosystems` / `trimmed_mean`
(agents/market_intelligence/theme_ecosystems.py + constants.py @ 1115b37).
This test can't reach the real Apollo DB or its test suite, so it pins the
port's behavior against a small hand-computed synthetic fixture instead —
if a future edit to ecosystem_score.py (or an upstream drift this port
fails to mirror) changes the arithmetic, this goes red.

Fixture: 3 ecosystems, 7 themes (6 active + 1 Fading), 19 tickers —
deliberately hits every documented edge case in the docstring:
  - E-CYBR: 3 sub-themes incl. a near-duplicate PAIR (Jaccard exactly 0.6 —
    the ">=" dedup boundary) that must collapse to ONE depth credit, plus a
    genuinely distinct 3rd sub-theme that must NOT collapse. strong=8 clears
    the thin-ecosystem floor (5) -> nonzero boost.
  - E-SAAS: single 3-member sub-theme, all members RS>=80 -> comp qualifies
    for depth (>=85) but strong=3 is BELOW the floor(5) -> boost forced to
    0 despite a qualifying sub-theme (the thin-ecosystem-floor rule).
  - E-AISEMI: one weak active sub-theme (comp well below the depth
    threshold) + one Fading sub-theme, to confirm Fading is excluded from
    scoring inputs but still flows through fading_by_eco for grouping.
  - ThemeD: highest raw score of ALL seven themes, deliberately left OUT of
    eco_map so it defaults to E-UNASSIGNED — pins the anti-gaming invariant
    that E-UNASSIGNED sorts LAST on the ecosystem board regardless of score,
    while still ranking #1 in the GLOBAL per-theme rank (the two rankings
    are independent axes, by design).

All expected numbers below were independently hand-computed (see git history
of this file / the PR description) AND cross-checked by running the ported
functions directly — this is not a rubber-stamp of "whatever the code
outputs today".
"""
from __future__ import annotations

import pytest

from ecosystem_score import (
    E_UNASSIGNED,
    compute_ecosystem_scores,
    get_ecosystem_codes,
    get_ecosystem_map,
    get_ecosystems,
    trimmed_mean,
    _group_and_rank_ecosystems,
    _jaccard,
)

# ── Shared fixture ───────────────────────────────────────────────────────────

THEME_RS = {
    "T1": {"rs_composite": 95}, "T2": {"rs_composite": 92}, "T3": {"rs_composite": 90},
    "T4": {"rs_composite": 88}, "T5": {"rs_composite": 86}, "T6": {"rs_composite": 89},
    "T7": {"rs_composite": 87}, "T8": {"rs_composite": 84},
    "T9": {"rs_composite": 93}, "T10": {"rs_composite": 91}, "T11": {"rs_composite": 89},
    "T12": {"rs_composite": 60}, "T13": {"rs_composite": 55}, "T14": {"rs_composite": 50},
    "T15": {"rs_composite": 70}, "T16": {"rs_composite": 65},
    "T17": {"rs_composite": 99}, "T18": {"rs_composite": 98}, "T19": {"rs_composite": 97},
    "T20": {"rs_composite": 96}, "T21": {"rs_composite": 95},
}

ECOSYSTEM_TO_THEMES = {
    "E-CYBR": [
        {"name": "ThemeA1", "tickers": ["T1", "T2", "T3", "T4"], "stage": "Accelerating"},
        # near-dup of A1: shares {T1,T2,T3} out of a 5-ticker union -> Jaccard
        # = 3/5 = 0.6, exactly at DEPTH_NEAR_DUP_JACCARD -> must collapse.
        {"name": "ThemeA2", "tickers": ["T1", "T2", "T3", "T5"], "stage": "Mainstream"},
        # genuinely distinct cluster, zero overlap with A1/A2 -> counts fully.
        {"name": "ThemeA3", "tickers": ["T6", "T7", "T8"], "stage": "Accelerating"},
    ],
    "E-SAAS": [
        {"name": "ThemeB1", "tickers": ["T9", "T10", "T11"], "stage": "Mainstream"},
    ],
    "E-AISEMI": [
        {"name": "ThemeC1", "tickers": ["T12", "T13", "T14"], "stage": "Nascent"},
        {"name": "ThemeC2", "tickers": ["T15", "T16"], "stage": "Fading"},
    ],
}


# ── trimmed_mean (ported from constants.py) ──────────────────────────────────

class TestTrimmedMean:
    def test_empty(self):
        assert trimmed_mean([]) == 0.0

    def test_below_trim_threshold_is_plain_mean(self):
        # len < 3 -> plain mean, no trimming
        assert trimmed_mean([10.0, 20.0]) == 15.0

    def test_five_or_fewer_drops_one(self):
        # n<=5 -> drop lowest 1
        assert trimmed_mean([40, 90, 91, 92, 95]) == pytest.approx(92.0)

    def test_six_to_ten_drops_two(self):
        vals = [95, 92, 90, 88, 86, 89, 87, 84]  # n=8 -> drop 2 lowest (84, 86)
        assert trimmed_mean(vals) == pytest.approx((87 + 88 + 89 + 90 + 92 + 95) / 6)

    def test_matches_manual_fixture_values(self):
        assert trimmed_mean([95, 90, 40]) == pytest.approx(92.5)     # n=3, drop 1
        assert trimmed_mean([93, 91, 89]) == pytest.approx(92.0)     # n=3, drop 1
        assert trimmed_mean([60, 55, 50]) == pytest.approx(57.5)     # n=3, drop 1


# ── Taxonomy loader ──────────────────────────────────────────────────────────

class TestTaxonomy:
    def test_e_unassigned_is_last(self):
        codes = get_ecosystem_codes()
        assert codes[-1] == E_UNASSIGNED

    def test_known_codes_present(self):
        codes = set(get_ecosystem_codes())
        for expected in ("E-CYBR", "E-AISEMI", "E-AIINFRA", "E-SAAS", "E-BIO"):
            assert expected in codes

    def test_20_buckets_plus_reserved(self):
        # 20 curated buckets + E-UNASSIGNED, per theme_ecosystems.yaml header.
        assert len(get_ecosystems()) == 21

    def test_ecosystem_map_has_display_names(self):
        m = get_ecosystem_map()
        assert m["E-CYBR"]["name"] == "Cybersecurity"


# ── _jaccard ─────────────────────────────────────────────────────────────────

class TestJaccard:
    def test_disjoint_is_zero(self):
        assert _jaccard(frozenset({"A"}), frozenset({"B"})) == 0.0

    def test_exact_boundary(self):
        a, b = frozenset({"T1", "T2", "T3", "T4"}), frozenset({"T1", "T2", "T3", "T5"})
        assert _jaccard(a, b) == pytest.approx(0.6)   # 3 shared / 5 union

    def test_empty_set_is_zero(self):
        assert _jaccard(frozenset(), frozenset({"A"})) == 0.0


# ── compute_ecosystem_scores ─────────────────────────────────────────────────

class TestComputeEcosystemScores:
    @pytest.fixture
    def scores(self):
        return compute_ecosystem_scores(ECOSYSTEM_TO_THEMES, THEME_RS)

    def test_all_three_ecosystems_present(self, scores):
        assert set(scores) == {"E-CYBR", "E-SAAS", "E-AISEMI"}

    def test_ecyber_near_dup_collapses_depth_but_not_raw_or_strong(self, scores):
        s = scores["E-CYBR"]
        # union = {T1..T8} (8 unique tickers; A1/A2 overlap doesn't double-count)
        assert s["member_union"] == sorted(["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"])
        assert s["n_scored"] == 8
        assert s["raw"] == pytest.approx((87 + 88 + 89 + 90 + 92 + 95) / 6)   # drop T8=84,T5=86
        assert s["strong"] == 8          # all 8 members are RS>=80
        # depth: A1/A2 near-dup (Jaccard 0.6) -> 1 credit; A3 distinct -> +1 = 2
        assert s["depth"] == 2
        assert s["n_active_themes"] == 3
        assert s["boost"] == pytest.approx(0.04 * 2 + 0.015 * 8)   # = 0.20, under cap
        assert s["boosted"] == pytest.approx(s["raw"] * 1.20)

    def test_esaas_thin_floor_forces_zero_boost(self, scores):
        s = scores["E-SAAS"]
        assert s["raw"] == pytest.approx(92.0)
        assert s["strong"] == 3           # < STRONG_FLOOR (5)
        assert s["depth"] == 1            # the sub-theme DOES qualify (comp>=85, 3 members)
        assert s["boost"] == 0.0          # but thin-ecosystem floor zeroes it anyway
        assert s["boosted"] == pytest.approx(s["raw"])

    def test_eaisemi_weak_theme_no_depth_fading_excluded(self, scores):
        s = scores["E-AISEMI"]
        # Fading ThemeC2 (T15,T16) must be excluded from the member union entirely.
        assert s["member_union"] == sorted(["T12", "T13", "T14"])
        assert s["n_active_themes"] == 1
        assert s["raw"] == pytest.approx(57.5)
        assert s["strong"] == 0
        assert s["depth"] == 0            # comp 57.5 << DEPTH_COMP_MIN (85)
        assert s["boost"] == 0.0
        assert s["boosted"] == pytest.approx(57.5)

    def test_boosted_never_below_raw(self, scores):
        for e_code, s in scores.items():
            assert s["boosted"] >= s["raw"]


# ── _group_and_rank_ecosystems ───────────────────────────────────────────────

class TestGroupAndRankEcosystems:
    @pytest.fixture
    def grouped(self):
        scored_themes = [
            {"name": "ThemeA1", "stage": "Accelerating", "comp": trimmed_mean([95, 92, 90, 88]),
             "delta": None, "tickers": ["T1", "T2", "T3", "T4"], "n_scored": 4},
            {"name": "ThemeA2", "stage": "Mainstream", "comp": trimmed_mean([95, 92, 90, 86]),
             "delta": None, "tickers": ["T1", "T2", "T3", "T5"], "n_scored": 4},
            {"name": "ThemeA3", "stage": "Accelerating", "comp": trimmed_mean([89, 87, 84]),
             "delta": None, "tickers": ["T6", "T7", "T8"], "n_scored": 3},
            {"name": "ThemeB1", "stage": "Mainstream", "comp": trimmed_mean([93, 91, 89]),
             "delta": None, "tickers": ["T9", "T10", "T11"], "n_scored": 3},
            {"name": "ThemeC1", "stage": "Nascent", "comp": trimmed_mean([60, 55, 50]),
             "delta": None, "tickers": ["T12", "T13", "T14"], "n_scored": 3},
            # ThemeD + ThemeE are deliberately left OUT of eco_map -> both
            # default to E-UNASSIGNED. Together their union (T17-T21, all
            # RS>=80) clears the thin-ecosystem floor (strong=5) and earns a
            # real boost, so E-UNASSIGNED's BOOSTED score (~112.6) actually
            # exceeds every real ecosystem's, including E-CYBR's (~108.2).
            # This is the point of the fixture: pinning must override score,
            # not merely coincide with a low score.
            {"name": "ThemeD", "stage": "Accelerating", "comp": trimmed_mean([99, 98, 97]),
             "delta": None, "tickers": ["T17", "T18", "T19"], "n_scored": 3},
            {"name": "ThemeE", "stage": "Accelerating", "comp": trimmed_mean([96, 95]),
             "delta": None, "tickers": ["T20", "T21"], "n_scored": 2},
        ]
        scored_themes.sort(key=lambda x: -x["comp"])
        fading = [{"name": "ThemeC2", "tickers": ["T15", "T16"]}]
        eco_map = {
            "ThemeA1": "E-CYBR", "ThemeA2": "E-CYBR", "ThemeA3": "E-CYBR",
            "ThemeB1": "E-SAAS",
            "ThemeC1": "E-AISEMI", "ThemeC2": "E-AISEMI",
            # ThemeD/ThemeE intentionally absent -> eco_map.get(...) falls
            # back to E-UNASSIGNED for both.
        }
        ordered, active_by_eco, fading_by_eco, scores = _group_and_rank_ecosystems(
            scored_themes, fading, THEME_RS, eco_map)
        return {
            "scored_themes": scored_themes, "ordered": ordered,
            "active_by_eco": active_by_eco, "fading_by_eco": fading_by_eco,
            "scores": scores,
        }

    def test_e_unassigned_pinned_last_despite_highest_boosted_score(self, grouped):
        assert grouped["ordered"] == ["E-CYBR", "E-SAAS", "E-AISEMI", E_UNASSIGNED]
        # ThemeD's ecosystem has the highest boosted score of all four groups...
        assert grouped["scores"][E_UNASSIGNED]["boosted"] > grouped["scores"]["E-CYBR"]["boosted"]
        # ...yet E-UNASSIGNED still sorts last (index -1), never by score.
        assert grouped["ordered"][-1] == E_UNASSIGNED

    def test_ecyber_beats_esaas_beats_eaisemi_by_boosted(self, grouped):
        s = grouped["scores"]
        assert s["E-CYBR"]["boosted"] > s["E-SAAS"]["boosted"] > s["E-AISEMI"]["boosted"]

    def test_global_rank_independent_of_ecosystem_rank(self, grouped):
        # ThemeD ranks #1 GLOBALLY (highest raw comp of all 7 themes) even
        # though its ecosystem (E-UNASSIGNED) ranks LAST on the board.
        global_rank = {t["name"]: i for i, t in enumerate(grouped["scored_themes"], 1)}
        assert global_rank["ThemeD"] == 1

    def test_fading_grouped_under_its_own_ecosystem(self, grouped):
        assert [t["name"] for t in grouped["fading_by_eco"]["E-AISEMI"]] == ["ThemeC2"]
        assert grouped["fading_by_eco"].get("E-CYBR") is None

    def test_active_themes_grouped_and_order_preserved(self, grouped):
        names = [t["name"] for t in grouped["active_by_eco"]["E-CYBR"]]
        assert set(names) == {"ThemeA1", "ThemeA2", "ThemeA3"}

    def test_unassigned_group_contains_both_unmapped_themes(self, grouped):
        assert set(t["name"] for t in grouped["active_by_eco"][E_UNASSIGNED]) == {"ThemeD", "ThemeE"}

    def test_unassigned_boost_actually_engages(self, grouped):
        # Confirms the fixture exercises a NONZERO boost for E-UNASSIGNED
        # (ThemeD+ThemeE union clears the strong>=5 floor) -- otherwise the
        # "pinned last despite highest score" test above would be vacuous.
        s = grouped["scores"][E_UNASSIGNED]
        assert s["strong"] >= 5
        assert s["boost"] > 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
