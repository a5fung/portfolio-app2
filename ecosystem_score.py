"""ADR 0032 theme-ecosystem D3 score — PORTED VERBATIM from Apollo
`agents/market_intelligence/theme_ecosystems.py` @ git short sha `1115b37`
(apollo_the_wise, 2026-07-19). Keep in sync — this must produce the exact same
numbers as the Telegram `/themes` v2 board, or the two-level Streamlit view
below silently drifts from production.

Ported as-is, DB/async plumbing stripped:
  - `compute_ecosystem_scores` (pure D3 scorer) — UNCHANGED arithmetic.
  - `_group_and_rank_ecosystems` (shared grouping + ranking) — UNCHANGED.
  - `_jaccard` (near-dup helper it calls) — UNCHANGED.
  - `trimmed_mean` — ported verbatim from
    `agents/market_intelligence/constants.py` (same file, same sha) since
    `compute_ecosystem_scores` depends on it and portfolio-app2 must not
    import from the Apollo repo at runtime.
  - The D3 score constants (STRONG_RS_MIN, DEPTH_COMP_MIN, ELITE_PAIR_MIN_RS,
    STRONG_FLOOR, BOOST_CAP, BOOST_DEPTH_COEF, BOOST_STRONG_COEF,
    DEPTH_NEAR_DUP_JACCARD) — UNCHANGED values.

NOT ported (out of scope for this read-only dashboard port — see Apollo
source for the full picture):
  - Haiku/keyword theme->ecosystem ASSIGNMENT (`assign_theme_to_ecosystem`,
    `keyword_fallback_ecosystem`, `ensure_theme_ecosystems`, ...). The
    snapshot JSON already carries a resolved `e_code` per theme (#194) —
    this dashboard only ever CONSUMES that assignment, never computes it.
  - `format_ecosystem_board` / `format_ecosystem_scorecard_compact` — those
    render Telegram Markdown (strikethrough, STAGE_EMOJI, conviction
    suffixes from briefing.py) and are not reusable outside Telegram. The
    Streamlit renderer (theme_ecosystem_view.py) is a NEW build that mirrors
    their STRUCTURE (ecosystems ranked by boosted score, sub-themes nested
    with global rank, raw->boosted delta, Fading struck-through inside its
    ecosystem, E-UNASSIGNED pinned last) using the verbatim-ported functions
    below for the actual numbers.

Taxonomy: embedded as a plain Python list (`_TAXONOMY` below), transcribed
from Apollo's `theme_ecosystems.yaml` (operator-signed SSoT, same commit).
Embedded rather than parsed from a bundled .yaml at runtime because
portfolio-app2's requirements.txt does not carry PyYAML and Streamlit Cloud
installs strictly from that file — adding a new runtime dependency for one
static ~20-row taxonomy is not worth the install-risk. A verbatim copy of
`theme_ecosystems.yaml` is ALSO kept in this repo (theme_ecosystems.yaml) as
a human-diffable reference against Apollo's copy; it is not read by any code
here. If the taxonomy changes upstream, update BOTH by hand.
"""
from __future__ import annotations

from typing import Any

# ═════════════════════════════════════════════════════════════════════════
# Taxonomy — transcribed from apollo_the_wise/theme_ecosystems.yaml @ 1115b37
# (20 buckets + reserved E-UNASSIGNED, operator-signed 2026-07-14). Order
# matters: it is the taxonomy tiebreak used by _group_and_rank_ecosystems.
# ═════════════════════════════════════════════════════════════════════════

E_UNASSIGNED = "E-UNASSIGNED"

_TAXONOMY: list[dict[str, Any]] = [
    {"e_code": "E-CYBR", "name": "Cybersecurity"},
    {"e_code": "E-AISEMI", "name": "AI silicon & semiconductors"},
    {"e_code": "E-AIINFRA", "name": "AI cloud & datacenter"},
    {"e_code": "E-SAAS", "name": "Enterprise software"},
    {"e_code": "E-BIO", "name": "Biotech & therapeutics"},
    {"e_code": "E-MEDTECH", "name": "Devices & diagnostics"},
    {"e_code": "E-INS", "name": "Insurance"},
    {"e_code": "E-BANKFIN", "name": "Banks / fintech / brokers"},
    {"e_code": "E-REIT", "name": "Real estate / REITs"},
    {"e_code": "E-ENER", "name": "Energy"},
    {"e_code": "E-METAL", "name": "Metals & mining"},
    {"e_code": "E-DEF", "name": "Defense & space"},
    {"e_code": "E-TRANS", "name": "Transport & logistics"},
    {"e_code": "E-CONS", "name": "Consumer / retail / dining"},
    {"e_code": "E-INDL", "name": "Industrials / power"},
    {"e_code": "E-CRYPTO", "name": "Crypto infrastructure"},
    {"e_code": "E-COMM", "name": "Communications & media"},
    {"e_code": "E-HLTH", "name": "Healthcare services"},
    {"e_code": "E-QUANTUM", "name": "Quantum computing"},
    {"e_code": "E-ROBOT", "name": "Robotics & automation"},
    {"e_code": E_UNASSIGNED, "name": "Unassigned"},
]


def get_ecosystems() -> list[dict[str, Any]]:
    """The ordered ecosystem list (order = display tiebreak). Parity with
    Apollo's `get_ecosystems()` signature; reads the embedded list instead
    of the YAML file (see module docstring)."""
    return _TAXONOMY


def get_ecosystem_map() -> dict[str, dict[str, Any]]:
    """e_code -> taxonomy entry lookup."""
    return {e["e_code"]: e for e in _TAXONOMY}


def get_ecosystem_codes() -> list[str]:
    """Ordered e_codes (taxonomy order)."""
    return [e["e_code"] for e in _TAXONOMY]


# ═════════════════════════════════════════════════════════════════════════
# trimmed_mean — ported VERBATIM from
# agents/market_intelligence/constants.py @ 1115b37
# ═════════════════════════════════════════════════════════════════════════

def trimmed_mean(values: list[float]) -> float:
    """
    Trimmed mean — drop the bottom 20% of values, then average the rest.
    Resists 1-2 outliers dragging down a strong theme while still reflecting
    broad weakness if many stocks are fading.

    <=5 stocks: drop lowest 1. 6-10: drop lowest 2. 11+: drop bottom 20%.
    Minimum 3 values required for trimming; below that, plain mean.
    """
    if not values:
        return 0.0
    if len(values) < 3:
        return sum(values) / len(values)

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n <= 5:
        drop = 1
    elif n <= 10:
        drop = 2
    else:
        drop = max(1, int(n * 0.2))

    trimmed = sorted_vals[drop:]
    return sum(trimmed) / len(trimmed)


# ═════════════════════════════════════════════════════════════════════════
# D3 ecosystem score — PORTED VERBATIM from theme_ecosystems.py @ 1115b37
# ═════════════════════════════════════════════════════════════════════════

# ── D3 score constants (ADR 0032 — illustrative pins; CHANGE_PROCESS N>=10
#    backtest before any of these gates a live decision. They gate NOTHING
#    live today: the score is a read-model for /themes + briefs.) ───────────
STRONG_RS_MIN = 80.0          # member counts as "strong" at rs_composite >= 80
DEPTH_COMP_MIN = 85.0         # sub-theme must post trimmed-mean comp >= 85
ELITE_PAIR_MIN_RS = 90.0      # 2-member sub-theme qualifies only if both >= 90
STRONG_FLOOR = 5              # strong(E) < 5 -> boost 0 (thin-ecosystem floor)
BOOST_CAP = 0.30
BOOST_DEPTH_COEF = 0.04
BOOST_STRONG_COEF = 0.015
# Depth near-dup dedup (implementation refinement of D3, documented deviation):
# the literal "count qualifying sub-themes" is gameable by re-listing the same
# cohort under N near-identical names (raw/strong are union-safe; depth alone
# was not). Two qualifying sub-themes whose member sets overlap at Jaccard >=
# this threshold count ONCE (highest-comp survivor). Genuine sub-structure —
# disjoint clusters, or PARENT_CHILD subsets like a 3-member vuln-mgmt child
# inside a 12-member parent (Jaccard 0.25) — still counts fully, which is the
# Marios "inherited from Themes" pop the depth term exists for.
DEPTH_NEAR_DUP_JACCARD = 0.6


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def compute_ecosystem_scores(
    ecosystem_to_themes: dict[str, list[dict]],
    theme_rs_data: dict[str, dict],
) -> dict[str, dict]:
    """D3 (ADR 0032) — per-ecosystem score aggregate. PURE: no I/O.

    Inputs:
      ecosystem_to_themes: e_code -> theme dicts (need "name", "tickers",
        "stage"; Fading/Retired themes may be included — they are excluded
        from members/depth here).
      theme_rs_data: ticker -> {"rs_composite": float, ...} (the
        get_rs_for_tickers shape). The stored mi_themes `score` column is
        deliberately ignored (mixed-scale landmine).

    Returns e_code -> {raw, strong, depth, boost, boosted, member_union,
    n_scored, n_active_themes}.

      members(E) = dedup union of tickers across active non-Fading sub-themes
      raw(E)     = trimmed_mean(rs_composite of members(E))
      strong(E)  = |{m : rs_composite >= 80}|
      depth(E)   = |{T : (>=3 members OR elite pair) AND comp >= 85}|,
                   near-duplicate member-sets (Jaccard >= 0.6) counted once
      boost(E)   = 0 if strong < 5 else min(0.30, 0.04*depth + 0.015*strong)
      boosted(E) = raw * (1 + boost)

    Anti-fragmentation: the union base means splitting a cohort into N
    near-dups changes neither raw nor strong, and the depth near-dup dedup
    means it doesn't inflate depth either — the same members score the same
    whether they live in 1 sub-theme or 8 near-copies. Genuinely distinct
    sub-clusters DO raise depth (the intended Marios boost).
    """
    def _rs(tk: str) -> float | None:
        d = theme_rs_data.get(tk) or {}
        return d.get("rs_composite")

    out: dict[str, dict] = {}
    for e_code, themes in ecosystem_to_themes.items():
        active = [t for t in (themes or [])
                  if t.get("stage") not in ("Fading", "Retired")]

        union: set[str] = set()
        for t in active:
            union.update(t.get("tickers") or [])
        member_union = sorted(union)

        rs_vals = [v for v in (_rs(tk) for tk in member_union) if v is not None]
        raw = trimmed_mean(rs_vals) if rs_vals else 0.0
        strong = sum(1 for v in rs_vals if v >= STRONG_RS_MIN)

        # depth — qualifying sub-themes, near-dup member-sets counted once
        qualifying: list[tuple[float, str, frozenset]] = []
        for t in active:
            tks = list(t.get("tickers") or [])
            t_rs = [v for v in (_rs(tk) for tk in tks) if v is not None]
            if not t_rs:
                continue
            comp = trimmed_mean(t_rs)
            if comp < DEPTH_COMP_MIN:
                continue
            is_elite_pair = (len(tks) == 2 and len(t_rs) == 2
                             and min(t_rs) >= ELITE_PAIR_MIN_RS)
            if len(tks) >= 3 or is_elite_pair:
                qualifying.append((comp, t.get("name") or "", frozenset(tks)))
        qualifying.sort(key=lambda q: (-q[0], q[1]))   # deterministic survivor order
        counted: list[frozenset] = []
        for _comp, _name, tkset in qualifying:
            if any(_jaccard(tkset, prev) >= DEPTH_NEAR_DUP_JACCARD for prev in counted):
                continue
            counted.append(tkset)
        depth = len(counted)

        boost = 0.0 if strong < STRONG_FLOOR else min(
            BOOST_CAP, BOOST_DEPTH_COEF * depth + BOOST_STRONG_COEF * strong)
        out[e_code] = {
            "raw": raw,
            "strong": strong,
            "depth": depth,
            "boost": boost,
            "boosted": raw * (1.0 + boost),
            "member_union": member_union,
            "n_scored": len(rs_vals),
            "n_active_themes": len(active),
        }
    return out


def _group_and_rank_ecosystems(
    scored_themes: list[dict],
    fading: list[dict],
    theme_rs_data: dict[str, dict],
    eco_map: dict[str, str],
) -> tuple[list[str], dict[str, list[dict]], dict[str, list[dict]], dict[str, dict]]:
    """Shared grouping + D3 scoring + display order for every ecosystem render
    (full /themes board AND the compact brief scorecard — ONE scoring path,
    #473; the surfaces must never disagree on grouping or rank).

    Returns (ordered_codes, active_by_eco, fading_by_eco, scores). Codes are
    sorted boosted-desc with taxonomy-order tiebreak, E-UNASSIGNED pinned last.
    """
    active_by_eco: dict[str, list[dict]] = {}
    for st in scored_themes:
        active_by_eco.setdefault(eco_map.get(st["name"], E_UNASSIGNED), []).append(st)
    fading_by_eco: dict[str, list[dict]] = {}
    for t in fading:
        fading_by_eco.setdefault(eco_map.get(t.get("name"), E_UNASSIGNED), []).append(t)

    # Score input: every theme (compute filters Fading itself).
    eco_to_all: dict[str, list[dict]] = {}
    for code in set(active_by_eco) | set(fading_by_eco):
        eco_to_all[code] = (
            [{"name": st["name"], "tickers": st.get("tickers") or [],
              "stage": st.get("stage", "")} for st in active_by_eco.get(code, [])]
            + [{"name": t.get("name"), "tickers": t.get("tickers") or [],
                "stage": "Fading"} for t in fading_by_eco.get(code, [])]
        )
    scores = compute_ecosystem_scores(eco_to_all, theme_rs_data)

    tax_order = {code: i for i, code in enumerate(get_ecosystem_codes())}

    def _sort_key(code: str):
        # E-UNASSIGNED pinned last; else boosted desc, taxonomy order tiebreak.
        if code == E_UNASSIGNED:
            return (1, 0.0, 0, code)
        s = scores.get(code) or {}
        return (0, -s.get("boosted", 0.0), tax_order.get(code, 999), code)

    return sorted(eco_to_all, key=_sort_key), active_by_eco, fading_by_eco, scores
