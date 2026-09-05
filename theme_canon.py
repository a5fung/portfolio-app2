"""Cross-day theme canonicalization — cohort identity (#315 R3; rewritten as a
MODEL under #555 / #553, 2026-09-05).

PROBLEM. The upstream theme engine re-mints each theme's NAME nightly (an LLM
description), so one stock cohort shows up as "U.S. Government/Defense Spending
Surge" one day and "...Contract Surge" the next; it also reuses a description
for an unrelated basket weeks later ("Satellite Mobile & IoT Connectivity
Services" was six defense/satellite names in March and four biotechs on
2026-04-16). Every over-time view (Rank Flow, Bump Chart, Forward Returns)
needs ONE stable identity per cohort. This module attaches it:
`canonical_id` (the cohort key) and `canonical_name` (the label it is shown
under).

The previous version decided identity by greedy day-to-day ticker overlap
with nine stacked guards and an ungated exact-name path; the guards fought
each other (a 0.70 first-contact floor sat 0.014 above the worst false merge)
and the ungated name path let a cohort walk from specialty chemicals to a
fertilizer basket in a week. All three false merges the operator evidenced
(#553) still reproduced across the March-April series. This rewrite replaces
the guards with a two-axis model and a slow-moving reference set.

## THE MODEL — a theme is a basket WITH a description; both must agree

An observation is one day's row: (name, ticker set). It continues a cohort
when BOTH hold:

  1. BASKET — it is mostly the same basket as the cohort's CORE (defined
     below): Jaccard(observation, core) >= `_OVERLAP_THRESHOLD` (0.50) — the
     shared tickers are at least half of everything either side holds.
  2. DESCRIPTION — its name does not contradict the cohort: it is a name the
     cohort has already worn, or it shares a SUBJECT word with one of them
     (`_subject_words`; form words like "platforms"/"producers", size words
     like "large-cap"/"pure-play" and narrative words like "recovery"/"surge"
     do not count — see `_NON_SUBJECT_WORDS`).

and the match must be carried by strong evidence on at least one axis:

  - SAME NAME (strong description): the engine is saying "same slot". The
    basket then only has to share at least ONE ticker with the core — a label
    reused on a basket sharing nothing with what the theme has mostly been is
    a different theme (the biotech "Satellite Mobile" above splits; so does
    "Semiconductor Probe Card & Front-End Test Equipment" when the engine
    moved that label from the broad semicap basket to {ONTO, TER} on
    2026-04-21 and kept it there for three months). A theme growing or
    narrowing around its core under its own label does not split. This is
    deliberate: under a single label the dashboard honors the engine's own
    continuity claim rather than second-guessing how wide the engine drew the
    basket that day — and a row always prefers its own label's cohort over a
    look-alike basket elsewhere.
  - NEW NAME: the basket must be mostly the same (axis 1) AND the description
    must agree (axis 2). Neither alone is enough at any size: a 2-ticker
    basket coincides across unrelated themes ({ADEA, RYAM} was both "IP
    Licensing & Ad-Tech Royalty Software" and "Niche Specialty Chemicals" on
    2026-05-12 — no shared word, split), and a shared word on a basket that
    is mostly different is a different theme. The old `min_shared = 3`
    floor existed to stop tiny-basket coincidences; the description axis
    does that job, so the floor is gone and a 2-ticker theme can be renamed
    like any other ("Domestic Steel Producers" / "U.S. Domestic Steel
    Producers", identical 2-set, shares "steel").

WHY BOTH AXES — the subset-vs-coincidence crux. Membership alone cannot tell
a small theme that is genuinely a subset of a larger one from a small theme
that merely sits inside an unrelated one: on 2026-03-25 "Satellite Mobile &
IoT Connectivity Services" {GD, IRDM, LMT, NOC, PL, RTX} contained all of
"U.S. Defense Primes & Aerospace" {GD, LMT, NOC, RTX} at Jaccard 0.67, and on
2026-03-26 "Nylon & Engineered Polymer Intermediates" contained all of "Niche
Specialty Chemicals & Industrial Intermediates" at Jaccard 0.71 — the same
geometry (an established basket plus two newcomers under a new label), and
the operator's judgment is opposite: the first is a different theme, the
second a rename. What separates them is the description: "Nylon ... Polymer
Intermediates" still describes the old members; "Satellite Mobile & IoT"
describes the two newcomers. Measured on the whole snapshot, every correct
multi-name cohort shares a subject word between its names, and every
evidenced false merge (satellite/defense, chemicals/ad-tech, the nylon ->
nitrogen -> agri chain, and a same-day absorb of "Optical Networking" into a
"Semiconductor Foundry" blob on 2026-04-06) shares none. Identical baskets
under unrelated names are NOT exempt — those are exactly the false merges
({ADEA, RYAM}; the 15-ticker blob the engine labelled Nylon, then Nitrogen,
then Agricultural on 04-08/04-10/04-13).

## THE CORE — how chaining is prevented

A cohort's reference set is not its latest observation and not its first: it
is the STRICT MAJORITY of its life — every ticker present in more than half
of the cohort's representative observations so far. Matching is always
against the core, and only the day's representative observation votes into
it. Consequences:

  - a single blob day (the engine attaching a 15-57 ticker basket to a small
    theme) cannot move the core, so it cannot become the reference for the
    next match — no separate "max set size" guard is needed;
  - a walk A -> B -> C needs the drifted basket to persist for more than
    half the cohort's life before the core follows it, at which point the
    cohort genuinely IS the new basket (a theme evolving over months, or the
    engine redefining what a label covers — "Optical Components &
    Transceiver Manufacturers" seeded as 4 tickers and became an 18-ticker
    basket within a week), while a chain of individually plausible one-day
    hops (the #553 fertilizer chain: 5-ticker chemicals to a 15-ticker agri
    basket in six hops) is impossible — each hop is measured against the
    origin-weighted core, not the previous hop. The one thing a label CAN
    do is carry its own cohort onto a different basket by persisting there
    (the engine kept "Independent Semiconductor Foundry" on an optical blob
    for three weeks after four days on {PLAB, TSEM}); the dashboard follows
    the engine's slot rather than second-guessing it, and the few seed rows
    then wear the label the cohort ends up under.

This is "origin plus a drift budget" without a budget knob: the budget is
"become the majority".

## SAME-DAY rows

Rows on one day are matched to live cohorts first; a cohort may take several
rows in a day when each independently passes the test (two descriptions of
one basket in one engine run — the 2026-08-18/19 double-run duplicates). The
best-evidence row is that day's representative; the others are same-day
aliases (they get the cohort's id but do not vote into the core). A row that
matches no core but IS the same theme as a cohort's representative row today
(row against row, same test) is also that cohort's alias — the day's second
wording of one emitted basket. Rows still unclaimed are deduplicated among
themselves with the same two-axis test (larger basket absorbs smaller) and
then start new cohorts. Finally, a cohort that took a row today and is the
same theme core-to-core as any other live cohort collapses into it (the
older keeps its id) — the persistent-duplicate case, two engine slots
emitting one basket under two wordings for weeks, which label continuity
would otherwise keep apart forever; tested against every live cohort, not
just today's, because duplicate slots often alternate days. Grid's own
same-day dedup (`theme_data.dedup_themes`, the operator's slider) is a
separate feature and is untouched.

## EMPTY rows

A row with no tickers (the engine's Retired marker; 293 of 4,468 rows, all
with null rs_avg) has no basket evidence. It attaches by name to the live
cohort that has worn that name; otherwise it starts a cohort of its own.
This is what makes a same-day duplicate stay merged on its Retired day
("Spending Surge" / "Contract Surge" both retired empty on 2026-08-05).

## NAMING

`canonical_name` is the cohort's most recently worn name among names worn on
at least two days (a young cohort with nothing else falls back to its latest
name). Previously it was the bare most-recent name, which let one Retired row
on 2026-04-13 relabel nine days of "Satellite Mobile" as "U.S. Defense Primes
& Aerospace". A real rename still takes over as soon as it recurs, so the
label matches the name on the Grid / Telegram board; a one-day visitor never
does.

## Knobs (two numbers; down from six numbers plus three structural rules)

`_OVERLAP_THRESHOLD = 0.50` reuses the operator-reviewed value from
`theme_data.dedup_themes` (theme_grid.py's slider default). `_MAX_GAP_DAYS =
10` is ~2x the largest real cadence gap (4 days over a holiday weekend); a
name and basket returning after longer is a new surfacing, not a
continuation. `_NON_SUBJECT_WORDS` is a word list, not a number — it is the
one thing here that encodes judgment about language and the one an operator
may want to edit. Retired with the old matcher: `_MIN_SHARED`,
`_MAX_SET_SIZE`, `_FIRST_CONTACT_THRESHOLD`, the anchor-set check, the
structural "old name still present" guard, the ungated exact-name tier and
`dedup_themes(jaccard_floor=)`.

Residual, accepted with reason: on 2026-03-20 the one-day parent "Oil & Gas"
{COP, EOG, FANG, MPC, VLO, XOM} is the union of two live sub-themes,
"Downstream Oil Refining & Midstream" {FANG, MPC, VLO} and "Large-Cap
Upstream Oil & Gas E&P" {COP, EOG, XOM}. It ties with both (each is exactly
half of it and shares "oil"), attaches to whichever is older, and the other
keeps its own identity — previously all three sub-themes (and "Permian Basin
Pure-Play E&P") collapsed into "Oil & Gas". Which half a one-day parent
attaches to is a tie-break; a sub-theme inside a broad parent is a
parent-child question (#505) the apollo-side engine owns.

`test_theme_canon.py` pins the model on synthetic fixtures (realistic names —
the description axis makes fixture names load-bearing) and on the real
snapshot: the three evidenced false merges stay split, the defense duplicate
stays merged on every day it appears, and Grid is byte-identical.
"""
from __future__ import annotations

import re
from collections import Counter
from datetime import date

import pandas as pd

# ── Tunables (see module docstring "Knobs") ────────────────────────────────
_OVERLAP_THRESHOLD = 0.50  # Jaccard floor vs the cohort core — reuses dedup_themes' value
_MAX_GAP_DAYS = 10          # ~2x the observed max real-cadence gap (4 days)

# Words that describe a basket's FORM, SIZE or NARRATIVE rather than its
# SUBJECT. Two names agree on description only when they share a word that is
# NOT in this list. "Space Launch & Orbital Services" vs "Satellite Mobile &
# IoT Connectivity Services" share only "services" — no agreement; "Hydraulic
# Fracturing & Well Completion Services" vs "Pressure Pumping & Completion
# Services" share "completion" — agreement. Connectors and the fragments that
# tokenizing "U.S." / "E&P" / "P&C" leave behind are here too.
_NON_SUBJECT_WORDS = frozenset({
    # connectors / tokenizer fragments
    "and", "or", "of", "the", "for", "in", "on", "to", "at", "by", "with",
    "via", "vs", "its", "u", "s", "us", "e", "p", "c", "non", "re",
    # size / tier / positioning
    "large", "mid", "small", "micro", "mega", "cap", "tier", "pure", "play",
    "senior", "junior", "major", "leading", "top", "diversified",
    "independent", "integrated", "specialty", "niche", "select", "core",
    "focused", "based", "driven", "related", "adjacent", "oriented", "stage",
    "high", "performance", "quality", "premium", "next", "gen", "generation",
    "new", "emerging", "other", "broad", "global", "domestic", "international",
    # form words — what kind of company, not what it does
    "platform", "service", "infrastructure", "system", "operator", "provider",
    "company", "manufacturer", "manufacturing", "producer", "developer",
    "distributor", "processing", "solution", "industry", "sector", "theme",
    "basket", "stock", "name",
    "group", "business", "holding", "player", "vendor", "supplier",
    "enabler", "beneficiary", "leader", "pick", "plays", "names",
    # narrative words — the story around a basket, not its subject
    "recovery", "rotation", "rating", "surge", "revival", "momentum",
    "breakout", "rebound", "rally", "turnaround", "comeback", "resurgence",
    "boom", "cycle", "upcycle", "trade", "rerating", "catalyst", "reflation",
    "reopening", "winner", "laggard",
})


def _singular(word: str) -> str:
    """Crude English singular so 'polymers' == 'polymer', 'companies' ==
    'company'. Only strips plural suffixes; never touches 3-letter words
    ('gas') or words ending in a double-s ('business')."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _subject_words(name: str) -> frozenset[str]:
    """The words of a theme name that carry its SUBJECT (see
    `_NON_SUBJECT_WORDS`). Lowercased, split on anything non-alphanumeric
    (so 'Gene-Editing' and 'Gene Editing' agree), singularized."""
    out = set()
    for raw in re.split(r"[^a-z0-9]+", name.lower()):
        if len(raw) < 2 or raw in _NON_SUBJECT_WORDS:
            continue
        word = _singular(raw)
        if word not in _NON_SUBJECT_WORDS:
            out.add(word)
    return frozenset(out)


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _same_theme(
    tks: frozenset, words: frozenset, ref_set: frozenset, ref_words: frozenset,
    *, exact_name: date | None, overlap_threshold: float,
) -> tuple | None:
    """The two-axis test. Returns an evidence key — higher sorts first — or
    None when the observation is not the same theme as the reference (a
    cohort's core, or another same-day row). `exact_name` is the date the
    reference last wore this exact name (None = never)."""
    shared = len(tks & ref_set)
    j = _jaccard(tks, ref_set)
    if exact_name is not None:
        # Same label: the engine's own continuity claim. Refuse only a label
        # reused on a basket sharing nothing with what the theme has mostly
        # been (its core) — or accept blindly when there is no basket on
        # record yet. Ranks above any basket-only match, and among cohorts
        # that wore the label, the one that wore it most recently wins: a
        # row stays where its label was yesterday rather than migrating to a
        # look-alike basket.
        if ref_set and shared < 1:
            return None
        return (1, exact_name, j, shared)
    if j < overlap_threshold:
        return None
    if not (words & ref_words):
        return None
    return (0, date.min, j, shared)


def _core(counts: Counter, n_obs: int) -> frozenset:
    """Tickers present in MORE than half of the cohort's representative
    observations (strict majority; one observation -> its own set)."""
    return frozenset(t for t, c in counts.items() if 2 * c > n_obs)


def canonicalize_themes(
    df: pd.DataFrame,
    *,
    overlap_threshold: float = _OVERLAP_THRESHOLD,
    max_gap_days: int = _MAX_GAP_DAYS,
) -> pd.DataFrame:
    """Attach `canonical_id` / `canonical_name` to every (name, theme_date) row.

    Input: the raw themes frame (`theme_date` as a python date, `name` str,
    `tickers` a list — the shape theme_data._load() produces). Output: a COPY
    with two new columns. `canonical_id` is a stable synthetic key ("K0001",
    ...) shared by every row belonging to the same tracked cohort, in ANY
    name it wore. `canonical_name` is that cohort's most recently worn name
    among names worn on two or more days (module docstring, NAMING) — the
    same value on every row of that cohort, so grouping by either column
    gives one continuous series per cohort.

    Pure function of its input (deterministic given the same df + params) —
    no I/O, no Streamlit dependency. Callers cache it (see
    theme_data.get_canonical_themes). The model is in the module docstring.
    """
    if df.empty:
        out = df.copy()
        out["canonical_id"] = pd.Series(dtype="object")
        out["canonical_name"] = pd.Series(dtype="object")
        return out

    work = df.sort_values("theme_date").reset_index(drop=True)

    # cohorts[cid] = {
    #   "counts": Counter  ticker -> number of representative observations containing it
    #   "n_obs":  int      representative observations with a non-empty basket
    #   "core":   frozenset  strict-majority set (derived; see _core)
    #   "names":  Counter  name -> days worn (representative or alias)
    #   "name_last": dict  name -> last date worn (naming tie-break)
    #   "words":  frozenset union of _subject_words over every name worn
    #   "last_date": date }
    cohorts: dict[str, dict] = {}
    next_id = 1
    cid_by_key: dict[tuple[date, str], str] = {}
    merged_into: dict[str, str] = {}   # duplicate cohort -> the cohort that absorbed it
    words_of: dict[str, frozenset] = {}

    def _words(name: str) -> frozenset:
        w = words_of.get(name)
        if w is None:
            w = words_of[name] = _subject_words(name)
        return w

    def _root(cid: str) -> str:
        while cid in merged_into:
            cid = merged_into[cid]
        return cid

    for day, day_rows in work.groupby("theme_date", sort=True):
        rows: list[tuple[str, frozenset]] = [
            (row["name"], frozenset(row["tickers"] or ())) for _, row in day_rows.iterrows()
        ]
        live = {
            cid: c for cid, c in cohorts.items()
            if 0 < (day - c["last_date"]).days <= max_gap_days
        }

        # 1. Every row against every live cohort; best evidence claims first.
        #    A cohort may take several rows in one day (same-day aliases of
        #    one basket); a row belongs to at most one cohort.
        scored: list[tuple[tuple, str, str]] = []
        for name, tks in rows:
            if not tks:
                # Retired marker (no basket): name-only evidence — the live
                # cohort that most recently wore this name, else nothing.
                wearers = [cid for cid, c in live.items() if name in c["names"]]
                if wearers:
                    cid = max(wearers, key=lambda k: live[k]["name_last"][name])
                    scored.append(((1, live[cid]["name_last"][name], -1.0, 0), name, cid))
                continue
            for cid, c in live.items():
                ev = _same_theme(
                    tks, _words(name), c["core"], c["words"],
                    exact_name=c["name_last"].get(name),
                    overlap_threshold=overlap_threshold,
                )
                if ev is not None:
                    scored.append((ev, name, cid))
        # Best evidence first; an exact tie goes to the OLDER cohort (same
        # rule as the duplicate-cohort fold below: the older id survives).
        scored.sort(key=lambda x: (x[0], -int(x[2][1:]), x[1]), reverse=True)
        assigned: dict[str, str] = {}
        rep_of: dict[str, tuple[str, frozenset]] = {}   # cid -> today's representative row
        for _ev, name, cid in scored:
            if name in assigned:
                continue
            assigned[name] = cid
            tks = dict(rows)[name]
            if tks:   # an empty Retired marker never represents the day's basket
                rep_of.setdefault(cid, (name, tks))

        # 1b. Same-day duplicates of what a cohort emitted TODAY: a row that
        #     is the same theme as a cohort's representative row (row vs
        #     row, same test) is that cohort's alias today even when the
        #     cohort's core says otherwise — e.g. a label that narrowed to 2
        #     tickers for weeks (core = the 2) re-emitting its old 5-ticker
        #     basket under two wordings on one day. Aliases never vote.
        for name, tks in rows:
            if name in assigned or not tks:
                continue
            best = None
            for cid, (rep_name, rep_tks) in rep_of.items():
                if not rep_tks:
                    continue
                ev = _same_theme(
                    tks, _words(name), rep_tks, _words(rep_name),
                    exact_name=cohorts[cid]["name_last"].get(name),
                    overlap_threshold=overlap_threshold,
                )
                if ev is not None and (best is None or (ev, -int(cid[1:])) > (best[0], -int(best[1][1:]))):
                    best = (ev, cid)
            if best is not None:
                assigned[name] = best[1]

        # 2. Rows no live cohort claimed: dedup among themselves with the
        #    same test (larger basket absorbs smaller), then new cohorts.
        new_rows = sorted(
            [(n, t) for n, t in rows if n not in assigned],
            key=lambda nt: (-len(nt[1]), nt[0]),
        )
        parent: dict[str, str] = {n: n for n, _ in new_rows}
        for i, (s_name, s_tks) in enumerate(new_rows):
            if not s_tks:
                continue
            best = None
            for l_name, l_tks in new_rows[:i]:
                if parent[l_name] != l_name or not l_tks:
                    continue
                ev = _same_theme(
                    s_tks, _words(s_name), l_tks, _words(l_name), exact_name=None,
                    overlap_threshold=overlap_threshold,
                )
                if ev is not None and (best is None or ev > best[0]):
                    best = (ev, l_name)
            if best is not None:
                parent[s_name] = best[1]
        for name, tks in new_rows:
            if parent[name] == name:
                cid = f"K{next_id:04d}"
                next_id += 1
                cohorts[cid] = {
                    "counts": Counter(), "n_obs": 0, "core": frozenset(),
                    "names": Counter(), "name_last": {}, "words": frozenset(),
                    "last_date": day,
                }
                assigned[name] = cid
                rep_of[cid] = (name, tks)
        for name, _tks in new_rows:
            if parent[name] != name:
                assigned[name] = assigned[parent[name]]

        # 3. Commit: the representative votes into the core (strict majority
        #    of everything the cohort has been — a few blob days cannot move
        #    it; a basket that persists for more than half the cohort's life
        #    becomes it). Every row (representative and alias) records its
        #    name on the cohort.
        for cid, (_name, tks) in rep_of.items():
            if tks:
                c = cohorts[cid]
                c["counts"].update(tks)
                c["n_obs"] += 1
                c["core"] = _core(c["counts"], c["n_obs"])
        for name, cid in assigned.items():
            c = cohorts[cid]
            c["names"][name] += 1
            c["name_last"][name] = day
            c["words"] = c["words"] | _words(name)
            c["last_date"] = day
            cid_by_key[(day, name)] = cid

        # 4. Duplicate cohorts. A cohort that took a row today and is the
        #    same theme core-to-core (same two-axis test) as another live
        #    cohort collapses into it; the older keeps its id and absorbs the
        #    other's whole history. This is the only place a PERSISTENT
        #    duplicate can be caught — two engine slots emitting one basket
        #    under two wordings for weeks ("Domestic Steel Producers" / "U.S.
        #    Domestic Steel Producers", {NUE, STLD} for a month): label
        #    continuity keeps each label on its own cohort, so row matching
        #    alone would never re-test them after their first (failed)
        #    contact. Cores are majorities of whole histories, so two cohorts
        #    only fold when their histories have mostly been the same basket.
        for a in sorted(set(assigned.values())):
            a = _root(a)   # may already have been folded earlier in this loop
            for b in sorted(cohorts):
                if b == a or b not in cohorts or a not in cohorts:
                    continue
                ca, cb = cohorts[a], cohorts[b]
                if (day - cb["last_date"]).days > max_gap_days:
                    continue
                if not ca["core"] or not cb["core"]:
                    continue
                if _same_theme(
                    cb["core"], cb["words"], ca["core"], ca["words"],
                    exact_name=None, overlap_threshold=overlap_threshold,
                ) is None:
                    continue
                keep, drop = (a, b) if a < b else (b, a)
                ck, cd = cohorts[keep], cohorts[drop]
                ck["counts"].update(cd["counts"])
                ck["n_obs"] += cd["n_obs"]
                ck["core"] = _core(ck["counts"], ck["n_obs"])
                ck["names"].update(cd["names"])
                for n, d in cd["name_last"].items():
                    ck["name_last"][n] = max(d, ck["name_last"].get(n, d))
                ck["words"] = ck["words"] | cd["words"]
                ck["last_date"] = max(ck["last_date"], cd["last_date"])
                merged_into[drop] = keep
                del cohorts[drop]
                a = keep

    def _label(c: dict) -> str:
        # The most recently worn name, ignoring names worn on a single day
        # (a one-day visitor is noise, not a rename) unless nothing else
        # exists yet. Tracks a real rename as soon as it has recurred, so
        # the label matches the name the Grid/Telegram board shows today.
        steady = [n for n, days in c["names"].items() if days >= 2]
        pool = steady or list(c["names"])
        return max(pool, key=lambda n: (c["name_last"][n], c["names"][n], n))

    canonical_name_of = {cid: _label(c) for cid, c in cohorts.items()}

    out = work.copy()
    out["canonical_id"] = [_root(cid_by_key[(row.theme_date, row.name)]) for row in out.itertuples()]
    out["canonical_name"] = out["canonical_id"].map(canonical_name_of)
    return out


def cohort_aliases(canon_df: pd.DataFrame) -> pd.DataFrame:
    """One row per canonical_id that wore >1 distinct raw name — for a
    transparency expander (mirrors theme_grid.py's "Dedup detail"). Columns:
    canonical_id, canonical_name, aliases (list[str], excludes canonical_name
    itself), first_date, last_date, n_names.

    No separate empty-input early return: every real caller passes
    `canonicalize_themes()`'s output, which always carries a `canonical_id`
    column (its own empty-input path attaches one — see that function). An
    empty frame shaped that way produces zero groupby groups, so `rows`
    stays empty and the `if not rows:` branch below already returns the
    identical empty frame."""
    g = canon_df.groupby("canonical_id")
    rows = []
    for cid, grp in g:
        names = sorted(grp["name"].unique())
        if len(names) < 2:
            continue
        canonical_name = grp["canonical_name"].iloc[0]
        aliases = [n for n in names if n != canonical_name]
        rows.append({
            "canonical_id": cid,
            "canonical_name": canonical_name,
            "aliases": aliases,
            "first_date": grp["theme_date"].min(),
            "last_date": grp["theme_date"].max(),
            "n_names": len(names),
        })
    if not rows:
        return pd.DataFrame(columns=[
            "canonical_id", "canonical_name", "aliases", "first_date", "last_date", "n_names",
        ])
    return pd.DataFrame(rows).sort_values("n_names", ascending=False).reset_index(drop=True)
