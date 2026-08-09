"""Cross-day theme canonicalization (#315 R3).

PROBLEM: the upstream theme engine re-mints each theme's NAME nightly (an LLM
description), so the same underlying stock cohort shows up as "U.S.
Government/Defense Spending Surge" one day and "U.S. Government/Defense
Contract Surge" the next. Every over-time view (bump chart, forward-return
study) needs ONE stable identity per cohort, not one row per name-variant.
`theme_data.py`'s existing `dedup_themes()` already solves the SAME-DAY half
of this (near-duplicate names co-existing in one snapshot, merged by ticker
overlap) — this module solves the CROSS-DAY half: a name that disappears
and a new name that appears nearby, driven by the same tickers.

KEY DECISION (per the task, and the direction the upstream theme engine
itself went): canonicalize by TICKER-SET, not by name. A cohort's membership
is far more stable day to day than its LLM-generated description.

## The algorithm (one pass, oldest day -> newest)

Per day:
  1. Intra-day collapse — reuse `dedup_themes()` (verbatim, same threshold/
     min_shared as the Grid view) so same-day aliases resolve to ONE
     representative row before cross-day matching ever sees them.
  2. Tier 1 — EXACT NAME continuation: if a representative's name exactly
     matches the most-recently-seen name of a still-live cohort (within
     `max_gap_days`), it continues that cohort automatically, no ticker
     check needed. An LLM-generated multi-word theme description recurring
     verbatim is near-zero-false-positive evidence of "same slot in the
     engine's output" — stronger than any overlap threshold, and it is what
     lets a cohort survive a same-name ticker swing that would otherwise
     dip below the overlap floor (e.g. a theme going from 5 members to 2
     members while keeping its exact name is obviously still itself).
  3. Tier 2 — TICKER-OVERLAP continuation for whatever a name-match didn't
     claim: greedy best-Jaccard-first matching against remaining live
     cohorts, one cohort claimed per day at most (mirrors `dedup_themes`'
     "each theme merges into at most one parent" invariant, applied across
     time instead of within a day).
  4. Anything still unmatched starts a NEW cohort.

Cohorts unseen for more than `max_gap_days` stop being match candidates —
old identities don't get silently resurrected by a coincidental overlap
months later (normal engine cadence is a 1-day gap, occasionally 3-4 over a
weekend/holiday — see the `_MAX_GAP_DAYS` paragraph below).

## Thresholds — WHY these numbers, with the real counter-examples that set them

`_OVERLAP_THRESHOLD = 0.50` and `_MIN_SHARED = 3` reuse `dedup_themes()`'s
already-operator-reviewed values verbatim (theme_grid.py's Dedup slider
default) — no new number invented for the "how much overlap counts"
question. Three NEW guards were required beyond that, each one found by
running this algorithm against the real snapshot and inspecting the
resulting cohorts (not picked from first principles):

  - **Symmetric Jaccard, not containment.** `dedup_themes` uses containment
    from the smaller side (`|S ∩ L| / |S|`) because same-day dedup is a
    subset relationship (a fragment theme is fully inside its parent).
    Applied across TIME that measure over-merges: a 2-ticker theme fully
    contained in an unrelated 12-ticker theme scores a perfect 1.0. Real
    example from this snapshot: on 2026-03-20, "Oil & Gas" (6 tickers) fully
    contains three genuinely distinct sub-themes ("Large-Cap Upstream Oil &
    Gas E&P", "Downstream Oil Refining & Midstream", "Permian Basin
    Pure-Play E&P") at containment 1.0 each — Jaccard for the same pairs
    tops out at 0.5, which the size-ratio guard below still catches.
  - **`_SIZE_RATIO_CAP = 2.5`** (`max(|A|,|B|) / min(|A|,|B|)` must not
    exceed this) and **both sides must have >= 2 tickers to attempt overlap
    matching at all.** A 1-ticker reference set is not a real signature: any
    2-ticker set sharing that one ticker clears Jaccard 0.5 trivially. Real
    example: "Crypto Recovery" (tickers={CRCL}) vs "CLO & Structured Credit
    Income" (tickers={CRCL, XFLT}) — Jaccard 0.5, `min_shared` floor
    satisfied, and yet these are unrelated themes that happen to share one
    stock. Blocked by requiring `min(|A|,|B|) >= 2`.
  - **`_MAX_SET_SIZE = 20`** — rows above this are excluded from BOTH
    matching and from updating a cohort's reference ticker set. 58/3093 rows
    (1.9%) in this snapshot have >20 tickers, and inspecting them shows the
    upstream engine occasionally attaches a near-universe-wide basket to a
    normally-tiny theme for a single day (e.g. "Robo-Advisor & AI-Driven
    Wealth Management Platforms" is {BETR, WLTH} on every day except
    2026-04-10, when the SAME name briefly carried 57 tickers spanning half
    the energy sector — an upstream data glitch, not a real membership
    change). Left unguarded, that one glitched day bridges two 50+-ticker
    baskets under different names, which then chain-merges dozens of
    unrelated themes transitively (observed directly while calibrating this
    module: "Aerospace MRO & Defense Parts Distribution" and "Robo-Advisor &
    AI-Driven Wealth Management Platforms" ended up in the same cohort as
    plain Oil & Gas names before this guard was added). The cap does not
    drop the row from the OUTPUT — it still renders with its real ticker
    count — it only stops that one day's reading from being trusted as a
    matching signature.
  - **Structural guard — a new name may only absorb a cohort via ticker
    overlap if that cohort's last-seen name is ABSENT from today's
    snapshot.** If the old and new names co-exist on the same day, the
    engine itself is saying they're different themes (this is exactly what
    Tier 1/intra-day dedup are for), not that one replaced the other.

`_MAX_GAP_DAYS = 10`: this snapshot's day-to-day cadence is calendar gap 1
on 67/84 transitions, 2-4 on the rest (weekends, holidays) — max observed 4.
10 is a deliberate ~2x buffer over that observed max so a slow week doesn't
sever a real cohort, while staying far short of "coincidental reuse months
later" (some exact-name pairs in this snapshot are 60+ days apart with no
ticker relationship at all — those must NOT be treated as one continuous
cohort, and gap alone stops the ticker-overlap tier from trying).

Calibration evidence: the counter-examples above (Oil & Gas containment
over-merge, Crypto Recovery/CLO weak-reference merge, Robo-Advisor glitch-day
chain) were found by running earlier drafts of this algorithm against the
real `apollo_themes_snapshot.json` and inspecting the resulting cohorts —
`test_theme_canon.py` pins the fixed behavior against small synthetic
fixtures encoding each one, plus a smoke test against the live snapshot.

## #553 fix — cohort matching glued unrelated themes together

Measured on the real snapshot (2026-08-08): 34 of 311 cohorts (11%) were
stitched from more than one raw name, and three of those were plainly wrong
— e.g. "Satellite Mobile & IoT Connectivity Services" absorbed "U.S. Defense
Primes & Aerospace" ({GD, LMT, NOC, RTX}) on pure ticker overlap despite
being different real themes. The guards above (symmetric Jaccard, size-ratio
cap, structural guard) all predate this fix and did NOT catch these three —
four NEW guards were required, found by instrumenting every merge decision
against the real snapshot (not guessed):

  - **`_intraday_representatives` now passes `jaccard_floor=overlap_threshold`
    to `dedup_themes()`** (see that function's docstring). Without it, the
    SAME-DAY collapse step (step 1 of the algorithm above) can glue two
    unrelated themes together at containment 1.0 / Jaccard as low as ~0.3
    BEFORE cross-day matching ever runs — real example: on 2026-04-13,
    "Niche Specialty Chemicals & Industrial Intermediates" (6 tickers) was
    same-day contained in "Agricultural Commodities & Agri-Business" (15
    tickers) at containment 1.0 but Jaccard 0.4, seeding part of the
    Nitrogen/chemicals chain-merge below. `theme_grid.py`'s own dedup call
    does NOT pass this — Grid's behavior is provably unchanged (see
    `test_grid_output_unchanged`).
  - **The Tier 2 `min_shared` relaxation was removed** — it used to accept
    `shared >= min(min_shared, denom)`, letting a 2-ticker set match on only
    2 shared tickers. Real example: "Niche Specialty Chemicals & Industrial
    Intermediates" and "IP Licensing & Ad-Tech Royalty Software" both
    reduced to the identical 2-ticker set {ADEA, RYAM} at different points
    and merged at Jaccard 1.0 purely because 2/2 cleared the relaxed floor —
    same failure shape as the 1-ticker CRCL/XFLT case above, just one notch
    bigger. Tier 2 now requires the SAME unrelaxed `min_shared` floor
    `dedup_themes()` already enforces — there is no principled reason
    cross-day matching should trust weaker evidence than same-day matching
    does.
  - **`_FIRST_CONTACT_THRESHOLD = 0.70`** — a rep_name that has NEVER before
    been recorded under a candidate cohort (as its representative OR as a
    same-day alias absorbed into it) must clear a higher Jaccard bar than one
    that has. Real example: "Satellite Mobile & IoT Connectivity Services"
    had never touched the "U.S. Defense Primes & Aerospace" cohort before,
    yet grabbed it at Jaccard 0.667 (4 of 6 tickers pre-existing, 2 brand
    new) on first contact — exactly the situation where a stable NAME (Tier
    1) or a prior track record is the only real evidence a match is genuine,
    and neither existed. 0.70 sits strictly above 0.667 (blocks the bad
    case) and strictly below 0.714/0.778 (the real first-contact renames
    this snapshot also contains, e.g. "Theme Alpha" → "Theme Alpha Redux"
    shape and "Niche Specialty Chemicals" → "Nylon & Engineered Polymer
    Intermediates" both still pass). A cohort's OWN alias history (recorded
    every day, not just for its current representative) counts as prior
    contact, so a same-day handoff established via intraday dedup still
    carries its lower 0.50 bar forward on the day the old name retires (see
    `test_old_name_retiring_hands_off_to_its_own_alias`).
  - **Anchor-set check** — every cohort freezes `anchor_tickers` (its ticker
    set at creation, never updated). Tier 2 must clear `overlap_threshold`
    against BOTH the cohort's latest set AND its anchor set. This is what
    stops chain drift: each single hop in a chain can look individually
    fine (Jaccard 0.7-1.0 against whatever the cohort currently holds) while
    the cumulative walk ends up somewhere unrelated to where the cohort
    started. Real example: by 2026-04-10, a cohort that began as "Niche
    Specialty Chemicals & Industrial Intermediates" ({ASIX, CC, CE, LXU,
    RYAM}) had — through 6 individually-plausible hops — drifted to a
    15-ticker nitrogen-fertilizer/agri-business set sharing only LXU with
    where it started; "Nitrogen & Specialty Crop Nutrient Producers" then
    matched that DRIFTED set at Jaccard 1.0. Checked against the frozen
    anchor, that same match scores Jaccard 0.333 and is blocked. Applies
    ONLY to Tier 2 — Tier 1 (exact-name) drift is deliberately left
    ungated, unchanged from the original design (a stable label is treated
    as strong evidence on its own, per the Tier 1 rationale above).

Residual (documented, not fixed here): a handful of same-day merges at
exactly the 0.50 Jaccard boundary survive by design — e.g. "Oil & Gas"
(6 tickers) still absorbs the 3 sub-themes cited above at Jaccard exactly
0.5 each (the size-ratio guard does NOT catch this one — 6/3 = 2.0 is under
the 2.5 cap; that claim in the paragraph above predates this fix and turned
out to be wrong once actually checked against the real data). Tightening
the boundary to `>` would flip `test_old_name_retiring_hands_off_to_its_own_
alias`, which is pinned at exactly Jaccard 0.50 for a real, wanted merge —
so this stays a documented limitation rather than a further threshold
change (avoids re-litigating one number against two conflicting real
examples with no data to break the tie). Also unfixed by design: a single
anomalous later row can still re-attach to a cohort it was split from, if by
then the two genuinely do share ~80%+ of their tickers (e.g. one stray
2026-04-13 "U.S. Defense Primes & Aerospace" row re-joins the Satellite
Mobile cohort it was split from at 03-25, because by then their baskets
really had converged to 83% overlap) — ticker-only matching cannot
distinguish that from a genuine rename; the ORIGINAL 4-ticker capture the
operator flagged is fixed and stays fixed for its entire run.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from theme_data import dedup_themes

# ── Tunables (see module docstring for the WHY behind each number) ─────────
_OVERLAP_THRESHOLD = 0.50  # Jaccard floor — reuses dedup_themes' value
_MIN_SHARED = 3             # |intersection| floor — reuses dedup_themes' value
_MAX_GAP_DAYS = 10          # ~2x the observed max real-cadence gap (4 days)
_SIZE_RATIO_CAP = 2.5       # blocks tiny-set-fully-inside-huge-set false merges
_MAX_SET_SIZE = 20          # excludes glitched near-universe-wide basket rows
_FIRST_CONTACT_THRESHOLD = 0.70  # #553: Tier 2 bar for a name/cohort pair with
                                  # no prior track record (see module docstring)


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _intraday_representatives(
    day_rows: pd.DataFrame, threshold: float, min_shared: int, max_set_size: int
) -> tuple[dict[str, str], dict[str, frozenset]]:
    """One dedup_themes() pass for a single day. Returns (parent_of, rep_tickers)
    — rep_tickers is each REPRESENTATIVE's own ticker set that day (dedup_themes
    only decides which name is the parent; it does not union ticker sets).

    Rows above `max_set_size` are withheld from dedup_themes entirely (can
    neither absorb nor be absorbed) — the same glitch-day protection Tier 2
    applies, needed here too: two coincidentally-huge same-day baskets can
    otherwise dedup into one alias pair before the cross-day matcher ever
    runs, which defeats the Tier-2 guard downstream (see module docstring's
    Robo-Advisor example — the same glitch row can strike intra-day)."""
    name_tickers = {
        row["name"]: tuple(row["tickers"])
        for _, row in day_rows.iterrows()
        if row["tickers"] and len(row["tickers"]) <= max_set_size
    }
    # #553: jaccard_floor=threshold turns dedup_themes' containment-only match
    # into a containment-AND-Jaccard gate here (Grid's own call in
    # theme_grid.py does NOT pass this, so Grid is unaffected — see that
    # function's docstring + test_grid_output_unchanged).
    parent_of = dedup_themes(
        name_tickers, threshold=threshold, min_shared=min_shared, jaccard_floor=threshold
    )
    rep_tickers = {
        name: frozenset(tks)
        for name, tks in name_tickers.items()
    }
    return parent_of, rep_tickers


def canonicalize_themes(
    df: pd.DataFrame,
    *,
    overlap_threshold: float = _OVERLAP_THRESHOLD,
    min_shared: int = _MIN_SHARED,
    max_gap_days: int = _MAX_GAP_DAYS,
    size_ratio_cap: float = _SIZE_RATIO_CAP,
    max_set_size: int = _MAX_SET_SIZE,
    first_contact_threshold: float = _FIRST_CONTACT_THRESHOLD,
) -> pd.DataFrame:
    """Attach `canonical_id` / `canonical_name` to every (name, theme_date) row.

    Input: the raw themes frame (`theme_date` as a python date, `name` str,
    `tickers` a list — the shape theme_data._load() produces). Output: a COPY
    with two new columns. `canonical_id` is a stable synthetic key ("K0001",
    ...) shared by every row belonging to the same tracked cohort, in ANY
    name it wore. `canonical_name` is that cohort's MOST RECENT name (the
    freshest LLM description) — the same value on every row of that cohort,
    so grouping by either column gives one continuous series per cohort.

    Pure function of its input (deterministic given the same df + params) —
    no I/O, no Streamlit dependency. Callers cache it (see
    theme_data.get_canonical_themes).
    """
    if df.empty:
        out = df.copy()
        out["canonical_id"] = pd.Series(dtype="object")
        out["canonical_name"] = pd.Series(dtype="object")
        return out

    work = df.sort_values("theme_date").reset_index(drop=True)

    # cohorts[cid] = {"tickers": frozenset (last USABLE reference set),
    #                 "last_date": date, "last_name": str,
    #                 "anchor_tickers": frozenset (FROZEN at creation, #553
    #                     Fix D — never updated again; the chain-drift check
    #                     below compares against this, not just "tickers"),
    #                 "ever_names": set[str] (#553 Fix C — every raw name ever
    #                     recorded under this cid, rep or alias; a name with
    #                     no entry here is "first contact" and needs the
    #                     higher first_contact_threshold bar)}
    cohorts: dict[str, dict] = {}
    next_id = 1
    # canonical_id assigned per (theme_date, name) row — built up day by day.
    cid_by_key: dict[tuple[date, str], str] = {}

    for day, day_rows in work.groupby("theme_date", sort=True):
        today_names = set(day_rows["name"])

        parent_of, rep_tickers = _intraday_representatives(
            day_rows, overlap_threshold, min_shared, max_set_size
        )
        rep_names = sorted({parent_of.get(n, n) for n in day_rows["name"]})

        claimed_today: set[str] = set()
        assigned: dict[str, str] = {}   # rep_name -> cid, this day only

        # Tier 1 — exact-name continuation (see module docstring).
        # Candidate cohorts still "live" within the gap window, keyed by
        # their last-registered name; ties broken by most-recently-seen.
        name_live: dict[str, list[tuple[date, str]]] = {}
        for cid, c in cohorts.items():
            gap = (day - c["last_date"]).days
            if 0 < gap <= max_gap_days:
                name_live.setdefault(c["last_name"], []).append((c["last_date"], cid))
        for rep_name in rep_names:
            candidates = name_live.get(rep_name)
            if not candidates:
                continue
            cid = max(candidates, key=lambda x: x[0])[1]
            if cid in claimed_today:
                continue
            assigned[rep_name] = cid
            claimed_today.add(cid)

        # Tier 2 — ticker-overlap continuation for whatever Tier 1 left open.
        scored: list[tuple[float, int, str, str]] = []  # (score, shared, rep, cid)
        for rep_name in rep_names:
            if rep_name in assigned:
                continue
            a = rep_tickers.get(rep_name, frozenset())
            if len(a) < 2 or len(a) > max_set_size:
                continue
            for cid, c in cohorts.items():
                if cid in claimed_today:
                    continue
                gap = (day - c["last_date"]).days
                if gap <= 0 or gap > max_gap_days:
                    continue
                # Structural guard: only absorb a cohort by overlap if its old
                # name genuinely stopped being emitted today (a real rename,
                # not two co-existing distinct themes that happen to overlap).
                if c["last_name"] in today_names and c["last_name"] != rep_name:
                    continue
                b = c["tickers"]
                if len(b) < 2 or len(b) > max_set_size:
                    continue
                denom = min(len(a), len(b))
                shared = len(a & b)
                # #553 Fix B: NEVER relax below the plain min_shared floor. A
                # 2-ticker set hitting shared==2 used to slip through here
                # (min(min_shared, denom) == denom when denom < min_shared) —
                # real example: two totally unrelated theme names both
                # reduced to the identical {ADEA, RYAM} pair and merged on a
                # "full" 2/2 match that carries no more evidence than the
                # already-rejected 1-ticker CRCL/XFLT case below. Cross-day
                # matching should never trust weaker evidence than same-day
                # dedup_themes does (which never relaxes this floor).
                if shared < min_shared:
                    continue
                if max(len(a), len(b)) / denom > size_ratio_cap:
                    continue
                score = _jaccard(a, b)
                if score < overlap_threshold:
                    continue
                # #553 Fix C: first contact between this raw name and this
                # cohort needs a higher bar than a name/cohort pair with a
                # track record (Tier 1 exact-name persistence or a prior
                # same-day alias — see cohort_aliases's "ever_names" comment
                # above). Real example: "Satellite Mobile & IoT Connectivity
                # Services" had never touched the "U.S. Defense Primes &
                # Aerospace" cohort before, yet grabbed it at Jaccard 0.667
                # (2 of its 6 tickers brand new) on first contact alone.
                if rep_name not in c.get("ever_names", ()):
                    if score < first_contact_threshold:
                        continue
                # #553 Fix D: also require the match against the cohort's
                # FROZEN anchor set (its tickers at creation), not just its
                # latest (possibly already-drifted) set — otherwise a chain
                # of individually-plausible day-to-day hops can walk a
                # cohort's identity far from where it started while every
                # single hop clears the bar. Tier 1 (exact-name) drift is
                # deliberately exempt — see module docstring.
                anchor = c.get("anchor_tickers")
                if anchor and _jaccard(a, anchor) < overlap_threshold:
                    continue
                scored.append((score, shared, rep_name, cid))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        for _score, _shared, rep_name, cid in scored:
            if rep_name in assigned or cid in claimed_today:
                continue
            assigned[rep_name] = cid
            claimed_today.add(cid)

        # New cohorts for anything still unmatched (deterministic order).
        for rep_name in rep_names:
            if rep_name not in assigned:
                cid = f"K{next_id:04d}"
                next_id += 1
                cohorts[cid] = {
                    "tickers": frozenset(), "last_date": day, "last_name": rep_name,
                    "anchor_tickers": None, "ever_names": set(),
                }
                assigned[rep_name] = cid

        # Commit today's state + row -> canonical_id map (rep AND aliases).
        for rep_name, cid in assigned.items():
            tks = rep_tickers.get(rep_name, frozenset())
            if tks and len(tks) <= max_set_size:
                cohorts[cid]["tickers"] = tks
                if cohorts[cid].get("anchor_tickers") is None:
                    cohorts[cid]["anchor_tickers"] = tks  # #553 Fix D: frozen once, at creation
            cohorts[cid]["last_date"] = day
            cohorts[cid]["last_name"] = rep_name

        for _, row in day_rows.iterrows():
            rep_name = parent_of.get(row["name"], row["name"])
            cid = assigned[rep_name]
            cid_by_key[(day, row["name"])] = cid
            # #553 Fix C: record EVERY raw name seen today under this cid
            # (rep and aliases alike) so a same-day handoff (e.g. the
            # old-name-retires-to-its-alias case) carries prior-contact
            # status forward, not just the day's chosen representative.
            cohorts[cid].setdefault("ever_names", set()).add(row["name"])

    canonical_name_of = {cid: c["last_name"] for cid, c in cohorts.items()}

    out = work.copy()
    out["canonical_id"] = [
        cid_by_key[(row.theme_date, row.name)] for row in out.itertuples()
    ]
    out["canonical_name"] = out["canonical_id"].map(canonical_name_of)
    return out


def cohort_aliases(canon_df: pd.DataFrame) -> pd.DataFrame:
    """One row per canonical_id that wore >1 distinct raw name — for a
    transparency expander (mirrors theme_grid.py's "Dedup detail"). Columns:
    canonical_id, canonical_name, aliases (list[str], excludes canonical_name
    itself), first_date, last_date, n_names."""
    if canon_df.empty:
        return pd.DataFrame(columns=[
            "canonical_id", "canonical_name", "aliases", "first_date", "last_date", "n_names",
        ])
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
