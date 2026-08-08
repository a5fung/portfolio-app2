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
    parent_of = dedup_themes(name_tickers, threshold=threshold, min_shared=min_shared)
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
    #                 "last_date": date, "last_name": str}
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
                if shared < min(min_shared, denom):
                    continue
                if max(len(a), len(b)) / denom > size_ratio_cap:
                    continue
                score = _jaccard(a, b)
                if score >= overlap_threshold:
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
                cohorts[cid] = {"tickers": frozenset(), "last_date": day, "last_name": rep_name}
                assigned[rep_name] = cid

        # Commit today's state + row -> canonical_id map (rep AND aliases).
        for rep_name, cid in assigned.items():
            tks = rep_tickers.get(rep_name, frozenset())
            if tks and len(tks) <= max_set_size:
                cohorts[cid]["tickers"] = tks
            cohorts[cid]["last_date"] = day
            cohorts[cid]["last_name"] = rep_name

        for _, row in day_rows.iterrows():
            rep_name = parent_of.get(row["name"], row["name"])
            cid_by_key[(day, row["name"])] = assigned[rep_name]

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
