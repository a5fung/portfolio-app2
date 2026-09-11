"""Weekly Movers — named rank gainers, replacing Rank Flow (#561).

THE ASK, in the operator's own words (2026-08-10, on his phone, looking at
Rank Flow): *"tbh I don't know how to read it."* PLAN.md #561 diagnosed WHY,
and this view is built to that diagnosis, not a re-derivation of it:

  - Rank Flow's ink is dominated by the BORING case (cohorts staying in their
    band) while the promotions that matter are hairline threads.
  - Every ribbon reads as the same pale green — nothing signals which move
    matters.
  - 4 bands x 11 weeks is too many ribbons to trace one cohort through.
  - The real defect isn't cosmetic: Rank Flow answers "how many cohorts
    changed band" (an aggregate count). The operator asks "which themes are
    climbing" (a name). A count can never name a theme.

THIS VIEW answers the actual question: a plain-text, per-week list of named
cohorts ranked by how much their rank IMPROVED — not by where they ended up
(see `compute_weekly_movers`'s docstring for why that ordering is the load-
bearing part). New arrivals into the top board get their own call-out
because a first appearance is the earliest signal there is (the north star:
subtle RS -> early theme -> buy before mainstream), independent of how big
the numeric jump was.

Reuses `theme_data.get_canonical_weekly_grid` — the SAME canonical-identity
weekly rank grid Rank Flow (theme_flow.py) and the Bump Chart (theme_bump.py)
already compute (#315 R3 / #555 identity rewrite) — and Rank Flow's own
`_usable_weeks` filter and `_OUTSIDE_BOUND` (=30) board-size constant, so
"on the board" means the same thing in both views. No new data pipeline.

Deliberately does NOT surface decliners or board exits: #561 asked for "the
biggest rank GAINERS", not a full leaderboard of change in both directions.

Rank Flow (theme_flow.py) stays reachable in the sidebar — this is an
addition, not a replacement, until the operator confirms this view is the
one that works (#561 explicit instruction: don't remove his fallback before
he's approved what replaces it). Rank Flow itself got a readability fix
2026-09-11 (#640 — the same known-vs-no-rank split this view already made,
plus legible labels and direction-colored ribbons), but it is still the
same view, not a new one.
"""
from __future__ import annotations

import re
from datetime import date

import pandas as pd
import streamlit as st

from theme_data import get_canonical_weekly_grid, get_canonical_weeks_on_file
from theme_flow import _OUTSIDE_BOUND, _usable_weeks

_TOP_N = 8  # cap per list, per week — phone-screen sized, see module docstring
_MD_SPECIAL = re.compile(r"([\\*_\[\]`])")


def _md_escape(text: str) -> str:
    """Escape CommonMark special characters in a theme name before it goes
    into an f-string handed to st.markdown (not st.markdown(unsafe_allow_html)
    — HTML escaping doesn't apply here, Markdown's own does). No cohort name
    in the current snapshot needs this (checked directly), but a name is
    engine-generated text, not a literal this code controls, and an
    unescaped "_" or "*" pair would silently break the bold/line formatting
    this whole view exists to make readable — the exact failure mode this
    task is about, just moved into a different character set."""
    return _MD_SPECIAL.sub(r"\\\1", str(text))


def compute_weekly_movers(
    grid: pd.DataFrame,
    weeks: list[date],
    board_size: int = _OUTSIDE_BOUND,
    top_n: int = _TOP_N,
) -> list[dict]:
    """One entry per consecutive week-pair in `weeks`, NEWEST TRANSITION FIRST.

    `grid` is a canonical weekly rank grid (the shape `get_canonical_weekly_grid`
    returns: one row per (canonical_id, week_start) with `week_rank`, 1 = best).
    `weeks` is the caller's usable-week list (already filtered — see
    theme_flow._usable_weeks) so this stays a pure function over what's passed
    in and is testable without touching Streamlit or the snapshot file.

    RANKING, the load-bearing part: sorted by the magnitude of rank CHANGE
    (prev_rank - curr_rank), never by curr_rank. #561's diagnosis was that the
    old view surfaced the boring "stayed put" case because a band-crossing
    COUNT can't distinguish "moved a lot" from "always been here" — ranking by
    raw position would repeat that mistake one level down (a cohort sitting at
    rank 3 both weeks would outrank a cohort that just jumped 22 -> 8).

    Returns a list of:
      {"week_start", "prev_week_start",
       "gainers": [{"canonical_id","name","prev_rank","curr_rank","delta"}, ...],
       "gainers_total": int,
       "entrants": [{"canonical_id","name","curr_rank"}, ...],
       "entrants_total": int}

    Two buckets, matching the operator's own example format in PLAN.md #561
    ("Defense Primes 22 -> 8", "Neoclouds outside -> 14"):
      - "gainers": on the board (rank <= board_size) in BOTH weeks, with
        curr_rank strictly better (smaller) than prev_rank. Sorted by delta
        descending, truncated to `top_n`; `gainers_total` is the count before
        truncation so a caller can show "+N more".
      - "entrants": on the board THIS week and with NO RANK AT ALL last week —
        absent from the grid, or present but unranked. Only these are a true
        first appearance, so only these carry no numeric "delta" ("outside"
        isn't a number to subtract from). Sorted by curr_rank ascending,
        truncated the same way.
        ⚠ A cohort that WAS ranked last week, even at 34 — past the board cut —
        is a GAINER with a real delta, not an entrant. The rank is known, so
        claiming it came from "outside" overstates the churn: measured
        2026-09-10, 12 of 19 apparent entrants had been on last week's list
        below rank 30, and only 7 were a genuinely new basket.
    A cohort that is on the board both weeks with an unchanged or WORSE rank,
    or that falls off the board entirely, appears in neither list — see the
    module docstring for why exits/decliners are out of scope here.
    """
    if grid.empty or len(weeks) < 2:
        return []

    sub = grid[grid["week_start"].isin(weeks)]
    names = sub.drop_duplicates("canonical_id").set_index("canonical_id")["canonical_name"]
    piv = sub.pivot_table(
        index="canonical_id", columns="week_start", values="week_rank", aggfunc="first"
    ).reindex(columns=weeks)

    out: list[dict] = []
    for i in range(len(weeks) - 1, 0, -1):  # newest transition first
        w0, w1 = weeks[i - 1], weeks[i]
        gainers: list[dict] = []
        entrants: list[dict] = []
        for cid in piv.index:
            curr_rank = piv.at[cid, w1]
            if pd.isna(curr_rank) or curr_rank > board_size:
                continue  # not on the board this week — nothing to report
            curr_rank = int(curr_rank)
            name = names.get(cid, cid)
            prev_rank = piv.at[cid, w0]
            # A KNOWN PRIOR RANK IS A KNOWN PRIOR RANK, even past the board cut (2026-09-10).
            # This read `prev_rank <= board_size`, so a cohort ranked 34 last week — a number we
            # HAVE — was thrown into "entrants" and rendered "outside -> 12" as if it had appeared
            # from nowhere. Measured that day: of 19 entrants, 12 were on last week's list below
            # rank 30, and only 7 were a genuinely new basket. The board is a top-30 cut of 143
            # themes, so most "entrants" are ordinary rotation past a line, and calling them new
            # overstates the churn — the operator spotted the discrepancy against his daily
            # summaries and was right.
            # A true entrant is one with NO rank at all last week (absent or unranked).
            was_ranked = pd.notna(prev_rank)
            if was_ranked:
                prev_rank = int(prev_rank)
                if curr_rank < prev_rank:
                    gainers.append({
                        "canonical_id": cid, "name": name,
                        "prev_rank": prev_rank, "curr_rank": curr_rank,
                        "delta": prev_rank - curr_rank,
                    })
                # unchanged or worsened but still on the board -> not reported
            else:
                entrants.append({
                    "canonical_id": cid, "name": name, "curr_rank": curr_rank,
                })
        gainers.sort(key=lambda m: m["delta"], reverse=True)
        entrants.sort(key=lambda m: m["curr_rank"])
        out.append({
            "week_start": w1,
            "prev_week_start": w0,
            "gainers": gainers[:top_n],
            "gainers_total": len(gainers),
            "entrants": entrants[:top_n],
            "entrants_total": len(entrants),
        })
    return out


def _render_week(entry: dict) -> None:
    gainers, entrants = entry["gainers"], entry["entrants"]
    if not gainers and not entrants:
        st.write(f"No cohort strengthened into the top {_OUTSIDE_BOUND} this week.")
        return
    if entrants:
        st.markdown(f"**New to the top {_OUTSIDE_BOUND}**")
        for m in entrants:
            st.markdown(f"- **{_md_escape(m['name'])}** outside → {m['curr_rank']}")
        if entry["entrants_total"] > len(entrants):
            st.caption(f"+{entry['entrants_total'] - len(entrants)} more")
    if gainers:
        st.markdown("**Climbing**")
        for m in gainers:
            st.markdown(
                f"- **{_md_escape(m['name'])}** {m['prev_rank']} → {m['curr_rank']} "
                f"(+{m['delta']})"
            )
        if entry["gainers_total"] > len(gainers):
            st.caption(f"+{entry['gainers_total'] - len(gainers)} more")


def render_movers() -> None:
    st.header("Theme Weekly Movers")
    st.caption(
        "Named rank gainers, one week at a time — which themes got "
        "stronger, not how many cohorts crossed a band line. Replaces Rank "
        "Flow for reading week over week (2026-08-10: “tbh I don't know "
        "how to read it”) — same canonical identity as the Bump "
        "Chart and Rank Flow (#315). Rank Flow stays in the sidebar for now."
    )

    with st.sidebar:
        st.subheader("Weekly movers")
        weeks_n = st.slider(
            "Weeks of history", min_value=3, max_value=16, value=8, step=1,
            help="How many trailing weeks feed the movers list. Each week's "
                 "movers are ranked by how much they moved, not by where "
                 "they ended up.",
            key="movers_weeks_n",
        )

    canon_grid = get_canonical_weekly_grid(weeks=weeks_n)
    if canon_grid.empty:
        st.info("No canonical theme data in this window.")
        return

    weeks = _usable_weeks(canon_grid)
    if len(weeks) < 2:
        st.info("Fewer than 2 usable weeks in this window — nothing to compare yet.")
        return

    weekly = compute_weekly_movers(canon_grid, weeks)
    if not weekly:
        st.info("No week-over-week data to compare in this window.")
        return

    st.caption(
        f"{len(weeks)} week(s) of {get_canonical_weeks_on_file(weeks=24)} total on file. "
        "⚠ Small/young cohorts (most theme rows carry under 3 tickers) can only be "
        "matched across days by name, not ticker set, so a rename can still misfire — "
        "read a single week's list directionally, don't over-index one row."
    )

    latest, *older = weekly
    st.subheader(f"Week of {latest['week_start']}")
    _render_week(latest)

    for entry in older:
        with st.expander(f"Week of {entry['week_start']}"):
            _render_week(entry)
