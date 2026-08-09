"""Forward RS movement — "what happened after a theme surfaced" (#315 R4).

NOT a price return — say so up top, not buried in a caption (a mislabeled
panel gets cited authoritatively later). This snapshot has no historical
per-ticker close-price series (`stock_scores` is a single cross-section for
one `score_date` — see theme_detail.py's V5 note), so a real $ or % forward
return cannot be computed from this app's data. `rs_avg` — each theme's own
trimmed-mean relative-strength composite, tracked weekly — is the largest
HONEST substitute available: this view reports how that score (and rank)
moved after a cohort first entered a rank band, using theme_data's
get_forward_rs_movement (theme_canon.py canonical identity underneath, so a
cohort's full lineage counts once, not fragmented by a name change).

Selection-effect framing (see get_forward_rs_movement's docstring): every
number here is shown ALONGSIDE the same measurement for a baseline band —
themes that reached rank 11-30 and, in the whole history this snapshot has
seen, NEVER made the top 10. That exclusion makes the two groups disjoint
(a genuine comparison population, not the same cohorts measured from two
different weeks of their own life — an earlier draft of this view did the
latter by mistake: 40% of "baseline" cohorts on the real snapshot were also
"surfaced" cohorts, counted in both columns). rs_avg is a 0-100 percentile
score, so conditioning on "just entered the top 10" selects on an
already-high score — a bare top-10 number mean-reverts by construction and
reads as "the board picks badly" if shown alone.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from theme_data import get_forward_rs_movement

_BAND_LABEL = {
    "surfaced_top": "Made the surfaced band",
    "baseline": "Reached baseline, never surfaced",
}


def _fmt_delta(v) -> str:
    if v is None or pd.isna(v):
        return "—"
    # `+ 0.0` normalizes a `-0.0` input to `0.0` (IEEE-754: adding positive
    # zero flips a negative zero's sign) so the f-string below never prints
    # a confusing "-0.0". Applies unconditionally — the old `if v != 0`
    # guard never actually ran this line for -0.0 (`-0.0 != 0` is False in
    # Python), so the hardcoded `else: 0.0` was silently doing the real work
    # for that case; `float(v) + 0.0` alone covers both cases identically.
    v = float(v) + 0.0
    return f"{v:+.1f}"


def _fmt_pct(v) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{v:.0f}%"


def render_forward() -> None:
    st.header("Forward RS Movement")
    st.warning(
        "⚠ This is NOT a price return. It tracks the theme engine's own "
        "relative-strength score (rs_avg, 0–100) and rank after a cohort "
        "first appears in a rank band — a proxy for what the RS engine did "
        "next, not a realized dollar or percent gain. True forward $/％ "
        "returns would need a per-ticker historical price join "
        "(`mi_signal_outcomes`) that this snapshot does not carry.",
        icon="⚠️",
    )

    with st.sidebar:
        st.subheader("Forward movement")
        surface_rank = st.slider(
            "Surfaced = reached rank <= N", min_value=3, max_value=20, value=10, step=1,
            help="A cohort's 'surfaced' event is the first week its canonical "
                 "rank fell at or inside this line.",
        )
        baseline_lo, baseline_hi = st.slider(
            "Baseline comparison band", min_value=1, max_value=60, value=(11, 30), step=1,
            help="Same study, run on themes that reached this band and NEVER "
                 "made the surfaced band — a genuine comparison group, not "
                 "the same cohorts measured twice (see the warning above).",
        )

    result = get_forward_rs_movement(
        surface_rank=surface_rank, baseline_lo=baseline_lo, baseline_hi=baseline_hi,
    )
    weeks_available = result["weeks_available"]
    debut_share = result.get("debut_share")
    debut_note = (
        f" **{debut_share * 100:.0f}%** of surfaced events are a cohort's first-ever "
        "recorded week already landing in that band — a strong debut, not a "
        "climb from lower in the pack; the rest are genuine promotions."
        if debut_share is not None else ""
    )
    st.caption(
        f"📏 Depth available: **{weeks_available} week(s)** of canonical history. "
        "The 4-week-forward figures below only include cohorts old enough to "
        "already HAVE a point 4 tracked weeks later — nothing is interpolated "
        "or projected forward." + debut_note
    )

    summary = result["summary"]
    events = result["events"]
    if summary.empty:
        st.info("No surfaced-theme events in the available history yet.")
        return

    # ── Summary: band x horizon, side by side ────────────────────────────
    for horizon in sorted(summary["horizon_weeks"].unique()):
        st.subheader(f"+{horizon} week(s) later")
        cols = st.columns(2)
        for col, band in zip(cols, ["surfaced_top", "baseline"]):
            row = summary[(summary["horizon_weeks"] == horizon) & (summary["band"] == band)]
            with col:
                st.markdown(f"**{_BAND_LABEL[band]}**")
                if row.empty or row.iloc[0]["n"] == 0:
                    st.caption("No observations yet at this horizon.")
                    continue
                r = row.iloc[0]
                st.metric(
                    "Median RS-score change", _fmt_delta(r["median_delta_rs"]),
                    help="rs_avg is 0-100. A cohort at 96 has almost no room "
                         "to gain further — compare against the baseline "
                         "column, not this number alone.",
                )
                st.caption(
                    f"Median rank change {_fmt_delta(-r['median_delta_rank']) if pd.notna(r['median_delta_rank']) else '—'} "
                    f"(positive = climbed) · {_fmt_pct(r['pct_rank_improved'])} of "
                    f"{int(r['n'])} cohort(s) ranked BETTER {horizon} week(s) later"
                )
        st.divider()

    st.caption(
        "Reading this: the two columns are DIFFERENT cohorts — themes that "
        "made the top band vs. themes that got close and never did. If the "
        "surfaced column looks worse, that can still be expected (a 0-100 "
        "score near its ceiling has less room to climb); the baseline column "
        "is what to check it against, not a claim either column is 'right'."
    )

    # ── Full event list ───────────────────────────────────────────────────
    with st.expander(f"All surfaced events ({len(events)})"):
        band_pick = st.radio(
            "Band", ["surfaced_top", "baseline"], horizontal=True,
            format_func=lambda b: _BAND_LABEL[b], key="fwd_band_pick",
        )
        show = events[events["band"] == band_pick].copy()
        show = show.sort_values("surfaced_week", ascending=False)
        display_cols = {
            "canonical_name": "Theme",
            "surfaced_week": "Surfaced",
            "rank_at_surfaced": "Rank then",
            "rs_avg_at_surfaced": "RS then",
        }
        for col in show.columns:
            if col.startswith("rank_fwd_"):
                n = col.replace("rank_fwd_", "").replace("w", "")
                display_cols[col] = f"Rank +{n}w"
            elif col.startswith("delta_rs_"):
                n = col.replace("delta_rs_", "").replace("w", "")
                display_cols[col] = f"Δ RS +{n}w"
        show = show[[c for c in display_cols if c in show.columns]].rename(columns=display_cols)
        for c in show.columns:
            if c.startswith("RS then") or c.startswith("Δ RS"):
                show[c] = show[c].apply(lambda v: f"{v:.1f}" if pd.notna(v) else "—")
            elif c.startswith("Rank"):
                show[c] = show[c].apply(lambda v: f"#{int(v)}" if pd.notna(v) else "—")
        st.dataframe(show, width='stretch', hide_index=True,
                     height=min(560, 60 + 36 * len(show)))
