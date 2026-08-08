"""Canonical rank heatmap — per-cohort trajectory, dense (operator ask 8/8).

Companion to theme_flow.py's alluvial, not a replacement for it: Flow shows
BOARD-LEVEL movement (how much is promoting/demoting in aggregate); this view
shows PER-COHORT trajectory (exactly where a specific cohort sat, week by
week) — the ask's own framing ("flow = board-level movement, heatmap =
per-theme trajectory"). It also scales past the ~8-line legibility ceiling
that motivated the bump chart's redesign in the first place: a heatmap adds
rows, not crossing lines, so 20-40 cohorts at once is normal here.

This is NOT a new visual language — it reuses theme_grid.py's `_rank_color`
/ `_rank_delta` (the exact same "brighter green = better rank" gradient and
Δ-arrow convention already on screen in the Grid tab) so a cell here and a
cell in Grid mean the same thing at a glance. The only thing that changes is
the ROW axis: Grid rows are the raw theme `name` (fragments across an LLM
rename); this view's rows are `canonical_id` (theme_canon.py) — one row
survives a rename, which is the entire reason #315 built canonical identity.
"""
from __future__ import annotations

import html as _html
from datetime import date

import pandas as pd
import streamlit as st

from theme_data import get_canonical_weekly_grid, get_top_members_by_rs
from theme_grid import _rank_color, _rank_delta   # single source of truth — see module docstring
from theme_palette import active


def _usable_weeks(canon_grid: pd.DataFrame) -> list[date]:
    """Same precedent as theme_grid.render_grid's own filter, applied to the
    canonical grid — drops pre-rs_avg-engine weeks where NO cohort has a rank
    at all (kept local rather than imported from theme_flow.py so a failure
    in one new view's module can never take the other down — see
    pages/Apollo_Themes.py's per-tab lazy-import isolation note)."""
    return [
        w for w in sorted(canon_grid["week_start"].unique())
        if canon_grid.loc[canon_grid["week_start"] == w, "week_rank"].notna().any()
    ]


def render_canon_heatmap() -> None:
    st.header("Canonical Rank Heatmap")
    st.caption(
        "Same weekly rank grid as the Grid tab, but rows are the canonical "
        "cohort (#315) instead of the raw theme name — a cohort keeps ONE row "
        "across an engine rename instead of fragmenting into several. "
        "Brighter cell = better rank, same gradient as Grid."
    )

    with st.sidebar:
        st.subheader("Canonical heatmap")
        weeks_n = st.slider("Weeks of history", min_value=4, max_value=24, value=12, step=1)
        top_n = st.slider(
            "Top-N by current rank", min_value=5, max_value=60, value=25, step=5,
            help="No line-chart color cap here (unlike Bump Chart / Flow) — "
                 "a heatmap row is just a row, so this can go wide.",
        )
        only_ranked_now = st.checkbox(
            "Hide cohorts with no current rank", value=True,
            help="Hides cohorts that did not appear in the most recent week. "
                 "Never hides an 'Also include' pick — an explicit search "
                 "result vanishing on you reads as broken, not filtered.",
        )

    grid = get_canonical_weekly_grid(weeks=weeks_n)
    if grid.empty:
        st.info("No canonical theme data in this window.")
        return

    weeks = _usable_weeks(grid)
    if not weeks:
        st.info("No week in this window has a usable rank yet.")
        return
    sub = grid[grid["week_start"].isin(weeks)]

    latest_week = max(weeks)
    latest = sub[sub["week_start"] == latest_week].set_index("canonical_id")

    also_include = st.sidebar.multiselect(
        "Also include (beyond Top-N)",
        options=sorted(sub["canonical_name"].unique()),
    )

    ranked_now = latest["week_rank"].dropna().sort_values()
    top_ids = list(ranked_now.index[:top_n])
    name_to_id = sub.drop_duplicates("canonical_name").set_index("canonical_name")["canonical_id"]
    extra_ids = [name_to_id[n] for n in also_include if n in name_to_id.index]
    selected_ids: list[str] = []
    for cid in top_ids + extra_ids:
        if cid not in selected_ids:
            selected_ids.append(cid)

    if only_ranked_now:
        # A cohort the operator explicitly typed into "Also include" must
        # never silently vanish because of this checkbox — that reads as a
        # bug ("I searched it and it's just... not there"), not a filter
        # doing its job. Only the Top-N-derived population is filtered.
        selected_ids = [
            cid for cid in selected_ids
            if cid in ranked_now.index or cid in extra_ids
        ]

    if not selected_ids:
        st.info("No ranked cohorts in this window (try unchecking 'Hide cohorts with no current rank').")
        return

    id_to_name = sub.drop_duplicates("canonical_id").set_index("canonical_id")["canonical_name"]

    pivot_rank = sub[sub["canonical_id"].isin(selected_ids)].pivot_table(
        index="canonical_id", columns="week_start", values="week_rank", aggfunc="first"
    )
    pivot_rank = pivot_rank.reindex(index=selected_ids, columns=weeks)

    # Sort by current rank ascending (best first); cohorts unranked today
    # (only reachable when the "hide" checkbox above is off) sort last.
    current_rank = pivot_rank[latest_week] if latest_week in pivot_rank.columns else pd.Series(dtype=float)
    order = current_rank.sort_values(na_position="last").index
    pivot_rank = pivot_rank.reindex(order)

    deltas: dict[str, tuple[str, str]] = {}
    for cid in pivot_rank.index:
        ranks = [
            (None if pd.isna(pivot_rank.at[cid, wk]) else float(pivot_rank.at[cid, wk]))
            for wk in weeks
        ]
        deltas[cid] = _rank_delta(ranks)

    latest_tickers: dict[str, tuple[str, ...]] = {}
    for cid in pivot_rank.index:
        tk = latest.at[cid, "tickers"] if cid in latest.index else None
        if tk:
            latest_tickers[id_to_name.get(cid, cid)] = tuple(tk)
    member_preview = get_top_members_by_rs(latest_tickers, n=4) if latest_tickers else {}

    P = active()
    _bd = P["border"]
    _head_bg = P["sidebar_bg"]
    _txt = P["text"]

    def _cell(bg, color, val, weight="normal"):
        return (
            f'<td style="background:{bg};color:{color};text-align:center;'
            f'padding:3px 6px;font-weight:{weight};border:1px solid {_bd};'
            f'font-variant-numeric:tabular-nums">{val}</td>'
        )

    def _th(label, align="center"):
        return (
            f'<th style="text-align:{align};padding:4px 6px;color:{_txt};'
            f'background:{_head_bg};border:1px solid {_bd};font-size:11px;'
            f'white-space:nowrap">{label}</th>'
        )

    rows_html = []
    for cid in pivot_rank.index:
        name = str(id_to_name.get(cid, cid))
        name_esc = _html.escape(name)
        members_esc = _html.escape(member_preview.get(name, ""))
        cells = [
            (f'<td style="text-align:left;padding:3px 8px;border:1px solid {_bd};'
             f'background:{P["cell_blank_bg"]};color:{_txt};max-width:300px">{name_esc}</td>'),
            (f'<td style="text-align:left;padding:3px 8px;border:1px solid {_bd};'
             f'background:{P["cell_blank_bg"]};color:{_txt};font-size:12px;'
             f'white-space:nowrap">{members_esc}</td>'),
        ]
        now_rank = pivot_rank.at[cid, latest_week] if latest_week in pivot_rank.columns else None
        nbg, ntxt = _rank_color(now_rank)
        nval = f"#{int(now_rank)}" if now_rank is not None and not pd.isna(now_rank) else ""
        cells.append(_cell(nbg, ntxt, nval, "600"))
        dtext, dcolor = deltas[cid]
        cells.append(
            f'<td style="text-align:center;color:{dcolor};font-weight:600;'
            f'background:{P["cell_blank_bg"]};border:1px solid {_bd};'
            f'padding:3px 6px">{_html.escape(dtext)}</td>'
        )
        for wk in weeks:
            bg, txt = _rank_color(pivot_rank.at[cid, wk])
            val = pivot_rank.at[cid, wk]
            val_s = f"#{int(val)}" if pd.notna(val) else ""
            cells.append(_cell(bg, txt, val_s))
        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    head = _th("Theme", "left") + _th("Top members", "left") + _th("Now") + _th("Δ")
    for wk in weeks:
        head += _th(wk.isoformat()[5:])

    st.markdown(
        f'<div style="overflow-x:auto;max-height:760px;overflow-y:auto">'
        f'<table style="border-collapse:collapse;font-size:13px;width:100%;color:{_txt}">'
        f'<thead><tr>{head}</tr></thead><tbody>{"".join(rows_html)}</tbody></table></div>',
        unsafe_allow_html=True,
    )

    with st.expander("Legend"):
        st.markdown(
            "- **Bright green** = top rank (#1); fades through the same gradient Grid uses\n"
            "- **Grey cell** = ranked outside top 50 that week\n"
            "- **Black/blank cell** = no snapshot for that ISO week for this cohort — "
            "not interpolated, not implied\n"
            "\n**Δ** = rank change from the earliest visible week to the most recent."
        )

    st.caption(
        f"Latest week: **{latest_week}** · {len(pivot_rank)} cohort(s) shown · "
        f"{len(weeks)} of {len(_usable_weeks(get_canonical_weekly_grid(weeks=24)))} "
        "week(s) on file. "
        "⚠ Canonical identity is ticker-set matching — most theme rows carry under "
        "3 tickers and can only be stitched by name, so small/young cohorts fragment "
        "more than one row per cohort implies; treat the row count as directional."
    )
