"""Bump chart — rank-over-time by CANONICAL identity (#315 R2).

"Which themes are gaining strength" at a glance: one line per canonical
cohort, y = its weekly rank (inverted, rank 1 at top — same convention as
theme_detail.py's rank arc), x = ISO week. Depends on R3
(theme_canon.py / theme_data.get_canonical_weekly_grid) — a bump chart keyed
on the raw, nightly-churning `name` would draw a new line every time the
theme engine re-describes the same cohort, which is exactly the fragmentation
R3 exists to fix. This view is new; Grid/Ecosystems/Detail stay on raw
`name` and are unchanged (see theme_data.py's "Canonical identity" section).

Chart-building notes:
  - Plotly (already the app's charting library — theme_detail.py, Portfolio.py
    — no new dependency).
  - Colors: the dataviz skill's validated 8-slot categorical palette (fixed
    hue order, CVD-checked for line/adjacent-pair charts in both light and
    dark). Capped at 8 concurrent lines for exactly that reason — a 9th
    series folds into "not shown, widen the filter" rather than cycling a
    hue (never cycle categorical hues).
  - A cohort's color is keyed to its position in the ALPHABETICAL order of
    the CURRENTLY SELECTED set (not to its rank) — ranks reshuffle week to
    week, and repainting a survivor's color every time the board moves would
    defeat "color follows the entity". Changing the selection itself can
    still reassign slots; that's an explicit user action, not a silent
    repaint.
  - `connectgaps=False`: a cohort with a >1-week gap in its recorded history
    must show a broken line, not a straight interpolated one implying
    continuity through weeks it has no data for.
"""
from __future__ import annotations

import html as _html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_theme import is_dark
from theme_data import get_canonical_weekly_grid, get_top_members_by_rs
from theme_palette import active

# Dataviz skill reference palette, categorical slots (validated adjacent-pair
# CVD order — see dataviz skill references/palette.md). Light/dark are the
# SAME 8 hues, stepped for each surface, not separate palettes.
_SERIES_LIGHT = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]
_SERIES_DARK = [
    "#3987e5", "#d95926", "#199e70", "#c98500",
    "#d55181", "#008300", "#9085e9", "#e66767",
]
_MAX_SERIES = 8       # the validated palette's adjacent-pair cap (see docstring)
_DIRECT_LABEL_CAP = 4  # dataviz rule: legend always; direct-label at most this many


def _series_palette() -> list[str]:
    return _SERIES_DARK if is_dark() else _SERIES_LIGHT


def render_bump() -> None:
    st.header("Theme Bump Chart")
    st.caption(
        "Rank over time by canonical theme identity (#315) — one line per "
        "underlying cohort, survives the theme engine renaming it day to "
        "day. Lower on the chart = better rank."
    )

    with st.sidebar:
        st.subheader("Bump chart")
        weeks = st.slider("Weeks of history", min_value=4, max_value=24, value=12, step=1)
        top_n = st.slider(
            "Top-N by current rank", min_value=2, max_value=_MAX_SERIES, value=6, step=1,
            help=f"Capped at {_MAX_SERIES} concurrent lines — the categorical "
                 "palette's validated colorblind-safe limit for a line chart "
                 "(dataviz skill). Widen with 'Also include' below instead of "
                 "raising this past the cap.",
        )

    grid = get_canonical_weekly_grid(weeks=weeks)
    if grid.empty:
        st.info("No canonical theme data in this window.")
        return

    latest_week = grid["week_start"].max()
    latest = grid[grid["week_start"] == latest_week].set_index("canonical_id")

    also_include = st.sidebar.multiselect(
        "Also include (beyond Top-N)",
        options=sorted(grid["canonical_name"].unique()),
        help=f"Total plotted lines are still capped at {_MAX_SERIES}.",
    )

    ranked_now = latest["week_rank"].dropna().sort_values()
    top_ids = list(ranked_now.index[:top_n])
    name_to_id = grid.drop_duplicates("canonical_name").set_index("canonical_name")["canonical_id"]
    extra_ids = [name_to_id[n] for n in also_include if n in name_to_id.index]
    selected_ids: list[str] = []
    for cid in top_ids + extra_ids:
        if cid not in selected_ids:
            selected_ids.append(cid)
    selected_ids = selected_ids[:_MAX_SERIES]

    if not selected_ids:
        st.info("No ranked themes in this window.")
        return

    sub = grid[grid["canonical_id"].isin(selected_ids)]
    pivot_rank = sub.pivot_table(
        index="canonical_id", columns="week_start", values="week_rank", aggfunc="first"
    )
    week_cols = sorted(pivot_rank.columns)
    pivot_rank = pivot_rank[week_cols]

    id_to_name = grid.drop_duplicates("canonical_id").set_index("canonical_id")["canonical_name"]

    # Stable color assignment: alphabetical by name within the CURRENT
    # selection, not by rank (see module docstring).
    ordered_ids = sorted(selected_ids, key=lambda cid: id_to_name.get(cid, cid))
    palette = _series_palette()
    color_of = {cid: palette[i % len(palette)] for i, cid in enumerate(ordered_ids)}

    # Direct-label only the best-ranked few (dataviz: legend always, direct
    # labels selective) — pick by final available rank in the window.
    final_rank = {
        cid: pivot_rank.loc[cid].dropna().iloc[-1] if pivot_rank.loc[cid].notna().any() else float("inf")
        for cid in selected_ids
    }
    direct_label_ids = set(
        sorted(selected_ids, key=lambda c: final_rank[c])[:_DIRECT_LABEL_CAP]
    )

    P = active()
    fig = go.Figure()
    x_vals = [pd.Timestamp(w) for w in week_cols]
    for cid in ordered_ids:
        y_vals = [pivot_rank.loc[cid, w] if pd.notna(pivot_rank.loc[cid, w]) else None for w in week_cols]
        name = id_to_name.get(cid, cid)
        color = color_of[cid]
        text = [""] * len(y_vals)
        # Label at the LAST non-null point only.
        last_idx = max((i for i, v in enumerate(y_vals) if v is not None), default=None)
        if last_idx is not None and cid in direct_label_ids:
            text[last_idx] = f"  {name}"
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals, mode="lines+markers+text",
            text=text, textposition="middle right",
            textfont=dict(color=color, size=11),
            connectgaps=False,   # a data gap must show as a BROKEN line, not a straight one
            line=dict(color=color, width=2.5),
            marker=dict(size=6, color=color),
            name=name,
            hovertemplate=f"<b>{_html.escape(name)}</b><br>Week %{{x|%Y-%m-%d}}<br>Rank #%{{y}}<extra></extra>",
        ))

    fig.update_yaxes(autorange="reversed", title="Rank", zeroline=False, gridcolor=P["border"])
    fig.update_xaxes(title=None, gridcolor=P["border"])
    fig.update_layout(
        height=560, margin=dict(l=10, r=140, t=20, b=10),
        paper_bgcolor=P["chart_paper"], plot_bgcolor=P["chart_plot"],
        font=dict(color=P["chart_font"]),
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        showlegend=True,
    )
    st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})

    # Member preview strip — same batched lookup pattern as theme_grid.py.
    latest_tickers: dict[str, tuple] = {}
    for cid in selected_ids:
        tk = latest.at[cid, "tickers"] if cid in latest.index else None
        if tk:
            latest_tickers[id_to_name.get(cid, cid)] = tuple(tk)
    preview = get_top_members_by_rs(latest_tickers, n=4) if latest_tickers else {}
    if preview:
        with st.expander("Top members (current)"):
            for name in sorted(preview):
                st.caption(f"**{name}** — {preview[name]}")

    st.caption(
        f"Latest week: **{latest_week}** · {len(selected_ids)} theme(s) plotted · "
        f"window {weeks} week(s). Broken lines = the cohort had no ranked "
        "snapshot that week (not interpolated)."
    )
