"""Rank-BAND flow (alluvial) — canonical identity, over time (operator ask 8/8).

PROBLEM WITH THE BUMP CHART: it plots one continuous line per cohort at its
exact weekly rank. With up to 8 concurrent lines constantly crossing (a rank
board reshuffles most weeks), the eye can't hold a single line's path through
a crossing — the operator's own words: "not easiest to read... see if
there's better visuals... like Sankey." This view answers the SAME question
("which themes are gaining strength, where in the arc") with an alluvial:
weeks on the x-axis, four FIXED rank bands as rows (never reordered), ribbons
between adjacent weeks whose THICKNESS is the count of cohorts making that
band-to-band move. A promotion is a ribbon rising toward the top row; a
demotion is a ribbon sinking — read directly off ribbon shape, no per-line
tracing required, and it scales to however many cohorts are on the board
(spaghetti in the bump chart, just thicker ribbons here).

Same canonical-identity data source as theme_bump.py
(theme_data.get_canonical_weekly_grid → theme_canon.py) — a cohort survives
the theme engine renaming it day to day. See theme_canon.py's docstring for
the known limitation this inherits: 55% of raw theme rows carry <3 tickers
and can only be stitched by name, so small/young cohorts fragment more than
this view's ribbon thickness implies. Disclosed in the caption below, not
just here — the ask was explicit that the UI, not only the code comment,
must carry it.

## The bands

Top 5 / 6-15 / 16-30 / Outside top 30 — the operator's own suggested cut
points. A cohort with no usable rank that week (never seen, a data gap, or
genuinely fell out of the engine's top ranks) reads as "Outside top 30" too —
same treatment the bump chart already gives a gap (no rank = no claim of a
better position), just folded into the worst band instead of a broken line.

## Two rendering decisions found by building this against the real snapshot

1. **The "Outside → Outside" self-loop is dropped from the plotted links.**
   Most cohorts spend most of their tracked life outside the top 30 — keeping
   that self-loop drew one dominant grey ribbon that visually swamped every
   promotion/demotion ribbon (verified against this snapshot: it carried the
   large majority of total plotted flow while containing zero "did this
   cohort move" information — a cohort staying outside the board is not the
   question the operator is asking). Every OTHER self-loop (a cohort holding
   its band) and every entry/exit to/from Outside stays — those are real
   signal ("still #2", "fell out of the top 30 entirely").
2. **Empty-band nodes get an invisible zero-width keep-alive self-loop.**
   Plotly's `go.Sankey` with `arrangement="fixed"` locks node position — but
   ONLY for nodes that carry at least one link. A node with literally zero
   throughput that week (e.g. no cohort landed in "16-30" that particular
   week) collapses out of the layout entirely and its column's OTHER nodes
   silently shift to fill the gap — discovered empirically while prototyping
   this view (a real Plotly behavior, not a hypothetical): the result reads
   as the band ROWS reordering week to week, which would be actively
   misleading for an alluvial whose entire point is "row = fixed meaning."
   A near-zero, fully-transparent self-loop (`rgba(0,0,0,0)`, value 0.3)
   gives the node enough throughput to keep its row locked without drawing
   anything visible.
"""
from __future__ import annotations

import html as _html
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from theme_data import get_canonical_weekly_grid, get_canonical_weeks_on_file, get_top_members_by_rs
# Reuse Grid's exact rank→color mapping (not a new ramp) so this view and the
# Grid/Heatmap read as ONE visual system — "brighter green = better rank" is
# already the app's established language; inventing a second one here would
# make the same number mean two different things on two tabs.
from theme_grid import _rank_color
from theme_palette import active

_OUTSIDE_BOUND = 30   # rank <= this is "on the board"; matches the population
                      # theme_bump.py's Top-N selector also treats as meaningful
_BAND_ORDER = ["Top 5", "6-15", "16-30", "Outside top 30"]   # fixed row order, top→bottom
_BAND_REPR_RANK = {"Top 5": 3, "6-15": 10, "16-30": 23}  # -> _rank_color(); Outside is special-cased below
_WORST_BAND = _BAND_ORDER[-1]   # the one whose self-loop gets dropped (see module docstring)


def _band_colors() -> dict[str, str]:
    """One representative fill per band. The first three route through
    _rank_color() (see import comment). "Outside top 30" deliberately does
    NOT use _rank_color's own out-of-range fill (`cell_out_bg`): that color
    was calibrated for a bordered TABLE CELL sitting next to other cells
    (Grid), where a near-background gray still reads because the border
    delineates it. Measured directly against the page background — where
    THIS band's floating, borderless Sankey nodes and ribbons actually sit —
    it comes out to ~1.15:1 contrast in both themes, i.e. nearly invisible,
    which would hide every "fell out of the top 30" / "climbed back in"
    ribbon: exactly the demotion signal this view exists to show. Its own
    palette's `cell_out_txt` (chosen for text legibility against that same
    near-background fill) measures ~2.6-2.9:1 against the page directly, so
    it's reused here as the fill instead."""
    P = active()
    colors = {b: _rank_color(_BAND_REPR_RANK[b])[0] for b in _BAND_ORDER if b in _BAND_REPR_RANK}
    colors[_WORST_BAND] = P["cell_out_txt"]
    return colors


def _band_of(rank: float | None) -> str:
    if rank is None or pd.isna(rank):
        return "Outside top 30"
    r = float(rank)
    if r <= 5:
        return "Top 5"
    if r <= 15:
        return "6-15"
    if r <= _OUTSIDE_BOUND:
        return "16-30"
    return "Outside top 30"


def _to_rgba(color: str, alpha: float) -> str:
    """Alpha-blend a color from _rank_color(), which returns either a hex
    string (theme_palette entries) or an "hsl(h, s%, l%)" string (the green
    gradient) — link ribbons need transparency, node bars don't, so only
    this call site needs both formats handled."""
    color = color.strip()
    if color.startswith("#"):
        h = color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    if color.startswith("hsl("):
        return "hsla(" + color[len("hsl("):-1] + f", {alpha})"
    return color  # unrecognized format — pass through rather than crash a render


def _usable_weeks(canon_grid: pd.DataFrame) -> list[date]:
    """Drop weeks where NO cohort has a usable rank (pre-rs_avg-engine dead
    columns) — identical precedent to theme_grid.render_grid's own
    `usable_weeks` filter, applied here to the canonical grid instead."""
    return [
        w for w in sorted(canon_grid["week_start"].unique())
        if canon_grid.loc[canon_grid["week_start"] == w, "week_rank"].notna().any()
    ]


def render_flow() -> None:
    st.header("Theme Rank Flow")
    st.caption(
        "Which cohorts are climbing toward the top of the board, and which are "
        "sliding out of it — same canonical identity as the Bump Chart (#315), "
        "shown as band-to-band flow instead of crossing lines. Ribbon thickness "
        "= how many cohorts made that move; rows never reorder, so a rising "
        "ribbon always means 'got stronger'."
    )

    with st.sidebar:
        st.subheader("Rank flow")
        weeks_n = st.slider(
            "Weeks of history", min_value=4, max_value=20, value=10, step=1,
            help="More weeks = more columns, not more detail per column — "
                 "past ~14 the ribbons get thin. Depth caption below shows "
                 "what's actually on file.",
        )

    canon_grid = get_canonical_weekly_grid(weeks=weeks_n)
    if canon_grid.empty:
        st.info("No canonical theme data in this window.")
        return

    weeks = _usable_weeks(canon_grid)
    if len(weeks) < 2:
        st.info("Fewer than 2 usable weeks in this window — nothing to flow between.")
        return

    sub = canon_grid[canon_grid["week_start"].isin(weeks)]
    id_to_name = sub.drop_duplicates("canonical_id").set_index("canonical_id")["canonical_name"]

    # Population: any cohort that reached the top-30 board at least once in
    # the visible window — mirrors the bump chart's own "only show cohorts
    # that mattered" philosophy instead of dragging in all ~300 tracked
    # cohorts (most of which never approach the board at all).
    on_board_ids = sub.loc[sub["week_rank"] <= _OUTSIDE_BOUND, "canonical_id"].unique()
    if len(on_board_ids) == 0:
        st.info(f"No cohort reached the top {_OUTSIDE_BOUND} in this window.")
        return

    piv = sub.pivot_table(
        index="canonical_id", columns="week_start", values="week_rank", aggfunc="first"
    ).reindex(index=on_board_ids, columns=weeks)
    band_piv = piv.map(_band_of)

    n_weeks, n_bands = len(weeks), len(_BAND_ORDER)
    node_index: dict[tuple[int, str], int] = {}
    labels: list[str] = []
    xs: list[float] = []
    ys: list[float] = []
    for wi in range(n_weeks):
        for bi, band in enumerate(_BAND_ORDER):
            node_index[(wi, band)] = len(labels)
            labels.append(band if wi == 0 else "")   # direct-label the leftmost column only
            xs.append(0.005 + wi / (n_weeks - 1) * 0.99)
            ys.append(0.02 + bi / (n_bands - 1) * 0.96)

    band_color = _band_colors()
    node_color = [band_color[b] for _wi in range(n_weeks) for b in _BAND_ORDER]

    # Links: (week_i -> week_i+1, band0 -> band1) -> {"count": n, "names": [...]}.
    # See module docstring point 1 for why the worst-band self-loop is excluded.
    links: dict[tuple[int, str, str], dict] = {}
    for i in range(n_weeks - 1):
        w0, w1 = weeks[i], weeks[i + 1]
        for cid, b0, b1 in zip(band_piv.index, band_piv[w0], band_piv[w1]):
            if b0 == _WORST_BAND and b1 == _WORST_BAND:
                continue
            key = (i, b0, b1)
            entry = links.setdefault(key, {"count": 0, "names": []})
            entry["count"] += 1
            entry["names"].append(str(id_to_name.get(cid, cid)))

    # node_link_total tracks ONLY whether a node has any plotted link touching
    # it — purely the zero-throughput predicate for the keep-alive fix below.
    # It is NOT a cohort count: an interior week's node collects both an
    # inbound link (from i-1) and an outbound link (to i+1) that both equal
    # that same band's population, so summing them double-counts; and the
    # worst band's dropped self-loop (module docstring point 1) means a node
    # can hold dozens of cohorts while touching zero plotted links. The real
    # per-node population is `band_piv[week].value_counts()` — computed
    # separately below as `node_population` and used for hover instead.
    node_link_total = {k: 0 for k in node_index}
    sources: list[int] = []
    targets: list[int] = []
    values: list[float] = []
    link_colors: list[str] = []
    hover_text: list[str] = []
    _NAME_PREVIEW = 6
    for (i, b0, b1), entry in links.items():
        s, t = node_index[(i, b0)], node_index[(i + 1, b1)]
        sources.append(s); targets.append(t); values.append(float(entry["count"]))
        link_colors.append(_to_rgba(band_color[b0], 0.45))
        names = sorted(entry["names"])
        shown = names[:_NAME_PREVIEW]
        more = len(names) - len(shown)
        preview = "<br>".join(_html.escape(n) for n in shown)
        if more > 0:
            preview += f"<br>…+{more} more"
        hover_text.append(
            f"<b>{_html.escape(b0)} → {_html.escape(b1)}</b><br>"
            f"{entry['count']} cohort(s)<br>{preview}<extra></extra>"
        )
        node_link_total[(i, b0)] += entry["count"]
        node_link_total[(i + 1, b1)] += entry["count"]

    # Keep-alive self-loops — see module docstring point 2. Fully transparent,
    # excluded from hover via an empty template.
    for key, total in node_link_total.items():
        if total == 0:
            n = node_index[key]
            sources.append(n); targets.append(n); values.append(0.3)
            link_colors.append("rgba(0,0,0,0)")
            hover_text.append("<extra></extra>")

    # Real per-node population — how many cohorts actually sat in that band
    # that week, straight off band_piv (not derived from links; see comment
    # above). Node hover uses this via customdata rather than `label` —
    # labels are blank past the first column to avoid 4 x n_weeks overlapping
    # text strings, but hover should still identify every node correctly.
    node_population = {
        (wi, band): int((band_piv[weeks[wi]] == band).sum())
        for wi in range(n_weeks) for band in _BAND_ORDER
    }
    node_customdata = [
        f"{band}, week of {weeks[wi]}<br>{node_population[(wi, band)]} cohort(s)"
        for wi in range(n_weeks) for band in _BAND_ORDER
    ]

    P = active()
    fig = go.Figure(go.Sankey(
        arrangement="fixed",
        node=dict(
            label=labels, x=xs, y=ys, color=node_color, pad=8, thickness=12,
            line=dict(width=0),
            customdata=node_customdata,
            hovertemplate="%{customdata}<extra></extra>",
        ),
        link=dict(
            source=sources, target=targets, value=values, color=link_colors,
            hovertemplate=hover_text,
        ),
    ))
    for wi, w in enumerate(weeks):
        fig.add_annotation(
            x=0.005 + wi / (n_weeks - 1) * 0.99, y=-0.06, xref="paper", yref="paper",
            text=w.strftime("%-m/%-d") if hasattr(w, "strftime") else str(w),
            showarrow=False, font=dict(size=10, color=P["chart_font"]), textangle=-45,
        )
    fig.update_layout(
        height=560, margin=dict(l=10, r=10, t=10, b=70),
        # Same theme-aware chart surface as theme_bump.py's Plotly figures
        # (not transparent) — a transparent paper_bgcolor would silently rely
        # on whatever sits behind it, and node labels ("Top 5" etc, rendered
        # in Plotly's own default dark text) need a KNOWN light-mode surface
        # to stay legible.
        paper_bgcolor=P["chart_paper"], plot_bgcolor=P["chart_plot"],
        font=dict(size=12, color=P["chart_font"]),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.caption(
        f"Rows top→bottom: {' · '.join(_BAND_ORDER)}. Latest week **{weeks[-1]}**, "
        # get_canonical_weeks_on_file (not a second get_canonical_weekly_grid
        # call) — weeks=24 here never shares a cache key with weeks_n above
        # (slider caps at 20), so a second full grid call would be a
        # guaranteed cache miss just to report a count; this reads the count
        # off get_canonical_themes() directly instead.
        f"{len(weeks)} week(s) of {get_canonical_weeks_on_file(weeks=24)} "
        "total on file · "
        f"{len(on_board_ids)} cohort(s) reached the top {_OUTSIDE_BOUND} in this window. "
        "⚠ Small/young cohorts (most theme rows carry under 3 tickers) can only be "
        "matched across days by name, not ticker set, so this count understates real "
        "fragmentation — read it as directional, not exact."
    )

    # ── Which cohorts moved bands in the most recent transition ────────────
    # A plain-text companion to the ribbons — answers "which ones, specifically"
    # without requiring a hover on every ribbon.
    last_w0, last_w1 = weeks[-2], weeks[-1]
    moves = []
    for cid in band_piv.index:
        b0, b1 = band_piv.at[cid, last_w0], band_piv.at[cid, last_w1]
        if b0 == b1:
            continue
        i0, i1 = _BAND_ORDER.index(b0), _BAND_ORDER.index(b1)
        moves.append({
            "Theme": id_to_name.get(cid, cid),
            "Was": b0, "Now": b1,
            "Direction": "↑ Promoted" if i1 < i0 else "↓ Demoted",
            "_mag": i0 - i1,
        })
    if moves:
        moves_df = pd.DataFrame(moves).sort_values("_mag", ascending=False).drop(columns="_mag")
        with st.expander(f"Band changes, {last_w0} → {last_w1} ({len(moves_df)})"):
            st.dataframe(moves_df, use_container_width=True, hide_index=True)

    # Top members of the CURRENT Top-5 band only — mirrors theme_bump.py's
    # "current members" expander, scoped tight (5 cohorts) instead of the
    # full on-board population.
    top5_now = band_piv[band_piv[weeks[-1]] == "Top 5"].index
    latest_tickers: dict[str, tuple] = {}
    for cid in top5_now:
        row = sub[(sub["canonical_id"] == cid) & (sub["week_start"] == weeks[-1])]
        if not row.empty and row.iloc[0]["tickers"]:
            latest_tickers[id_to_name.get(cid, cid)] = tuple(row.iloc[0]["tickers"])
    preview = get_top_members_by_rs(latest_tickers, n=4) if latest_tickers else {}
    if preview:
        with st.expander("Top members — current Top 5"):
            for name in sorted(preview):
                st.caption(f"**{name}** — {preview[name]}")
