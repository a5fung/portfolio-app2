"""Rank-BAND flow (alluvial) — canonical identity, over time (operator ask 8/8).

PROBLEM WITH THE BUMP CHART: it plots one continuous line per cohort at its
exact weekly rank. With up to 8 concurrent lines constantly crossing (a rank
board reshuffles most weeks), the eye can't hold a single line's path through
a crossing — the operator's own words: "not easiest to read... see if
there's better visuals... like Sankey." This view answers the SAME question
("which themes are gaining strength, where in the arc") with an alluvial:
weeks on the x-axis, five FIXED rank bands as rows (never reordered), ribbons
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

Top 5 / 6-15 / 16-30 / 31+ / No rank — five fixed rows, top→bottom. The old
"Outside top 30" band merged four different facts (rank 31, rank 140,
never-seen, and a data gap all read identically), which made every ribbon
into or out of it uninterpretable — fixed 2026-09-11 (#640) by splitting it:

- **"31+"** — a cohort WITH a known week_rank that is simply worse than the
  board cut. The number is real; we just don't show it precisely (matches
  `theme_grid._rank_color`'s own "cell_out" treatment for the same fact —
  a known-but-low rank is a distinct visual class from no data at all, and
  this view now draws that same line).
- **"No rank"** — no usable rank AT ALL that week: absent from that week's
  snapshot (never seen, or retired that week) or a data gap (row present,
  `rs_avg` null). This is a fact about our DATA, not about the cohort's
  strength, and is kept visually the dimmest row for exactly that reason.

Measured against the live snapshot at the page's own default (10 weeks): of
the cohort-weeks that would have fallen into the old merged band, 190 of
1230 (~15%) carried a KNOWN rank past the cut and were being told apart from
zero-data weeks by nothing. See `theme_movers.py`'s own 2026-09-10 fix for
the sibling defect this reuses the same distinction from (`pd.notna(rank)`
on the grid's `week_rank` column — a known prior rank counts even past the
cut, full stop).

## Rendering decisions found by building this against the real snapshot

1. **Only the "No rank → No rank" self-loop is dropped from the plotted
   links** (before 2026-09-11 this was the whole merged "Outside → Outside"
   self-loop). A cohort with zero data two weeks running carries zero "did
   this cohort move" information and, measured against the live snapshot,
   is overwhelmingly the majority case (826 of 980 dropped-candidate links
   at the 10-week default) — keeping it would still swamp every real move.
   A cohort sitting continuously in "31+" (a known, if unexciting, rank) is
   NOT dropped — that self-loop is small in practice (90 of the same 980)
   and, unlike "No rank", it is a real fact about the cohort, not an
   absence of one. Every entry/exit across any band boundary stays,
   including "31+ ↔ No rank" (a known-low cohort dropping off the map
   entirely, or reappearing) — that transition is real signal.
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
3. **Band labels are NOT Plotly's built-in node `label` — they're separate
   `paper`-space annotations in a reserved left gutter (`_LABEL_GUTTER`).**
   `go.Sankey` draws a leftmost-column node's label to ITS LEFT by default,
   but this view's leftmost node sits at x≈0.005 with no room to its left —
   verified empirically (screenshot, 2026-09-11): Plotly instead draws the
   text ON TOP of the node bar and the ribbons immediately behind it, which
   is the "labels are unreadable" defect reported from the phone. Shrinking
   the Sankey's own `domain` to `[_LABEL_GUTTER, 1.0]` and placing each band
   name as its own annotation at x=0 fixes this — Plotly's Sankey `y` and
   `paper`-space `y` run in OPPOSITE directions (confirmed empirically), so
   an annotation's y is `1 - <that node's y>`, not the same value.
4. **Ribbons are colored by DIRECTION, not by source band** (before
   2026-09-11 every ribbon was a shade of the source band's own color, i.e.
   mostly the same pale green — "nothing signals which move matters").
   Reuses `theme_grid`'s own `_DELTA_UP` / `_DELTA_DOWN` / `_DELTA_FLAT` —
   the SAME green/red/grey vocabulary the Grid view's own rank-change chips
   already use, so "green = improved, red = worsened" means the same thing
   on both tabs. Same-band ("flat") ribbons additionally get a much lower
   opacity (0.12 vs 0.5-0.55) — they're still drawn (a cohort holding a
   real rank is a fact, not a gap), but ink is weighted toward the moves
   that actually answer "which themes are climbing," per the operator's own
   question, rather than the boring "stayed put" case that used to dominate
   the page.
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
# make the same number mean two different things on two tabs. _DELTA_UP/DOWN/
# FLAT are the SAME reuse principle applied to ribbon color (see module
# docstring point 4) — Grid's own rank-change chips already use this
# green/red/grey vocabulary for "improved / worsened / flat."
from theme_grid import _DELTA_DOWN, _DELTA_FLAT, _DELTA_UP, _rank_color
from theme_palette import active

_OUTSIDE_BOUND = 30   # rank <= this is "on the board"; matches the population
                      # theme_bump.py's Top-N selector also treats as meaningful
_KNOWN_LOW_BAND = "31+"      # known week_rank, just worse than the board cut
_NO_RANK_BAND = "No rank"    # no usable rank at all that week — never seen or a data gap
_BAND_ORDER = ["Top 5", "6-15", "16-30", _KNOWN_LOW_BAND, _NO_RANK_BAND]   # fixed row order, top→bottom
_BAND_REPR_RANK = {"Top 5": 3, "6-15": 10, "16-30": 23}  # -> _rank_color(); the two lower bands are special-cased below
_BAND_RANK = {b: i for i, b in enumerate(_BAND_ORDER)}   # 0 = best row — used to sign a ribbon's direction
_LABEL_GUTTER = 0.11   # fraction of figure width reserved for band-name annotations (see module docstring point 3)


def _band_colors() -> dict[str, str]:
    """One representative fill per band. The first three route through
    _rank_color() (see import comment). The two lower bands deliberately do
    NOT use _rank_color's own out-of-range fill (`cell_out_bg`) or blank
    fill (`cell_blank_bg`): those were calibrated for a bordered TABLE CELL
    sitting next to other cells (Grid), where a near-background gray still
    reads because the border delineates it. Measured directly against the
    page background — where THIS band's floating, borderless Sankey nodes
    and ribbons actually sit — `cell_out_bg` comes out to ~1.15:1 contrast
    in both themes, i.e. nearly invisible, which would hide every "fell out
    of the top 30" / "climbed back in" ribbon: exactly the demotion signal
    this view exists to show.
      - "31+" (known, low rank) uses `cell_out_txt` (~2.6-2.9:1 against the
        page) — the same tone Grid's own "known but below the floor" cells
        use, chosen for legibility against that near-background fill.
      - "No rank" (no data at all) uses `border` (~1.4-1.55:1) — dimmer
        than "31+" on purpose (an absence of data should recede further
        than a known-bad fact), while still clearing the ~1.15:1 floor
        already established above as "nearly invisible" for this chart."""
    P = active()
    colors = {b: _rank_color(_BAND_REPR_RANK[b])[0] for b in _BAND_ORDER if b in _BAND_REPR_RANK}
    colors[_KNOWN_LOW_BAND] = P["cell_out_txt"]
    colors[_NO_RANK_BAND] = P["border"]
    return colors


def _band_of(rank: float | None) -> str:
    """`rank` is a row's `week_rank` — NaN/None means no usable rank that
    week (absent from the snapshot, or present with a null rs_avg); any
    other float is a real, known rank, however large. See module docstring
    ("The bands") for why these two cases must never collapse together."""
    if rank is None or pd.isna(rank):
        return _NO_RANK_BAND
    r = float(rank)
    if r <= 5:
        return "Top 5"
    if r <= 15:
        return "6-15"
    if r <= _OUTSIDE_BOUND:
        return "16-30"
    return _KNOWN_LOW_BAND


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


def compute_band_flow(
    grid: pd.DataFrame,
    weeks: list[date],
    board_size: int = _OUTSIDE_BOUND,
) -> tuple[pd.DataFrame, dict[tuple[int, str, str], dict]]:
    """Pure computation behind the Sankey — no Streamlit, no Plotly, testable
    against a synthetic grid shaped like `theme_data.get_canonical_weekly_grid`'s
    output (the same shape `theme_movers.compute_weekly_movers` takes, so a
    `_grid()` fixture helper transfers between the two test files).

    Returns (band_piv, links):
      - `band_piv`: index=canonical_id, columns=`weeks`, values=band name
        (`_band_of` applied to that week's rank). Population = any cohort
        with rank <= board_size in at least one of the given weeks (mirrors
        the bump chart's "only show cohorts that mattered" philosophy) — a
        cohort that never reached the board at all is dropped entirely, NOT
        folded into "No rank" (that band means "on our radar, no usable
        rank THIS week," not "never relevant").
      - `links`: keyed by (week_index, band_from, band_to) ->
        {"count": n, "names": [canonical_name, ...]}. The "No rank -> No
        rank" self-loop is excluded — see module docstring point 1 for why
        (it is the dominant, zero-signal case: a cohort with no data two
        weeks running). Every other self-loop and every cross-band move,
        including into/out of "No rank", is kept.

    Empty grid, fewer than 2 weeks, or nobody ever reaching the board
    returns (empty DataFrame, {}).
    """
    if grid.empty or len(weeks) < 2:
        return pd.DataFrame(), {}
    sub = grid[grid["week_start"].isin(weeks)]
    on_board_ids = sub.loc[sub["week_rank"] <= board_size, "canonical_id"].unique()
    if len(on_board_ids) == 0:
        return pd.DataFrame(), {}
    id_to_name = sub.drop_duplicates("canonical_id").set_index("canonical_id")["canonical_name"]
    piv = sub.pivot_table(
        index="canonical_id", columns="week_start", values="week_rank", aggfunc="first"
    ).reindex(index=on_board_ids, columns=weeks)
    band_piv = piv.map(_band_of)

    links: dict[tuple[int, str, str], dict] = {}
    for i in range(len(weeks) - 1):
        w0, w1 = weeks[i], weeks[i + 1]
        for cid, b0, b1 in zip(band_piv.index, band_piv[w0], band_piv[w1]):
            if b0 == _NO_RANK_BAND and b1 == _NO_RANK_BAND:
                continue
            key = (i, b0, b1)
            entry = links.setdefault(key, {"count": 0, "names": []})
            entry["count"] += 1
            entry["names"].append(str(id_to_name.get(cid, cid)))
    return band_piv, links


def _ribbon_color(b0: str, b1: str) -> str:
    """Ribbon fill for a band0->band1 transition — see module docstring
    point 4. Direction is signed by _BAND_RANK (lower index = a better
    row): a positive delta moved toward the top (promotion, green), a
    negative delta moved toward the bottom (demotion, red), zero held its
    band (flat, grey) at much lower opacity so ink is weighted toward the
    moves that actually answer "which themes are climbing" rather than the
    boring "stayed put" case."""
    # A MOVE INVOLVING "No rank" IS NOT A STRENGTH MOVE (2026-09-11). `_BAND_RANK` puts No-rank
    # at the bottom, so a plain delta paints "we stopped having data" red as a demotion and "data
    # came back" green as a promotion. That is this task's own defect — a data fact rendered as a
    # strength fact — relocated into the colour channel, so it is fixed rather than left as a
    # preference. Neutral, but at full opacity: appearing or disappearing IS worth seeing, it just
    # is not a climb or a fall. The card flagged this and left it; it is correctness, not styling.
    if _NO_RANK_BAND in (b0, b1) and b0 != b1:
        return _to_rgba(_DELTA_FLAT, 0.45)
    band_delta = _BAND_RANK[b0] - _BAND_RANK[b1]
    if band_delta > 0:
        return _to_rgba(_DELTA_UP, 0.55)
    if band_delta < 0:
        return _to_rgba(_DELTA_DOWN, 0.5)
    return _to_rgba(_DELTA_FLAT, 0.12)


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

    # Population + band-of-each-cohort-each-week + week-to-week transitions —
    # see compute_band_flow's own docstring; pulled out to a pure function so
    # it's testable without Streamlit/Plotly (test_theme_flow.py).
    band_piv, links = compute_band_flow(canon_grid, weeks)
    if band_piv.empty:
        st.info(f"No cohort reached the top {_OUTSIDE_BOUND} in this window.")
        return
    on_board_ids = band_piv.index

    n_weeks, n_bands = len(weeks), len(_BAND_ORDER)
    node_index: dict[tuple[int, str], int] = {}
    labels: list[str] = []
    xs: list[float] = []
    ys: list[float] = []
    for wi in range(n_weeks):
        for bi, band in enumerate(_BAND_ORDER):
            node_index[(wi, band)] = len(labels)
            labels.append("")   # no built-in node labels — see module docstring point 3
            xs.append(0.005 + wi / (n_weeks - 1) * 0.99)
            ys.append(0.02 + bi / (n_bands - 1) * 0.96)

    band_color = _band_colors()
    node_color = [band_color[b] for _wi in range(n_weeks) for b in _BAND_ORDER]

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
        # Color by DIRECTION, not by source band — see _ribbon_color/module
        # docstring point 4.
        link_colors.append(_ribbon_color(b0, b1))
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
    # above). Node hover uses this via customdata rather than `label` — every
    # node's `label` is now blank (module docstring point 3), so hover is the
    # only way to identify a node; this must stand in for it correctly.
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
        # Reserve a left gutter for the band-name annotations below instead
        # of Plotly's own node `label` — see module docstring point 3.
        domain=dict(x=[_LABEL_GUTTER, 1.0], y=[0, 1]),
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
    # Band-name row labels, once per row (not per week — a row's y position
    # is the same in every column). Sankey `y` runs top(0)→bottom(1) but
    # `paper`-space `y` runs bottom(0)→top(1) — confirmed empirically while
    # building this fix, hence the `1 - sankey_y` flip (module docstring
    # point 3). x=0.0 sits inside the reserved gutter, left of every node.
    for bi, band in enumerate(_BAND_ORDER):
        sankey_y = 0.02 + bi / (n_bands - 1) * 0.96
        fig.add_annotation(
            x=0.0, y=1 - sankey_y, xref="paper", yref="paper",
            text=band, showarrow=False, xanchor="left", yanchor="middle",
            font=dict(size=11, color=P["chart_font"]),
        )
    for wi, w in enumerate(weeks):
        # Node x is relative to the Sankey's OWN domain (shrunk to
        # [_LABEL_GUTTER, 1.0] above); a week annotation lives in full
        # `paper` space, so it needs the same domain rescale to land under
        # its column instead of drifting left of it.
        node_x = 0.005 + wi / (n_weeks - 1) * 0.99
        paper_x = _LABEL_GUTTER + node_x * (1 - _LABEL_GUTTER)
        fig.add_annotation(
            x=paper_x, y=-0.06, xref="paper", yref="paper",
            text=w.strftime("%-m/%-d") if hasattr(w, "strftime") else str(w),
            showarrow=False, font=dict(size=10, color=P["chart_font"]), textangle=-45,
        )
    fig.update_layout(
        height=560, margin=dict(l=10, r=10, t=10, b=70),
        # Same theme-aware chart surface as theme_bump.py's Plotly figures
        # (not transparent) — a transparent paper_bgcolor would silently rely
        # on whatever sits behind it, and the band/week annotations (rendered
        # in a fixed font color) need a KNOWN surface to stay legible.
        paper_bgcolor=P["chart_paper"], plot_bgcolor=P["chart_plot"],
        font=dict(size=12, color=P["chart_font"]),
    )
    st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})

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
            st.dataframe(moves_df, width='stretch', hide_index=True)

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

