"""Rank-BAND flow — ONE transition, readable on a phone (#640, rebuilt 2026-09-19).

THE QUESTION it answers is unchanged since the 8/8 ask: "which themes are gaining
strength, where in the arc" — a cohort's canonical identity (theme_canon.py, the
same source as theme_bump.py and theme_movers.py) tracked between two weeks, by
RANK BAND rather than exact rank, so the eye reads a ribbon instead of tracing a
line through a crossing.

WHY IT IS ONE TRANSITION, NOT ELEVEN (operator 2026-09-11 on the previous
build: *"still not readable, can this view even be fixed?"*, then, sending his
own Income → Expense flow: *"what I really want is a sankey chart like this
which is readable"*). His chart is three columns and every node names its own
value and share. The old Rank Flow drew every week at once — measured on the
live grid, 24 weeks of transitions is 357 ribbons across 218 cohorts; ONE
four-week hop is 17 ribbons across 55, and 95% of those cohorts actually moved.
Density was the defect, not styling. Design + sign-off:
`apollo_the_wise/docs/design/rank_flow_readable_sankey.md` (§ MOBILE PASS).

## What the page shows, top to bottom

1. **The answer in a sentence** — `25 climbed · 27 fell · 3 held`, then the
   single biggest move BY NAME (largest band distance; a climb beats a fall at
   equal distance; first name alphabetically inside that ribbon). Derived from
   `compute_band_flow`'s `links` by `summarize_transition`.
2. **The chart** — source bands on the left, target bands on the right, in a
   PORTRAIT figure sized for a 390px screen: labels live in fixed pixel margins
   either side, so the ribbon span is what is left of the width (~150px on a
   phone, ~350px on a desktop inside a capped container) and no label ever sits
   on a ribbon. Every band node is labelled with its own count and share
   (`16-30 · 15 · 27%`) PERMANENTLY — hover does not exist on a phone. Tapping a
   ribbon still lists its names.
3. **A caption** carrying the honesty about the `New / unranked` band (below).
4. **A movers list** — one line per ribbon, biggest band distance first, naming
   the cohorts that made that move; the stay-put cohorts collapse to one line.
   The full per-cohort table stays in an expander.

## The bands

Top 5 / 6-15 / 16-30 / 31+ / New / unranked — five fixed rows, top→bottom.
The old "Outside top 30" band merged four facts (rank 31, rank 140, never-seen,
and a data gap) and was split 2026-09-11:

- **"31+"** — a cohort WITH a known week_rank that is simply worse than the
  board cut. Matches `theme_grid._rank_color`'s own "cell_out" treatment.
- **"New / unranked"** (renamed from "No rank" 2026-09-19, operator ruling) —
  no usable rank at all that week: absent from that week's snapshot, or a row
  present with a null `rs_avg`. MEASURED over the chosen 08-17 → 09-14 hop, of
  the 20 cohorts that climbed out of this band 18 had NEVER been ranked in the
  19 weeks on file (a theme being born), 2 had a row but no rank, 0 were
  returning. So the fattest ribbon on the chart is real, and the ruling was to
  RENAME rather than add a sixth band for a two-cohort case: the honesty goes
  in the caption, computed for the displayed hop by `classify_unranked_edges`,
  not in the geometry. Found while building the caption: EVERY null-rank row
  in the snapshot (798 of 1895) is stage Fading or Retired — a null rank is
  the engine's own verdict on the theme, not a hole in our export — so the
  caption says "listed but unscored", not "data gap".

## Rendering decisions (each found by rendering against the real snapshot)

1. **Only the "New / unranked → New / unranked" self-loop is dropped** by
   `compute_band_flow` (zero information: no data two weeks running). Over a
   single hop it cannot occur anyway — the population is cohorts on the board
   in at least one of the two weeks — so every node's total is exactly its
   band population and the sentence, the labels and the ribbons all add up to
   the same cohort count.
2. **Node position is the node's CENTRE, and a zero coordinate is ignored.**
   Read off plotly.min.js 6.9.0's sankey renderer: with `arrangement="fixed"`
   it sets `y0 = y*height - h/2, y1 = y*height + h/2` for every node whose
   `x` AND `y` are truthy. `_node_layout` therefore reproduces d3-sankey's own
   height rule (`ky = (H - (n-1)*pad) / column_total`) so the margin labels
   land on their nodes, and no node is ever placed at exactly 0.
3. **An empty band still needs a link or Plotly drops the node** and the
   column's other nodes shift to fill the gap (the rows would appear to
   reorder). A ZERO-value link does not count — Plotly drops it AND the
   node, and the fixed positions then land on the wrong nodes (rendered and
   seen 2026-09-19). The old build's transparent 0.3-value SELF-loop kept
   the node but made the graph circular, which moves the node's column in
   the layout pass and throws the height rule in (2) off. So: a transparent
   0.3-value link from the empty band's SOURCE node to the same band's
   TARGET node. It adds 0.3 to both column totals alike, `_node_layout` is
   given the same values, and the empty node draws as a ~2px hairline under
   its label.
4. **Ribbons are colored by DIRECTION** (`theme_grid`'s `_DELTA_UP` /
   `_DELTA_DOWN` / `_DELTA_FLAT` — the same green/red/grey the Grid's
   rank-change chips use). **A move into or out of `New / unranked` is
   directional too (2026-09-19).** The 09-11 build painted those neutral,
   reasoning that a band meaning "absent OR a data gap" must not read as a
   climb or a fall. The measurement above answered that: leaving the band is
   a theme being born (18 of 20), and a null rank is the engine marking the
   theme fading/retired, never a snapshot hole. Painting `New / unranked →
   Top 5` grey would also contradict the sentence above it, which counts it
   as a climb — the header and the ribbons must agree. Held ribbons are
   drawn at lower opacity than moves, but only mildly: on a single hop the
   stayers are 3 of 55, not the bulk.
"""
from __future__ import annotations

import html as _html
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from theme_data import get_canonical_weekly_grid, get_top_members_by_rs
# Reuse Grid's exact rank→color mapping (not a new ramp) so this view and the
# Grid read as ONE visual system — "brighter green = better rank" is already
# the app's established language. _DELTA_UP/DOWN/FLAT are the same reuse
# principle applied to ribbon color (module docstring point 4).
from theme_grid import _DELTA_DOWN, _DELTA_FLAT, _DELTA_UP, _rank_color
from theme_palette import active

_OUTSIDE_BOUND = 30   # rank <= this is "on the board"; matches the population
                      # theme_bump.py's Top-N selector also treats as meaningful
_KNOWN_LOW_BAND = "31+"              # known week_rank, just worse than the board cut
_NO_RANK_BAND = "New / unranked"     # no usable rank at all that week — see module docstring "The bands"
_BAND_ORDER = ["Top 5", "6-15", "16-30", _KNOWN_LOW_BAND, _NO_RANK_BAND]   # fixed row order, top→bottom
_BAND_REPR_RANK = {"Top 5": 3, "6-15": 10, "16-30": 23}  # -> _rank_color(); the two lower bands are special-cased below
_BAND_RANK = {b: i for i, b in enumerate(_BAND_ORDER)}   # 0 = best row — used to sign a ribbon's direction

_DEFAULT_HOP = 4                # weeks back, by default — see module docstring (95% movers vs 85% at one week)
_HOP_OPTIONS = (1, 2, 4, 8)     # the picker STAYS: a fast one-week move must be one click away
_ALL_HISTORY_WEEKS = 520        # "everything on file" sentinel, same as theme_forward.py — the caption
                                # needs the full history to say whether a cohort was ever ranked before

# Chart geometry — pixels, not fractions, so the label gutters are the same
# width on a 390px phone and a 1400px desktop; only the ribbon span flexes.
_FIG_HEIGHT = 480
_MARGIN_T, _MARGIN_B = 30, 8
_LABEL_MARGIN = 118             # px each side — "New / unranked" at 11px bold clipped at 104 (rendered 2026-09-19)
_LABEL_GAP = 6                  # px between a node and its label
_NODE_PAD = 14                  # px between bands — a 2-line label on a 1-cohort node spills ~10px each side
_NODE_THICKNESS = 12
_X_SRC, _X_TGT = 0.02, 0.98     # never exactly 0: Plotly skips a falsy coordinate (module docstring point 2)
_KEEP_ALIVE = 0.3               # value of the transparent link that keeps an empty band's nodes (point 3)
_CHART_MAX_WIDTH = 560          # st.container cap — a phone still gets its full width
_ALPHA_UP, _ALPHA_DOWN, _ALPHA_HELD = 0.55, 0.5, 0.35
_LIST_NAMES = 1                 # names shown per ribbon in the movers list before "+N more" — the
                                # mockup's form; three a line ran to five wrapped rows per ribbon at 390px
_HOVER_NAMES = 6                # names shown in a ribbon's tap/hover text


def _band_colors() -> dict[str, str]:
    """One representative fill per band. The first three route through
    _rank_color() (see import comment). The two lower bands deliberately do
    NOT use _rank_color's own out-of-range fill (`cell_out_bg`) or blank
    fill (`cell_blank_bg`): those were calibrated for a bordered TABLE CELL
    sitting next to other cells (Grid), where a near-background gray still
    reads because the border delineates it. Measured directly against the
    page background — where THIS view's floating, borderless Sankey nodes
    and ribbons actually sit — `cell_out_bg` comes out to ~1.15:1 contrast
    in both themes, i.e. nearly invisible, which would hide every "fell out
    of the top 30" / "climbed back in" ribbon: exactly the demotion signal
    this view exists to show.
      - "31+" (known, low rank) uses `cell_out_txt` (~2.6-2.9:1 against the
        page) — the same tone Grid's own "known but below the floor" cells
        use, chosen for legibility against that near-background fill.
      - "New / unranked" (no data at all) uses `border` (~1.4-1.55:1) —
        dimmer than "31+" on purpose (an absence of data should recede
        further than a known-bad fact), while still clearing the ~1.15:1
        floor already established above as "nearly invisible" for this chart."""
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


def _direction(b0: str, b1: str) -> int:
    """+1 = climbed toward the top row, -1 = fell toward the bottom, 0 = held.
    Signed by _BAND_RANK (lower index = better row) — INCLUDING moves into
    or out of `New / unranked`; see module docstring point 4 for why that
    band is directional since 2026-09-19."""
    delta = _BAND_RANK[b0] - _BAND_RANK[b1]
    return (delta > 0) - (delta < 0)


def _ribbon_color(b0: str, b1: str) -> str:
    """Ribbon fill for a band0->band1 transition — module docstring point 4.
    A climb is green, a fall red, a hold grey at mildly lower opacity."""
    d = _direction(b0, b1)
    if d > 0:
        return _to_rgba(_DELTA_UP, _ALPHA_UP)
    if d < 0:
        return _to_rgba(_DELTA_DOWN, _ALPHA_DOWN)
    return _to_rgba(_DELTA_FLAT, _ALPHA_HELD)


def summarize_transition(links: dict[tuple[int, str, str], dict]) -> dict:
    """The sentence above the chart, from `compute_band_flow`'s `links`.

    Returns {"climbed", "fell", "held", "total": cohort counts, "biggest": the
    single biggest move or None}. "Biggest" = the largest band DISTANCE
    (`New / unranked → Top 5` is 4 rows); at equal distance a climb beats a
    fall (the operator's question is which themes are climbing); inside the
    winning ribbon the first name alphabetically is shown and the rest are
    reported as `others` (so the page can say "3 more made the same move").
    Deterministic, so the same links always name the same cohort.
    Sums every link passed, whatever its week index — the page passes a
    single hop, so the three counts add up to the cohort population."""
    climbed = fell = held = 0
    best: dict | None = None
    best_key: tuple = ()
    for (_i, b0, b1), entry in links.items():
        d = _direction(b0, b1)
        n = int(entry["count"])
        if d > 0:
            climbed += n
        elif d < 0:
            fell += n
        else:
            held += n
        if d == 0:
            continue
        distance = abs(_BAND_RANK[b0] - _BAND_RANK[b1])
        name = min(entry["names"])
        key = (distance, d > 0)
        if best is None or key > best_key or (key == best_key and name < best["name"]):
            best_key = key
            best = {
                "name": name, "from": b0, "to": b1, "count": n, "others": n - 1,
                "direction": "climb" if d > 0 else "fall", "distance": distance,
            }
    return {"climbed": climbed, "fell": fell, "held": held,
            "total": climbed + fell + held, "biggest": best}


def band_shares(band_piv: pd.DataFrame, week: date) -> dict[str, tuple[int, float]]:
    """{band: (count, share of the population)} for one week's column of
    `band_piv`, every band present (zero if empty) in `_BAND_ORDER` order.
    Share denominator = the whole population, the same for both weeks, so
    the two columns' percentages are comparable."""
    total = len(band_piv)
    counts = band_piv[week].value_counts() if total else {}
    return {
        b: (int(counts.get(b, 0)), (int(counts.get(b, 0)) / total) if total else 0.0)
        for b in _BAND_ORDER
    }


def band_label(band: str, count: int, share: float) -> str:
    """The permanent on-chart node label: `16-30 · 15 · 27%`."""
    return f"{band} · {count} · {share:.0%}"


def classify_unranked_edges(
    grid: pd.DataFrame, band_piv: pd.DataFrame, w0: date, w1: date,
) -> dict:
    """WHY cohorts crossed into or out of `New / unranked` on this hop — the
    caption's honesty, computed rather than hard-coded (operator ruling
    2026-09-19: rename the band, state the split in words).

    Returns {
      "left":    {"new": never had a ranked week before w0,
                  "unscored": a row exists at w0 but its rank is null,
                  "returning": ranked in some week before w0, no row at w0},
      "entered": {"unscored": a row exists at w1 but its rank is null,
                  "absent": no row at w1},
      "history_weeks": usable weeks in `grid` before w0 — the horizon "new"
                       is judged against (a theme first ranked before the
                       file starts reads as new: the share is a floor).
    }
    Reproduces the 18 / 2 / 0 measurement in the design doc for the
    2026-08-17 → 09-14 hop. `grid` must be the FULL history, not a window
    (get_canonical_weekly_grid(weeks=_ALL_HISTORY_WEEKS))."""
    left = {"new": 0, "unscored": 0, "returning": 0}
    entered = {"unscored": 0, "absent": 0}
    if band_piv.empty:
        return {"left": left, "entered": entered, "history_weeks": 0}
    ranked = grid[grid["week_rank"].notna()]
    ranked_before = set(ranked.loc[ranked["week_start"] < w0, "canonical_id"])
    rows_w0 = set(grid.loc[grid["week_start"] == w0, "canonical_id"])
    rows_w1 = set(grid.loc[grid["week_start"] == w1, "canonical_id"])

    for cid in band_piv.index[(band_piv[w0] == _NO_RANK_BAND) & (band_piv[w1] != _NO_RANK_BAND)]:
        if cid in rows_w0:            # a row that week, rank null
            left["unscored"] += 1
        elif cid in ranked_before:
            left["returning"] += 1
        else:
            left["new"] += 1
    for cid in band_piv.index[(band_piv[w1] == _NO_RANK_BAND) & (band_piv[w0] != _NO_RANK_BAND)]:
        if cid in rows_w1:
            entered["unscored"] += 1
        else:
            entered["absent"] += 1
    history_weeks = int(ranked.loc[ranked["week_start"] < w0, "week_start"].nunique())
    return {"left": left, "entered": entered, "history_weeks": history_weeks}


def _node_values(src_counts: list[int], tgt_counts: list[int]) -> tuple[list[float], list[float]]:
    """Per-node throughput as Plotly will see it: the band's cohort count
    plus `_KEEP_ALIVE` on BOTH sides wherever either side is empty (that is
    exactly the link `build_flow_figure` adds — module docstring point 3)."""
    keep = [_KEEP_ALIVE if (s == 0 or t == 0) else 0.0 for s, t in zip(src_counts, tgt_counts)]
    return [s + k for s, k in zip(src_counts, keep)], [t + k for t, k in zip(tgt_counts, keep)]


def _node_layout(
    src_vals: list[float], tgt_vals: list[float], plot_height: float, pad: float,
) -> tuple[list[float], list[float]]:
    """Centre y (fraction of the plot height, Sankey top-down) for each band
    in `_BAND_ORDER`, one list per column, reproducing d3-sankey's own
    height rule so the margin annotations sit on their nodes (module
    docstring point 2): `ky = (H - (n-1)*pad) / max column total`, node
    height = value * ky, nodes stacked from the top with `pad` between.
    `src_vals`/`tgt_vals` come from `_node_values` (counts plus keep-alive),
    so an empty band is a ~2px hairline and its label sits on it. Never
    returns exactly 0.0 — Plotly ignores a falsy coordinate."""
    n = len(src_vals)
    total = max(sum(src_vals), sum(tgt_vals), 1e-9)
    ky = (plot_height - (n - 1) * pad) / total
    out: list[list[float]] = []
    for vals in (src_vals, tgt_vals):
        y = 0.0
        centres = []
        for v in vals:
            h = v * ky
            centres.append(max((y + h / 2) / plot_height, 1e-6))
            y += h + pad
        out.append(centres)
    return out[0], out[1]


_MD_SPECIALS = str.maketrans({c: "\\" + c for c in "*_`[]<>#~"})


def _md(text: str) -> str:
    """A theme name inside st.markdown: backslash-escape the characters
    markdown would format (`*`, `_`, backtick, brackets…). Not HTML-escaped —
    `&amp;` would show literally in markdown, and HTML is off by default."""
    return str(text).translate(_MD_SPECIALS)


def movers_lines(links: dict[tuple[int, str, str], dict], names_per_line: int = _LIST_NAMES) -> list[str]:
    """The movers list under the chart, as markdown lines: one per ribbon
    that changed band, biggest band distance first (climbs before falls at
    equal distance, then the larger ribbon), each naming up to
    `names_per_line` cohorts and "+N more". The stay-put ribbons collapse to
    ONE trailing line with their count. Pure — testable without Streamlit."""
    movers, held_names = [], []
    for (_i, b0, b1), entry in links.items():
        d = _direction(b0, b1)
        if d == 0:
            held_names.extend(entry["names"])
            continue
        movers.append((abs(_BAND_RANK[b0] - _BAND_RANK[b1]), d, int(entry["count"]), b0, b1, sorted(entry["names"])))
    movers.sort(key=lambda m: (-m[0], -m[1], -m[2], m[3]))

    def _names(names: list[str]) -> str:
        shown = ", ".join(_md(n) for n in names[:names_per_line])
        more = len(names) - min(len(names), names_per_line)
        return shown + (f", +{more} more" if more > 0 else "")

    lines = [
        f"{':green[▲]' if d > 0 else ':red[▼]'} **{b0} → {b1}** · {n} — {_names(names)}"
        for _dist, d, n, b0, b1, names in movers
    ]
    if held_names:
        lines.append(f":grey[—] **held their band** · {len(held_names)} — {_names(sorted(held_names))}")
    return lines


def build_flow_figure(
    band_piv: pd.DataFrame,
    links: dict[tuple[int, str, str], dict],
    w0: date,
    w1: date,
    left_header: str,
    right_header: str,
) -> go.Figure:
    """The two-column Sankey for ONE hop (`links` from a two-week
    `compute_band_flow` call, so every key's week index is 0). Pure Plotly —
    testable without Streamlit. Geometry per the module docstring: fixed
    node centres from `_node_layout`, labels as annotations in the pixel
    margins, a zero-value transparent link keeping any empty band's nodes
    alive, ribbons coloured by `_ribbon_color`."""
    P = active()
    band_color = _band_colors()
    src, tgt = band_shares(band_piv, w0), band_shares(band_piv, w1)
    plot_h = _FIG_HEIGHT - _MARGIN_T - _MARGIN_B
    src_vals, tgt_vals = _node_values([src[b][0] for b in _BAND_ORDER], [tgt[b][0] for b in _BAND_ORDER])
    ys_src, ys_tgt = _node_layout(src_vals, tgt_vals, plot_h, _NODE_PAD)
    n_bands = len(_BAND_ORDER)
    node_index = {(0, b): i for i, b in enumerate(_BAND_ORDER)}
    node_index.update({(1, b): n_bands + i for i, b in enumerate(_BAND_ORDER)})
    xs = [_X_SRC] * n_bands + [_X_TGT] * n_bands
    ys = ys_src + ys_tgt
    node_color = [band_color[b] for b in _BAND_ORDER] * 2
    node_hover = [
        f"{_html.escape(b)} · {_html.escape(left_header)}<br>{src[b][0]} cohort(s) · {src[b][1]:.0%}"
        for b in _BAND_ORDER
    ] + [
        f"{_html.escape(b)} · {_html.escape(right_header)}<br>{tgt[b][0]} cohort(s) · {tgt[b][1]:.0%}"
        for b in _BAND_ORDER
    ]

    sources: list[int] = []
    targets: list[int] = []
    values: list[float] = []
    link_colors: list[str] = []
    hover_text: list[str] = []
    for (_i, b0, b1), entry in sorted(links.items(), key=lambda kv: (_BAND_RANK[kv[0][1]], _BAND_RANK[kv[0][2]])):
        sources.append(node_index[(0, b0)])
        targets.append(node_index[(1, b1)])
        values.append(float(entry["count"]))
        link_colors.append(_ribbon_color(b0, b1))
        names = sorted(entry["names"])
        preview = "<br>".join(_html.escape(n) for n in names[:_HOVER_NAMES])
        if len(names) > _HOVER_NAMES:
            preview += f"<br>…+{len(names) - _HOVER_NAMES} more"
        hover_text.append(
            f"<b>{_html.escape(b0)} → {_html.escape(b1)}</b><br>"
            f"{entry['count']} cohort(s)<br>{preview}<extra></extra>"
        )
    # Keep-alive for empty bands — module docstring point 3. Transparent, no
    # hover, and the SAME value `_node_values` assumed, or the labels drift.
    for b in _BAND_ORDER:
        if src[b][0] == 0 or tgt[b][0] == 0:
            sources.append(node_index[(0, b)])
            targets.append(node_index[(1, b)])
            values.append(_KEEP_ALIVE)
            link_colors.append("rgba(0,0,0,0)")
            hover_text.append("<extra></extra>")

    fig = go.Figure(go.Sankey(
        arrangement="fixed",
        node=dict(
            label=[""] * (2 * n_bands),   # labels are the margin annotations below, never Plotly's own
            x=xs, y=ys, color=node_color, pad=_NODE_PAD, thickness=_NODE_THICKNESS,
            line=dict(width=0),
            customdata=node_hover, hovertemplate="%{customdata}<extra></extra>",
        ),
        link=dict(source=sources, target=targets, value=values, color=link_colors, hovertemplate=hover_text),
    ))

    muted = _to_rgba(P["chart_font"], 0.7)
    # Sankey y runs top(0)→bottom(1); paper y runs bottom(0)→top(1) — hence 1 - y.
    for b, y in zip(_BAND_ORDER, ys_src):
        fig.add_annotation(
            x=0, xshift=-_LABEL_GAP, y=1 - y, xref="paper", yref="paper", xanchor="right", yanchor="middle",
            text=f"<b>{_html.escape(b)}</b><br><span style='font-size:9.5px;color:{muted}'>"
                 f"{src[b][0]} · {src[b][1]:.0%}</span>",
            showarrow=False, align="right", font=dict(size=11, color=P["chart_font"]),
        )
    for b, y in zip(_BAND_ORDER, ys_tgt):
        fig.add_annotation(
            x=1, xshift=_LABEL_GAP, y=1 - y, xref="paper", yref="paper", xanchor="left", yanchor="middle",
            text=f"<b>{_html.escape(b)}</b><br><span style='font-size:9.5px;color:{muted}'>"
                 f"{tgt[b][0]} · {tgt[b][1]:.0%}</span>",
            showarrow=False, align="left", font=dict(size=11, color=P["chart_font"]),
        )
    # Column headers hang from the figure's OUTER edges (the margin plus the
    # plot), so they never clip however narrow the ribbon span gets.
    fig.add_annotation(
        x=0, xshift=-(_LABEL_MARGIN - 2), y=1, yshift=12, xref="paper", yref="paper",
        xanchor="left", yanchor="bottom",
        text=_html.escape(left_header).upper(), showarrow=False, font=dict(size=9.5, color=muted),
    )
    fig.add_annotation(
        x=1, xshift=_LABEL_MARGIN - 2, y=1, yshift=12, xref="paper", yref="paper",
        xanchor="right", yanchor="bottom",
        text=_html.escape(right_header).upper(), showarrow=False, font=dict(size=9.5, color=muted),
    )
    fig.update_layout(
        height=_FIG_HEIGHT,
        margin=dict(l=_LABEL_MARGIN, r=_LABEL_MARGIN, t=_MARGIN_T, b=_MARGIN_B),
        # Same theme-aware chart surface as theme_bump.py's Plotly figures
        # (not transparent) — the annotations render in a fixed font color
        # and need a KNOWN surface to stay legible.
        paper_bgcolor=P["chart_paper"], plot_bgcolor=P["chart_plot"],
        font=dict(size=12, color=P["chart_font"]),
    )
    return fig


def _usable_weeks(canon_grid: pd.DataFrame) -> list[date]:
    """Drop weeks where NO cohort has a usable rank (pre-rs_avg-engine dead
    columns) — identical precedent to theme_grid.render_grid's own
    `usable_weeks` filter, applied here to the canonical grid instead."""
    return [
        w for w in sorted(canon_grid["week_start"].unique())
        if canon_grid.loc[canon_grid["week_start"] == w, "week_rank"].notna().any()
    ]


def _fmt_week(w: date) -> str:
    return w.strftime("%-d %b") if hasattr(w, "strftime") else str(w)


def render_flow() -> None:
    st.header("Theme Rank Flow")
    st.caption(
        "One hop, in words and one picture: which cohorts climbed toward the top "
        "of the board and which slid out of it between two weeks — same canonical "
        "identity as the Bump Chart and Weekly Movers. Ribbon width = how many "
        "cohorts made that move; rows never reorder."
    )

    grid = get_canonical_weekly_grid(weeks=_ALL_HISTORY_WEEKS)
    if grid.empty:
        st.info("No canonical theme data on file.")
        return
    weeks = _usable_weeks(grid)
    if len(weeks) < 2:
        st.info("Fewer than 2 usable weeks on file — nothing to flow between.")
        return

    with st.sidebar:
        st.subheader("Rank flow")
        w1 = st.selectbox(
            "Week", options=list(reversed(weeks)), index=0, format_func=_fmt_week, key="flow_week",
            help="The week on the right of the chart. Latest on file by default.",
        )
        hop = st.selectbox(
            "Compared with", options=list(_HOP_OPTIONS), index=_HOP_OPTIONS.index(_DEFAULT_HOP),
            format_func=lambda h: f"{h} week{'s' if h > 1 else ''} earlier", key="flow_hop",
            help="Four weeks is the default: the same number of ribbons as one week, "
                 "but 95% of the cohorts actually moved (85% at one week). A fast "
                 "one-week move is one click away.",
        )
    i1 = weeks.index(w1)
    i0 = max(0, i1 - hop)
    if i0 == i1:
        st.info(f"No usable week on file before {_fmt_week(w1)} to compare with.")
        return
    w0 = weeks[i0]
    actual_hop = i1 - i0

    band_piv, links = compute_band_flow(grid, [w0, w1])
    if band_piv.empty:
        st.info(f"No cohort reached the top {_OUTSIDE_BOUND} in either week.")
        return
    n_cohorts = len(band_piv)
    summary = summarize_transition(links)
    left_header = f"{actual_hop} week{'s' if actual_hop > 1 else ''} ago · {_fmt_week(w0)}"
    right_header = f"{_fmt_week(w1)}" + (" · now" if i1 == len(weeks) - 1 else "")

    # ── The answer in a sentence ───────────────────────────────────────────
    st.caption(
        f"{_fmt_week(w0)} → {_fmt_week(w1)} · {actual_hop}-week hop · {n_cohorts} cohorts"
        + (f" (only {actual_hop} week(s) on file before {_fmt_week(w1)})" if actual_hop < hop else "")
    )
    st.markdown(
        f"### :green[{summary['climbed']} climbed] · :red[{summary['fell']} fell] · "
        f":grey[{summary['held']} held]"
    )
    big = summary["biggest"]
    if big:
        others = f" ({big['others']} more made the same move)" if big["others"] > 0 else ""
        st.markdown(f"Biggest {big['direction']} — **{_md(big['name'])}**, {big['from']} → {big['to']}{others}.")

    # ── The chart ──────────────────────────────────────────────────────────
    with st.container(width=_CHART_MAX_WIDTH):
        fig = build_flow_figure(band_piv, links, w0, w1, left_header, right_header)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # ── The caption: legend + the New / unranked honesty, computed ─────────
    edges = classify_unranked_edges(grid, band_piv, w0, w1)
    left_n, entered_n = sum(edges["left"].values()), sum(edges["entered"].values())
    honesty = []
    if left_n:
        honesty.append(
            f"Of the {left_n} that climbed out of {_NO_RANK_BAND}: {edges['left']['new']} had never "
            f"been ranked in the {edges['history_weeks']} week(s) on file before {_fmt_week(w0)} "
            f"(new themes), {edges['left']['unscored']} were listed but unscored by the engine that "
            f"week, {edges['left']['returning']} had been ranked before and were missing."
        )
    if entered_n:
        honesty.append(
            f"Of the {entered_n} that dropped into it: {edges['entered']['unscored']} were still "
            f"listed but unscored (the engine marks a theme fading or retired), "
            f"{edges['entered']['absent']} were gone from the snapshot."
        )
    st.caption(
        ":green[green] = climbed · :red[red] = fell · :grey[grey] = held · ribbon width = cohorts · "
        "tap a ribbon for names. " + " ".join(honesty) + " ⚠ Small cohorts (most theme rows carry "
        "under 3 tickers) can only be matched across weeks by name, so a rename can read as one "
        "theme ending and another being born — directional, not exact."
    )

    # ── The movers list ────────────────────────────────────────────────────
    lines = movers_lines(links)
    if lines:
        st.markdown("\n".join(f"- {line}" for line in lines))

    # Full per-cohort table — every cohort that changed band, biggest move first.
    sub = grid[grid["week_start"].isin([w0, w1])]
    id_to_name = sub.drop_duplicates("canonical_id").set_index("canonical_id")["canonical_name"]
    moves = []
    for cid in band_piv.index:
        b0, b1 = band_piv.at[cid, w0], band_piv.at[cid, w1]
        if b0 == b1:
            continue
        i0b, i1b = _BAND_RANK[b0], _BAND_RANK[b1]
        moves.append({
            "Theme": id_to_name.get(cid, cid),
            "Was": b0, "Now": b1,
            "Direction": "↑ Climbed" if i1b < i0b else "↓ Fell",
            "_mag": i0b - i1b,
        })
    if moves:
        moves_df = pd.DataFrame(moves).sort_values("_mag", ascending=False).drop(columns="_mag")
        with st.expander(f"Every band change, {_fmt_week(w0)} → {_fmt_week(w1)} ({len(moves_df)})"):
            st.dataframe(moves_df, width="stretch", hide_index=True)

    # Top members of the CURRENT Top-5 band only — mirrors theme_bump.py's
    # "current members" expander, scoped tight (5 cohorts).
    top5_now = band_piv[band_piv[w1] == "Top 5"].index
    latest_tickers: dict[str, tuple] = {}
    for cid in top5_now:
        row = sub[(sub["canonical_id"] == cid) & (sub["week_start"] == w1)]
        if not row.empty and row.iloc[0]["tickers"]:
            latest_tickers[id_to_name.get(cid, cid)] = tuple(row.iloc[0]["tickers"])
    preview = get_top_members_by_rs(latest_tickers, n=4) if latest_tickers else {}
    if preview:
        with st.expander(f"Top members — Top 5 as of {_fmt_week(w1)}"):
            for name in sorted(preview):
                st.caption(f"**{name}** — {preview[name]}")
