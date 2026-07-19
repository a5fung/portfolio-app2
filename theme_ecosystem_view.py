"""Ecosystem view — ADR 0032 two-level (ecosystem -> sub-theme) board (#472).

NEW Streamlit render mirroring the STRUCTURE of Apollo's Telegram `/themes`
v2 board (`format_ecosystem_board` in agents/market_intelligence/
theme_ecosystems.py): ecosystems ranked by the D3 boosted score, each
expandable to its member sub-themes nested with their GLOBAL theme rank, a
raw->boosted (Δ) readout per ecosystem, Fading sub-themes struck-through
INSIDE their ecosystem, and E-UNASSIGNED pinned last. This is not a literal
port of `format_ecosystem_board` itself — that function renders Telegram
Markdown (STAGE_EMOJI / _conviction_suffix / strikethrough live in Apollo's
briefing.py and are Telegram-specific plumbing this dashboard has no use
for). The numbers driving this view — `compute_ecosystem_scores` and
`_group_and_rank_ecosystems` — ARE ported verbatim (see ecosystem_score.py),
so the ranking/scoring matches the Telegram board exactly; only the
rendering layer is a fresh Streamlit-native build.

Data comes from theme_data.get_ecosystem_board(), which reuses the ported
scorer over the committed snapshot (apollo_themes_snapshot.json).

2026-07-19 follow-up (operator feedback): the two-level board lost the flat
Grid view's week-over-week visibility, so each sub-theme line now also
carries a COMPACT movement cell (sparkline + window Δ, `_movement_html`
below) — a basic strengthening/weakening read, not a re-implementation of
the Grid's full weekly heatmap (theme_grid.py is UNCHANGED and stays the
place for that). Movement data comes from theme_data's reuse of
get_weekly_grid + ecosystem_score.compute_theme_movement (new, not ported —
see that module's docstring).
"""
from __future__ import annotations

import html as _html
from urllib.parse import quote

import streamlit as st

from ecosystem_score import E_UNASSIGNED
from theme_data import get_ecosystem_board, get_top_members_by_rs

_STAGE_EMOJI = {
    "Nascent": "🌱",
    "Accelerating": "⚡",
    "Mainstream": "📊",
    "Fading": "🔻",
}

_DEFAULT_EXPANDED = 3   # top-N ranked ecosystems auto-expanded; rest collapsed

# 8-level unicode block sparkline — a BASIC directional read (not the Grid's
# full heatmap), normalized per-theme (min->max of its own recent points).
_SPARK_CHARS = "▁▂▃▄▅▆▇█"
_MOVE_UP = "#3fb950"      # green — strengthening (RS rising)
_MOVE_DOWN = "#f85149"    # red — weakening
_MOVE_FLAT = "#8b949e"    # grey — flat / no usable trend


def _sparkline(points: list[float]) -> str:
    """Map a short list of values onto an 8-level unicode block string,
    normalized to THIS theme's own min/max (a per-row sparkline, not a
    cross-theme-comparable scale). <2 points -> "" (caller degrades to "—")."""
    if len(points) < 2:
        return ""
    lo, hi = min(points), max(points)
    if hi == lo:
        mid = _SPARK_CHARS[len(_SPARK_CHARS) // 2]
        return mid * len(points)
    span = hi - lo
    n = len(_SPARK_CHARS) - 1
    chars = []
    for v in points:
        idx = int(round((v - lo) / span * n))
        idx = max(0, min(n, idx))
        chars.append(_SPARK_CHARS[idx])
    return "".join(chars)


def _movement_html(movement: dict | None) -> str:
    """Compact W/W movement cell (#472 follow-up, operator ask 2026-07-19):
    an 8-level sparkline of the recent weekly rs_avg points + a window Δ,
    green when strengthening / red when weakening. Themes with <2 usable
    weekly points (too new for a trend) degrade to a dim "—" — never crashes
    on sparse data. This is a BASIC directional signal; the full weekly
    heatmap is theme_grid.py's job (Grid view, unchanged)."""
    if not movement or movement.get("n", 0) < 2:
        return '<span style="opacity:0.55">—</span>'
    points = movement["points"]
    delta = movement["delta"]
    spark = _sparkline(points)
    if delta > 0:
        color, arrow = _MOVE_UP, "↑"
    elif delta < 0:
        color, arrow = _MOVE_DOWN, "↓"
    else:
        color, arrow = _MOVE_FLAT, "—"
    return (
        f'<span style="font-family:monospace,monospace">{spark}</span> '
        f'<span style="color:{color};font-weight:600">{arrow}{abs(delta):.1f}</span>'
    )


def _render_theme_line(st_dict: dict, rank: int | None, preview: str) -> None:
    """One active sub-theme: rank + drill-link + stage tag + RS + Δ, with a
    member-preview caption underneath (mirrors Apollo's _theme_line, adapted
    to two Streamlit calls instead of two Telegram Markdown lines).

    Built as escaped HTML (not an f-string dropped into st.markdown's default
    markdown mode) — theme names are engine-generated free text and routinely
    contain '&', '_', '*', '[', ']' (e.g. "Satellite Imagery & Geospatial
    Intelligence"), any of which would corrupt unescaped markdown bold/link
    syntax. Mirrors theme_grid.py's own `_html.escape` + raw `<a href>`
    pattern rather than the markdown-link shortcut used elsewhere in this
    app (the search box), which does NOT escape and is a latent bug there.
    """
    stage = st_dict.get("stage", "?")
    emoji = _STAGE_EMOJI.get(stage, "")
    rank_str = f"#{rank} " if rank is not None else ""
    delta = st_dict.get("delta")
    delta_str = f"  Δ{delta:+.1f}" if delta is not None else ""
    name = st_dict["name"]
    drill = quote(name, safe="")
    name_esc = _html.escape(name)
    stage_esc = _html.escape(stage)
    movement_html = _movement_html(st_dict.get("movement"))
    st.markdown(
        f'{rank_str}{emoji} <a href="?drill={drill}" target="_self">'
        f'<b>{name_esc}</b></a> <i>[{stage_esc}]</i>  '
        f'RS {st_dict["comp"]:.0f}{delta_str}  {movement_html}',
        unsafe_allow_html=True,
    )
    if preview:
        st.caption(preview)


def _render_fading_line(t: dict, preview: str) -> None:
    """Fading sub-theme — struck-through, nested inside its ecosystem
    (mirrors Apollo's board: Fading shown INSIDE the group, not dropped).
    Escaped HTML for the same reason as _render_theme_line."""
    name_esc = _html.escape(t["name"])
    st.markdown(f"🔻 <s>{name_esc}</s> <i>(Fading)</i>", unsafe_allow_html=True)
    if preview:
        st.caption(preview)


def render_ecosystems() -> None:
    st.header("Theme Ecosystems")
    st.caption(
        "Ecosystems ranked by the D3 boosted score (member-union breadth-"
        "weighted, capped depth boost, thin-ecosystem floor below 5 strong "
        "members) — same scoring path as the Telegram `/themes` v2 board. "
        "Expand an ecosystem for its sub-themes, each shown with its "
        "GLOBAL rank across all active themes."
    )

    with st.sidebar:
        st.subheader("Ecosystems")
        stale_after_days = st.slider(
            "Recency window (days)", min_value=1, max_value=21, value=7, step=1,
            help="A theme counts as active if its latest snapshot row falls "
                 "within this many days (mirrors get_active_themes).",
        )
        show_unassigned = st.checkbox("Show E-UNASSIGNED", value=True)

    board = get_ecosystem_board(stale_after_days=stale_after_days)
    if not board:
        st.info(
            "No scored theme data in this window — try widening the "
            "recency slider, or check the snapshot freshness caption above."
        )
        return

    ordered = board["ordered_codes"]
    active_by_eco = board["active_by_eco"]
    fading_by_eco = board["fading_by_eco"]
    scores = board["scores"]
    global_rank = board["global_rank"]
    eco_display = board["eco_display"]

    # One batched get_top_members_by_rs call across every visible theme
    # (active + fading) — same batching pattern as theme_grid.py.
    all_theme_tickers: dict[str, tuple] = {}
    for group in list(active_by_eco.values()) + list(fading_by_eco.values()):
        for t in group:
            if t.get("tickers"):
                all_theme_tickers[t["name"]] = tuple(t["tickers"])
    member_preview = (
        get_top_members_by_rs(all_theme_tickers, n=5) if all_theme_tickers else {}
    )

    n_total_active = sum(len(v) for v in active_by_eco.values())
    n_total_fading = sum(len(v) for v in fading_by_eco.values())
    n_ecosystems = len([c for c in ordered if c != E_UNASSIGNED
                        and (active_by_eco.get(c) or fading_by_eco.get(c))])
    st.caption(
        f"{n_total_active} active theme(s) across {n_ecosystems} ecosystem(s) "
        f"· {n_total_fading} fading"
    )

    rank = 0
    for code in ordered:
        if code == E_UNASSIGNED and not show_unassigned:
            continue
        group_active = active_by_eco.get(code, [])
        group_fading = fading_by_eco.get(code, [])
        if not group_active and not group_fading:
            continue
        disp = (eco_display.get(code) or {}).get("name", "")

        if code == E_UNASSIGNED:
            n = len(group_active) + len(group_fading)
            title = f"❔ {code}{' ' + disp if disp else ''} — {n} theme(s) awaiting mapping"
            expanded = False
        else:
            rank += 1
            s = scores.get(code) or {}
            if not s.get("member_union"):
                stat = "all sub-themes Fading"
            else:
                raw_i, boosted_i = s["raw"], s["boosted"]
                if s.get("boost", 0) > 0:
                    stat = (f"raw {raw_i:.0f} → boosted {boosted_i:.0f} "
                            f"(Δ+{boosted_i - raw_i:.0f})")
                else:
                    stat = f"raw {raw_i:.0f}"
                stat += f" · {len(s['member_union'])} names · {s['strong']} RS80+"
            title = f"{rank}. {code}{' ' + disp if disp else ''} — {stat}"
            expanded = rank <= _DEFAULT_EXPANDED

        with st.expander(title, expanded=expanded):
            for st_dict in group_active:   # already comp-desc within the group
                _render_theme_line(
                    st_dict, global_rank.get(st_dict["name"]),
                    member_preview.get(st_dict["name"], ""),
                )
            for t in group_fading:
                _render_fading_line(t, member_preview.get(t["name"], ""))

    st.caption(
        "💡 Ecosystem rank = boosted D3 score. Sub-themes keep their GLOBAL "
        "rank across ALL active themes (not a per-ecosystem rank) — an "
        "ecosystem can outrank another while its top sub-theme ranks lower "
        "globally than a sub-theme in the ecosystem below it. The sparkline "
        "+ Δ next to each theme is its recent weekly rs_avg trend (green "
        "↑ = strengthening, red ↓ = weakening, — = too new for a trend) — "
        "for the FULL weekly heatmap, switch to the Grid view."
    )
