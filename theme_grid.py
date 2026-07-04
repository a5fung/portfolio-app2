"""Grid view — weekly rank heatmap with sparkline arc per theme.

Ported near-verbatim from rs-theme-dash/views/grid.py — only the data import is
swapped (live Postgres -> snapshot adapter). Logic/visuals unchanged.
"""
from __future__ import annotations

import html as _html
from urllib.parse import quote

import pandas as pd
import streamlit as st

from theme_data import TunnelDownError, dedup_themes, get_top_members_by_rs, get_weekly_grid
from theme_palette import active


_RANK_FLOOR = 50          # ranks worse than this collapse into "out" tone
# blank/out cell colors now come from theme_palette (dark/light) — see _rank_color.
_TXT_DIM = "#e8e8e8"      # light text on the green gradient cells (both modes)

_DELTA_UP = "#3fb950"      # green — rank improved
_DELTA_DOWN = "#f85149"    # red — rank declined
_DELTA_FLAT = "#8b949e"    # grey — sideways or insufficient data


def _rank_delta(ranks: list[float | None]) -> tuple[str, str]:
    """Compute rank change from earliest → latest non-null rank.

    Returns (display_text, css_color). Positive delta (rank number dropped) = green up arrow.
    """
    pts = [r for r in ranks if r is not None and not pd.isna(r)]
    if len(pts) < 2:
        return "—", _DELTA_FLAT
    delta = int(round(pts[0] - pts[-1]))  # positive → improved
    if delta > 0:
        return f"↑ {delta}", _DELTA_UP
    if delta < 0:
        return f"↓ {-delta}", _DELTA_DOWN
    return "—", _DELTA_FLAT


_RANK_KNEE = 15


def _rank_color(rank: float | None) -> tuple[str, str]:
    """Return (background, text) for a given rank. Brighter = better.

    Theme-aware: blank/out cells come from the active (dark/light) palette; the
    green gradient + its light text are mode-independent (dark-green cells read
    on either page background).

    Piecewise gradient: ranks 1..15 occupy the broad bright→mid range so the
    leaders are visually distinguished from each other. Ranks 16..50 fade
    quickly into the deep/dim range so they read as "background pack" rather
    than competing with the leaders.
    """
    P = active()
    if rank is None or pd.isna(rank):
        return P["cell_blank_bg"], P["cell_blank_txt"]
    r = int(rank)
    if r > _RANK_FLOOR:
        return P["cell_out_bg"], P["cell_out_txt"]
    if r <= _RANK_KNEE:
        # 1 → 50% (vivid), 15 → 26% (deep)
        t = (r - 1) / (_RANK_KNEE - 1)
        lightness = 50 - 24 * t
    else:
        # 16 → 22% (already dim), 50 → 12% (almost background)
        t = (r - _RANK_KNEE - 1) / (_RANK_FLOOR - _RANK_KNEE - 1)
        lightness = 22 - 10 * t
    bg = f"hsl(140, 55%, {lightness:.0f}%)"
    return bg, _TXT_DIM


def _format_value(value, encoding: str) -> str:
    if value is None or pd.isna(value):
        return ""
    if encoding == "rank":
        return f"#{int(value)}"
    if encoding == "rs_avg":
        return f"{value:.1f}"
    if encoding == "pct_above_20sma":
        return f"{value:.0f}%"
    if encoding == "member_count":
        return f"{int(value)}"
    return str(value)


def _render_young_strip(young_latest, min_age) -> None:
    # 🆕 New-themes strip — the thin-history themes filtered out of the grid above
    # (the post-6/25 discovery wave debuts with a single weekly snapshot; a grid row
    # of 11 empty cells reads as breakage). Compact, drill-linked, nothing hidden.
    if young_latest is None or young_latest.empty:
        return

    st.subheader(f"🆕 New themes ({len(young_latest)})")
    st.caption(
        f"Fewer than {min_age} weekly snapshots — too new for a rank arc. "
        "They join the grid as history accrues."
    )
    _lines = []
    for _, row in young_latest.head(40).iterrows():
        _nm = str(row["name"])
        _drill = quote(_nm, safe="")
        _members = len(row["tickers"]) if row["tickers"] else 0
        _rs = f"{row['rs_avg']:.0f}" if pd.notna(row["rs_avg"]) else "?"
        _lines.append(
            f'<a href="?drill={_drill}" target="_self">{_html.escape(_nm)}</a>'
            f' · {_html.escape(str(row["stage"]))} · RS {_rs} · {_members} members'
        )
    st.markdown("<br>".join(_lines), unsafe_allow_html=True)
    if len(young_latest) > 40:
        st.caption(f"…and {len(young_latest) - 40} more — use the search box above.")


def render_grid() -> None:
    st.header("Theme Rank Grid")
    st.caption("Weekly snapshots — last trading day per ISO week. Brighter cell = top rank.")

    with st.sidebar:
        st.subheader("Grid")
        weeks = st.slider("Weeks of history", min_value=4, max_value=24, value=12, step=1)
        encoding = st.selectbox(
            "Cell encoding",
            ["rank", "rs_avg", "pct_above_20sma", "member_count"],
            index=0,
        )
        stage_filter = st.multiselect(
            "Stage",
            ["Nascent", "Accelerating", "Mainstream", "Fading", "Retired"],
            default=["Nascent", "Accelerating", "Mainstream", "Fading"],
        )
        min_age = st.number_input(
            "Min weeks active (grid)", min_value=0, value=2, step=1,
            help="Themes with fewer weekly snapshots than this move to the compact "
                 "'New themes' strip below the grid instead of rendering "
                 "mostly-empty rows (the post-6/25 discovery wave debuts with 1 week "
                 "of history). Set 0 to put everything in the grid.")
        only_ranked_now = st.checkbox(
            "Hide themes with no current rank", value=True,
            help="Hides themes that did not appear in the most recent week."
        )
        # Single knob: floor on |intersection|. Lower = more aggressive merging.
        # 0 = off. The overlap-ratio threshold is fixed at 0.50 — empirically the
        # threshold slider was a no-op on real data (themes that share ≥N tickers
        # almost always have high overlap_ratio; pairs with low overlap_ratio
        # share only 1–2 tickers, which the floor already blocks).
        dedup_min_shared = st.slider(
            "Dedup: min shared tickers", min_value=0, max_value=6, value=3, step=1,
            help=(
                "Floor on |intersection| between two themes' member lists. "
                "Lower = more aggressive merging; 0 disables dedup. "
                "1–2 surfaces 'tiny alias' false positives (1-ticker themes "
                "trivially hit 100% overlap); 3 is the empirically safe default."
            ),
        )

    try:
        df = get_weekly_grid(weeks=weeks)
    except TunnelDownError:
        st.error("Can't reach Apollo theme data — snapshot missing or unreadable.")
        return
    except Exception as e:
        st.error(f"Theme data load failed: {e}")
        return

    if df.empty:
        st.info("No theme data in window.")
        return

    df = df.copy()
    df["member_count"] = df["tickers"].apply(lambda t: len(t) if t else 0)

    latest_week = df["week_start"].max()
    latest = df[df["week_start"] == latest_week].set_index("name")

    # Subset-aware dedup against latest snapshot. Aliases drop out of the universe;
    # ranks are recomputed downstream so the pack is contiguous (no gaps from absorbed dupes).
    alias_count: dict[str, int] = {}
    aliases_of: dict[str, list[str]] = {}
    dedup_on = dedup_min_shared > 0
    if dedup_on:
        latest_tickers_all = {
            name: tuple(row["tickers"])
            for name, row in latest.iterrows()
            if row["tickers"]
        }
        parent_of = dedup_themes(latest_tickers_all, threshold=0.50, min_shared=dedup_min_shared)
        for name, parent in parent_of.items():
            if name != parent:
                alias_count[parent] = alias_count.get(parent, 0) + 1
                aliases_of.setdefault(parent, []).append(name)
        reps = {name for name, parent in parent_of.items() if name == parent}
        # Themes with no tickers in latest snapshot weren't seen by dedup — keep them.
        df = df[df["name"].isin(reps) | ~df["name"].isin(parent_of)]

    if stage_filter:
        keep_names = latest[latest["stage"].isin(stage_filter)].index
        df = df[df["name"].isin(keep_names)]

    young_latest = None  # themes excluded from the grid for thin history -> the strip below
    if not df.empty:
        weeks_per_theme = df.groupby("name")["week_start"].nunique()
        keep_names = weeks_per_theme[weeks_per_theme >= min_age].index
        young_names = weeks_per_theme[weeks_per_theme < min_age].index
        _lat = df[df["week_start"] == df["week_start"].max()]
        young_latest = (_lat[_lat["name"].isin(young_names)]
                        .sort_values("rs_avg", ascending=False, na_position="last"))
        df = df[df["name"].isin(keep_names)]

    if df.empty:
        st.info("No themes match the current filters.")
        _render_young_strip(young_latest, min_age)
        return

    # When dedup is on, SQL ranks span the original (alias-polluted) universe — recompute
    # over survivors so ranks contract cleanly. When off, trust the adapter rank.
    if dedup_on:
        pivot_rs_universe = df.pivot_table(
            index="name", columns="week_start", values="rs_avg", aggfunc="first"
        )
        pivot_rank = pivot_rs_universe.rank(
            axis=0, method="min", ascending=False, na_option="keep"
        )
    else:
        pivot_rank = df.pivot_table(
            index="name", columns="week_start", values="week_rank", aggfunc="first"
        )

    value_col = {
        "rank": None,  # rank uses pivot_rank directly
        "rs_avg": "rs_avg",
        "pct_above_20sma": "pct_above_20sma",
        "member_count": "member_count",
    }[encoding]

    if encoding == "rank":
        pivot_value = pivot_rank.copy()
    else:
        pivot_value = df.pivot_table(
            index="name", columns="week_start", values=value_col, aggfunc="first"
        )

    # Sort by current week's rank (ascending — best first)
    if latest_week in pivot_rank.columns:
        current_rank = pivot_rank[latest_week]
        if only_ranked_now:
            keep = current_rank.dropna().index
            pivot_rank = pivot_rank.loc[keep]
            pivot_value = pivot_value.loc[keep]
            current_rank = current_rank.loc[keep]
        order = current_rank.sort_values(na_position="last").index
        pivot_value = pivot_value.reindex(order)
        pivot_rank = pivot_rank.reindex(order)

    # Drop weeks where no theme has a usable rank (e.g. pre-rs_avg-engine columns).
    usable_weeks = [wk for wk in pivot_rank.columns if pivot_rank[wk].notna().any()]
    pivot_rank = pivot_rank[sorted(usable_weeks)]
    pivot_value = pivot_value[[wk for wk in sorted(usable_weeks) if wk in pivot_value.columns]]

    week_cols = list(pivot_value.columns)

    deltas: dict[str, tuple[str, str]] = {}
    for theme in pivot_rank.index:
        ranks = [
            (None if pd.isna(pivot_rank.at[theme, wk]) else float(pivot_rank.at[theme, wk]))
            for wk in week_cols
        ]
        deltas[theme] = _rank_delta(ranks)

    # Top tickers per theme — most-recent snapshot, ordered by current RS.
    # `latest` (built above) already holds the latest-week row per theme, indexed
    # by name, so look up directly instead of re-scanning df per theme.
    latest_tickers: dict[str, tuple[str, ...]] = {}
    for theme in pivot_rank.index:
        tickers = latest.at[theme, "tickers"] if theme in latest.index else None
        if tickers:
            latest_tickers[theme] = tuple(tickers)
    member_preview = get_top_members_by_rs(latest_tickers, n=4) if latest_tickers else {}
    if alias_count:
        for theme, n_aliases in alias_count.items():
            if theme in member_preview:
                member_preview[theme] = f"{member_preview[theme]}  ⧉{n_aliases}"

    # Render as an HTML table (not st.dataframe) so the active palette fully
    # controls every cell color — it follows the native Streamlit theme. Drill-
    # down is a plain ?drill=<url-encoded theme> link the page catches; encoding
    # is required so a '&' in a theme name (e.g. "Satellite Imagery & Geospatial
    # Intelligence") doesn't split the query string.
    import html as _html

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
    for theme in pivot_rank.index:
        drill = quote(str(theme), safe="")
        name_esc = _html.escape(str(theme))
        members_esc = _html.escape(member_preview.get(theme, ""))
        cells = [
            (f'<td style="text-align:left;padding:3px 8px;border:1px solid {_bd};'
             f'background:{P["cell_blank_bg"]};max-width:300px">'
             f'<a href="?drill={drill}" target="_self" '
             f'style="color:{_txt};text-decoration:none">{name_esc}</a></td>'),
            (f'<td style="text-align:left;padding:3px 8px;border:1px solid {_bd};'
             f'background:{P["cell_blank_bg"]};color:{_txt};font-size:12px;'
             f'white-space:nowrap">{members_esc}</td>'),
        ]
        now_rank = pivot_rank.at[theme, latest_week] if latest_week in pivot_rank.columns else None
        nbg, ntxt = _rank_color(now_rank)
        nval = f"#{int(now_rank)}" if now_rank is not None and not pd.isna(now_rank) else ""
        cells.append(_cell(nbg, ntxt, nval, "600"))
        dtext, dcolor = deltas[theme]
        cells.append(
            f'<td style="text-align:center;color:{dcolor};font-weight:600;'
            f'background:{P["cell_blank_bg"]};border:1px solid {_bd};'
            f'padding:3px 6px">{_html.escape(dtext)}</td>'
        )
        for wk in week_cols:
            bg, txt = _rank_color(pivot_rank.at[theme, wk])
            val = _html.escape(str(_format_value(pivot_value.at[theme, wk], encoding)))
            cells.append(_cell(bg, txt, val))
        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    head = _th("Theme", "left") + _th("Top members", "left") + _th("Now") + _th("Δ")
    for wk in week_cols:
        head += _th(wk.isoformat()[5:])

    st.caption("👉 Click any theme name to open its detail view.")
    st.markdown(
        f'<div style="overflow-x:auto;max-height:760px;overflow-y:auto">'
        f'<table style="border-collapse:collapse;font-size:13px;width:100%;color:{_txt}">'
        f'<thead><tr>{head}</tr></thead><tbody>{"".join(rows_html)}</tbody></table></div>',
        unsafe_allow_html=True,
    )

    n_aliases_total = sum(alias_count.values())

    if aliases_of:
        with st.expander(
            f"Dedup detail — {n_aliases_total} aliases absorbed "
            f"(min_shared={dedup_min_shared})"
        ):
            # Recompute per-merge stats so the user can spot weak/spurious merges.
            for rep in sorted(aliases_of, key=lambda r: -len(aliases_of[r])):
                rep_set = set(latest_tickers_all.get(rep, ()))
                st.markdown(f"**{rep}**  _(size {len(rep_set)})_")
                rows = []
                for alias in sorted(aliases_of[rep]):
                    a_set = set(latest_tickers_all.get(alias, ()))
                    shared = a_set & rep_set
                    ratio = len(shared) / len(a_set) if a_set else 0
                    rows.append({
                        "Alias": alias,
                        "Size": len(a_set),
                        "Overlap": f"{ratio*100:.0f}%",
                        "Shared": ", ".join(sorted(shared)) or "—",
                    })
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.caption(
                f"**Overlap** = |alias ∩ rep| / |alias|. A pair merges when "
                f"|shared| ≥ {dedup_min_shared} AND overlap ≥ 50%. Lower the slider "
                "to absorb more aliases (1–2 will surface tiny-alias false positives)."
            )

    with st.expander("Legend"):
        st.markdown(
            "- **Bright green** = top rank (#1)\n"
            "- **Mid green** = top 15 — broad gradient so leaders are distinguishable\n"
            "- **Deep green** = ranks 16-50 — fades quickly into the background\n"
            "- **Grey cell** = ranked outside top 50 that week\n"
            "- **Black cell** = no snapshot for that ISO week\n"
            "\n**Δ** is the rank change from the earliest visible week → most recent "
            "(↑ green = improved, ↓ red = declined, — grey = no movement / insufficient data). "
            "The per-week color cells themselves form a horizontal arc — cells getting "
            "brighter left-to-right = theme is climbing.\n"
            "\nWeeks are ISO weeks (Monday-anchored). Cells show the last trading-day "
            "rank within each week — Friday by default, Thursday on Good Friday-style holidays. "
            "Weeks where no theme has a usable `rs_avg` (engine pre-March 2026) are dropped entirely."
        )

    if dedup_on:
        dedup_note = (
            f" · dedup min_shared={dedup_min_shared} absorbed {n_aliases_total} "
            f"alias{'es' if n_aliases_total != 1 else ''}"
            if n_aliases_total
            else f" · dedup min_shared={dedup_min_shared} (no merges)"
        )
    else:
        dedup_note = " · dedup off"
    st.caption(f"Latest week: **{latest_week}** · {len(pivot_rank)} themes shown{dedup_note}")

    _render_young_strip(young_latest, min_age)
