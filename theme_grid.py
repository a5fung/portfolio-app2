"""Grid view — weekly rank heatmap with sparkline arc per theme.

Ported near-verbatim from rs-theme-dash/views/grid.py — only the data import is
swapped (live Postgres -> snapshot adapter). Logic/visuals unchanged.
"""
from __future__ import annotations

from urllib.parse import quote

import pandas as pd
import streamlit as st

from theme_data import TunnelDownError, dedup_themes, get_top_members_by_rs, get_weekly_grid


_RANK_FLOOR = 50          # ranks worse than this collapse into "out" tone
_BG_OUT = "#1e1f24"       # dim grey for out-of-range
_BG_BLANK = "#0e1117"     # streamlit dark bg — no snapshot
_TXT_OUT = "#5a5d63"
_TXT_BLANK = "#2c2e34"
_TXT_BRIGHT = "#0a0a0a"   # near-black on bright cells (high contrast)
_TXT_DIM = "#e8e8e8"

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

    Piecewise gradient: ranks 1..15 occupy the broad bright→mid range so the
    leaders are visually distinguished from each other. Ranks 16..50 fade
    quickly into the deep/dim range so they read as "background pack" rather
    than competing with the leaders.
    """
    if rank is None or pd.isna(rank):
        return _BG_BLANK, _TXT_BLANK
    r = int(rank)
    if r > _RANK_FLOOR:
        return _BG_OUT, _TXT_OUT
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
        min_age = st.number_input("Min weeks active", min_value=0, value=0, step=1)
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

    if not df.empty:
        weeks_per_theme = df.groupby("name")["week_start"].nunique()
        keep_names = weeks_per_theme[weeks_per_theme >= min_age].index
        df = df[df["name"].isin(keep_names)]

    if df.empty:
        st.info("No themes match the current filters.")
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
    latest_tickers: dict[str, tuple[str, ...]] = {}
    for theme in pivot_rank.index:
        row = df[(df["name"] == theme) & (df["week_start"] == latest_week)]
        if not row.empty and row.iloc[0]["tickers"]:
            latest_tickers[theme] = tuple(row.iloc[0]["tickers"])
    member_preview = get_top_members_by_rs(latest_tickers, n=4) if latest_tickers else {}
    if alias_count:
        for theme, n_aliases in alias_count.items():
            if theme in member_preview:
                member_preview[theme] = f"{member_preview[theme]}  ⧉{n_aliases}"

    # Theme cell holds a URL-encoded drill value plus a readable fragment.
    # Encoding `?drill=` is mandatory — themes with `&` (e.g. "Satellite Imagery
    # & Geospatial Intelligence") would otherwise split the query string and
    # st.query_params would see only "Satellite Imagery ", so detail.py fell
    # back to themes[0] (Agri-Chemical) every time. The `#…` fragment is what
    # display_text's regex captures for rendering — fragments aren't sent to
    # the server, so st.query_params is unaffected.
    display = pd.DataFrame(
        {
            "Theme": [
                f"?drill={quote(t, safe='')}#{t}" for t in pivot_rank.index
            ],
            "Members": [member_preview.get(t, "") for t in pivot_rank.index],
            "Now": [
                int(pivot_rank.at[t, latest_week])
                if latest_week in pivot_rank.columns and pd.notna(pivot_rank.at[t, latest_week])
                else None
                for t in pivot_rank.index
            ],
            "Δ": [deltas[t][0] for t in pivot_rank.index],
        },
        index=pivot_rank.index,
    )
    for wk in week_cols:
        display[wk.isoformat()] = [
            _format_value(pivot_value.at[t, wk], encoding) for t in pivot_rank.index
        ]

    week_str_cols = [wk.isoformat() for wk in week_cols]

    def _style(_df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame("", index=_df.index, columns=_df.columns)
        for col_idx, wk in enumerate(week_cols):
            col_name = week_str_cols[col_idx]
            for theme in _df.index:
                bg, txt = _rank_color(pivot_rank.at[theme, wk])
                out.at[theme, col_name] = (
                    f"background-color: {bg}; color: {txt}; "
                    f"text-align: center; font-variant-numeric: tabular-nums;"
                )
        for theme in _df.index:
            now_rank = pivot_rank.at[theme, latest_week] if latest_week in pivot_rank.columns else None
            bg, txt = _rank_color(now_rank)
            out.at[theme, "Now"] = (
                f"background-color: {bg}; color: {txt}; "
                f"text-align: center; font-weight: 600;"
            )
            delta_color = deltas[theme][1]
            out.at[theme, "Δ"] = (
                f"color: {delta_color}; font-weight: 600; text-align: center;"
            )
        return out

    styled = display.style.apply(_style, axis=None)
    styled = styled.set_table_styles([
        {"selector": "th.col_heading", "props": "writing-mode: vertical-rl; transform: rotate(180deg); text-align: center; font-size: 11px;"},
        {"selector": "th.row_heading", "props": "text-align: left; max-width: 320px; font-size: 13px;"},
    ])

    n_rows = len(display)
    height = min(900, 60 + 32 * n_rows)

    st.caption("👉 Click any **theme name** to drill into the detail view.")

    st.dataframe(
        styled,
        height=height,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Theme": st.column_config.LinkColumn(
                "Theme",
                display_text=r"#(.+)$",
                help="Click to open detail view",
                width="large",
            ),
            "Members": st.column_config.TextColumn(
                "Top members",
                help=(
                    "Top 4 tickers by current RS · trailing +N is the rest · "
                    "⧉N marks N absorbed alias themes (subset-aware dedup)."
                ),
                width="medium",
            ),
        },
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
    st.caption(f"Latest week: **{latest_week}** · {len(display)} themes shown{dedup_note}")
