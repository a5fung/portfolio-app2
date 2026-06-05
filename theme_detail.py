"""Detail view — drill-down into one theme's narrative arc.

Ported near-verbatim from rs-theme-dash/views/detail.py — only the data import
is swapped (live Postgres -> snapshot adapter). Logic/visuals unchanged.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from theme_data import TunnelDownError, get_active_themes, get_correlated_themes, get_theme_detail


_STAGE_COLORS = {
    "Nascent": "#8b949e",
    "Accelerating": "#3fb950",
    "Mainstream": "#58a6ff",
    "Fading": "#d29922",
    "Retired": "#6e7681",
}


def _stage_pill(stage: str) -> str:
    color = _STAGE_COLORS.get(stage, "#8b949e")
    return (
        f'<span style="background:{color};color:#0a0a0a;padding:2px 10px;'
        f'border-radius:10px;font-size:12px;font-weight:600;">{stage}</span>'
    )


def _rank_arc_chart(arc: pd.DataFrame) -> go.Figure:
    """Rank line chart with inverted Y axis (rank 1 at top)."""
    df = arc.dropna(subset=["week_rank"]).copy()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["week_start"],
            y=df["week_rank"],
            mode="lines+markers",
            line=dict(color="#3fb950", width=2.5),
            marker=dict(size=8, color="#3fb950"),
            hovertemplate="<b>Week %{x}</b><br>Rank: #%{y}<extra></extra>",
            name="Rank",
        )
    )
    fig.update_yaxes(autorange="reversed", title="Rank", zeroline=False)
    fig.update_xaxes(title=None)
    fig.update_layout(
        height=280, margin=dict(l=10, r=10, t=30, b=10),
        title="Rank arc (lower = better)",
        showlegend=False, paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
        font=dict(color="#e8e8e8"),
    )
    return fig


def _rs_arc_chart(arc: pd.DataFrame) -> go.Figure:
    """RS_avg + breadth (pct_above_20sma) panel."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=arc["week_start"], y=arc["rs_avg"],
            mode="lines+markers", name="RS avg",
            line=dict(color="#58a6ff", width=2.5),
            marker=dict(size=7),
            hovertemplate="<b>Week %{x}</b><br>RS: %{y:.1f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=arc["week_start"], y=arc["pct_above_20sma"],
            mode="lines+markers", name="% above 20-SMA",
            line=dict(color="#d29922", width=2, dash="dot"),
            marker=dict(size=6),
            yaxis="y2",
            hovertemplate="<b>Week %{x}</b><br>Breadth: %{y:.0f}%<extra></extra>",
        )
    )
    fig.update_layout(
        height=280, margin=dict(l=10, r=10, t=30, b=10),
        title="RS avg + member breadth",
        paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
        font=dict(color="#e8e8e8"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(title="RS avg"),
        yaxis2=dict(title="Breadth %", overlaying="y", side="right",
                    range=[0, 100], showgrid=False),
    )
    return fig


def render_detail() -> None:
    back_col, header_col = st.columns([1, 6])
    with back_col:
        if st.button("← Back to grid", use_container_width=True):
            st.session_state["view"] = "Grid"
            st.rerun()
    with header_col:
        st.header("Theme Detail")

    try:
        themes = get_active_themes(stale_after_days=14)
    except TunnelDownError:
        st.error("Can't reach Apollo theme data — snapshot missing or unreadable.")
        return
    except Exception as e:
        st.error(f"Theme data load failed: {e}")
        return

    if not themes:
        st.info("No active themes in the last 14 days.")
        return

    selected = st.session_state.get("selected_theme")
    if selected not in themes:
        selected = themes[0]
    selected = st.selectbox("Theme", themes, index=themes.index(selected))
    st.session_state["selected_theme"] = selected

    detail = get_theme_detail(selected, weeks=12)
    if not detail:
        st.warning(f"No data for **{selected}** in the last 12 weeks.")
        return

    # Header strip — stage, rank, RS, breadth
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        st.markdown(_stage_pill(detail["stage"]), unsafe_allow_html=True)
    with c2:
        st.metric("Rank", f"#{detail['current_rank']}" if detail["current_rank"] else "—")
    with c3:
        rs = detail["rs_avg"]
        st.metric("RS avg", f"{rs:.1f}" if rs is not None and not pd.isna(rs) else "—")
    with c4:
        br = detail["pct_above_20sma"]
        st.metric("Above 20-SMA", f"{br:.0f}%" if br is not None and not pd.isna(br) else "—")

    if detail["thesis"]:
        st.caption(f"📝 {detail['thesis']}")

    st.divider()

    # Arcs
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(_rank_arc_chart(detail["arc"]), use_container_width=True)
    with col_b:
        st.plotly_chart(_rs_arc_chart(detail["arc"]), use_container_width=True)

    st.divider()

    # Members + Churn
    members = detail["members"]
    col_m, col_c = st.columns([3, 2])

    with col_m:
        st.subheader(f"Members ({len(detail['current_tickers'])})")
        if members.empty:
            st.caption(
                f"_No member RS data available for {detail['members_as_of']}._"
            )
            st.write(", ".join(detail["current_tickers"]) or "—")
        else:
            display_members = members.copy()
            display_members["close"] = display_members["close"].apply(
                lambda v: f"${v:.2f}" if v and not pd.isna(v) else "—"
            )
            display_members["rs_composite"] = display_members["rs_composite"].apply(
                lambda v: f"{v:.1f}" if v and not pd.isna(v) else "—"
            )
            display_members["rs_rank"] = display_members["rs_rank"].apply(
                lambda v: f"#{int(v)}" if v and not pd.isna(v) else "—"
            )
            display_members = display_members.rename(columns={
                "ticker": "Ticker",
                "rs_composite": "RS",
                "rs_rank": "Univ rank",
                "sector": "Sector",
                "close": "Close",
            })[["Ticker", "RS", "Univ rank", "Sector", "Close"]]
            st.dataframe(display_members, use_container_width=True, hide_index=True,
                         height=min(560, 60 + 36 * len(display_members)))
            st.caption(f"_RS scores as of {detail['members_as_of']}._")

    with col_c:
        st.subheader("Churn (last 4 weeks)")
        if detail["joined"]:
            st.markdown("**Joined**")
            st.success(", ".join(detail["joined"]))
        else:
            st.caption("_No new members in the trailing window._")

        if detail["exited"]:
            st.markdown("**Exited**")
            st.error(", ".join(detail["exited"]))
        else:
            st.caption("_No exits in the trailing window._")

    st.divider()

    # Correlated themes
    st.subheader("Correlated themes (member-overlap ≥ 30%)")
    try:
        corr = get_correlated_themes(selected, min_overlap=0.30)
    except Exception as e:
        st.warning(f"Correlated themes query failed: {e}")
        corr = pd.DataFrame()

    if corr.empty:
        st.caption("_No other current themes share ≥30% of this theme's members._")
    else:
        corr_display = corr.copy()
        corr_display["overlap_pct"] = corr_display["overlap_pct"].apply(lambda v: f"{v*100:.0f}%")
        corr_display["rs_avg"] = corr_display["rs_avg"].apply(
            lambda v: f"{v:.1f}" if v and not pd.isna(v) else "—"
        )
        corr_display["shared_tickers"] = corr_display["shared_tickers"].apply(
            lambda lst: ", ".join(lst) if lst else ""
        )
        corr_display = corr_display.rename(columns={
            "name": "Theme",
            "stage": "Stage",
            "rs_avg": "RS",
            "shared_count": "Shared",
            "other_size": "Members",
            "overlap_pct": "Overlap",
            "shared_tickers": "Shared tickers",
        })[["Theme", "Stage", "RS", "Overlap", "Shared", "Members", "Shared tickers"]]
        st.dataframe(corr_display, use_container_width=True, hide_index=True,
                     height=min(360, 60 + 36 * len(corr_display)))
        st.caption(
            "💡 High overlap is one signal of theme-engine fragmentation — "
            "two near-identical themes will show ≥80% overlap."
        )

    st.divider()
    st.caption(
        "**V5 placeholder:** Median member return since entering Top 10 — N/A until "
        "forward-return outcomes are wired (`mi_signal_outcomes` join)."
    )
