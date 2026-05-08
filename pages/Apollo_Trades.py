"""
Apollo Trades — long-term retrospection dashboard.

Phase 1 scaffold (2026-05-09): KPI strip + calendar P&L heatmap + setup-tagged
stats. Built against mock methodology-realistic data so layout decisions
generalize to live data when ≥30 closed live trades exist (~July 2026).

Phase 2 (later): equity curve, best/worst panel, drill-down.
Phase 3 (gated on Apollo MAE/MFE): excursion histograms.

Data source: apollo_data.load_trades() — toggled by APOLLO_DATA_MODE env var.
Mock by default; flip to 'db' when Tailscale + Postgres role are wired.

Color palette mirrors Portfolio.py (C_DARK / C_LIGHT) for visual continuity.
"""
from __future__ import annotations

import calendar
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from apollo_data import daily_pnl, load_trades, setup_stats

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Apollo Trades", layout="wide", page_icon="◆")

# Mirror Portfolio.py palette so this page feels native
C_DARK = {
    "bg": "#000000", "surface": "#111113", "surface2": "#18181B",
    "text": "#FFFFFF", "text_sec": "#D4D4D8", "text_muted": "#A1A1AA",
    "text_dim": "#52525B",
    "primary": "#00D26A", "primary_dim": "#00331B",
    "positive": "#00D26A", "positive_dim": "#004D26",
    "negative": "#F82C2C", "negative_dim": "#450A0A",
    "warning": "#F59E0B",
    "border": "#27272A", "grid": "#18181B",
}
C_LIGHT = {
    "bg": "#FFFFFF", "surface": "#F4F4F5", "surface2": "#E4E4E7",
    "text": "#09090B", "text_sec": "#27272A", "text_muted": "#52525B",
    "text_dim": "#71717A",
    "primary": "#00B85E", "primary_dim": "#DCFCE7",
    "positive": "#00B85E", "positive_dim": "#DCFCE7",
    "negative": "#DC2626", "negative_dim": "#FEE2E2",
    "warning": "#D97706",
    "border": "#D4D4D8", "grid": "#D4D4D8",
}

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
C = C_DARK if st.session_state.dark_mode else C_LIGHT

CHART_CONFIG = {"displayModeBar": False, "staticPlot": False, "scrollZoom": False}

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <style>
    body, .main {{ background: {C['bg']}; color: {C['text']}; }}
    .stApp {{ background: {C['bg']}; }}
    .kpi-card {{
        background: {C['surface']}; border: 1px solid {C['border']};
        border-radius: 8px; padding: 14px 16px; height: 100%;
    }}
    .kpi-label {{ color: {C['text_muted']}; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }}
    .kpi-value {{ color: {C['text']}; font-size: 24px; font-weight: 600; margin-top: 4px; }}
    .kpi-sub {{ color: {C['text_muted']}; font-size: 11px; margin-top: 2px; }}
    .scaffold-banner {{
        background: {C['surface']}; border-left: 3px solid {C['warning']};
        padding: 10px 14px; border-radius: 4px; color: {C['text_sec']};
        font-size: 12px; margin-bottom: 16px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

col_title, col_toggle = st.columns([4, 1])
with col_title:
    st.title("◆ Apollo Trades")
    st.caption(
        "Long-term retrospection · Calendar P&L · Setup-tagged performance"
    )
with col_toggle:
    if st.button("◐ Theme", use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

# Scaffold banner — clear visual marker that this is mock data
import os
data_mode = os.environ.get("APOLLO_DATA_MODE", "mock").lower()
if data_mode == "mock":
    st.markdown(
        f"""<div class="scaffold-banner">
        <strong>SCAFFOLD MODE</strong> · Methodology-realistic mock data.
        Switch to live by setting <code>APOLLO_DATA_MODE=db</code>
        (gated on ≥30 closed live trades).
        </div>""",
        unsafe_allow_html=True,
    )

# ── Sidebar filters ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    account_mode = st.selectbox(
        "Account", ["paper", "live"], index=0,
        help="Paper trading data while live cutover pending.",
    )
    days_back_options = {
        "Last 30 days": 30, "Last 60 days": 60, "Last 90 days": 90,
        "Year to date": (date.today() - date(date.today().year, 1, 1)).days,
        "All time": 365,
    }
    period_label = st.selectbox(
        "Period", list(days_back_options.keys()), index=2,
    )
    days_back = days_back_options[period_label]

# ── Load data ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def _cached_load(mode: str) -> pd.DataFrame:
    return load_trades(account_mode=mode)

df_all = _cached_load(account_mode)
cutoff = date.today() - timedelta(days=days_back)
df = df_all[df_all["alert_date"] >= cutoff].copy()

if df.empty:
    st.warning(f"No trades in the last {days_back} days. Adjust the period filter.")
    st.stop()

# ── KPI strip ───────────────────────────────────────────────────────────────
closed_df = df[df["status"].isin(["closed", "stopped"])].copy()
n_closed = len(closed_df)
n_open = len(df) - n_closed
total_pnl = closed_df["total_pnl"].sum()
wins = closed_df[closed_df["total_pnl"] > 0]
win_rate = len(wins) / max(1, n_closed)
profit_factor = (
    wins["total_pnl"].sum()
    / max(1.0, abs(closed_df[closed_df["total_pnl"] <= 0]["total_pnl"].sum()))
)
avg_r = closed_df["r_multiple"].mean() if n_closed else 0.0

# Equity curve for max-drawdown calc (cheap; avoids re-aggregating later)
if n_closed:
    sorted_closed = closed_df.sort_values("closed_at")
    cum = sorted_closed["total_pnl"].cumsum()
    running_peak = cum.cummax()
    drawdown = (cum - running_peak)
    max_dd = drawdown.min()
else:
    max_dd = 0.0


def _kpi(label: str, value: str, sub: str = "", color: str | None = None) -> str:
    val_color = color or C["text"]
    return (
        f"""<div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color:{val_color};">{value}</div>
        <div class="kpi-sub">{sub}</div>
        </div>"""
    )


pnl_color = C["positive"] if total_pnl >= 0 else C["negative"]
dd_color = C["negative"] if max_dd < 0 else C["text"]
wr_color = C["positive"] if win_rate >= 0.30 else C["warning"]

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(_kpi("Realized P&L", f"${total_pnl:+,.0f}",
                     f"{n_closed} closed · {n_open} open", color=pnl_color),
                unsafe_allow_html=True)
with k2:
    st.markdown(_kpi("Win rate", f"{win_rate*100:.1f}%",
                     f"{len(wins)} winners · {n_closed - len(wins)} losers",
                     color=wr_color),
                unsafe_allow_html=True)
with k3:
    st.markdown(_kpi("Avg R", f"{avg_r:+.2f}R",
                     "per closed trade"),
                unsafe_allow_html=True)
with k4:
    st.markdown(_kpi("Profit factor", f"{profit_factor:.2f}",
                     "gross win / gross loss"),
                unsafe_allow_html=True)
with k5:
    st.markdown(_kpi("Max drawdown", f"${max_dd:,.0f}",
                     "peak-to-trough", color=dd_color),
                unsafe_allow_html=True)

st.markdown(" ")

# ── Calendar P&L heatmap ────────────────────────────────────────────────────
st.subheader("Calendar P&L")
st.caption("Each cell = one trading day. Hover for tickers and net P&L.")

dpnl = daily_pnl(df)
if dpnl.empty:
    st.info("No closed trades in the selected period.")
else:
    # Build month-by-month grid: rows = weeks of the month, cols = Mon-Fri.
    # Plotly Heatmap with text annotations. One subplot per month for the
    # selected window (default 90d → up to 4 months).
    dpnl["date"] = pd.to_datetime(dpnl["date"])
    pnl_by_date = dict(zip(dpnl["date"].dt.date, dpnl["pnl"]))
    tickers_by_date = dict(zip(dpnl["date"].dt.date, dpnl["tickers"]))
    n_by_date = dict(zip(dpnl["date"].dt.date, dpnl["n_trades"]))

    # Determine months present in the period
    start_date = date.today() - timedelta(days=days_back)
    end_date = date.today()
    months: list[tuple[int, int]] = []
    cur = date(start_date.year, start_date.month, 1)
    while cur <= end_date:
        months.append((cur.year, cur.month))
        # advance to first of next month
        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)

    # Symmetric color scale based on max absolute daily P&L for nice contrast
    cmax = max(abs(dpnl["pnl"].min()), abs(dpnl["pnl"].max())) if not dpnl.empty else 1
    cmax = max(cmax, 1.0)

    cols = st.columns(min(len(months), 3) or 1)
    for idx, (yr, mo) in enumerate(months):
        col = cols[idx % len(cols)]
        with col:
            # Build the grid: one row per week of the month, col = Mon..Fri
            weeks = calendar.Calendar(firstweekday=0).monthdatescalendar(yr, mo)
            # Trim leading/trailing weeks that have no in-month days
            grid_dates: list[list[date | None]] = []
            for week in weeks:
                row = []
                for d in week[:5]:  # Mon-Fri only
                    row.append(d if d.month == mo else None)
                if any(d is not None for d in row):
                    grid_dates.append(row)

            z = []
            text = []
            hover = []
            for row in grid_dates:
                z_row, t_row, h_row = [], [], []
                for d in row:
                    if d is None:
                        z_row.append(None)
                        t_row.append("")
                        h_row.append("")
                    elif d in pnl_by_date:
                        pnl = pnl_by_date[d]
                        z_row.append(pnl)
                        t_row.append(f"{d.day}<br>${pnl:+,.0f}")
                        h_row.append(
                            f"<b>{d.isoformat()}</b><br>"
                            f"P&L: ${pnl:+,.2f}<br>"
                            f"Trades: {n_by_date[d]}<br>"
                            f"Tickers: {tickers_by_date[d]}"
                        )
                    else:
                        z_row.append(0)
                        t_row.append(str(d.day))
                        h_row.append(f"{d.isoformat()}<br>(no trades)")
                z.append(z_row)
                text.append(t_row)
                hover.append(h_row)

            # Custom diverging colorscale (red/black/green to match dark theme)
            colorscale = [
                [0.0, C["negative"]],
                [0.5, C["surface"] if st.session_state.dark_mode else "#F0F0F0"],
                [1.0, C["positive"]],
            ]

            fig = go.Figure(go.Heatmap(
                z=z,
                text=text,
                texttemplate="%{text}",
                textfont={"size": 10, "color": C["text"]},
                customdata=hover,
                hovertemplate="%{customdata}<extra></extra>",
                colorscale=colorscale,
                zmid=0, zmin=-cmax, zmax=cmax,
                showscale=False,
                xgap=2, ygap=2,
            ))

            month_name = calendar.month_name[mo]
            fig.update_layout(
                title=dict(
                    text=f"<b>{month_name} {yr}</b>",
                    font=dict(size=13, color=C["text"]),
                    x=0.02, y=0.95,
                ),
                xaxis=dict(
                    tickmode="array",
                    tickvals=list(range(5)),
                    ticktext=["Mon", "Tue", "Wed", "Thu", "Fri"],
                    showgrid=False, zeroline=False, fixedrange=True,
                    side="top",
                    tickfont=dict(color=C["text_muted"], size=10),
                ),
                yaxis=dict(
                    showgrid=False, zeroline=False, fixedrange=True,
                    autorange="reversed",
                    showticklabels=False,
                ),
                paper_bgcolor=C["bg"],
                plot_bgcolor=C["bg"],
                margin=dict(l=10, r=10, t=40, b=10),
                height=max(180, 60 + 50 * len(grid_dates)),
            )
            st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

# ── Setup stats ─────────────────────────────────────────────────────────────
st.subheader("Setup-tagged stats")
st.caption("Per-strategy performance · win rate, R-multiples, profit factor, expectancy, holding period")

stats = setup_stats(df)
if stats.empty:
    st.info("No closed trades in the selected period.")
else:
    # Format for display
    display = stats.copy()
    display["win_rate"] = display["win_rate"].apply(lambda x: f"{x*100:.1f}%")
    display["avg_r"] = display["avg_r"].apply(lambda x: f"{x:+.2f}R")
    display["avg_r_win"] = display["avg_r_win"].apply(lambda x: f"{x:+.2f}R")
    display["avg_r_loss"] = display["avg_r_loss"].apply(lambda x: f"{x:+.2f}R")
    display["profit_factor"] = display["profit_factor"].apply(lambda x: f"{x:.2f}")
    display["expectancy"] = display["expectancy"].apply(lambda x: f"${x:+,.0f}")
    display["avg_hold_days"] = display["avg_hold_days"].apply(lambda x: f"{x:.1f}d")
    display["largest_win"] = display["largest_win"].apply(lambda x: f"${x:+,.0f}")
    display["largest_loss"] = display["largest_loss"].apply(lambda x: f"${x:+,.0f}")

    display = display.rename(columns={
        "strategy": "Strategy",
        "n_trades": "Trades",
        "win_rate": "Win rate",
        "avg_r": "Avg R",
        "avg_r_win": "Avg R (win)",
        "avg_r_loss": "Avg R (loss)",
        "profit_factor": "Profit factor",
        "expectancy": "Expectancy",
        "avg_hold_days": "Avg hold",
        "largest_win": "Best",
        "largest_loss": "Worst",
    })

    st.dataframe(
        display, use_container_width=True, hide_index=True,
        column_config={
            "Strategy": st.column_config.TextColumn(width="small"),
            "Trades": st.column_config.NumberColumn(width="small"),
        },
    )

# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown(" ")
st.caption(
    f"Mode: {data_mode.upper()} · Period: {period_label} · Account: {account_mode} · "
    f"{len(df)} trades shown ({n_closed} closed, {n_open} open)"
)
