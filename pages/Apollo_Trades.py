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

from apollo_data import daily_pnl, excursion_stats, load_trades, setup_stats

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
    .strat-card {{
        background: {C['surface']}; border: 1px solid {C['border']};
        border-radius: 8px; padding: 16px; height: 100%;
    }}
    .strat-name {{
        color: {C['text']}; font-size: 14px; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.5px;
        padding-bottom: 8px; border-bottom: 1px solid {C['border']}; margin-bottom: 12px;
    }}
    .strat-net {{ color: {C['text_muted']}; font-size: 11px; text-transform: uppercase; }}
    .strat-net-val {{ font-size: 26px; font-weight: 700; margin-top: 2px; margin-bottom: 12px; }}
    .strat-row {{ display: flex; justify-content: space-between; font-size: 11px;
                  color: {C['text_muted']}; margin-top: 2px; }}
    .strat-row-val {{ color: {C['text_sec']}; font-weight: 500; }}
    .winrate-track {{
        background: {C['surface2']}; height: 8px; border-radius: 4px;
        position: relative; overflow: hidden; margin: 6px 0 12px 0;
    }}
    .winrate-fill {{ height: 100%; border-radius: 4px; transition: width 0.3s; }}
    .strat-section-label {{
        color: {C['text_muted']}; font-size: 10px;
        text-transform: uppercase; letter-spacing: 0.4px; margin-top: 12px;
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
    # ── Scorecards (one per strategy, max 4 across) ────────────────────────
    n_strats = len(stats)
    card_cols = st.columns(min(n_strats, 4) or 1)
    for idx, row in stats.iterrows():
        strat = row["strategy"]
        strat_trades = closed_df[closed_df["entry_strategy"] == strat]
        net_pnl = strat_trades["total_pnl"].sum()
        win_rate_pct = row["win_rate"] * 100
        wr_color = (
            C["positive"] if win_rate_pct >= 35
            else (C["warning"] if win_rate_pct >= 25 else C["negative"])
        )
        net_color = C["positive"] if net_pnl >= 0 else C["negative"]

        with card_cols[idx % len(card_cols)]:
            # Card header + net P&L
            st.markdown(
                f"""<div class="strat-card">
                <div class="strat-name">{strat}</div>
                <div class="strat-net">Net realized P&L</div>
                <div class="strat-net-val" style="color:{net_color};">${net_pnl:+,.0f}</div>
                <div class="strat-row"><span>Win rate</span><span class="strat-row-val">{win_rate_pct:.1f}%</span></div>
                <div class="winrate-track">
                  <div class="winrate-fill" style="width:{min(win_rate_pct, 100):.1f}%; background:{wr_color};"></div>
                </div>
                <div class="strat-section-label">R-multiple distribution</div>
                </div>""",
                unsafe_allow_html=True,
            )
            # Mini R-distribution histogram inside the card
            r_vals = strat_trades["r_multiple"].dropna()
            if len(r_vals):
                wins_r = r_vals[r_vals > 0]
                losses_r = r_vals[r_vals <= 0]
                fig_r = go.Figure()
                fig_r.add_trace(go.Histogram(
                    x=losses_r, marker_color=C["negative"],
                    xbins=dict(start=-3, end=12, size=1),
                    opacity=0.85, showlegend=False,
                    hovertemplate="R %{x}<br>%{y} trades<extra></extra>",
                ))
                fig_r.add_trace(go.Histogram(
                    x=wins_r, marker_color=C["positive"],
                    xbins=dict(start=-3, end=12, size=1),
                    opacity=0.85, showlegend=False,
                    hovertemplate="R %{x}<br>%{y} trades<extra></extra>",
                ))
                fig_r.update_layout(
                    height=110, barmode="stack",
                    paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
                    margin=dict(l=0, r=0, t=0, b=20),
                    xaxis=dict(
                        showgrid=False, zeroline=True, fixedrange=True,
                        zerolinecolor=C["text_dim"], zerolinewidth=1,
                        tickfont=dict(color=C["text_muted"], size=9),
                        tickvals=[-2, 0, 2, 4, 6, 8, 10],
                    ),
                    yaxis=dict(
                        showgrid=False, zeroline=False, fixedrange=True,
                        showticklabels=False,
                    ),
                )
                st.plotly_chart(fig_r, use_container_width=True, config=CHART_CONFIG)
            # Bottom stats row
            pf = row["profit_factor"]
            exp_v = row["expectancy"]
            avg_hold = row["avg_hold_days"]
            n_t = int(row["n_trades"])
            st.markdown(
                f"""<div class="strat-row" style="margin-top:6px;">
                <span>Profit factor</span><span class="strat-row-val">{pf:.2f}</span></div>
                <div class="strat-row">
                <span>Expectancy</span><span class="strat-row-val">${exp_v:+,.0f}</span></div>
                <div class="strat-row">
                <span>Trades · avg hold</span><span class="strat-row-val">{n_t} · {avg_hold:.1f}d</span></div>
                <div class="strat-row">
                <span>Avg R (win/loss)</span><span class="strat-row-val">{row['avg_r_win']:+.2f} / {row['avg_r_loss']:+.2f}</span></div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(" ")

    # ── Precise reference table (below scorecards) ─────────────────────────
    with st.expander("Precise reference table"):
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

# ── Worst / best price excursion ───────────────────────────────────────────
st.subheader("Worst-vs-you / best-in-favor")
st.caption(
    "How far each trade went underwater vs how high it ran, expressed in R "
    "(risk-multiples). Bottom-left quadrant = clean entries that ran far. "
    "Top-right = drag near stop before running. Y=X diagonal = exited at peak."
)

if "worst_r" in closed_df.columns:
    valid = closed_df.dropna(subset=["worst_r", "best_r"]).copy()
else:
    valid = pd.DataFrame()

if valid.empty:
    st.info(
        "No excursion data yet. Apollo started capturing this 2026-05-10; "
        "backfill closed paper trades via "
        "`docker exec apollo-market python -m scripts.backfill_position_extremes`."
    )
else:
    # Two-column row: scatter (left) + per-setup median table (right)
    ex_col1, ex_col2 = st.columns([3, 2])

    with ex_col1:
        # Color by entry_strategy
        strategy_colors = {
            "magna53": C["primary"],
            "9m_day2": "#3B82F6",  # blue contrast to primary green
        }
        ex_fig = go.Figure()
        for strat, sub in valid.groupby("entry_strategy"):
            ex_fig.add_trace(go.Scatter(
                x=sub["worst_r"],
                y=sub["best_r"],
                mode="markers",
                name=strat,
                marker=dict(
                    size=10,
                    color=strategy_colors.get(strat, C["text_muted"]),
                    line=dict(width=1, color=C["bg"]),
                    opacity=0.75,
                ),
                customdata=sub[["ticker", "r_multiple", "alert_date"]].values,
                hovertemplate=(
                    "<b>%{customdata[0]}</b> · "
                    + strat + "<br>"
                    "Worst: %{x:+.2f}R · Best: %{y:+.2f}R<br>"
                    "Exit: %{customdata[1]:+.2f}R<br>"
                    "Date: %{customdata[2]}<extra></extra>"
                ),
            ))
        # Diagonal y=x reference (only positive quadrant: trades that
        # reached peak above zero R)
        max_r = max(valid["best_r"].max(), 1.0) * 1.05
        ex_fig.add_trace(go.Scatter(
            x=[0, max_r], y=[0, max_r],
            mode="lines",
            line=dict(color=C["text_dim"], width=1, dash="dot"),
            name="y=x (exit at peak)",
            hoverinfo="skip",
            showlegend=False,
        ))
        # Axis crosshair lines at 0 and -1 (typical stop)
        ex_fig.add_hline(y=0, line_width=1, line_color=C["text_dim"])
        ex_fig.add_vline(x=0, line_width=1, line_color=C["text_dim"])
        ex_fig.add_vline(
            x=-1.0, line_width=1, line_color=C["negative"],
            line_dash="dash",
            annotation_text="typical stop -1R",
            annotation_position="bottom right",
            annotation_font=dict(color=C["text_muted"], size=10),
        )
        ex_fig.update_layout(
            height=440,
            paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
            font=dict(color=C["text"], size=12),
            xaxis=dict(
                title="Worst R during hold (drawdown vs entry)",
                gridcolor=C["grid"], zerolinecolor=C["text_dim"],
                range=[min(valid["worst_r"].min(), -1.2) * 1.05, 0.2],
            ),
            yaxis=dict(
                title="Best R during hold (peak vs entry)",
                gridcolor=C["grid"], zerolinecolor=C["text_dim"],
                range=[-0.2, max_r],
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1,
                bgcolor="rgba(0,0,0,0)",
            ),
            margin=dict(l=60, r=30, t=40, b=50),
            hoverlabel=dict(
                bgcolor=C["surface"], bordercolor=C["border"],
                font=dict(color=C["text"], size=12),
            ),
        )
        st.plotly_chart(ex_fig, use_container_width=True, config=CHART_CONFIG)

    with ex_col2:
        ex_stats = excursion_stats(closed_df)
        if ex_stats.empty:
            st.info("No per-strategy data yet.")
        else:
            # Format display
            ex_stats_disp = ex_stats.copy()
            ex_stats_disp["median_worst_r"] = ex_stats_disp["median_worst_r"].apply(
                lambda v: f"{v:+.2f}R" if pd.notna(v) else "—")
            ex_stats_disp["median_best_r"] = ex_stats_disp["median_best_r"].apply(
                lambda v: f"{v:+.2f}R" if pd.notna(v) else "—")
            ex_stats_disp["median_r"] = ex_stats_disp["median_r"].apply(
                lambda v: f"{v:+.2f}R" if pd.notna(v) else "—")
            ex_stats_disp["median_capture_pct"] = ex_stats_disp["median_capture_pct"].apply(
                lambda v: f"{v*100:.0f}%" if pd.notna(v) else "—")
            ex_stats_disp = ex_stats_disp.rename(columns={
                "strategy": "Strategy",
                "n": "N",
                "median_worst_r": "Worst",
                "median_best_r": "Best",
                "median_r": "Exit",
                "median_capture_pct": "Capture",
            })
            st.markdown(
                f"<div style='color:{C['text_muted']}; font-size:0.85em; "
                f"margin-top:0.5em;'>Per-strategy medians:</div>",
                unsafe_allow_html=True,
            )
            st.dataframe(
                ex_stats_disp,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Strategy": st.column_config.TextColumn(width="small"),
                    "N": st.column_config.NumberColumn(width="small"),
                },
            )
            st.markdown(
                f"<div style='color:{C['text_muted']}; font-size:0.8em; "
                f"margin-top:0.8em; line-height:1.5;'>"
                f"<b>Capture</b> = exit R / best R · "
                f"&lt;60% = exiting too early · "
                f"&gt;85% = trail discipline good<br>"
                f"<b>Worst</b> near -1R = trades often touch stop · "
                f"near 0 = clean entries<br>"
                f"<b>Best</b> much &gt; exit = wide opportunity left on table"
                f"</div>",
                unsafe_allow_html=True,
            )

# ── Equity curve with drawdown shading ─────────────────────────────────────
st.subheader("Equity curve")
st.caption("Cumulative realized P&L · drawdown shaded · open trades excluded")

if not closed_df.empty:
    eq = closed_df.sort_values("closed_at")[["closed_at", "total_pnl"]].copy()
    eq["cum_pnl"] = eq["total_pnl"].cumsum()
    eq["peak"] = eq["cum_pnl"].cummax()
    eq["drawdown"] = eq["cum_pnl"] - eq["peak"]

    fig_eq = go.Figure()
    # Drawdown band — peak line and current cum line; fill between to show DD
    fig_eq.add_trace(go.Scatter(
        x=eq["closed_at"], y=eq["peak"],
        mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip", name="peak",
    ))
    fig_eq.add_trace(go.Scatter(
        x=eq["closed_at"], y=eq["cum_pnl"],
        mode="lines", line=dict(width=0),
        fill="tonexty",
        fillcolor=C["negative_dim"] if st.session_state.dark_mode else "rgba(220,38,38,0.18)",
        showlegend=False, hoverinfo="skip", name="drawdown_fill",
    ))
    # Main equity line on top
    fig_eq.add_trace(go.Scatter(
        x=eq["closed_at"], y=eq["cum_pnl"],
        mode="lines",
        line=dict(color=C["positive"], width=2.5),
        name="Cumulative P&L",
        hovertemplate="%{x|%b %d}<br>$%{y:,.0f}<extra></extra>",
    ))
    # Zero line
    fig_eq.add_hline(y=0, line_dash="dot", line_color=C["text_dim"], line_width=1)

    fig_eq.update_layout(
        height=320,
        paper_bgcolor=C["bg"],
        plot_bgcolor=C["bg"],
        margin=dict(l=10, r=10, t=10, b=30),
        xaxis=dict(
            showgrid=False, zeroline=False, fixedrange=True,
            tickfont=dict(color=C["text_muted"], size=10),
        ),
        yaxis=dict(
            showgrid=True, gridcolor=C["grid"], zeroline=False, fixedrange=True,
            tickprefix="$", tickformat=",",
            tickfont=dict(color=C["text_muted"], size=10),
        ),
        showlegend=False,
    )
    st.plotly_chart(fig_eq, use_container_width=True, config=CHART_CONFIG)
else:
    st.info("No closed trades for equity curve.")

# ── Best / worst trades ────────────────────────────────────────────────────
st.subheader("Best & worst trades")
st.caption("Top-10 by realized P&L (left) and by R-multiple (right) · regime + catalyst context")

if not closed_df.empty:
    bw_cols = ["ticker", "entry_strategy", "alert_date", "regime",
               "catalyst_quality", "gap_pct", "entry_price", "exit_price",
               "r_multiple", "total_pnl", "holding_days"]

    def _format_bw(d: pd.DataFrame) -> pd.DataFrame:
        out = d[bw_cols].copy()
        out["alert_date"] = out["alert_date"].apply(lambda x: x.strftime("%b %d") if hasattr(x, "strftime") else str(x))
        out["gap_pct"] = out["gap_pct"].apply(lambda v: f"{v:+.1f}%" if pd.notna(v) else "—")
        out["entry_price"] = out["entry_price"].apply(lambda v: f"${v:.2f}")
        out["exit_price"] = out["exit_price"].apply(lambda v: f"${v:.2f}" if pd.notna(v) else "—")
        out["r_multiple"] = out["r_multiple"].apply(lambda v: f"{v:+.2f}R")
        out["total_pnl"] = out["total_pnl"].apply(lambda v: f"${v:+,.0f}")
        out["holding_days"] = out["holding_days"].apply(lambda v: f"{int(v)}d" if pd.notna(v) else "—")
        return out.rename(columns={
            "ticker": "Ticker", "entry_strategy": "Strategy", "alert_date": "Date",
            "regime": "Regime", "catalyst_quality": "Catalyst", "gap_pct": "Gap",
            "entry_price": "Entry", "exit_price": "Exit", "r_multiple": "R",
            "total_pnl": "P&L", "holding_days": "Hold",
        })

    bc, wc = st.columns(2)
    with bc:
        st.markdown(f"**Top 10 by P&L**")
        top_pnl = closed_df.nlargest(10, "total_pnl")
        st.dataframe(_format_bw(top_pnl), use_container_width=True, hide_index=True, height=380)
    with wc:
        st.markdown(f"**Bottom 10 by P&L**")
        bot_pnl = closed_df.nsmallest(10, "total_pnl")
        st.dataframe(_format_bw(bot_pnl), use_container_width=True, hide_index=True, height=380)

    bc2, wc2 = st.columns(2)
    with bc2:
        st.markdown(f"**Top 10 by R-multiple**")
        top_r = closed_df.nlargest(10, "r_multiple")
        st.dataframe(_format_bw(top_r), use_container_width=True, hide_index=True, height=380)
    with wc2:
        st.markdown(f"**Bottom 10 by R-multiple**")
        bot_r = closed_df.nsmallest(10, "r_multiple")
        st.dataframe(_format_bw(bot_r), use_container_width=True, hide_index=True, height=380)
else:
    st.info("No closed trades for best/worst panel.")

# ── Holding-period histogram ───────────────────────────────────────────────
st.subheader("Holding period")
st.caption("Distribution of (closed_at − filled_at) days · split by win/loss")

if not closed_df.empty and closed_df["holding_days"].notna().any():
    holds = closed_df[closed_df["holding_days"].notna()].copy()
    wins_h = holds[holds["total_pnl"] > 0]["holding_days"]
    losses_h = holds[holds["total_pnl"] <= 0]["holding_days"]

    fig_h = go.Figure()
    fig_h.add_trace(go.Histogram(
        x=losses_h, name="Losses",
        marker_color=C["negative"],
        xbins=dict(start=0, end=25, size=1),
        opacity=0.85,
    ))
    fig_h.add_trace(go.Histogram(
        x=wins_h, name="Winners",
        marker_color=C["positive"],
        xbins=dict(start=0, end=25, size=1),
        opacity=0.85,
    ))
    fig_h.update_layout(
        height=280,
        barmode="stack",
        paper_bgcolor=C["bg"],
        plot_bgcolor=C["bg"],
        margin=dict(l=10, r=10, t=10, b=30),
        xaxis=dict(
            title=dict(text="Holding days", font=dict(color=C["text_muted"], size=11)),
            showgrid=False, zeroline=False, fixedrange=True,
            tickfont=dict(color=C["text_muted"], size=10),
        ),
        yaxis=dict(
            title=dict(text="Trade count", font=dict(color=C["text_muted"], size=11)),
            showgrid=True, gridcolor=C["grid"], zeroline=False, fixedrange=True,
            tickfont=dict(color=C["text_muted"], size=10),
        ),
        legend=dict(
            orientation="h", yanchor="top", y=1.05, xanchor="right", x=1,
            font=dict(color=C["text_muted"], size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    st.plotly_chart(fig_h, use_container_width=True, config=CHART_CONFIG)

    # Median lines text
    med_w = wins_h.median() if len(wins_h) else None
    med_l = losses_h.median() if len(losses_h) else None
    cap_parts = []
    if med_w is not None:
        cap_parts.append(f"Winner median **{med_w:.0f}d**")
    if med_l is not None:
        cap_parts.append(f"Loser median **{med_l:.0f}d**")
    if cap_parts:
        st.markdown(" · ".join(cap_parts))
else:
    st.info("No closed trades with holding-period data.")

# ── Trade drill-down ───────────────────────────────────────────────────────
with st.expander(f"Trade-level drill-down ({len(df)} trades · click to expand)"):
    drill = df.copy()
    # Optional ticker filter
    tickers_in_period = sorted(drill["ticker"].unique())
    selected = st.multiselect(
        "Filter by ticker (empty = all)",
        tickers_in_period, default=[],
    )
    if selected:
        drill = drill[drill["ticker"].isin(selected)]

    drill_cols = [
        "ticker", "entry_strategy", "alert_date", "filled_at", "closed_at",
        "status", "entry_price", "stop_price", "exit_price", "entry_shares",
        "regime", "catalyst_quality", "gap_pct", "ep_score",
        "r_multiple", "total_pnl", "holding_days",
    ]
    drill_disp = drill[drill_cols].copy()
    drill_disp["alert_date"] = drill_disp["alert_date"].apply(
        lambda x: x.strftime("%Y-%m-%d") if hasattr(x, "strftime") else str(x))
    drill_disp["filled_at"] = drill_disp["filled_at"].apply(
        lambda x: x.strftime("%m-%d %H:%M") if pd.notna(x) and hasattr(x, "strftime") else "—")
    drill_disp["closed_at"] = drill_disp["closed_at"].apply(
        lambda x: x.strftime("%m-%d %H:%M") if pd.notna(x) and hasattr(x, "strftime") else "—")
    for col in ("entry_price", "stop_price", "exit_price"):
        drill_disp[col] = drill_disp[col].apply(
            lambda v: f"${v:.2f}" if pd.notna(v) else "—")
    drill_disp["gap_pct"] = drill_disp["gap_pct"].apply(
        lambda v: f"{v:+.1f}%" if pd.notna(v) else "—")
    drill_disp["ep_score"] = drill_disp["ep_score"].apply(
        lambda v: f"{v:.0f}" if pd.notna(v) else "—")
    drill_disp["r_multiple"] = drill_disp["r_multiple"].apply(
        lambda v: f"{v:+.2f}" if pd.notna(v) else "—")
    drill_disp["total_pnl"] = drill_disp["total_pnl"].apply(
        lambda v: f"${v:+,.0f}" if pd.notna(v) else "—")
    drill_disp["holding_days"] = drill_disp["holding_days"].apply(
        lambda v: f"{int(v)}" if pd.notna(v) else "—")

    drill_disp = drill_disp.rename(columns={
        "ticker": "Ticker", "entry_strategy": "Strategy", "alert_date": "Date",
        "filled_at": "Filled", "closed_at": "Closed", "status": "Status",
        "entry_price": "Entry", "stop_price": "Stop", "exit_price": "Exit",
        "entry_shares": "Sh", "regime": "Regime", "catalyst_quality": "Catalyst",
        "gap_pct": "Gap", "ep_score": "Score", "r_multiple": "R",
        "total_pnl": "P&L", "holding_days": "Hold",
    })

    st.dataframe(
        drill_disp.sort_values("Date", ascending=False),
        use_container_width=True, hide_index=True, height=500,
    )

# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown(" ")
st.caption(
    f"Mode: {data_mode.upper()} · Period: {period_label} · Account: {account_mode} · "
    f"{len(df)} trades shown ({n_closed} closed, {n_open} open)"
)
