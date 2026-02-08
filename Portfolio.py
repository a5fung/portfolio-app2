"""
Portfolio Command Center v3
Dark theme · Data labels · Merged risk view
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, datetime

# --- CONFIG ---
st.set_page_config(page_title="Portfolio Dashboard", layout="wide", page_icon="◆")

CHART_CONFIG = {"displayModeBar": False, "staticPlot": False, "scrollZoom": False}
WARNING_THRESHOLD = 0.93
DANGER_THRESHOLD = 0.85
DATA_CACHE_TTL = 60

# --- DARK PALETTE ---
C = {
    "bg":           "#0F1117",
    "surface":      "#1A1D27",
    "surface2":     "#232733",
    "primary":      "#3B82F6",
    "primary_dim":  "#1E3A5F",
    "positive":     "#22C55E",
    "positive_dim": "#16A34A",
    "negative":     "#EF4444",
    "negative_dim": "#DC2626",
    "warning":      "#F59E0B",
    "text":         "#F1F5F9",
    "text_muted":   "#94A3B8",
    "border":       "#2D3348",
    "grid":         "#1E2235",
}

ACCENT_RAMP = [
    "#3B82F6", "#06B6D4", "#8B5CF6", "#14B8A6",
    "#A78BFA", "#2DD4BF", "#6366F1", "#34D399",
]

# --- STYLES ---
st.markdown(f"""
<style>
    .stApp {{ background-color: {C["bg"]} !important; }}
    section[data-testid="stSidebar"] {{
        background-color: {C["surface"]} !important;
        border-right: 1px solid {C["border"]};
    }}

    /* Typography */
    h1 {{ color: {C["text"]} !important; font-weight: 700 !important; letter-spacing: -0.025em !important; }}
    h2, h3, h4 {{ color: {C["text"]} !important; font-weight: 600 !important; }}
    p, li, div, span, label {{ color: {C["text"]} !important; }}

    /* Tabs */
    button[data-baseweb="tab"] {{
        background-color: transparent !important;
        color: {C["text_muted"]} !important;
        font-weight: 600; font-size: 15px;
        padding-bottom: 12px !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {C["primary"]} !important;
        border-bottom: 2px solid {C["primary"]} !important;
    }}

    /* Metric cards */
    [data-testid="stMetric"] {{
        background: {C["surface"]};
        border: 1px solid {C["border"]};
        border-radius: 12px;
        padding: 16px 20px;
    }}
    [data-testid="stMetricLabel"] {{
        font-weight: 600; font-size: clamp(10px, 1.2vw, 13px);
        text-transform: uppercase; letter-spacing: 0.04em;
        color: {C["text_muted"]} !important;
    }}
    [data-testid="stMetricValue"] {{
        font-weight: 700; color: {C["text"]} !important;
        font-size: clamp(16px, 2.2vw, 28px) !important;
        white-space: nowrap;
        overflow: visible;
    }}
    [data-testid="stMetricDelta"] {{
        font-size: clamp(10px, 1.1vw, 14px) !important;
    }}

    /* Dataframe */
    .stDataFrame {{ background: {C["surface"]} !important; border-radius: 8px; }}

    /* Dividers */
    hr {{ border-color: {C["border"]} !important; opacity: 0.4; }}

    /* Expander */
    details {{ background: {C["surface"]} !important; border: 1px solid {C["border"]} !important; border-radius: 8px !important; }}

    /* Sparkline container */
    .spark-row {{
        display: flex; align-items: center; justify-content: space-between;
        padding: 6px 0; border-bottom: 1px solid {C["border"]};
    }}
    .spark-label {{ font-size: 12px; color: {C["text_muted"]} !important; font-weight: 600; }}
    .spark-value {{ font-size: 13px; color: {C["text"]} !important; font-weight: 700; }}

    /* Timestamp */
    .timestamp {{
        font-size: 12px; color: {C["text_muted"]} !important;
        text-align: right; padding: 4px 0 12px 0;
    }}

    /* Hide chrome */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)


# --- AUTH ---
def check_password():
    if st.session_state.get("password_correct"):
        return True
    password = st.text_input("Enter Password", type="password", key="pw")
    if password:
        try:
            if password == st.secrets["app_password"]:
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("Incorrect password")
        except KeyError:
            st.error("Password not configured in secrets")
    return False


if not check_password():
    st.stop()


# --- CHART HELPERS ---
def style_chart(fig, height=None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C["text_muted"], size=12),
        margin=dict(l=0, r=0, t=8, b=0),
        xaxis=dict(
            showgrid=False,
            showline=True, linecolor=C["border"],
            tickfont=dict(color=C["text_muted"], size=11),
            fixedrange=True,
        ),
        yaxis=dict(
            showgrid=True, gridcolor=C["grid"], gridwidth=1,
            showline=False, zeroline=False,
            tickfont=dict(color=C["text_muted"], size=11),
            fixedrange=True,
            tickprefix="$", tickformat=",.0f",
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(color=C["text_muted"], size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=C["surface2"], font_size=12, bordercolor=C["border"]),
        dragmode=False,
        yaxis2=dict(fixedrange=True),
    )
    if height:
        fig.update_layout(height=height)
    return fig


def ytd_color(val):
    """Return green for positive YTD, red for negative."""
    return C["positive"] if val >= 0 else C["negative"]


def drawdown_chart(data, date_col="Date", value_col="Total Value", height=None, show_legend=True, show_labels=False):
    data = data.sort_values(date_col).copy()
    data["Peak"] = data[value_col].cummax()

    fig = go.Figure()

    # Danger zone fill
    fig.add_trace(go.Scatter(
        x=data[date_col], y=data["Peak"] * DANGER_THRESHOLD,
        fill=None, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=data[date_col], y=data["Peak"] * WARNING_THRESHOLD,
        fill="tonexty", fillcolor="rgba(239,68,68,0.08)",
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))

    # Warning zone fill
    fig.add_trace(go.Scatter(
        x=data[date_col], y=data["Peak"],
        fill="tonexty", fillcolor="rgba(245,158,11,0.06)",
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))

    # Threshold lines
    fig.add_trace(go.Scatter(
        x=data[date_col], y=data["Peak"] * WARNING_THRESHOLD,
        name="-7%", line=dict(dash="dot", color=C["warning"], width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=data[date_col], y=data["Peak"] * DANGER_THRESHOLD,
        name="-15%", line=dict(dash="dot", color=C["negative"], width=1.5),
    ))

    # Peak
    fig.add_trace(go.Scatter(
        x=data[date_col], y=data["Peak"],
        name="Peak", line=dict(dash="dash", color=C["text_muted"], width=1.5),
    ))

    # Actual value
    value_trace = dict(
        x=data[date_col], y=data[value_col],
        name="Value", line=dict(color=C["primary"], width=2.5),
        mode="lines+markers", marker=dict(size=5),
    )
    if show_labels:
        value_trace.update(
            mode="lines+markers+text",
            text=data[value_col],
            texttemplate="%{y:.2s}",
            textposition="top center",
            textfont=dict(color=C["primary"], size=11),
        )
    fig.add_trace(go.Scatter(**value_trace))

    fig = style_chart(fig, height=height)
    if not show_legend:
        fig.update_layout(showlegend=False)
    return fig


def make_sparkline_svg(values, color=None, width=80, height=24):
    """Generate a tiny inline SVG sparkline."""
    if not values or len(values) < 2:
        return ""
    color = color or C["primary"]
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1
    n = len(values)
    points = []
    for i, v in enumerate(values):
        x = (i / (n - 1)) * width
        y = height - ((v - mn) / rng) * (height - 4) - 2
        points.append(f"{x:.1f},{y:.1f}")
    polyline = " ".join(points)
    return f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}"><polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>'


# --- DATA ---
@st.cache_data(ttl=DATA_CACHE_TTL)
def load_data():
    try:
        url = st.secrets.get("public_sheet_url")
        if not url or "PASTE_YOUR" in url:
            st.error("Google Sheet URL not configured in secrets")
            return pd.DataFrame()
        return pd.read_csv(url)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


def clean_data(df):
    if df.empty:
        return df
    df.columns = df.columns.str.strip()
    df = df[[c for c in df.columns if "Unnamed" not in c]]
    if "Bucket" not in df.columns:
        st.error("Missing required 'Bucket' column")
        return pd.DataFrame()
    df = df.dropna(subset=["Bucket"])
    for col in ["Total Value", "Cash", "Margin Balance"]:
        if col in df.columns:
            s = df[col].astype(str).str.replace(r'[$,\s]', '', regex=True)
            df[col] = pd.to_numeric(s.str.replace('-', '0'), errors='coerce').fillna(0)
    if "YTD" in df.columns:
        df["YTD"] = pd.to_numeric(df["YTD"].astype(str).str.replace('%', ''), errors='coerce').fillna(0)
    else:
        df["YTD"] = 0.0
    if "Date" not in df.columns:
        st.error("Missing required 'Date' column")
        return pd.DataFrame()
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    df = df.dropna(subset=["Date"]).sort_values("Date")
    return df


def validate_data(df):
    if df.empty:
        st.error("No data loaded.")
        return False
    missing = [c for c in ["Date", "Bucket", "Account", "Total Value"] if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        return False
    return True


# --- LOAD ---
load_time = datetime.now()
df = load_data()
df = clean_data(df)
if not validate_data(df):
    st.stop()


# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### Filters")
    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()
    default_start = max(date(2026, 1, 1), min_date)

    date_range = st.date_input(
        "Date Range",
        [default_start, max_date],
        min_value=min_date, max_value=max_date,
    )

    st.markdown("---")

    # Sparklines — rendered after filtering via placeholder
    spark_placeholder = st.empty()

    st.markdown("---")
    with st.expander("Data Quality"):
        st.caption(f"**{len(df)}** rows · {df['Date'].min():%Y-%m-%d} → {df['Date'].max():%Y-%m-%d}")
        st.caption(f"Missing values: {df.isnull().sum().sum()}")
    st.caption(f"Auto-refreshes every {DATA_CACHE_TTL}s")


# --- APPLY FILTERS ---
if len(date_range) == 2:
    start_date, end_date = date_range
    fdf = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))]
elif len(date_range) == 1:
    st.warning("Select both start and end dates")
    fdf = df
else:
    fdf = df

if fdf.empty:
    st.warning("No data for selected range.")
    st.stop()

# --- KEY METRICS ---
latest = fdf[fdf["Date"] == fdf["Date"].max()]
account_order = latest.groupby("Account")["Total Value"].sum().sort_values(ascending=False).index.tolist()
total_value = latest["Total Value"].sum()
total_cash = latest["Cash"].sum()
total_margin = latest["Margin Balance"].sum()

# Value-weighted portfolio YTD: each row's YTD weighted by its share of total value
portfolio_ytd = (latest["YTD"] * latest["Total Value"]).sum() / total_value if total_value else 0

# Deltas from previous period
dates_sorted = sorted(fdf["Date"].unique())
if len(dates_sorted) >= 2:
    prev = fdf[fdf["Date"] == dates_sorted[-2]]
    prev_value = prev["Total Value"].sum()
    prev_cash = prev["Cash"].sum()
    prev_margin = prev["Margin Balance"].sum()
    delta_value = f"{((total_value / prev_value) - 1) * 100:+.1f}%" if prev_value else None
    delta_cash = f"${total_cash - prev_cash:+,.0f}" if prev_cash else None
    delta_margin = f"${total_margin - prev_margin:+,.0f}" if prev_margin else None
else:
    delta_value = delta_cash = delta_margin = None

# --- SIDEBAR SPARKLINES ---
with spark_placeholder.container():
    st.markdown("### Accounts")
    for acct in account_order:
        hist = fdf[fdf["Account"] == acct].groupby("Date")["Total Value"].sum().sort_index()
        vals = hist.tolist()
        current = vals[-1] if vals else 0
        trend_color = C["positive"] if len(vals) >= 2 and vals[-1] >= vals[0] else C["negative"]
        svg = make_sparkline_svg(vals, color=trend_color)
        st.markdown(
            f'<div class="spark-row">'
            f'<span class="spark-label">{acct}</span>'
            f'{svg}'
            f'<span class="spark-value">${current:,.0f}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


# --- HEADER ---
st.title("Portfolio Command Center")
st.markdown(f'<div class="timestamp">Last updated: {load_time.strftime("%b %d, %Y  %I:%M %p")}</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Net Liquidity", f"${total_value:,.0f}", delta=delta_value)
c2.metric("Cash", f"${total_cash:,.0f}", delta=delta_cash)
c3.metric("Margin", f"${total_margin:,.0f}", delta=delta_margin)
c4.metric("YTD Return", f"{portfolio_ytd:+.1f}%")

st.markdown("")

# --- TABS (3 tabs: Overview merged with Risk) ---
tab1, tab2, tab3 = st.tabs(["Overview", "Performance", "Allocation"])

# ═══════════════════════════════════════════
# TAB 1: OVERVIEW + RISK
# ═══════════════════════════════════════════
with tab1:
    st.subheader("Portfolio Growth")
    trend = fdf.groupby("Date")["Total Value"].sum().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend["Date"], y=trend["Total Value"],
        mode="lines+markers+text",
        line=dict(color=C["primary"], width=2.5),
        marker=dict(size=5, color=C["primary"]),
        fill="tozeroy",
        fillcolor="rgba(59,130,246,0.08)",
        name="Total Value",
        text=trend["Total Value"],
        texttemplate="%{y:.2s}",
        textposition="top center",
        textfont=dict(color=C["primary"], size=11),
    ))
    fig = style_chart(fig, height=350)
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG, key="ov_growth")

    # Global Risk Monitor
    st.subheader("Risk Monitor")
    daily_totals = fdf.groupby("Date")["Total Value"].sum().reset_index()
    fig_risk = drawdown_chart(daily_totals, height=300, show_labels=True)
    st.plotly_chart(fig_risk, use_container_width=True, config=CHART_CONFIG, key="ov_risk")

    # Per-account drawdown grid
    cols = st.columns(2)
    for i, account in enumerate(account_order):
        with cols[i % 2]:
            st.markdown(f"**{account}**")
            acct_hist = df[df["Account"] == account].groupby("Date")["Total Value"].sum().reset_index()
            if len(date_range) == 2:
                acct_view = acct_hist[
                    (acct_hist["Date"] >= pd.to_datetime(date_range[0]))
                    & (acct_hist["Date"] <= pd.to_datetime(date_range[1]))
                ]
            else:
                acct_view = acct_hist
            if not acct_view.empty:
                fig = drawdown_chart(acct_view, height=220, show_legend=False)
                st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG, key=f"risk_{i}")
            else:
                st.info("No data for selected range")


# ═══════════════════════════════════════════
# TAB 2: PERFORMANCE
# ═══════════════════════════════════════════
with tab2:
    st.subheader("Account Performance")

    for i, account in enumerate(account_order):
        with st.container():
            st.markdown(f"#### {account}")

            acct_df = fdf[fdf["Account"] == account]
            daily = acct_df.groupby("Date")[
                ["Total Value", "Cash", "Margin Balance", "YTD"]
            ].sum().reset_index()

            latest_acct = acct_df.iloc[-1]
            current_value = latest_acct["Total Value"]
            current_ytd = latest_acct["YTD"]

            k1, k2 = st.columns(2)
            k1.metric("Current Value", f"${current_value:,.0f}")
            k2.metric("YTD Return", f"{current_ytd:+.1f}%")

            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # Value bars with labels
            fig.add_trace(go.Bar(
                x=daily["Date"], y=daily["Total Value"],
                name="Value", marker_color=C["primary_dim"],
                marker_line=dict(color=C["primary"], width=1),
                text=daily["Total Value"],
                texttemplate="%{y:.2s}",
                textposition="inside",
                textfont=dict(color=C["primary"], size=11),
            ), secondary_y=False)

            # Cash bars
            fig.add_trace(go.Bar(
                x=daily["Date"], y=daily["Cash"],
                name="Cash", marker_color="rgba(34,197,94,0.2)",
                marker_line=dict(color=C["positive"], width=1),
            ), secondary_y=False)

            # YTD line — color-coded green/red
            ytd_line_color = ytd_color(daily["YTD"].iloc[-1]) if not daily.empty else C["text_muted"]
            fig.add_trace(go.Scatter(
                x=daily["Date"], y=daily["YTD"],
                name="YTD %", mode="lines+markers+text",
                line=dict(color=ytd_line_color, width=2.5),
                marker=dict(size=6, color=ytd_line_color),
                text=daily["YTD"],
                texttemplate="%{y:.1f}%",
                textposition="top center",
                textfont=dict(color=ytd_line_color, size=11),
            ), secondary_y=True)

            fig = style_chart(fig, height=320)
            fig.update_layout(barmode="group")
            fig.update_yaxes(title_text="Value ($)", secondary_y=False)
            fig.update_yaxes(
                title_text="YTD (%)", secondary_y=True,
                tickprefix="", ticksuffix="%", tickformat=".1f",
                showgrid=False,
            )
            st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG, key=f"perf_{i}")
            st.divider()


# ═══════════════════════════════════════════
# TAB 3: ALLOCATION
# ═══════════════════════════════════════════
with tab3:
    col_chart, col_table = st.columns([1, 1])

    with col_chart:
        st.subheader("Breakdown")
        fig_sun = px.sunburst(
            latest, path=["Bucket", "Account"], values="Total Value",
            color_discrete_sequence=ACCENT_RAMP,
        )
        fig_sun.update_traces(
            textinfo="label+percent entry",
            insidetextorientation="radial",
        )
        fig_sun.update_layout(
            margin=dict(l=0, r=0, t=8, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=C["text"]),
        )
        st.plotly_chart(fig_sun, use_container_width=True, config=CHART_CONFIG, key="alloc_sun")

    with col_table:
        st.subheader("By Bucket")
        pivot = latest.groupby("Bucket")[["Total Value", "Cash"]].sum()
        pivot = pivot.sort_values("Total Value", ascending=False).reset_index()
        pivot["Allocation"] = (pivot["Total Value"] / total_value * 100)

        display_pivot = pivot.copy()
        display_pivot["Total Value"] = display_pivot["Total Value"].apply(lambda x: f"${x:,.0f}")
        display_pivot["Cash"] = display_pivot["Cash"].apply(lambda x: f"${x:,.0f}")
        display_pivot["Allocation"] = display_pivot["Allocation"].apply(lambda x: f"{x:.1f}%")

        st.dataframe(
            display_pivot, use_container_width=True, hide_index=True, height=400,
        )

    # Bucket allocation over time
    st.subheader("Bucket Allocation Over Time")
    bucket_trend = fdf.groupby(["Date", "Bucket"])["Total Value"].sum().reset_index()
    date_totals = bucket_trend.groupby("Date")["Total Value"].sum().rename("DateTotal")
    bucket_trend = bucket_trend.join(date_totals, on="Date")
    bucket_trend["Pct"] = bucket_trend["Total Value"] / bucket_trend["DateTotal"] * 100

    fig_bt = px.area(
        bucket_trend, x="Date", y="Pct", color="Bucket",
        color_discrete_sequence=ACCENT_RAMP,
        labels={"Pct": "Allocation %"},
    )
    fig_bt.update_traces(line=dict(width=1))
    fig_bt = style_chart(fig_bt, height=300)
    fig_bt.update_yaxes(tickprefix="", ticksuffix="%", tickformat=".0f")
    st.plotly_chart(fig_bt, use_container_width=True, config=CHART_CONFIG, key="alloc_trend")
