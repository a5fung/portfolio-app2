"""
Portfolio Command Center v3
Dark theme · Data labels · Merged risk view
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, datetime, timedelta
import yfinance as yf
import json
from pathlib import Path

# --- CONFIG ---
st.set_page_config(page_title="Portfolio Dashboard", layout="wide", page_icon="◆")

CHART_CONFIG = {"displayModeBar": False, "staticPlot": False, "scrollZoom": False}
WARNING_THRESHOLD = 0.93
DANGER_THRESHOLD = 0.85
DATA_CACHE_TTL = 300  # 5 minutes

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

    /* Quick range pill buttons */
    div[data-testid="stHorizontalBlock"]:has(> div [data-testid="stRadio"]) [data-testid="stRadio"] > div {{
        gap: 0 !important;
        display: flex !important;
        flex-wrap: nowrap !important;
    }}
    div[data-testid="stHorizontalBlock"]:has(> div [data-testid="stRadio"]) [data-testid="stRadio"] label {{
        background: {C["surface"]} !important;
        border: 1px solid {C["border"]} !important;
        border-radius: 0 !important;
        padding: 6px 16px !important;
        cursor: pointer !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        color: {C["text_muted"]} !important;
        transition: all 0.15s ease !important;
        flex: 1 0 auto !important;
        text-align: center !important;
        justify-content: center !important;
        white-space: nowrap !important;
    }}
    div[data-testid="stHorizontalBlock"]:has(> div [data-testid="stRadio"]) [data-testid="stRadio"] label:first-of-type {{
        border-radius: 8px 0 0 8px !important;
    }}
    div[data-testid="stHorizontalBlock"]:has(> div [data-testid="stRadio"]) [data-testid="stRadio"] label:last-of-type {{
        border-radius: 0 8px 8px 0 !important;
    }}
    div[data-testid="stHorizontalBlock"]:has(> div [data-testid="stRadio"]) [data-testid="stRadio"] label:has(input:checked) {{
        background: {C["primary"]} !important;
        border-color: {C["primary"]} !important;
        color: white !important;
    }}
    div[data-testid="stHorizontalBlock"]:has(> div [data-testid="stRadio"]) [data-testid="stRadio"] label p {{
        color: inherit !important;
    }}
    div[data-testid="stHorizontalBlock"]:has(> div [data-testid="stRadio"]) [data-testid="stRadio"] > label {{
        display: none !important;
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


def add_annotation_markers(fig, date_min, date_max, annotations):
    """Add vertical dashed lines + text labels for annotations within date range."""
    if not annotations:
        return fig
    for date_str, text in annotations.items():
        try:
            ann_dt = pd.to_datetime(date_str)
        except Exception:
            continue
        if ann_dt < pd.to_datetime(date_min) or ann_dt > pd.to_datetime(date_max):
            continue
        label = text[:30] + ("..." if len(text) > 30 else "")
        fig.add_vline(
            x=ann_dt.timestamp() * 1000,
            line_dash="dash", line_color=C["warning"], line_width=1.5,
            opacity=0.7,
        )
        fig.add_annotation(
            x=ann_dt, y=1, yref="paper", yanchor="bottom",
            text=label, showarrow=False,
            font=dict(color=C["warning"], size=10),
            bgcolor=C["surface2"], bordercolor=C["warning"],
            borderwidth=1, borderpad=3, opacity=0.9,
        )
    return fig


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


@st.cache_data(ttl=DATA_CACHE_TTL)
def load_benchmark(start_date, end_date):
    """Fetch S&P 500 and QQQ data for benchmark comparison."""
    try:
        data = yf.download(["SPY", "QQQ"], start=start_date, end=end_date, progress=False)
        if data.empty:
            return pd.DataFrame()

        # Handle both single and multi-ticker downloads
        if isinstance(data.columns, pd.MultiIndex):
            result = pd.DataFrame({
                "Date": data.index,
                "SPY": data[("Close", "SPY")].values if ("Close", "SPY") in data.columns else data["Close"]["SPY"].values,
                "QQQ": data[("Close", "QQQ")].values if ("Close", "QQQ") in data.columns else data["Close"]["QQQ"].values,
            })
        else:
            data = data.reset_index()
            result = data[["Date", "Close"]].rename(columns={"Close": "SPY"})
            result["QQQ"] = 0

        return result.reset_index(drop=True)
    except Exception as e:
        st.warning(f"Could not load benchmark data: {e}")
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
    for col in ["Total Value", "Cash", "Margin Balance", "W/D"]:
        if col in df.columns:
            s = df[col].astype(str).str.replace(r'[$,\s]', '', regex=True)
            df[col] = pd.to_numeric(s.str.replace('-', '0'), errors='coerce').fillna(0)
    if "W/D" not in df.columns:
        df["W/D"] = 0.0
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


# --- SESSION STATE ---
if "quick_range" not in st.session_state:
    st.session_state.quick_range = "YTD"

def _on_date_picker_change():
    """Clear quick range when user manually picks dates."""
    st.session_state.quick_range = None

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### Filters")

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()
    default_start = max(date(datetime.now().year, 1, 1), min_date)

    date_range = st.date_input(
        "Date Range",
        [default_start, max_date],
        min_value=min_date, max_value=max_date,
        on_change=_on_date_picker_change,
    )

    st.markdown("---")

    # Sparklines — rendered after filtering via placeholder
    spark_placeholder = st.empty()

    st.markdown("---")

    # Annotations (persisted to JSON)
    ANNOTATIONS_FILE = Path(__file__).parent / "annotations.json"

    def _load_annotations():
        if ANNOTATIONS_FILE.exists():
            try:
                return json.loads(ANNOTATIONS_FILE.read_text())
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_annotations(data):
        ANNOTATIONS_FILE.write_text(json.dumps(data, indent=2))

    if "annotations" not in st.session_state:
        st.session_state.annotations = _load_annotations()

    with st.expander("📝 Annotations"):
        st.caption("Track important events")

        # Add annotation
        ann_date = st.date_input("Date", max_date, key="ann_date")
        ann_text = st.text_input("Note", key="ann_text", placeholder="What happened?")

        if st.button("Add Note", use_container_width=True):
            if ann_text:
                st.session_state.annotations[str(ann_date)] = ann_text
                _save_annotations(st.session_state.annotations)
                st.rerun()

        # Show existing annotations
        if st.session_state.annotations:
            st.markdown("**Recent Notes:**")
            for date_str in sorted(st.session_state.annotations.keys(), reverse=True)[:5]:
                st.caption(f"**{date_str}:** {st.session_state.annotations[date_str]}")

    st.markdown("---")
    with st.expander("Data Quality"):
        st.caption(f"**{len(df)}** rows · {df['Date'].min():%Y-%m-%d} → {df['Date'].max():%Y-%m-%d}")
        st.caption(f"Missing values: {df.isnull().sum().sum()}")
    st.caption(f"Auto-refreshes every {DATA_CACHE_TTL//60}min")


# --- APPLY FILTERS ---
def _resolve_date_range(quick_range, date_range, min_date, max_date):
    """Resolve start/end dates from quick range or sidebar picker."""
    today = max_date
    if quick_range:
        ranges = {
            "1W": today - timedelta(weeks=1),
            "1M": today - timedelta(days=30),
            "3M": today - timedelta(days=90),
            "YTD": date(today.year, 1, 1),
            "1Y": today - timedelta(days=365),
            "All": min_date,
        }
        start = ranges.get(quick_range, min_date)
        if isinstance(start, date):
            start = max(start, min_date)
        return start, today
    if len(date_range) == 2:
        return date_range[0], date_range[1]
    return min_date, today

active_start, active_end = _resolve_date_range(
    st.session_state.quick_range, date_range, min_date, max_date
)
fdf = df[(df["Date"] >= pd.to_datetime(active_start)) & (df["Date"] <= pd.to_datetime(active_end))]

if len(date_range) == 1 and not st.session_state.quick_range:
    st.warning("Select both start and end dates")
    fdf = df

if fdf.empty:
    st.warning("No data for selected range.")
    st.stop()

# --- BENCHMARK DATA ---
# Load full year for YTD comparison (Jan 1 of current year to today)
current_year = datetime.now().year
ytd_start = pd.Timestamp(current_year, 1, 1)
ytd_end = pd.Timestamp(datetime.now().date())
benchmark_df_ytd = load_benchmark(ytd_start, ytd_end)

# --- KEY METRICS ---
latest = fdf[fdf["Date"] == fdf["Date"].max()]
account_order = latest.groupby("Account")["Total Value"].sum().sort_values(ascending=False).index.tolist()
total_value = latest["Total Value"].sum()
total_cash = latest["Cash"].sum()
total_margin = latest["Margin Balance"].sum()

# Value-weighted portfolio YTD: each row's YTD weighted by its share of total value
portfolio_ytd = (latest["YTD"] * latest["Total Value"]).sum() / total_value if total_value else 0

# Net deposits/withdrawals for the period
net_deposits = fdf["W/D"].sum()

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

# Value attribution: Market returns vs deposits
if len(dates_sorted) >= 1:
    first = fdf[fdf["Date"] == dates_sorted[0]]
    first_value = first["Total Value"].sum()
    total_change = total_value - first_value
    market_returns = total_change - net_deposits
else:
    total_change = market_returns = 0

# Benchmark YTD returns (from Jan 1)
spy_return = 0
qqq_return = 0
if not benchmark_df_ytd.empty and len(benchmark_df_ytd) > 1:
    try:
        spy_return = ((benchmark_df_ytd["SPY"].iloc[-1] / benchmark_df_ytd["SPY"].iloc[0]) - 1) * 100
        qqq_return = ((benchmark_df_ytd["QQQ"].iloc[-1] / benchmark_df_ytd["QQQ"].iloc[0]) - 1) * 100
    except Exception as e:
        st.warning(f"Could not calculate benchmark returns: {e}")
        spy_return = qqq_return = 0

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

# Helper: Generate the "Bubble" style for deltas
def pill_html(value, color_key):
    # Define colors for the pill background/text
    if color_key == "positive":
        bg = "rgba(34, 197, 94, 0.15)"  # Green tint
        fg = C["positive"]
    elif color_key == "negative":
        bg = "rgba(239, 68, 68, 0.15)"  # Red tint
        fg = C["negative"]
    else:
        bg = "rgba(148, 163, 184, 0.15)" # Grey tint
        fg = C["text_muted"]
        
    return f'<span style="background: {bg}; color: {fg}; padding: 2px 8px; border-radius: 12px; font-weight: 600; font-size: 12px;">{value}</span>'

# Helper: Card Container
def card_html(label, value, rows):
    # Flattened HTML to prevent Markdown rendering errors
    content = "".join(rows)
    return f'<div style="background: {C["surface"]}; border: 1px solid {C["border"]}; border-radius: 12px; padding: 16px 20px; margin-bottom: 16px; height: 100%;"><div style="font-weight: 600; font-size: 13px; text-transform: uppercase; letter-spacing: 0.04em; color: {C["text_muted"]}; margin-bottom: 8px;">{label}</div><div style="font-weight: 700; font-size: 28px; color: {C["text"]}; margin-bottom: 8px;">{value}</div><div style="display: flex; flex-direction: column; gap: 6px;">{content}</div></div>'

# Columns
c1, c2, c3, c4 = st.columns(4)

# 1. Net Liquidity
with c1:
    delta_type = "positive" if delta_value and "+" in delta_value else "negative"
    pill = pill_html(delta_value, delta_type) if delta_value else "-"
    row = f'<div style="display: flex; justify-content: space-between; align-items: center; font-size: 12px;"><span style="color: {C["text_muted"]};">vs Prev Period</span>{pill}</div>'
    st.markdown(card_html("Net Liquidity", f"${total_value:,.0f}", [row]), unsafe_allow_html=True)

# 2. Margin
with c2:
    pill = pill_html(delta_margin, "text_muted") if delta_margin else "-"
    row = f'<div style="display: flex; justify-content: space-between; align-items: center; font-size: 12px;"><span style="color: {C["text_muted"]};">Change</span>{pill}</div>'
    st.markdown(card_html("Margin", f"${total_margin:,.0f}", [row]), unsafe_allow_html=True)

# 3. Cash & Deposits
with c3:
    # Row 1: Net Deposits
    r1 = f'<div style="display: flex; justify-content: space-between; align-items: center; font-size: 12px;"><span style="color: {C["text_muted"]};">Net Deposits</span><span style="color: {C["text"]}; font-weight: 600;">${net_deposits:,.0f}</span></div>'
    # Row 2: vs Prev (Pill)
    delta_type = "text_muted" # Default to grey for cash unless you want logic
    pill = pill_html(delta_cash, delta_type) if delta_cash else "-"
    r2 = f'<div style="display: flex; justify-content: space-between; align-items: center; font-size: 12px;"><span style="color: {C["text_muted"]};">vs Prev</span>{pill}</div>'
    st.markdown(card_html("CASH & DEPOSITS", f"${total_cash:,.0f}", [r1, r2]), unsafe_allow_html=True)

# 4. YTD Return
with c4:
    # SPY Row
    spy_diff = portfolio_ytd - spy_return
    spy_pill = pill_html(f"{spy_diff:+.1f}%", "positive" if spy_diff >= 0 else "negative")
    r1 = f'<div style="display: flex; justify-content: space-between; align-items: center; font-size: 12px;"><span style="color: {C["text_muted"]};">vs SPY <span style="opacity:0.5; font-size: 10px;">({spy_return:+.1f}%)</span></span>{spy_pill}</div>'
    
    # QQQ Row
    qqq_diff = portfolio_ytd - qqq_return
    qqq_pill = pill_html(f"{qqq_diff:+.1f}%", "positive" if qqq_diff >= 0 else "negative")
    r2 = f'<div style="display: flex; justify-content: space-between; align-items: center; font-size: 12px;"><span style="color: {C["text_muted"]};">vs QQQ <span style="opacity:0.5; font-size: 10px;">({qqq_return:+.1f}%)</span></span>{qqq_pill}</div>'
    
    st.markdown(card_html("YTD RETURN", f"{portfolio_ytd:+.1f}%", [r1, r2]), unsafe_allow_html=True)

# Footer
if len(dates_sorted) >= 1:
    period_start = pd.Timestamp(dates_sorted[0]).strftime("%b %#d")
    period_end = pd.Timestamp(dates_sorted[-1]).strftime("%b %#d")
    st.caption(f"Change ({period_start} – {period_end}): **\\${total_change:+,.0f}** = Market Returns **\\${market_returns:+,.0f}** + Net Deposits **\\${net_deposits:+,.0f}**")
    
# --- DRAWDOWN ALERT BANNER ---
_all_time_totals = df.groupby("Date")["Total Value"].sum().sort_index()
_all_time_peak = _all_time_totals.cummax().iloc[-1]
_current_total = _all_time_totals.iloc[-1]
_drawdown_pct = (1 - _current_total / _all_time_peak) * 100 if _all_time_peak > 0 else 0

if _drawdown_pct > 15:
    _peak_idx = _all_time_totals[_all_time_totals == _all_time_peak].index[-1]
    st.markdown(
        f'<div style="background: rgba(239,68,68,0.15); border: 1px solid {C["negative"]}; border-radius: 8px; '
        f'padding: 12px 20px; margin-bottom: 12px; display: flex; align-items: center; gap: 10px;">'
        f'<span style="font-size: 18px;">🔴</span>'
        f'<span style="color: {C["negative"]}; font-weight: 600; font-size: 14px;">'
        f'Portfolio is {_drawdown_pct:.1f}% below peak — entered drawdown on {_peak_idx:%b %d, %Y}</span></div>',
        unsafe_allow_html=True,
    )
elif _drawdown_pct > 7:
    _peak_idx = _all_time_totals[_all_time_totals == _all_time_peak].index[-1]
    st.markdown(
        f'<div style="background: rgba(245,158,11,0.15); border: 1px solid {C["warning"]}; border-radius: 8px; '
        f'padding: 12px 20px; margin-bottom: 12px; display: flex; align-items: center; gap: 10px;">'
        f'<span style="font-size: 18px;">🟡</span>'
        f'<span style="color: {C["warning"]}; font-weight: 600; font-size: 14px;">'
        f'Portfolio is {_drawdown_pct:.1f}% below peak — entered drawdown on {_peak_idx:%b %d, %Y}</span></div>',
        unsafe_allow_html=True,
    )

st.markdown("")

# --- QUICK RANGE BUTTONS ---
def _on_quick_range_change():
    """Update quick_range from radio selection."""
    st.session_state.quick_range = st.session_state._quick_range_radio

qr_options = ["1W", "1M", "3M", "YTD", "1Y", "All"]
current_idx = qr_options.index(st.session_state.quick_range) if st.session_state.quick_range in qr_options else None

qr_col1, qr_col2 = st.columns([1, 2])
with qr_col1:
    if current_idx is not None:
        st.radio(
            "Time Range", qr_options, index=current_idx,
            horizontal=True, key="_quick_range_radio",
            on_change=_on_quick_range_change, label_visibility="collapsed",
        )
    else:
        st.radio(
            "Time Range", qr_options, index=3,
            horizontal=True, key="_quick_range_radio",
            on_change=_on_quick_range_change, label_visibility="collapsed",
        )

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
    add_annotation_markers(fig, fdf["Date"].min(), fdf["Date"].max(), st.session_state.get("annotations", {}))
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG, key="ov_growth")

    # Benchmark Comparison
    if not benchmark_df_ytd.empty and len(benchmark_df_ytd) > 1:
        st.subheader("vs Benchmarks (YTD)")
        # Build portfolio daily totals for YTD period
        ytd_portfolio = df[(df["Date"] >= ytd_start) & (df["Date"] <= ytd_end)]
        if not ytd_portfolio.empty:
            port_daily = ytd_portfolio.groupby("Date")["Total Value"].sum().reset_index()
            port_daily = port_daily.sort_values("Date")
            port_daily["Portfolio"] = (port_daily["Total Value"] / port_daily["Total Value"].iloc[0] - 1) * 100

            bench = benchmark_df_ytd.copy()
            bench["SPY_pct"] = (bench["SPY"] / bench["SPY"].iloc[0] - 1) * 100
            bench["QQQ_pct"] = (bench["QQQ"] / bench["QQQ"].iloc[0] - 1) * 100

            fig_bench = go.Figure()
            fig_bench.add_trace(go.Scatter(
                x=port_daily["Date"], y=port_daily["Portfolio"],
                name="Portfolio", line=dict(color=C["primary"], width=2.5),
                mode="lines",
            ))
            fig_bench.add_trace(go.Scatter(
                x=bench["Date"], y=bench["SPY_pct"],
                name="SPY", line=dict(color=C["warning"], width=1.5, dash="dash"),
                mode="lines",
            ))
            fig_bench.add_trace(go.Scatter(
                x=bench["Date"], y=bench["QQQ_pct"],
                name="QQQ", line=dict(color=C["positive"], width=1.5, dash="dash"),
                mode="lines",
            ))
            fig_bench = style_chart(fig_bench, height=280)
            fig_bench.update_yaxes(tickprefix="", ticksuffix="%", tickformat="+.1f")
            fig_bench.update_layout(hovermode="x unified")
            st.plotly_chart(fig_bench, use_container_width=True, config=CHART_CONFIG, key="ov_bench")

    # Global Risk Monitor
    st.subheader("Risk Monitor")
    daily_totals = fdf.groupby("Date")["Total Value"].sum().reset_index()
    fig_risk = drawdown_chart(daily_totals, height=300, show_labels=True)
    add_annotation_markers(fig_risk, fdf["Date"].min(), fdf["Date"].max(), st.session_state.get("annotations", {}))
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
    # --- Performance Attribution Table ---
    st.subheader("Performance Attribution")
    _attr_rows = []
    _first_date = fdf["Date"].min()
    _last_date = fdf["Date"].max()
    _first_snap = fdf[fdf["Date"] == _first_date]
    _last_snap = fdf[fdf["Date"] == _last_date]
    _total_dollar_change = _last_snap["Total Value"].sum() - _first_snap["Total Value"].sum()

    for acct in account_order:
        s_val = _first_snap.loc[_first_snap["Account"] == acct, "Total Value"].sum()
        e_val = _last_snap.loc[_last_snap["Account"] == acct, "Total Value"].sum()
        d_change = e_val - s_val
        p_change = ((e_val / s_val) - 1) * 100 if s_val else 0
        contrib = (d_change / _total_dollar_change) * 100 if _total_dollar_change else 0
        _attr_rows.append({
            "Account": acct,
            "Start Value": s_val,
            "Current Value": e_val,
            "$ Change": d_change,
            "% Change": p_change,
            "Contribution": contrib,
        })

    _attr_df = pd.DataFrame(_attr_rows).sort_values("Current Value", ascending=False)

    _attr_display = _attr_df.copy()
    _attr_display["Start Value"] = _attr_display["Start Value"].apply(lambda x: f"${x:,.0f}")
    _attr_display["Current Value"] = _attr_display["Current Value"].apply(lambda x: f"${x:,.0f}")
    _attr_display["$ Change"] = _attr_display["$ Change"].apply(lambda x: f"${x:+,.0f}")
    _attr_display["% Change"] = _attr_display["% Change"].apply(lambda x: f"{x:+.1f}%")
    _attr_display["Contribution"] = _attr_display["Contribution"].apply(lambda x: f"{x:.1f}%")

    st.dataframe(_attr_display, use_container_width=True, hide_index=True)
    st.markdown("")

    # --- Normalized Account Comparison ---
    st.subheader("Normalized Comparison (Rebased to 100)")
    _NORM_COLORS = [
        "#3B82F6",  # blue
        "#EF4444",  # red
        "#22C55E",  # green
        "#F59E0B",  # amber
        "#A855F7",  # purple
        "#EC4899",  # pink
        "#06B6D4",  # cyan
        "#F97316",  # orange
    ]
    fig_norm = go.Figure()
    for idx, acct in enumerate(account_order):
        acct_hist = fdf[fdf["Account"] == acct].groupby("Date")["Total Value"].sum().sort_index()
        if len(acct_hist) < 2:
            continue
        base = acct_hist.iloc[0]
        if base == 0:
            continue
        normalized = (acct_hist / base) * 100
        fig_norm.add_trace(go.Scatter(
            x=normalized.index, y=normalized.values,
            name=acct, mode="lines",
            line=dict(color=_NORM_COLORS[idx % len(_NORM_COLORS)], width=2.5),
        ))
    # Baseline at 100
    fig_norm.add_hline(
        y=100, line_dash="dot", line_color=C["text_muted"],
        line_width=1, opacity=0.5,
    )
    fig_norm = style_chart(fig_norm, height=350)
    fig_norm.update_yaxes(tickprefix="", tickformat=",.0f")
    st.plotly_chart(fig_norm, use_container_width=True, config=CHART_CONFIG, key="perf_norm")
    st.markdown("")

    st.subheader("Account Performance")

    for i, account in enumerate(account_order):
        with st.container():
            st.markdown(f"#### {account}")

            acct_df = fdf[fdf["Account"] == account]
            daily = acct_df.groupby("Date")[
                ["Total Value", "Cash", "Margin Balance", "YTD", "W/D"]
            ].sum().reset_index()

            latest_acct = acct_df.iloc[-1]
            current_value = latest_acct["Total Value"]
            current_ytd = latest_acct["YTD"]
            acct_net_deposits = acct_df["W/D"].sum()

            k1, k2, k3 = st.columns(3)
            k1.metric("Current Value", f"${current_value:,.0f}")
            k2.metric("Net Deposits", f"${acct_net_deposits:,.0f}")
            k3.metric("YTD Return", f"{current_ytd:+.1f}%")

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
