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
st.set_page_config(page_title="Portfolio", layout="wide", page_icon="◆")

CHART_CONFIG = {"displayModeBar": False, "staticPlot": False, "scrollZoom": False}
WARNING_THRESHOLD = 0.93
DANGER_THRESHOLD = 0.85
DATA_CACHE_TTL = 300  # 5 minutes

# --- ROBINHOOD-INSPIRED PALETTE ---
# --- ROBINHOOD-INSPIRED PALETTE (Final) ---
C = {
    # Base Colors
    "bg":           "#000000", # True Black
    "surface":      "#111113", # Almost Black Card
    "surface2":     "#18181B", # Hover state
    
    # Text (New & Old Keys mapped together)
    "text":         "#FFFFFF", # Pure White
    "text_sec":     "#A1A1AA", # Zinc-400 (Secondary)
    "text_muted":   "#A1A1AA", # Zinc-400 (Old key mapped to new color)
    "text_dim":     "#52525B", # Zinc-600 (Tertiary)
    
    # Status Colors
    "primary":      "#00D26A", # Growth Green
    "primary_dim":  "#00331B", # Dark Green (for bar charts)
    "positive":     "#00D26A", 
    "positive_dim": "#004D26",
    "negative":     "#F82C2C", # Alert Red
    "negative_dim": "#450A0A",
    "warning":      "#F59E0B", # Amber
    
    # Structural
    "border":       "#27272A", # Zinc-800
    "grid":         "#18181B", # Very subtle grid
}

ACCENT_RAMP = [
    "#3B82F6", "#06B6D4", "#8B5CF6", "#14B8A6",
    "#A78BFA", "#2DD4BF", "#6366F1", "#34D399",
]

# --- STYLES ---
st.markdown(f"""
<style>
    /* 1. FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');

    /* 2. THE VOID */
    .stApp {{
        background-color: #000000 !important;
        font-family: 'Inter', sans-serif !important;
    }}
    
    /* 3. MONOSPACE NUMBERS */
    .mono {{
        font-family: 'JetBrains Mono', monospace !important;
        font-feature-settings: "zero" 1;
    }}

    /* 4. FORCED HORIZONTAL ROW (The Fix for Screenshot 2) */
    .flex-row {{
        display: flex !important;
        flex-direction: row !important;
        justify-content: space-between;
        align-items: flex-end;
        gap: 12px;
        width: 100%;
        margin-bottom: 16px;
        overflow-x: hidden; /* Prevent horizontal scroll */
    }}
    
    /* 5. METRIC LABELS & VALUES */
    .sub-label {{ font-size: 10px; color: #666; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }}
    .sub-val {{ font-family: 'JetBrains Mono', monospace; font-size: 16px; color: #FFF; font-weight: 500; white-space: nowrap; }}
    
    /* Mobile text scaling for the row */
    @media (max-width: 600px) {{
        .sub-val {{ font-size: 14px; }} /* Slightly smaller on phone to fit 3 in a row */
    }}

    /* 6. SPARKLINE GRID */
    .spark-row {{
        display: grid;
        grid-template-columns: 1fr 60px 80px;
        align-items: center;
        gap: 12px;
        padding: 8px 0;
        border-bottom: 1px solid #111;
    }}
    .spark-label {{ font-size: 12px; color: #888; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .spark-val {{ font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #FFF; text-align: right; }}

    /* 7. METRIC GRID (Header) */
    .metric-grid {{
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 16px 24px;
        margin-bottom: 24px;
    }}
    @media (min-width: 768px) {{ .metric-grid {{ grid-template-columns: repeat(4, 1fr); }} }}
    .metric-item {{ border-top: 1px solid #222; padding-top: 8px; }}

    /* UI CLEANUP */
    [data-testid="stMetric"] {{ background: transparent !important; border: none !important; padding: 0 !important; }}
    section[data-testid="stSidebar"] {{ background-color: #000000 !important; border-right: 1px solid #222; }}
    #MainMenu, footer, header {{ visibility: hidden; }}
    .block-container {{ padding-top: 1.5rem !important; }}
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
    # 1. Base Layout
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#666", size=10, family="JetBrains Mono"),
        margin=dict(l=0, r=0, t=10, b=0),
        
        # 2. LOCK INTERACTIONS (The Fix)
        dragmode=False,     # Disable panning/zooming via drag
        hovermode="x unified",
        
        xaxis=dict(
            showgrid=False, 
            showline=False, 
            fixedrange=True, # Disable Zoom on X
            visible=True
        ),
        yaxis=dict(
            showgrid=True, 
            gridcolor="#222", 
            gridwidth=1, 
            showline=False,
            fixedrange=True, # Disable Zoom on Y
            tickprefix="$"
        ),
        showlegend=False,
    )
    
    # 3. Remove the floating toolbar completely
    fig.update_layout(modebar_remove=["zoom", "pan", "select", "lasso2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"])
    
    if height: fig.update_layout(height=height)
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
        
        # Color logic: Green if up, Red if down
        is_up = len(vals) >= 2 and vals[-1] >= vals[0]
        trend_color = "#00D26A" if is_up else "#F82C2C"
        
        # Generate SVG
        svg = make_sparkline_svg(vals, color=trend_color, width=60, height=20)
        
        # Render using the Grid Class
        st.markdown(
            f"""
            <div class="spark-row">
                <div class="spark-label" title="{acct}">{acct}</div>
                <div style="display: flex; align-items: center;">{svg}</div>
                <div class="spark-val">${current:,.0f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
# --- HEADER: THE HUD (Responsive) ---
# 1. The "Hero" Row
c_hero, c_delta = st.columns([1.5, 2])

with c_hero:
    st.markdown(f"""
    <div style="margin-bottom: 4px;">
        <span style="font-size: 10px; color: #666; letter-spacing: 0.15em; text-transform: uppercase; white-space: nowrap;">Net Liquidity</span>
    </div>
    <div class="mono" style="
        font-size: clamp(32px, 8vw, 64px); /* Slightly smaller clamp for tighter mobile fit */
        font-weight: 700; color: #FFF; line-height: 1.1; white-space: nowrap; letter-spacing: -0.04em;
    ">
        ${total_value:,.0f}
    </div>
    """, unsafe_allow_html=True)

with c_delta:
    is_pos = "+" in str(delta_value) if delta_value else False
    d_color = "#00D26A" if is_pos else "#F82C2C"
    
    st.markdown(f"""
    <div style="height: 100%; min-height: 60px; display: flex; align-items: flex-end; gap: 16px; padding-bottom: 4px; flex-wrap: wrap;">
        <div>
            <div style="font-size: 9px; color: #666; margin-bottom: 2px; text-transform: uppercase;">Period Change</div>
            <div class="mono" style="font-size: clamp(16px, 4vw, 24px); color: {d_color}; white-space: nowrap;">{delta_value}</div>
        </div>
        <div style="width: 1px; height: 24px; background: #333; opacity: 0.5; margin-bottom: 4px;"></div>
        <div>
            <div style="font-size: 9px; color: #666; margin-bottom: 2px; text-transform: uppercase;">YTD Return</div>
            <div class="mono" style="font-size: clamp(16px, 4vw, 24px); color: {ytd_color(portfolio_ytd)}; white-space: nowrap;">{portfolio_ytd:+.1f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)

# 2. THE COMPACT METRIC GRID (2x2 on Mobile)
alpha = portfolio_ytd - spy_return
alpha_col = "#00D26A" if alpha >= 0 else "#F82C2C"

st.markdown(f"""
<div class="metric-grid">
    <div class="metric-item">
        <div style="font-size: 9px; color: #666; letter-spacing: 0.05em; margin-bottom: 2px;">BUYING POWER</div>
        <div class="mono" style="font-size: 16px; color: #DDD;">${total_cash:,.0f}</div>
    </div>
    <div class="metric-item">
        <div style="font-size: 9px; color: #666; letter-spacing: 0.05em; margin-bottom: 2px;">MARGIN USED</div>
        <div class="mono" style="font-size: 16px; color: #DDD;">${total_margin:,.0f}</div>
    </div>
    <div class="metric-item">
        <div style="font-size: 9px; color: #666; letter-spacing: 0.05em; margin-bottom: 2px;">NET DEPOSITS</div>
        <div class="mono" style="font-size: 16px; color: #DDD;">${net_deposits:,.0f}</div>
    </div>
    <div class="metric-item">
        <div style="font-size: 9px; color: #666; letter-spacing: 0.05em; margin-bottom: 2px;">ALPHA (vs SPY)</div>
        <div class="mono" style="font-size: 16px; color: {alpha_col};">{alpha:+.1f}%</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- DRAWDOWN ALERT BANNER & SYSTEM STATUS ---
# 1. Calculate Overall Portfolio Drawdown
_all_time_totals = df.groupby("Date")["Total Value"].sum().sort_index()
if not _all_time_totals.empty:
    _peak = _all_time_totals.cummax().iloc[-1]
    _curr = _all_time_totals.iloc[-1]
    _port_dd_pct = (1 - _curr / _peak) * 100 if _peak > 0 else 0

    # 2. Main "Hero" Alert (The Big Banner)
    if _port_dd_pct > 7:
        _color = "#F82C2C" if _port_dd_pct > 15 else "#F59E0B"
        _icon = "🔴" if _port_dd_pct > 15 else "🟡"
        _peak_idx = _all_time_totals[_all_time_totals == _peak].index[-1]
        
        # Flattened HTML to prevent code-block rendering
        st.markdown(f'<div style="background: rgba({("248, 44, 44" if _port_dd_pct > 15 else "245, 158, 11")}, 0.1); border: 1px solid {_color}; border-radius: 8px; padding: 12px 16px; margin-bottom: 16px; display: flex; align-items: center; gap: 12px;"><span style="font-size: 16px;">{_icon}</span><div><div style="color: {_color}; font-weight: 600; font-size: 13px; letter-spacing: 0.02em;">PORTFOLIO DRAWDOWN ACTIVE</div><div style="color: {_color}; font-size: 12px; opacity: 0.9;">Current level is <span class="mono" style="font-weight: 700;">-{_port_dd_pct:.1f}%</span> from peak ({_peak_idx:%b %d}).</div></div></div>', unsafe_allow_html=True)

    # 3. Account Risk Matrix (The "System Status" Grid)
    status_items = []

    for acct in account_order:
        # Get account history
        a_hist = df[df["Account"] == acct].groupby("Date")["Total Value"].sum().sort_index()
        if a_hist.empty: continue
        
        a_peak = a_hist.cummax().iloc[-1]
        a_curr = a_hist.iloc[-1]
        a_dd = (1 - a_curr / a_peak) * 100 if a_peak > 0 else 0
        
        # --- THE STATUS LOGIC ---
        if a_dd < 1.0:
            # ATH MODE (The "Winning" State)
            # Bright Cyan/Green, Full Opacity
            s_color = "#00D26A" # Growth Green
            s_bg = "rgba(0, 210, 106, 0.1)"
            s_border = "#00D26A"
            s_opacity = "1.0"
            s_text = "ATH" # Display "ATH" instead of -0.0%
        elif a_dd > 15:
            # DANGER
            s_color = "#F82C2C"
            s_bg = "rgba(248, 44, 44, 0.15)"
            s_border = "#F82C2C"
            s_opacity = "1.0"
            s_text = f"-{a_dd:.1f}%"
        elif a_dd > 7:
            # WARNING
            s_color = "#F59E0B"
            s_bg = "rgba(245, 158, 11, 0.15)"
            s_border = "#F59E0B"
            s_opacity = "1.0"
            s_text = f"-{a_dd:.1f}%"
        else:
            # NORMAL / DORMANT (The "Ghost" State)
            # This is for accounts that are down, but not critically.
            # We dim them to reduce noise.
            s_color = "#666" 
            s_bg = "transparent"
            s_border = "#333"
            s_opacity = "0.5"
            s_text = f"-{a_dd:.1f}%"

        # Render
        status_items.append(f'<div style="display: flex; align-items: center; justify-content: space-between; gap: 10px; background: {s_bg}; border: 1px solid {s_border}; border-radius: 4px; padding: 6px 10px; opacity: {s_opacity}; min-width: 140px; flex: 1;"><span style="font-size: 10px; font-weight: 600; color: {s_color}; text-transform: uppercase; letter-spacing: 0.05em;">{acct}</span><span class="mono" style="font-size: 11px; color: {s_color};">{s_text}</span></div>')

    # Render the Grid
    st.markdown(f'<div style="display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 24px;">{"".join(status_items)}</div>', unsafe_allow_html=True)

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
# Tab 1: Overview Chart
    fig = go.Figure()
    
    # Gradient Fill ("The Glow")
    fig.add_trace(go.Scatter(
        x=trend["Date"], y=trend["Total Value"],
        mode="lines",
        line=dict(color="#00D26A", width=2), # Robinhood Green
        fill="tozeroy",
        fillcolor="rgba(0, 210, 106, 0.1)", # Subtle radioactive glow
        name="Total Value",
    ))
    
    # Update style to match the green theme
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#666", size=10, family="JetBrains Mono"),
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False, showline=False),
        yaxis=dict(showgrid=True, gridcolor="#222", gridwidth=1, showline=False),
        hovermode="x unified",
        showlegend=False,
    )
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

    # --- Global Risk Monitor ---
    st.subheader("Risk Monitor")
    
    # 1. Overall Portfolio Risk
    daily_totals = fdf.groupby("Date")["Total Value"].sum().reset_index()
    fig_risk = drawdown_chart(daily_totals, height=300, show_labels=True)
    add_annotation_markers(fig_risk, fdf["Date"].min(), fdf["Date"].max(), st.session_state.get("annotations", {}))
    st.markdown("**Total Portfolio Drawdown**")
    st.plotly_chart(fig_risk, use_container_width=True, config=CHART_CONFIG, key="ov_risk")

    st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True) # Spacer

    # 2. Split Risk: Growth vs Stable
    # We stack them to keep the "same size" (Full Resolution) as requested
    
    # GROWTH RISK
    daily_growth = fdf[fdf["Bucket"] == "Growth"].groupby("Date")["Total Value"].sum().reset_index()
    if not daily_growth.empty:
        st.markdown("**Growth Bucket Risk**")
        fig_growth = drawdown_chart(daily_growth, height=300, show_labels=True)
        # We can add a specific color override if you want Growth to look 'hotter', 
        # but for now we keep the uniform "HUD" style.
        st.plotly_chart(fig_growth, use_container_width=True, config=CHART_CONFIG, key="risk_growth")
        st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)

    # STABLE RISK
    daily_stable = fdf[fdf["Bucket"] == "Stable"].groupby("Date")["Total Value"].sum().reset_index()
    if not daily_stable.empty:
        st.markdown("**Stable Bucket Risk**")
        fig_stable = drawdown_chart(daily_stable, height=300, show_labels=True)
        st.plotly_chart(fig_stable, use_container_width=True, config=CHART_CONFIG, key="risk_stable")


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
            # Header
            st.markdown(f"#### {account}")

            acct_df = fdf[fdf["Account"] == account]
            daily = acct_df.groupby("Date")[
                ["Total Value", "Cash", "Margin Balance", "YTD", "W/D"]
            ].sum().reset_index()
            daily["Invested"] = daily["Total Value"] - daily["Cash"]

            latest_acct = acct_df.iloc[-1]
            current_value = latest_acct["Total Value"]
            current_ytd = latest_acct["YTD"]
            acct_net_deposits = acct_df["W/D"].sum()
            
            # --- THE FIX: Custom HTML Row instead of st.columns ---
            # This guarantees 3 items in one row on mobile
            ytd_c = "#00D26A" if current_ytd >= 0 else "#F82C2C"
            
            st.markdown(f"""
            <div class="flex-row">
                <div>
                    <div class="sub-label">Current Value</div>
                    <div class="sub-val">${current_value:,.0f}</div>
                </div>
                <div style="text-align: center;">
                    <div class="sub-label">Net Deposits</div>
                    <div class="sub-val" style="color: #DDD;">${acct_net_deposits:,.0f}</div>
                </div>
                <div style="text-align: right;">
                    <div class="sub-label">YTD Return</div>
                    <div class="sub-val" style="color: {ytd_c};">{current_ytd:+.1f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            # ------------------------------------------------------

            # The Chart (Code remains similar, but using locked style)
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            fig.add_trace(go.Bar(
                x=daily["Date"], y=daily["Cash"],
                name="Cash", marker_color="#27272A", 
                marker_line_width=0, opacity=0.8,
            ), secondary_y=False)

            fig.add_trace(go.Bar(
                x=daily["Date"], y=daily["Invested"],
                name="Invested", marker_color="#064E3B", 
                marker_line_width=0, opacity=0.9,
            ), secondary_y=False)

            curr_ytd_val = daily["YTD"].iloc[-1]
            line_color = "#00D26A" if curr_ytd_val >= 0 else "#F82C2C"
            
            fig.add_trace(go.Scatter(
                x=daily["Date"], y=daily["YTD"],
                name="YTD %", mode="lines",
                line=dict(color=line_color, width=2),
                fill="tozeroy", fillcolor=f"rgba({('0, 210, 106' if curr_ytd_val >= 0 else '248, 44, 44')}, 0.05)",
            ), secondary_y=True)

            # Apply strict locked style
            fig = style_chart(fig, height=280) # Slightly shorter for mobile
            fig.update_layout(barmode="stack")
            
            # Axis formatting
            fig.update_yaxes(title_text="", showgrid=True, gridcolor="#18181B", gridwidth=1, tickfont=dict(color="#52525B"), secondary_y=False)
            fig.update_yaxes(title_text="", showgrid=False, tickformat="+.1f", ticksuffix="%", tickfont=dict(color=line_color), secondary_y=True)

            # Important: config=CHART_CONFIG ensures staticPlot=False but we disabled interactions in layout
            st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG, key=f"perf_{i}")
            
            st.markdown('<div style="height: 24px; border-bottom: 1px solid #111; margin-bottom: 24px;"></div>', unsafe_allow_html=True)
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
