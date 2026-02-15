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
import numpy as np
import math

# --- CONFIG ---
st.set_page_config(page_title="Portfolio", layout="wide", page_icon="◆")

CHART_CONFIG = {"displayModeBar": False, "staticPlot": False, "scrollZoom": False}
WARNING_THRESHOLD = 0.93
DANGER_THRESHOLD = 0.85
DATA_CACHE_TTL = 300  # 5 minutes

# --- ROBINHOOD-INSPIRED PALETTE ---
C_DARK = {
    # Base Colors
    "bg":           "#000000", # True Black
    "surface":      "#111113", # Almost Black Card
    "surface2":     "#18181B", # Hover state

    # Text hierarchy
    "text":         "#FFFFFF", # Pure White
    "text_sec":     "#D4D4D8", # Zinc-300 (Secondary — values, numbers)
    "text_muted":   "#A1A1AA", # Zinc-400 (Tertiary — chart labels, axes)
    "text_dim":     "#52525B", # Zinc-600 (Quaternary — micro labels)

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
    "#3B82F6", "#EF4444", "#22C55E", "#F59E0B",
    "#A855F7", "#EC4899", "#06B6D4", "#F97316",
]

C_LIGHT = {
    "bg":           "#FFFFFF",
    "surface":      "#F4F4F5",
    "surface2":     "#E4E4E7",
    "text":         "#09090B",
    "text_sec":     "#27272A",
    "text_muted":   "#52525B",
    "text_dim":     "#71717A",   # Zinc-500 (readable on white)
    "primary":      "#00B85E",
    "primary_dim":  "#DCFCE7",
    "positive":     "#00B85E",
    "positive_dim": "#DCFCE7",
    "negative":     "#DC2626",
    "negative_dim": "#FEE2E2",
    "warning":      "#D97706",
    "border":       "#D4D4D8",
    "grid":         "#D4D4D8",   # Slightly darker grid for white bg
}

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
C = C_DARK if st.session_state.dark_mode else C_LIGHT
_svg_stroke = C["text"].replace("#", "%23")

# --- NATIVE MOBILE CHROME ---
st.markdown(f"""
<meta name="theme-color" content="{C["bg"]}">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<script>
(function() {{
    var vp = document.querySelector('meta[name="viewport"]');
    if (vp) vp.setAttribute('content', 'width=device-width, initial-scale=1.0, viewport-fit=cover, maximum-scale=1.0, user-scalable=no');
}})();
</script>
""", unsafe_allow_html=True)

# --- STYLES ---
st.markdown(f"""
<style>
    /* 1. FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');

    /* 2. THE VOID */
    .stApp {{
        background-color: {C["bg"]} !important;
        font-family: 'Inter', sans-serif !important;
        color: {C["text"]} !important;
    }}

    /* 2b. GLOBAL TEXT COLOR OVERRIDE (avoid div/span to preserve inline styles) */
    .stApp p, .stApp label,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stText"],
    .stApp .stRadio label p,
    .stApp .stSelectbox label p,
    .stApp .stDateInput label p,
    .stApp .stTextInput label p,
    .stApp .stNumberInput label p {{
        color: {C["text"]} !important;
    }}
    section[data-testid="stSidebar"] {{
        color: {C["text"]} !important;
    }}
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] .stMarkdown {{
        color: {C["text"]} !important;
    }}
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] small {{
        color: {C["text_muted"]} !important;
    }}
    /* Toggle label text */
    .stApp [data-testid="stToggle"] label span {{
        color: {C["text"]} !important;
    }}
    /* Expander header text */
    .stApp [data-testid="stExpander"] summary span {{
        color: {C["text"]} !important;
    }}
    /* Radio pill text fix — inherit from parent label which has its own color */
    .stMainBlockContainer [data-testid="stRadio"] > div > label p {{
        color: {C["text"]} !important;
    }}
    /* Button text */
    .stApp button[kind="secondary"] {{
        color: {C["text"]} !important;
        border-color: {C["border"]} !important;
    }}
    /* Selectbox / Date input text */
    .stApp [data-baseweb="select"] span,
    .stApp [data-baseweb="input"] input {{
        color: {C["text"]} !important;
    }}
    /* Dataframe text */
    .stApp [data-testid="stDataFrame"] {{
        color: {C["text"]} !important;
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
    .sub-label {{ font-size: 10px; color: {C["text_dim"]}; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }}
    .sub-val {{ font-family: 'JetBrains Mono', monospace; font-size: 16px; color: {C["text"]}; font-weight: 500; white-space: nowrap; }}
    
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
        border-bottom: 1px solid {C["border"]};
    }}
    .spark-row:last-child {{ border-bottom: none; }}
    .spark-label {{ font-size: 12px; color: {C["text_muted"]}; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .spark-val {{ font-family: 'JetBrains Mono', monospace; font-size: 12px; color: {C["text"]}; text-align: right; }}

    /* 7. METRIC GRID (Header) */
    .metric-grid {{
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 16px 24px;
        margin-bottom: 24px;
    }}
    @media (min-width: 768px) {{ .metric-grid {{ grid-template-columns: repeat(4, 1fr); }} }}
    .metric-item {{ border-top: 1px solid {C["border"]}; padding-top: 8px; }}

    /* 8. QUICK RANGE PILLS — text only, Robinhood style */
    .stMainBlockContainer [data-testid="stRadio"] > label {{ display: none !important; }}
    .stMainBlockContainer [data-testid="stRadio"] > div {{
        gap: 0 !important;
        display: flex !important;
        flex-wrap: nowrap !important;
    }}
    .stMainBlockContainer [data-testid="stRadio"] > div > label {{
        flex: 1 !important;
        text-align: center !important;
        justify-content: center !important;
        background: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        padding: 10px 0 !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        color: {C["text_dim"]} !important;
        white-space: nowrap !important;
        cursor: pointer !important;
        transition: color 0.15s ease !important;
        min-height: 44px !important;
    }}
    .stMainBlockContainer [data-testid="stRadio"] > div > label:has(input:checked) {{
        color: {C["text"]} !important;
        font-weight: 700 !important;
    }}
    /* 9. TABS */
    button[data-baseweb="tab"] {{
        background-color: transparent !important;
        color: {C["text_dim"]} !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        padding: 12px 16px !important;
        letter-spacing: 0.04em !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {C["text"]} !important;
        border-bottom: 2px solid {C["primary"]} !important;
    }}
    [data-baseweb="tab-list"] {{
        border-bottom: 1px solid {C["border"]} !important;
        gap: 0 !important;
    }}

    /* UI CLEANUP */
    [data-testid="stMetric"] {{ background: transparent !important; border: none !important; padding: 0 !important; }}
    section[data-testid="stSidebar"] {{ background-color: {C["bg"]} !important; border-right: 1px solid {C["border"]}; }}
    #MainMenu, footer, [data-testid="stAppDeployButton"] {{ visibility: hidden; }}
    .block-container {{ padding-top: 1.5rem !important; }}

    /* 10. HERO HUD — desktop: side-by-side */
    .hero-hud {{
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        gap: 16px;
        width: 100%;
    }}

    /* 11. MOBILE REFINEMENTS */
    @media (max-width: 768px) {{
        /* Safe area insets */
        .block-container {{
            padding-left: max(1rem, env(safe-area-inset-left)) !important;
            padding-right: max(1rem, env(safe-area-inset-right)) !important;
            padding-bottom: calc(110px + env(safe-area-inset-bottom)) !important;
        }}
        .metric-grid {{ gap: 12px 16px; }}

        /* Kill Streamlit's floating chrome (keep sidebar toggle) */
        .stBottom,
        .stBottomBlockContainer,
        .stStatusWidget,
        .stElementToolbar {{
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
            overflow: hidden !important;
            pointer-events: none !important;
        }}

        /* Force sidebar toggle visible on mobile */
        [data-testid="collapsedControl"] {{
            display: flex !important;
            visibility: visible !important;
            position: fixed !important;
            top: 8px !important;
            left: 8px !important;
            z-index: 999999 !important;
            background: {C["border"]} !important;
            border-radius: 8px !important;
            padding: 4px !important;
            opacity: 0.85 !important;
        }}

        /* Bottom tab bar */
        [data-testid="stTabs"] {{
            overflow: visible !important;
        }}
        [data-baseweb="tab-list"] {{
            position: fixed !important;
            bottom: 0 !important;
            left: 0 !important;
            right: 0 !important;
            z-index: 999999 !important;
            background: {C["bg"]} !important;
            border-top: 1px solid {C["border"]} !important;
            border-bottom: none !important;
            padding: 0 !important;
            padding-bottom: calc(36px + env(safe-area-inset-bottom)) !important;
            justify-content: stretch !important;
        }}
        button[data-baseweb="tab"] {{
            flex: 1 !important;
            min-height: 64px !important;
            padding: 10px 4px 2px !important;
            justify-content: flex-start !important;
            align-items: center !important;
            flex-direction: column !important;
            display: flex !important;
            gap: 2px !important;
            border-bottom: none !important;
            border-top: 2px solid transparent !important;
        }}
        button[data-baseweb="tab"] p {{
            font-size: 10px !important;
            line-height: 1 !important;
            margin: 0 !important;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            border-bottom: none !important;
            border-top: 2px solid {C["primary"]} !important;
        }}
        [data-baseweb="tab-highlight"] {{
            display: none !important;
        }}
        [data-baseweb="tab-panel"] {{
            padding-bottom: calc(110px + env(safe-area-inset-bottom)) !important;
        }}

        /* Tab icons via ::before pseudo-elements */
        button[data-baseweb="tab"]::before {{
            content: '';
            display: block;
            width: 22px;
            height: 22px;
            background-size: contain;
            background-repeat: no-repeat;
            background-position: center;
            opacity: 0.45;
            flex-shrink: 0;
        }}
        button[data-baseweb="tab"][aria-selected="true"]::before {{
            opacity: 1;
        }}
        /* Overview — line chart icon */
        button[data-baseweb="tab"]:nth-of-type(1)::before {{
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='{_svg_stroke}' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M3 3v18h18'/%3E%3Cpath d='m19 9-5 5-4-4-3 3'/%3E%3C/svg%3E");
        }}
        /* Performance — bar chart icon */
        button[data-baseweb="tab"]:nth-of-type(2)::before {{
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='{_svg_stroke}' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='5' y='14' width='4' height='7' rx='0.5'/%3E%3Crect x='10' y='4' width='4' height='17' rx='0.5'/%3E%3Crect x='15' y='9' width='4' height='12' rx='0.5'/%3E%3C/svg%3E");
        }}
        /* Allocation — pie chart icon */
        button[data-baseweb="tab"]:nth-of-type(3)::before {{
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='{_svg_stroke}' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21.21 15.89A10 10 0 1 1 8 2.83'/%3E%3Cpath d='M22 12A10 10 0 0 0 12 2v10z'/%3E%3C/svg%3E");
        }}
        /* Cash Flow — wallet icon */
        button[data-baseweb="tab"]:nth-of-type(4)::before {{
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='{_svg_stroke}' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M19 7V4a1 1 0 0 0-1-1H5a2 2 0 0 0 0 4h15a1 1 0 0 1 1 1v4h-3a2 2 0 0 0 0 4h3a1 1 0 0 0 1-1v-2a1 1 0 0 0-1-1'/%3E%3Cpath d='M3 5v14a2 2 0 0 0 2 2h15a1 1 0 0 0 1-1v-4'/%3E%3C/svg%3E");
        }}
        /* Trading — candlestick chart icon */
        button[data-baseweb="tab"]:nth-of-type(5)::before {{
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='{_svg_stroke}' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cline x1='6' y1='4' x2='6' y2='20'/%3E%3Crect x='4' y='7' width='4' height='6' rx='0.5'/%3E%3Cline x1='12' y1='6' x2='12' y2='18'/%3E%3Crect x='10' y='9' width='4' height='5' rx='0.5'/%3E%3Cline x1='18' y1='3' x2='18' y2='16'/%3E%3Crect x='16' y='5' width='4' height='7' rx='0.5'/%3E%3C/svg%3E");
        }}

        /* Sticky hero header */
        .hero-hud {{
            position: sticky;
            top: 0;
            z-index: 99;
            flex-direction: column;
            align-items: flex-start;
            gap: 8px;
            background: {C["bg"]};
            padding: 12px 0;
            margin: -12px 0 0 0;
        }}

        /* Allocation columns stack vertically */
        [data-testid="stHorizontalBlock"] {{
            flex-direction: column !important;
        }}

        /* Breathing room & touch polish */
        .flex-row {{
            flex-wrap: wrap !important;
        }}
        [data-testid="stPlotlyChart"] {{
            min-height: 280px !important;
        }}
        [data-testid="stDataFrame"] {{
            overflow-x: auto !important;
            -webkit-overflow-scrolling: touch !important;
        }}
        button[data-testid="stBaseButton-headerNoPadding"] {{
            opacity: 0.5 !important;
        }}
    }}
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
def style_chart(fig, height=None, dollar_format=True):
    # 1. Base Layout
    fig.update_layout(
        template="plotly_dark" if st.session_state.dark_mode else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C["text_dim"], size=10, family="JetBrains Mono"),
        margin=dict(l=0, r=0, t=10, b=0),
        
        # 2. LOCK INTERACTIONS (The Fix)
        dragmode=False,     # Disable panning/zooming via drag
        hovermode=False if st.session_state.get("privacy_mode") else "x unified",
        hoverlabel=dict(bgcolor=C["surface2"], font_size=12, font_family="JetBrains Mono", bordercolor=C["border"]),

        xaxis=dict(
            showgrid=False,
            showline=False,
            fixedrange=True, # Disable Zoom on X
            visible=True,
            tickfont=dict(color=C["text_muted"]),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=C["grid"],
            gridwidth=1,
            showline=False,
            fixedrange=True, # Disable Zoom on Y
            tickprefix="$" if dollar_format else "",
            hoverformat=".3s" if dollar_format else "",
            showticklabels=not st.session_state.get("privacy_mode", False),
            tickfont=dict(color=C["text_muted"]),
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


def _fmt(val):
    """Compact dollar format: $1.2M, $234K, $5.3K, $900."""
    if abs(val) >= 1e6:
        return f"${val/1e6:.1f}M"
    if abs(val) >= 1e3:
        return f"${val/1e3:.0f}K"
    return f"${val:,.0f}"


def _mask(val, fmt="dollar"):
    """Return masked string when privacy mode is on, otherwise passthrough."""
    if not st.session_state.get("privacy_mode", False):
        return val
    if fmt == "dollar":
        return "$***"
    elif fmt == "pct":
        return "**%"
    elif fmt == "num":
        return "***"
    return "***"


def _hex_to_rgba(hex_color, alpha=0.3):
    """Convert hex color to rgba string."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


GOALS_FILE = Path(__file__).parent / "goals.json"

def _load_goals():
    if GOALS_FILE.exists():
        try:
            data = json.loads(GOALS_FILE.read_text())
            if isinstance(data, dict):
                return [data] if data.get("target") else []
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return []

def _save_goals(data):
    GOALS_FILE.write_text(json.dumps(data, indent=2))

if "goals" not in st.session_state:
    st.session_state.goals = _load_goals()


def section_label(text):
    """Render a thin, uppercase section label — native app style."""
    st.markdown(
        f'<div style="font-size: 11px; font-weight: 600; color: {C["text_dim"]}; '
        f'text-transform: uppercase; letter-spacing: 0.1em; margin: 24px 0 8px 0;">{text}</div>',
        unsafe_allow_html=True,
    )


def drawdown_chart(data, date_col="Date", value_col="Total Value", wd_col=None, height=None, show_legend=True, show_labels=False):
    data = data.sort_values(date_col).copy()
    data["Peak"] = data[value_col].cummax()
    if wd_col and wd_col in data.columns:
        data["Peak"] = data["Peak"] + data[wd_col]

    fig = go.Figure()

    # Danger zone fill
    fig.add_trace(go.Scatter(
        x=data[date_col], y=data["Peak"] * DANGER_THRESHOLD,
        fill=None, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=data[date_col], y=data["Peak"] * WARNING_THRESHOLD,
        fill="tonexty", fillcolor=_hex_to_rgba(C["negative"], 0.08),
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))

    # Warning zone fill
    fig.add_trace(go.Scatter(
        x=data[date_col], y=data["Peak"],
        fill="tonexty", fillcolor=_hex_to_rgba(C["warning"], 0.06),
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
        n = len(data)
        vals = data[value_col].values
        text_arr = [""] * n
        # Last value label
        if n >= 2:
            text_arr[-1] = _mask(_fmt(vals[-1]))
        # Peak label — show where all-time high was reached
        if n >= 3:
            peak_idx = int(data[value_col].idxmax())
            # Convert from DataFrame index to positional index
            peak_pos = data.index.get_loc(peak_idx)
            # Only add if peak isn't the last point (avoid overlap)
            if peak_pos != n - 1:
                text_arr[peak_pos] = _mask(_fmt(vals[peak_pos]))
        value_trace.update(
            mode="lines+markers+text",
            text=text_arr,
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


@st.cache_data(ttl=DATA_CACHE_TTL)
def load_transactions():
    """Load transaction data from Tiller Money Google Sheet."""
    try:
        url = st.secrets.get("transactions_sheet_url")
        if not url:
            return pd.DataFrame()
        return pd.read_csv(url)
    except Exception:
        return pd.DataFrame()


def clean_transactions(tdf):
    """Clean and normalize transaction data from Tiller."""
    if tdf.empty:
        return tdf
    tdf.columns = tdf.columns.str.strip()
    required = {"Date", "Description", "Category", "Amount"}
    if not required.issubset(set(tdf.columns)):
        return pd.DataFrame()
    tdf = tdf.copy()
    tdf["Date"] = pd.to_datetime(tdf["Date"], errors="coerce")
    tdf = tdf.dropna(subset=["Date"])
    tdf["Amount"] = pd.to_numeric(
        tdf["Amount"].astype(str).str.replace(r'[$,\s]', '', regex=True),
        errors="coerce",
    ).fillna(0)
    tdf["Type"] = tdf["Amount"].apply(lambda x: "Income" if x > 0 else "Expense")
    tdf["AbsAmount"] = tdf["Amount"].abs()
    # Fall back to Category Hint (Plaid auto-detect) when Category is empty
    # Subcategory overrides: extract specific subcategories as top-level
    SUBCATEGORY_MAP = {
        # Loan Payments
        "LOAN_PAYMENTS_MORTGAGE_PAYMENT": "Mortgage",
        "LOAN_PAYMENTS_CREDIT_CARD_PAYMENT": "Credit Card Payment",
        "LOAN_PAYMENTS_CAR_PAYMENT": "Car Payment",
        "LOAN_PAYMENTS_STUDENT_LOAN_PAYMENT": "Student Loan",
        "LOAN_PAYMENTS_PERSONAL_LOAN_PAYMENT": "Loan Repayment",
        # Food & Drink
        "FOOD_AND_DRINK_RESTAURANT": "Restaurants",
        "FOOD_AND_DRINK_FAST_FOOD": "Fast Food",
        "FOOD_AND_DRINK_COFFEE": "Coffee",
        "FOOD_AND_DRINK_GROCERIES": "Groceries",
        "FOOD_AND_DRINK_BEER_AND_LIQUOR": "Restaurants",
        # General Services
        "GENERAL_SERVICES_INSURANCE": "Insurance",
        "GENERAL_SERVICES_ACCOUNTING_AND_FINANCIAL_PLANNING": "Financial Services",
        "GENERAL_SERVICES_AUTOMOTIVE": "Auto Maintenance",
        "GENERAL_SERVICES_CHILDCARE": "Childcare",
        "GENERAL_SERVICES_CONSULTING_AND_LEGAL": "Legal Services",
        "GENERAL_SERVICES_EDUCATION": "Education",
        "GENERAL_SERVICES_POSTAGE_AND_SHIPPING": "Shipping",
        "GENERAL_SERVICES_STORAGE": "Storage",
        "GENERAL_SERVICES_OTHER_GENERAL_SERVICES": "Services",
        # General Merchandise
        "GENERAL_MERCHANDISE_CLOTHING_AND_ACCESSORIES": "Clothing",
        "GENERAL_MERCHANDISE_DEPARTMENT_STORES": "Department Stores",
        "GENERAL_MERCHANDISE_DISCOUNT_STORES": "Discount Stores",
        "GENERAL_MERCHANDISE_ELECTRONICS": "Electronics",
        "GENERAL_MERCHANDISE_GIFTS_AND_NOVELTIES": "Gifts",
        "GENERAL_MERCHANDISE_OFFICE_SUPPLIES": "Office Supplies",
        "GENERAL_MERCHANDISE_ONLINE_MARKETPLACES": "Online Shopping",
        "GENERAL_MERCHANDISE_PET_SUPPLIES": "Pets",
        "GENERAL_MERCHANDISE_SPORTING_GOODS": "Sporting Goods",
        "GENERAL_MERCHANDISE_SUPERSTORES": "Superstores",
        "GENERAL_MERCHANDISE_BOOKSTORES_AND_NEWSSTANDS": "Books",
        "GENERAL_MERCHANDISE_CONVENIENCE_STORES": "Convenience Stores",
        "GENERAL_MERCHANDISE_TOBACCO_AND_VAPE": "Tobacco",
        "GENERAL_MERCHANDISE_OTHER_GENERAL_MERCHANDISE": "Shopping",
    }
    tdf["Category"] = tdf["Category"].astype(str)
    if "Category Hint" in tdf.columns:
        mask = tdf["Category"].str.strip().isin(["", "nan", "NaN", "None"])
        hints = tdf.loc[mask, "Category Hint"].astype(str)
        # Try subcategory match first, else fall back to top-level
        subcats = hints.str.split(":").str[-1].str.strip()
        top_cats = hints.str.split(":").str[0].str.replace("_", " ").str.strip().str.title()
        resolved = subcats.map(SUBCATEGORY_MAP).fillna(top_cats)
        tdf.loc[mask, "Category"] = resolved
    tdf["Category"] = tdf["Category"].astype(str).str.strip().str.title()
    tdf["Category"] = tdf["Category"].replace({"": "Uncategorized", "Nan": "Uncategorized"})
    # Filter out transfers
    transfer_cats = {"Transfer", "Credit Card Payment", "Transfer Out", "Transfer In"}
    tdf = tdf[~tdf["Category"].isin(transfer_cats)]
    # Filter out investment accounts (tracked in portfolio tabs)
    if "Account" in tdf.columns:
        exclude_accounts = {"Robinhood individual", "Robinhood managed individual"}
        tdf = tdf[~tdf["Account"].str.lower().isin({a.lower() for a in exclude_accounts})]
    tdf = tdf.sort_values("Date", ascending=False)
    return tdf


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
    if "W/D" in df.columns:
        s = df["W/D"].astype(str).str.replace(r'[$,\s]', '', regex=True)
        s = s.str.replace(r'^-$', '0', regex=True)
        s = s.str.replace(r'^\(([0-9.]+)\)$', r'-\1', regex=True)
        df["W/D"] = pd.to_numeric(s, errors='coerce').fillna(0)
    else:
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
df["_WD_neg"] = df["W/D"].clip(upper=0)
df["Cum_WD"] = df.sort_values("Date").groupby("Account")["_WD_neg"].cumsum()
df["Adjusted Value"] = df["Total Value"] - df["Cum_WD"]
df = df.drop(columns=["_WD_neg"])

# --- TRANSACTION DATA ---
tdf_raw = load_transactions()
tdf = clean_transactions(tdf_raw)

# --- SESSION STATE ---
if "quick_range" not in st.session_state:
    st.session_state.quick_range = "YTD"

def _on_date_picker_change():
    """Clear quick range when user manually picks dates."""
    st.session_state.quick_range = None

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### Filters")
    st.toggle("Dark Mode", value=st.session_state.dark_mode, key="dark_mode")
    st.toggle("Privacy Mode", value=False, key="privacy_mode")

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

    with st.expander("🎯 Goal Tracker"):
        st.caption("Set portfolio targets")

        goal_label = st.text_input("Goal Name", value="", placeholder="e.g. Half Million")
        goal_target = st.number_input("Target ($)", min_value=0, step=10000, value=0)

        if st.button("Add Goal", use_container_width=True):
            if goal_target > 0:
                st.session_state.goals.append({"target": int(goal_target), "label": goal_label or f"${goal_target:,.0f}"})
                _save_goals(st.session_state.goals)
                st.rerun()
            else:
                st.warning("Enter a target amount greater than 0")

        if st.session_state.goals:
            st.markdown("---")
            for gi, g in enumerate(st.session_state.goals):
                gcol1, gcol2 = st.columns([4, 1])
                gcol1.caption(f"**{g['label']}** — ${g['target']:,.0f}")
                if gcol2.button("✕", key=f"del_goal_{gi}"):
                    st.session_state.goals.pop(gi)
                    _save_goals(st.session_state.goals)
                    st.rerun()

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
        trend_color = C["positive"] if is_up else C["negative"]
        
        # Generate SVG
        svg = make_sparkline_svg(vals, color=trend_color, width=60, height=20)
        
        # Render using the Grid Class
        st.markdown(
            f"""
            <div class="spark-row">
                <div class="spark-label" title="{acct}">{acct}</div>
                <div style="display: flex; align-items: center;">{svg}</div>
                <div class="spark-val">{_mask(f"${current:,.0f}")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
# --- HEADER: THE HUD (Responsive) ---
# 1. The "Hero" Row — single block, flexbox desktop / sticky mobile
is_pos = "+" in str(delta_value) if delta_value else False
d_color = C["positive"] if is_pos else C["negative"]

st.markdown(f"""
<div class="hero-hud">
    <div>
        <div style="margin-bottom: 4px;">
            <span style="font-size: 10px; color: {C["text_dim"]}; letter-spacing: 0.15em; text-transform: uppercase; white-space: nowrap;">Net Liquidity</span>
        </div>
        <div class="mono" style="
            font-size: clamp(32px, 8vw, 64px);
            font-weight: 700; color: {C["text"]}; line-height: 1.1; white-space: nowrap; letter-spacing: -0.04em;
        ">
            {_mask(f"${total_value:,.0f}")}
        </div>
    </div>
    <div style="display: flex; align-items: flex-end; gap: 16px; padding-bottom: 4px; flex-wrap: wrap;">
        <div>
            <div style="font-size: 10px; color: {C["text_dim"]}; margin-bottom: 2px; text-transform: uppercase;">Period Change</div>
            <div class="mono" style="font-size: clamp(16px, 4vw, 24px); color: {d_color}; white-space: nowrap;">{_mask(delta_value, "pct")}</div>
        </div>
        <div style="width: 1px; height: 24px; background: {C["border"]}; opacity: 0.5; margin-bottom: 4px;"></div>
        <div>
            <div style="font-size: 10px; color: {C["text_dim"]}; margin-bottom: 2px; text-transform: uppercase;">YTD Return</div>
            <div class="mono" style="font-size: clamp(16px, 4vw, 24px); color: {ytd_color(portfolio_ytd)}; white-space: nowrap;">{_mask(f"{portfolio_ytd:+.1f}%", "pct")}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)

# 2. THE COMPACT METRIC GRID (2x2 on Mobile)
alpha = portfolio_ytd - spy_return
alpha_col = C["positive"] if alpha >= 0 else C["negative"]

st.markdown(f"""
<div class="metric-grid">
    <div class="metric-item">
        <div style="font-size: 10px; color: {C["text_dim"]}; letter-spacing: 0.05em; margin-bottom: 2px;">BUYING POWER</div>
        <div class="mono" style="font-size: 16px; color: {C["text_sec"]};">{_mask(f"${total_cash:,.0f}")}</div>
    </div>
    <div class="metric-item">
        <div style="font-size: 10px; color: {C["text_dim"]}; letter-spacing: 0.05em; margin-bottom: 2px;">MARGIN USED</div>
        <div class="mono" style="font-size: 16px; color: {C["text_sec"]};">{_mask(f"${total_margin:,.0f}")}</div>
    </div>
    <div class="metric-item">
        <div style="font-size: 10px; color: {C["text_dim"]}; letter-spacing: 0.05em; margin-bottom: 2px;">NET DEPOSITS</div>
        <div class="mono" style="font-size: 16px; color: {C["text_sec"]};">{_mask(f"${net_deposits:,.0f}")}</div>
    </div>
    <div class="metric-item">
        <div style="font-size: 10px; color: {C["text_dim"]}; letter-spacing: 0.05em; margin-bottom: 2px;">ALPHA (vs SPY)</div>
        <div class="mono" style="font-size: 16px; color: {alpha_col};">{_mask(f"{alpha:+.1f}%", "pct")}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Goal Tracker Progress Bars ──
for goal in st.session_state.goals:
    g_target = goal["target"]
    g_label = goal.get("label", f"${g_target:,.0f}")
    g_pct = min((total_value / g_target) * 100, 100) if g_target > 0 else 0
    g_remaining = max(g_target - total_value, 0)

    # CAGR-based projection
    g_projected = ""
    if len(dates_sorted) >= 2 and total_value < g_target:
        first_date = dates_sorted[0]
        last_date = dates_sorted[-1]
        first_val = fdf[fdf["Date"] == first_date]["Total Value"].sum()
        if first_val > 0 and total_value > first_val:
            days_elapsed = (pd.Timestamp(last_date) - pd.Timestamp(first_date)).days
            if days_elapsed > 30:
                cagr = (total_value / first_val) ** (365.25 / days_elapsed) - 1
                if cagr > 0:
                    years_to_goal = math.log(g_target / total_value) / math.log(1 + cagr)
                    proj_date = pd.Timestamp.now() + pd.DateOffset(years=int(years_to_goal), months=int((years_to_goal % 1) * 12))
                    g_projected = proj_date.strftime("%b %Y")

    if g_pct >= 100:
        g_bar_color = C["positive"]
    elif g_pct >= 50:
        g_bar_color = C["primary"]
    else:
        g_bar_color = C["text_muted"]

    proj_html = f'<span style="font-size: 10px; color: {C["text_dim"]};">Est. {g_projected}</span>' if g_projected else ""

    _goal_html = f"""<div style="background: {C["surface"]}; border: 1px solid {C["border"]}; border-radius: 8px; padding: 12px 16px; margin-bottom: 8px;">
<div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px;">
<span style="font-size: 11px; font-weight: 600; color: {C["text_muted"]}; text-transform: uppercase; letter-spacing: 0.05em;">{g_label}</span>
<span class="mono" style="font-size: 11px; color: {C["text_sec"]};">{_mask(f"${total_value:,.0f}")} / {_mask(f"${g_target:,.0f}")}</span>
</div>
<div style="height: 6px; background: {C["surface2"]}; border-radius: 3px; overflow: hidden; margin-bottom: 6px;">
<div style="width: {g_pct:.1f}%; height: 100%; background: {g_bar_color}; border-radius: 3px; transition: width 0.3s ease;"></div>
</div>
<div style="display: flex; justify-content: space-between; align-items: center;">
<span class="mono" style="font-size: 11px; color: {g_bar_color};">{_mask(f"{g_pct:.0f}%", "pct")}</span>
<div style="display: flex; gap: 12px; align-items: center;">
<span style="font-size: 10px; color: {C["text_dim"]};">{_mask(f"${g_remaining:,.0f}")} to go</span>
{proj_html}
</div>
</div>
</div>"""
    st.markdown(_goal_html, unsafe_allow_html=True)

# --- DRAWDOWN ALERT BANNER & SYSTEM STATUS ---
# 1. Calculate Overall Portfolio Drawdown (YTD scope)
_ytd_start = pd.Timestamp(datetime.now().year, 1, 1)
_ytd_df = df[df["Date"] >= _ytd_start]
_all_time_totals = _ytd_df.groupby("Date")["Adjusted Value"].sum().sort_index()
if not _all_time_totals.empty:
    _peak = _all_time_totals.cummax().iloc[-1]
    _curr = _all_time_totals.iloc[-1]
    _port_dd_pct = (1 - _curr / _peak) * 100 if _peak > 0 else 0

    # 2. Main "Hero" Alert (The Big Banner)
    if _port_dd_pct > 7:
        _color = C["negative"] if _port_dd_pct > 15 else C["warning"]
        _peak_idx = _all_time_totals[_all_time_totals == _peak].index[-1]

        st.markdown(f'<div style="background: {_hex_to_rgba(_color, 0.1)}; border: 1px solid {_color}; border-radius: 8px; padding: 12px 16px; margin-bottom: 16px; display: flex; align-items: center; gap: 12px;"><span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: {_color}; flex-shrink: 0;"></span><div><div style="color: {_color}; font-weight: 600; font-size: 13px; letter-spacing: 0.02em;">PORTFOLIO DRAWDOWN ACTIVE</div><div style="color: {_color}; font-size: 12px; opacity: 0.9;">Current level is <span class="mono" style="font-weight: 700;">{_mask(f"-{_port_dd_pct:.1f}%", "pct")}</span> from peak ({_peak_idx:%b %d}).</div></div></div>', unsafe_allow_html=True)

    # 3. Account Risk Matrix (The "System Status" Grid)
    status_items = []

    for acct in account_order:
        # Get account history
        a_hist = _ytd_df[_ytd_df["Account"] == acct].groupby("Date")["Adjusted Value"].sum().sort_index()
        if a_hist.empty: continue
        
        a_peak = a_hist.cummax().iloc[-1]
        a_curr = a_hist.iloc[-1]
        a_dd = (1 - a_curr / a_peak) * 100 if a_peak > 0 else 0
        
        # --- THE STATUS LOGIC ---
        if a_dd < 1.0:
            s_color = C["positive"]
            s_bg = _hex_to_rgba(C["positive"], 0.1)
            s_border = C["positive"]
            s_opacity = "1.0"
            s_text = _mask("ATH", "pct")
        elif a_dd > 15:
            s_color = C["negative"]
            s_bg = _hex_to_rgba(C["negative"], 0.15)
            s_border = C["negative"]
            s_opacity = "1.0"
            s_text = _mask(f"-{a_dd:.1f}%", "pct")
        elif a_dd > 7:
            s_color = C["warning"]
            s_bg = _hex_to_rgba(C["warning"], 0.15)
            s_border = C["warning"]
            s_opacity = "1.0"
            s_text = _mask(f"-{a_dd:.1f}%", "pct")
        else:
            s_color = C["text_dim"]
            s_bg = "transparent"
            s_border = C["border"]
            s_opacity = "0.5"
            s_text = _mask(f"-{a_dd:.1f}%", "pct")

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

st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)

# --- TABS (3 tabs: Overview merged with Risk) ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Performance", "Allocation", "Cash Flow", "Trading Journal"])

# ═══════════════════════════════════════════
# TAB 1: OVERVIEW + RISK
# ═══════════════════════════════════════════
with tab1:
    section_label("Portfolio Growth")
    trend = fdf.groupby("Date")["Total Value"].sum().reset_index()

    fig = go.Figure()

    # Lower bound for fill (tight band under line instead of fill-to-zero)
    y_min = trend["Total Value"].min() * 0.98
    fig.add_trace(go.Scatter(
        x=trend["Date"], y=[y_min] * len(trend),
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=trend["Date"], y=trend["Total Value"],
        mode="lines",
        line=dict(color=C["primary"], width=2),
        fill="tonexty",
        fillcolor=_hex_to_rgba(C["primary"], 0.08),
        name="Total Value",
    ))
    fig = style_chart(fig, height=350)
    # Last-value end-point label
    if len(trend) >= 1:
        _last_val = trend["Total Value"].iloc[-1]
        fig.add_annotation(
            x=trend["Date"].iloc[-1], y=_last_val,
            text=_mask(_fmt(_last_val)), showarrow=False,
            font=dict(color=C["primary"], size=11, family="JetBrains Mono"),
            yshift=14,
        )
    add_annotation_markers(fig, fdf["Date"].min(), fdf["Date"].max(), st.session_state.get("annotations", {}))
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG, key="ov_growth")

    st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)

    # Benchmark Comparison
    if not benchmark_df_ytd.empty and len(benchmark_df_ytd) > 1:
        section_label("YTD vs Benchmarks")
        ytd_portfolio = df[(df["Date"] >= ytd_start) & (df["Date"] <= ytd_end)]
        if not ytd_portfolio.empty:
            port_daily = ytd_portfolio.groupby("Date")["Total Value"].sum().reset_index()
            port_daily = port_daily.sort_values("Date")
            port_daily["Portfolio"] = ((port_daily["Total Value"] / port_daily["Total Value"].iloc[0] - 1) * 100).round(2)

            bench = benchmark_df_ytd.copy()
            bench["SPY_pct"] = ((bench["SPY"] / bench["SPY"].iloc[0] - 1) * 100).round(2)
            bench["QQQ_pct"] = ((bench["QQQ"] / bench["QQQ"].iloc[0] - 1) * 100).round(2)

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
                name="QQQ", line=dict(color="#3B82F6", width=1.5, dash="dash"),
                mode="lines",
            ))
            fig_bench = style_chart(fig_bench, height=280, dollar_format=False)
            fig_bench.update_layout(showlegend=True, legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(color=C["text_muted"], size=11), bgcolor="rgba(0,0,0,0)",
            ))
            fig_bench.update_yaxes(tickprefix="", ticksuffix="%", tickformat="+.1f")
            st.plotly_chart(fig_bench, use_container_width=True, config=CHART_CONFIG, key="ov_bench")

    st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)

    # --- Global Risk Monitor ---
    section_label("Risk Monitor")
    
    # 1. Overall Portfolio Risk
    daily_totals = fdf.groupby("Date").agg({"Total Value": "sum", "Cum_WD": "sum"}).reset_index()
    fig_risk = drawdown_chart(daily_totals, wd_col="Cum_WD", height=300, show_labels=True)
    add_annotation_markers(fig_risk, fdf["Date"].min(), fdf["Date"].max(), st.session_state.get("annotations", {}))
    st.markdown(f'<div style="font-size: 12px; font-weight: 600; color: {C["text_muted"]}; margin-bottom: 4px;">Total Portfolio</div>', unsafe_allow_html=True)
    st.plotly_chart(fig_risk, use_container_width=True, config=CHART_CONFIG, key="ov_risk")

    st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True) # Spacer

    # 2. Split Risk: Growth vs Stable
    # We stack them to keep the "same size" (Full Resolution) as requested
    
    # GROWTH RISK
    daily_growth = fdf[fdf["Bucket"] == "Growth"].groupby("Date").agg({"Total Value": "sum", "Cum_WD": "sum"}).reset_index()
    if not daily_growth.empty:
        st.markdown(f'<div style="font-size: 12px; font-weight: 600; color: {C["text_muted"]}; margin-bottom: 4px;">Growth Bucket</div>', unsafe_allow_html=True)
        fig_growth = drawdown_chart(daily_growth, wd_col="Cum_WD", height=300, show_labels=True)
        # We can add a specific color override if you want Growth to look 'hotter', 
        # but for now we keep the uniform "HUD" style.
        st.plotly_chart(fig_growth, use_container_width=True, config=CHART_CONFIG, key="risk_growth")
        st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)

    # STABLE RISK
    daily_stable = fdf[fdf["Bucket"] == "Stable"].groupby("Date").agg({"Total Value": "sum", "Cum_WD": "sum"}).reset_index()
    if not daily_stable.empty:
        st.markdown(f'<div style="font-size: 12px; font-weight: 600; color: {C["text_muted"]}; margin-bottom: 4px;">Stable Bucket</div>', unsafe_allow_html=True)
        fig_stable = drawdown_chart(daily_stable, wd_col="Cum_WD", height=300, show_labels=True)
        st.plotly_chart(fig_stable, use_container_width=True, config=CHART_CONFIG, key="risk_stable")


# ═══════════════════════════════════════════
# TAB 2: PERFORMANCE
# ═══════════════════════════════════════════
with tab2:
    # --- Performance Attribution Table ---
    section_label("Performance Attribution")
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

    # Custom HTML table — mobile-friendly, 4 essential columns
    _thead = (
        f'<tr style="border-bottom: 1px solid {C["border"]};">'
        f'<th style="text-align: left; padding: 6px 8px 6px 0; font-size: 10px; font-weight: 600; color: {C["text_dim"]}; text-transform: uppercase; letter-spacing: 0.05em;">Account</th>'
        f'<th style="text-align: right; padding: 6px 8px; font-size: 10px; font-weight: 600; color: {C["text_dim"]}; text-transform: uppercase; letter-spacing: 0.05em;">Value</th>'
        f'<th style="text-align: right; padding: 6px 8px; font-size: 10px; font-weight: 600; color: {C["text_dim"]}; text-transform: uppercase; letter-spacing: 0.05em;">Change</th>'
        f'<th style="text-align: right; padding: 6px 0 6px 8px; font-size: 10px; font-weight: 600; color: {C["text_dim"]}; text-transform: uppercase; letter-spacing: 0.05em;">Contrib</th>'
        f'</tr>'
    )
    _trows = ""
    for _, row in _attr_df.iterrows():
        chg_color = C["positive"] if row["% Change"] >= 0 else C["negative"]
        _trows += (
            f'<tr style="border-bottom: 1px solid {C["border"]};">'
            f'<td style="padding: 8px 8px 8px 0; font-size: 13px; font-weight: 500; color: {C["text"]};">{row["Account"]}</td>'
            f'<td class="mono" style="text-align: right; padding: 8px; font-size: 13px; color: {C["text_sec"]};">{_mask(_fmt(row["Current Value"]))}</td>'
            f'<td class="mono" style="text-align: right; padding: 8px; font-size: 13px; color: {chg_color};">{_mask(f"{row['% Change']:+.1f}%", "pct")}</td>'
            f'<td class="mono" style="text-align: right; padding: 8px 0 8px 8px; font-size: 13px; color: {C["text_muted"]};">{_mask(f"{row['Contribution']:.0f}%", "pct")}</td>'
            f'</tr>'
        )
    st.markdown(
        f'<table style="width: 100%; border-collapse: collapse; margin-bottom: 16px;">'
        f'<thead>{_thead}</thead><tbody>{_trows}</tbody></table>',
        unsafe_allow_html=True,
    )

    # --- Normalized Account Comparison ---
    section_label("Normalized Comparison")
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
    fig_norm.update_layout(showlegend=True, legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        font=dict(color=C["text_muted"], size=11), bgcolor="rgba(0,0,0,0)",
    ))
    fig_norm.update_yaxes(tickprefix="", tickformat=",.0f")
    st.plotly_chart(fig_norm, use_container_width=True, config=CHART_CONFIG, key="perf_norm")
    st.markdown("")

    section_label("Account Performance")

    for i, account in enumerate(account_order):

        with st.container():
            st.markdown(f'<div style="font-size: 14px; font-weight: 600; color: {C["text"]}; letter-spacing: 0.02em; margin-bottom: 4px;">{account}</div>', unsafe_allow_html=True)

            acct_df = fdf[fdf["Account"] == account]
            if acct_df.empty:
                continue
            daily = acct_df.groupby("Date")[
                ["Total Value", "Cash", "Margin Balance", "YTD", "W/D"]
            ].sum().reset_index()
            daily["Invested"] = daily["Total Value"] - daily["Cash"]

            latest_acct = acct_df.iloc[-1]
            current_value = latest_acct["Total Value"]
            current_ytd = latest_acct["YTD"]
            acct_net_deposits = acct_df["W/D"].sum()
            
            ytd_c = C["positive"] if current_ytd >= 0 else C["negative"]

            st.markdown(f"""
            <div class="flex-row">
                <div>
                    <div class="sub-label">Current Value</div>
                    <div class="sub-val">{_mask(f"${current_value:,.0f}")}</div>
                </div>
                <div style="text-align: center;">
                    <div class="sub-label">Net Deposits</div>
                    <div class="sub-val" style="color: {C["text_sec"]};">{_mask(f"${acct_net_deposits:,.0f}")}</div>
                </div>
                <div style="text-align: right;">
                    <div class="sub-label">YTD Return</div>
                    <div class="sub-val" style="color: {ytd_c};">{_mask(f"{current_ytd:+.1f}%", "pct")}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            # ------------------------------------------------------

            # The Chart (Code remains similar, but using locked style)
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            fig.add_trace(go.Bar(
                x=daily["Date"], y=daily["Cash"],
                name="Cash", marker_color=C["border"],
                marker_line_width=0, opacity=0.8,
            ), secondary_y=False)

            fig.add_trace(go.Bar(
                x=daily["Date"], y=daily["Invested"],
                name="Invested", marker_color=C["primary_dim"],
                marker_line_width=0, opacity=0.9,
            ), secondary_y=False)

            curr_ytd_val = daily["YTD"].iloc[-1]
            line_color = C["positive"] if curr_ytd_val >= 0 else C["negative"]

            fig.add_trace(go.Scatter(
                x=daily["Date"], y=daily["YTD"],
                name="YTD %", mode="lines",
                line=dict(color=line_color, width=2),
                fill="tozeroy", fillcolor=_hex_to_rgba(line_color, 0.05),
            ), secondary_y=True)

            fig = style_chart(fig, height=280)
            fig.update_layout(barmode="stack")

            fig.update_yaxes(title_text="", showgrid=True, gridcolor=C["grid"], gridwidth=1, tickfont=dict(color=C["text_dim"]), secondary_y=False)
            fig.update_yaxes(title_text="", showgrid=False, tickformat="+.1f", ticksuffix="%", tickfont=dict(color=line_color), secondary_y=True)

            st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG, key=f"perf_{i}")

            # Divider between accounts, not after last
            if i < len(account_order) - 1:
                st.markdown(f'<div style="height: 24px; border-bottom: 1px solid {C["border"]}; margin-bottom: 24px;"></div>', unsafe_allow_html=True)
# ═══════════════════════════════════════════
# TAB 3: ALLOCATION
# ═══════════════════════════════════════════
with tab3:
    col_chart, col_table = st.columns([1, 1])

    with col_chart:
        section_label("Breakdown")
        fig_sun = px.sunburst(
            latest, path=["Bucket", "Account"], values="Total Value",
            color_discrete_sequence=ACCENT_RAMP,
        )
        fig_sun.update_traces(
            textinfo="label+percent entry",
            insidetextorientation="horizontal",
        )
        fig_sun.update_layout(
            margin=dict(l=0, r=0, t=8, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=C["text"]),
        )
        st.plotly_chart(fig_sun, use_container_width=True, config=CHART_CONFIG, key="alloc_sun")

    with col_table:
        section_label("By Bucket")
        pivot = latest.groupby("Bucket")[["Total Value", "Cash"]].sum()
        pivot = pivot.sort_values("Total Value", ascending=False).reset_index()
        pivot["Allocation"] = (pivot["Total Value"] / total_value * 100)

        display_pivot = pivot.copy()
        display_pivot["Total Value"] = display_pivot["Total Value"].apply(lambda x: _mask(f"${x:,.0f}"))
        display_pivot["Cash"] = display_pivot["Cash"].apply(lambda x: _mask(f"${x:,.0f}"))
        display_pivot["Allocation"] = display_pivot["Allocation"].apply(lambda x: _mask(f"{x:.1f}%", "pct"))

        st.dataframe(
            display_pivot, use_container_width=True, hide_index=True, height=400,
        )

    # Bucket allocation over time
    section_label("Allocation Over Time")
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
    fig_bt = style_chart(fig_bt, height=300, dollar_format=False)
    fig_bt.update_layout(showlegend=True, legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        font=dict(color=C["text_muted"], size=11), bgcolor="rgba(0,0,0,0)",
    ))
    fig_bt.update_yaxes(tickprefix="", ticksuffix="%", tickformat=".0f")
    st.plotly_chart(fig_bt, use_container_width=True, config=CHART_CONFIG, key="alloc_trend")

# ═══════════════════════════════════════════
# TAB 4: CASH FLOW
# ═══════════════════════════════════════════
BUDGETS_FILE = Path(__file__).parent / "budgets.json"

def _load_budgets():
    if BUDGETS_FILE.exists():
        try:
            return json.loads(BUDGETS_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}

def _save_budgets(data):
    BUDGETS_FILE.write_text(json.dumps(data, indent=2))

if "budgets" not in st.session_state:
    st.session_state.budgets = _load_budgets()


def render_cashflow_tab():
    """Render the Cash Flow tab. Returns early if no transaction data."""
    if tdf.empty:
        st.info("Connect Tiller Money to enable cash flow tracking. Add `transactions_sheet_url` to your Streamlit secrets.")
        return

    # Month selector
    available_months = pd.PeriodIndex(tdf["Date"].dt.to_period("M").unique()).sort_values(ascending=False)
    month_labels = [p.strftime("%b %Y") for p in available_months]
    if not month_labels:
        st.warning("No transactions available.")
        return
    selected_label = st.selectbox("Month", month_labels, key="cf_month_select", label_visibility="collapsed")
    selected_period = available_months[month_labels.index(selected_label)]
    month_start = selected_period.start_time
    month_end = selected_period.end_time

    ftdf = tdf[
        (tdf["Date"] >= month_start) & (tdf["Date"] <= month_end)
    ]
    if ftdf.empty:
        st.warning("No transactions for the selected month.")
        return

    # --- Computed values ---
    total_income = ftdf.loc[ftdf["Type"] == "Income", "Amount"].sum()
    total_spending = ftdf.loc[ftdf["Type"] == "Expense", "AbsAmount"].sum()
    net_flow = total_income - total_spending
    savings_rate = (net_flow / total_income * 100) if total_income > 0 else 0

    # ── 1. HERO METRICS ──
    net_color = C["positive"] if net_flow >= 0 else C["negative"]
    sr_color = C["positive"] if savings_rate >= 0 else C["negative"]

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-item">
            <div style="font-size: 10px; color: {C["text_dim"]}; letter-spacing: 0.05em; margin-bottom: 2px;">NET CASH FLOW</div>
            <div class="mono" style="font-size: 16px; color: {net_color};">{_mask(f'{"-" if net_flow < 0 else ""}${abs(net_flow):,.0f}')}</div>
        </div>
        <div class="metric-item">
            <div style="font-size: 10px; color: {C["text_dim"]}; letter-spacing: 0.05em; margin-bottom: 2px;">TOTAL INCOME</div>
            <div class="mono" style="font-size: 16px; color: {C["positive"]};">{_mask(f"${total_income:,.0f}")}</div>
        </div>
        <div class="metric-item">
            <div style="font-size: 10px; color: {C["text_dim"]}; letter-spacing: 0.05em; margin-bottom: 2px;">TOTAL SPENDING</div>
            <div class="mono" style="font-size: 16px; color: {C["negative"]};">{_mask(f"${total_spending:,.0f}")}</div>
        </div>
        <div class="metric-item">
            <div style="font-size: 10px; color: {C["text_dim"]}; letter-spacing: 0.05em; margin-bottom: 2px;">SAVINGS RATE</div>
            <div class="mono" style="font-size: 16px; color: {sr_color};">{_mask(f"{savings_rate:+.1f}%", "pct")}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 2. SANKEY CHART ──
    section_label("Income → Expense Flow")

    income_df = ftdf[ftdf["Type"] == "Income"].groupby("Category")["AbsAmount"].sum()
    expense_df = ftdf[ftdf["Type"] == "Expense"].groupby("Category")["AbsAmount"].sum()

    # Filter out categories < 1% of their respective totals
    if not income_df.empty:
        income_df = income_df[income_df / income_df.sum() >= 0.01]
    if not expense_df.empty:
        expense_df = expense_df[expense_df / expense_df.sum() >= 0.01]

    # ── Subcategory → Parent Group mapping ──
    # Keys are Title Case (clean_transactions applies .str.title() to all categories).
    # Includes Plaid fallback names (e.g. "Food And Drink" from FOOD_AND_DRINK hint).
    EXPENSE_GROUPS = {
        # Housing
        "Mortgage": "Housing", "Rent": "Housing", "Home Improvement": "Housing",
        "Hoa": "Housing", "Hoa Monthly Dues": "Housing",
        "Home Maintenance": "Housing", "Home Repair": "Housing",
        "Property Tax": "Housing", "Rent And Utilities": "Housing",
        # Utilities
        "Gas & Electric": "Utilities", "Electric": "Utilities",
        "Internet & Cable": "Utilities", "Internet": "Utilities",
        "Phone": "Utilities", "Water": "Utilities", "Garbage": "Utilities",
        "Sewer": "Utilities", "Trash": "Utilities", "Cable": "Utilities",
        # Food & Drink
        "Groceries": "Food & Drink", "Restaurants": "Food & Drink",
        "Restaurants & Bars": "Food & Drink", "Fast Food": "Food & Drink",
        "Coffee": "Food & Drink", "Dining": "Food & Drink",
        "Food And Drink": "Food & Drink",
        # Transportation
        "Gas": "Transportation", "Fuel": "Transportation",
        "Auto Insurance": "Transportation", "Auto Payment": "Transportation",
        "Parking": "Transportation", "Public Transit": "Transportation",
        "Auto Maintenance": "Transportation", "Car Maintenance": "Transportation",
        "Rideshare": "Transportation", "Transportation": "Transportation",
        # Financial (includes car/loan payments)
        "Loan Repayment": "Financial", "Student Loan": "Financial",
        "Student Loan Payment": "Financial", "Car Payment": "Financial",
        "Insurance": "Financial", "Life Insurance": "Financial",
        "Bank Fee": "Financial", "Bank Fees": "Financial",
        "Late Fee": "Financial", "Interest": "Financial",
        "Tax": "Financial", "Taxes": "Financial",
        "Loan Payments": "Financial", "General Services": "Financial",
        "Government And Non Profit": "Financial",
        "Financial Services": "Financial", "Legal Services": "Financial",
        # Shopping
        "Shopping": "Shopping", "Clothing": "Shopping",
        "General Merchandise": "Shopping", "Personal Care": "Shopping",
        "Department Stores": "Shopping", "Discount Stores": "Shopping",
        "Electronics": "Shopping", "Gifts": "Shopping",
        "Office Supplies": "Shopping", "Online Shopping": "Shopping",
        "Sporting Goods": "Shopping", "Superstores": "Shopping",
        "Books": "Shopping", "Convenience Stores": "Shopping",
        "Tobacco": "Shopping", "Shipping": "Shopping",
        # Travel (standalone)
        "Travel": "Travel",
        # Lifestyle
        "Pets": "Lifestyle", "Pet Supplies": "Lifestyle",
        "Entertainment": "Lifestyle", "Streaming": "Lifestyle",
        "Streaming Subscription": "Lifestyle",
        "Gym": "Lifestyle", "Fitness": "Lifestyle",
        "Health": "Lifestyle", "Medical": "Lifestyle", "Pharmacy": "Lifestyle",
        "Subscription": "Lifestyle", "Software": "Lifestyle",
        "Membership": "Lifestyle", "Recreation": "Lifestyle", "Community": "Lifestyle",
        "Childcare": "Lifestyle", "Education": "Lifestyle", "Storage": "Lifestyle",
    }

    # Friendly display names for confusing Plaid fallback categories
    SUBCAT_DISPLAY = {
        "Food And Drink": "Other Dining",
        "General Services": "Services",
        "General Merchandise": "Merchandise",
        "Government And Non Profit": "Government",
        "Rent And Utilities": "Rent & Utilities",
    }

    GROUP_COLORS = {
        "Housing": "#2271B1",
        "Utilities": "#009688",
        "Food & Drink": "#4CAF50",
        "Transportation": "#3366CC",
        "Financial": "#7B42BC",
        "Shopping": "#F5A623",
        "Travel": "#E25C5C",
        "Lifestyle": "#45C9C1",
        "Savings": "#22C55E",
        "Other": "#78909C",
    }

    # Monarch-inspired palette for individual categories (donut + income nodes)
    SANKEY_PALETTE = [
        "#2271B1", "#4CAF50", "#F5A623", "#E25C5C", "#7B42BC",
        "#45C9C1", "#E84393", "#3366CC", "#8BC34A", "#009688",
        "#FF7043", "#5C6BC0", "#FFCA28", "#26A69A", "#2E7D32",
        "#AB47BC", "#42A5F5", "#66BB6A", "#EF5350", "#78909C",
    ]

    has_income = not income_df.empty
    has_expense = not expense_df.empty

    if has_income or has_expense:
        expense_cats = expense_df.index.tolist() if has_expense else []
        total_inc = float(income_df.sum()) if has_income else 0
        total_exp = float(expense_df.sum()) if has_expense else 0
        ref_total = total_exp if total_exp > 0 else total_inc

        # Group expenses into parent categories
        expense_grouped = {}
        for cat in expense_cats:
            grp = EXPENSE_GROUPS.get(cat, "Other")
            expense_grouped.setdefault(grp, []).append(cat)
        sorted_groups = sorted(
            expense_grouped.keys(),
            key=lambda g: -sum(float(expense_df[c]) for c in expense_grouped[g]),
        )

        savings = total_inc - total_exp
        has_savings = has_income and savings > 0

        # Identify multi-subcat groups (fan out to col 3) vs single-subcat (terminal at col 2)
        multi_groups = [g for g in sorted_groups if len(expense_grouped[g]) > 1]

        def _lbl(name, amt):
            display = SUBCAT_DISPLAY.get(name, name)
            pct = (amt / ref_total * 100) if ref_total > 0 else 0
            return f"{display}\n{_mask(f'${amt:,.0f}')}\n{_mask(f'{pct:.0f}%', 'pct')}"

        # ── 3-column layout: Income → Groups (+Savings) → Subcategories ──
        labels, node_colors, node_x, node_y, node_customdata = [], [], [], [], []

        X_COL1, X_COL2, X_COL3 = 0.01, 0.40, 0.99
        Y_TOP, Y_BOTTOM = 0.02, 0.98
        USABLE = Y_BOTTOM - Y_TOP
        COL2_GAP = 0.02

        # ── Col 2 items: groups + savings ──
        col2_items = []  # (name, value, color, subcats)
        for g in sorted_groups:
            g_total = sum(float(expense_df[c]) for c in expense_grouped[g])
            g_color = GROUP_COLORS.get(g, GROUP_COLORS["Other"])
            col2_items.append((g, g_total, g_color, expense_grouped[g]))
        if has_savings:
            col2_items.append(("Savings", savings, GROUP_COLORS["Savings"], []))

        N_col2 = len(col2_items)
        total_col2 = sum(item[1] for item in col2_items)

        # Col 2 y-positions (proportional, tracking start/center/end per node)
        col2_pad = COL2_GAP * max(N_col2 - 1, 0)
        col2_space = USABLE - col2_pad
        y_cur = Y_TOP
        col2_ys, col2_yc, col2_ye = [], [], []
        for item in col2_items:
            prop = item[1] / total_col2 if total_col2 > 0 else 1.0 / max(N_col2, 1)
            h = prop * col2_space
            col2_ys.append(y_cur)
            col2_yc.append(y_cur + h / 2.0)
            col2_ye.append(y_cur + h)
            y_cur += h + COL2_GAP

        # Count col 3 nodes for dynamic height
        n_col3 = sum(len(expense_grouped[g]) for g in multi_groups)
        max_nodes = max(N_col2, n_col3) if n_col3 > 0 else N_col2
        chart_h = max(520, min(900, 300 + max_nodes * 48))

        # ── Node 0: Income (col 1, vertically centered) ──
        income_idx = 0
        labels.append(f"Income\n{_mask(f'${total_inc:,.0f}')}\n100%")
        node_colors.append(C["text_muted"])
        node_x.append(X_COL1)
        node_y.append(0.5)
        inc_lines = [f"  {c}: {_mask(f'${float(income_df[c]):,.0f}')}" for c in income_df.sort_values(ascending=False).index]
        node_customdata.append(["Income", _mask(f"{total_inc:,.0f}", "num"), "100%", "<br>".join(inc_lines)])

        # ── Col 2 nodes: groups + savings ──
        group_node_idx = {}
        savings_node_idx = None
        col2_annotations = []  # (label_text, y_center) for multi-group nodes → rendered as left-side annotations
        for i, (name, val, color, subcats) in enumerate(col2_items):
            idx = len(labels)
            if name == "Savings":
                sav_pct = (savings / total_inc * 100) if total_inc > 0 else 0
                labels.append(f"Savings\n{_mask(f'${val:,.0f}')}\n{_mask(f'{sav_pct:.0f}%', 'pct')}")
                savings_node_idx = idx
                node_customdata.append(["Savings", _mask(f"{val:,.0f}", "num"), _mask(f"{sav_pct:.1f}%", "pct"), ""])
            elif len(subcats) == 1:
                # Single subcat — show the specific category name
                labels.append(_lbl(subcats[0], val))
                group_node_idx[name] = idx
                dn = SUBCAT_DISPLAY.get(subcats[0], subcats[0])
                pct = (val / ref_total * 100) if ref_total > 0 else 0
                node_customdata.append([dn, _mask(f"{val:,.0f}", "num"), _mask(f"{pct:.1f}%", "pct"), ""])
            else:
                # Multi-subcat group: blank Sankey label, use annotation on left side
                labels.append("")
                col2_annotations.append((_lbl(name, val), col2_yc[i]))
                group_node_idx[name] = idx
                sub_lines = []
                for c in sorted(subcats, key=lambda c: -float(expense_df[c])):
                    dn = SUBCAT_DISPLAY.get(c, c)
                    sub_lines.append(f"  {dn}: {_mask(f'${float(expense_df[c]):,.0f}')}")
                pct = (val / ref_total * 100) if ref_total > 0 else 0
                node_customdata.append([name, _mask(f"{val:,.0f}", "num"), _mask(f"{pct:.1f}%", "pct"), "<br>".join(sub_lines)])
            node_colors.append(color)
            node_x.append(X_COL2)
            node_y.append(col2_yc[i])

        # ── Col 3 nodes: subcategories (multi-subcat groups only) ──
        # Minimum height per node prevents label overlap on small categories
        MIN_SUB_H = 0.05
        subcat_node_idx = {}
        for i, (name, val, color, subcats) in enumerate(col2_items):
            if name == "Savings" or len(subcats) <= 1:
                continue
            sorted_subs = sorted(subcats, key=lambda c: -float(expense_df[c]))
            n_subs = len(sorted_subs)
            y_s, y_e = col2_ys[i], col2_ye[i]
            sub_gap = 0.018
            sub_total_gap = sub_gap * max(n_subs - 1, 0)
            sub_space = (y_e - y_s) - sub_total_gap
            sub_total_val = sum(float(expense_df[c]) for c in sorted_subs)
            # First pass: compute raw proportional heights, enforce minimum
            raw_heights = []
            for cat in sorted_subs:
                cat_val = float(expense_df[cat])
                sub_prop = cat_val / sub_total_val if sub_total_val > 0 else 1.0 / n_subs
                raw_heights.append(max(sub_prop * sub_space, MIN_SUB_H))
            # Normalize so heights fit within available space
            raw_sum = sum(raw_heights)
            scale = sub_space / raw_sum if raw_sum > 0 else 1.0
            sub_y = y_s
            for j, cat in enumerate(sorted_subs):
                cat_val = float(expense_df[cat])
                sub_h = raw_heights[j] * scale
                sidx = len(labels)
                subcat_node_idx[cat] = sidx
                labels.append(_lbl(cat, cat_val))
                node_colors.append(color)
                node_x.append(X_COL3)
                node_y.append(sub_y + sub_h / 2.0)
                pct = (cat_val / ref_total * 100) if ref_total > 0 else 0
                node_customdata.append([SUBCAT_DISPLAY.get(cat, cat), _mask(f"{cat_val:,.0f}", "num"), _mask(f"{pct:.1f}%", "pct"), ""])
                sub_y += sub_h + sub_gap

        # ── Links ──
        sources, targets, values, link_colors = [], [], [], []

        # Income → Groups (all groups)
        for g in sorted_groups:
            g_total = sum(float(expense_df[c]) for c in expense_grouped[g])
            g_color = GROUP_COLORS.get(g, GROUP_COLORS["Other"])
            sources.append(income_idx)
            targets.append(group_node_idx[g])
            values.append(g_total)
            link_colors.append(_hex_to_rgba(g_color, 0.45))

        # Income → Savings
        if savings_node_idx is not None:
            sources.append(income_idx)
            targets.append(savings_node_idx)
            values.append(savings)
            link_colors.append(_hex_to_rgba(GROUP_COLORS["Savings"], 0.5))

        # Groups → Subcategories (multi-subcat groups only)
        for g in multi_groups:
            g_color = GROUP_COLORS.get(g, GROUP_COLORS["Other"])
            for cat in expense_grouped[g]:
                sources.append(group_node_idx[g])
                targets.append(subcat_node_idx[cat])
                values.append(float(expense_df[cat]))
                link_colors.append(_hex_to_rgba(g_color, 0.3))

        # ── Hover templates ──
        _priv = st.session_state.get("privacy_mode", False)
        node_hover = (
            "<b>%{customdata[0]}</b><br>"
            "$***<br>"
            "**% of spending"
            "<extra></extra>"
        ) if _priv else (
            "<b>%{customdata[0]}</b><br>"
            "$%{customdata[1]}<br>"
            "%{customdata[2]} of spending"
            "<br>%{customdata[3]}"
            "<extra></extra>"
        )
        link_hover = (
            "%{source.label} → %{target.label}<br>"
            "$***"
            "<extra></extra>"
        ) if _priv else (
            "%{source.label} → %{target.label}<br>"
            "$%{value:,.0f}"
            "<extra></extra>"
        )

        fig_sankey = go.Figure(go.Sankey(
            arrangement="fixed",
            node=dict(
                pad=15,
                thickness=26,
                line=dict(width=0),
                label=labels,
                color=node_colors,
                x=node_x,
                y=node_y,
                customdata=node_customdata,
                hovertemplate=node_hover,
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors,
                hovertemplate=link_hover,
            ),
        ))
        # Add left-side annotations for multi-group col 2 nodes
        # Sankey y: 0=top, 1=bottom; paper y: 0=bottom, 1=top → invert
        for ann_text, ann_y in col2_annotations:
            fig_sankey.add_annotation(
                x=X_COL2 - 0.02,
                y=1 - ann_y,
                xref="paper",
                yref="paper",
                text=ann_text.replace("\n", "<br>"),
                showarrow=False,
                xanchor="right",
                yanchor="middle",
                font=dict(color=C["text_sec"], size=12, family="Inter"),
            )
        fig_sankey.update_layout(
            height=chart_h,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=C["text_sec"], size=12, family="Inter"),
        )
        st.plotly_chart(fig_sankey, use_container_width=True, config=CHART_CONFIG, key="cf_sankey")
    else:
        st.caption("No transaction data for Sankey chart.")

    st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)

    # ── 3. BUDGET TRACKER ──
    section_label("Budget Tracker")

    # Selected month spending (already filtered by month selector above)
    month_mask = ftdf["Type"] == "Expense"
    month_spending = ftdf[month_mask].groupby("Category")["AbsAmount"].sum()

    with st.expander("Set Budgets"):
        all_expense_cats = sorted(ftdf[ftdf["Type"] == "Expense"]["Category"].unique().tolist())
        if all_expense_cats:
            budget_cat = st.selectbox("Category", all_expense_cats, key="budget_cat_select")
            budget_amt = st.number_input("Monthly Limit ($)", min_value=0, step=50, key="budget_amt_input")
            if st.button("Save Budget", use_container_width=True, key="budget_save_btn"):
                if budget_amt > 0:
                    st.session_state.budgets[budget_cat] = budget_amt
                    _save_budgets(st.session_state.budgets)
                    st.rerun()

            # Show existing budgets with delete buttons
            if st.session_state.budgets:
                st.markdown("**Current Budgets:**")
                for bcat in list(st.session_state.budgets.keys()):
                    bcol1, bcol2 = st.columns([3, 1])
                    with bcol1:
                        st.caption(f"{bcat}: {_mask(f'${st.session_state.budgets[bcat]:,.0f}')}")
                    with bcol2:
                        if st.button("✕", key=f"del_budget_{bcat}"):
                            del st.session_state.budgets[bcat]
                            _save_budgets(st.session_state.budgets)
                            st.rerun()

    # Progress bars
    if st.session_state.budgets:
        budget_html = ""
        for bcat, limit in sorted(st.session_state.budgets.items()):
            spent = float(month_spending.get(bcat, 0))
            pct = (spent / limit * 100) if limit > 0 else 0
            bar_width = min(pct, 100)
            if pct > 100:
                bar_color = C["negative"]
            elif pct >= 80:
                bar_color = C["warning"]
            else:
                bar_color = C["positive"]

            budget_html += f"""
            <div style="margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px;">
                    <span style="font-size: 12px; color: {C["text_muted"]};">{bcat}</span>
                    <span class="mono" style="font-size: 11px; color: {C["text_sec"]};">{_mask(f"${spent:,.0f}")} / {_mask(f"${limit:,.0f}")}</span>
                </div>
                <div style="height: 6px; background: {C["surface2"]}; border-radius: 3px; overflow: hidden;">
                    <div style="width: {bar_width}%; height: 100%; background: {bar_color}; border-radius: 3px; transition: width 0.3s ease;"></div>
                </div>
            </div>
            """
        st.markdown(budget_html, unsafe_allow_html=True)

    st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)

    # ── 4. SPENDING BREAKDOWN ──
    section_label("Spending by Category")
    expenses_only = ftdf[ftdf["Type"] == "Expense"]

    if not expenses_only.empty:
        cat_totals = expenses_only.groupby("Category")["AbsAmount"].sum().sort_values(ascending=False)

        # Show top N initially, group the rest into "Everything else"
        show_all_key = "cf_show_all_cats"
        if show_all_key not in st.session_state:
            st.session_state[show_all_key] = False
        INITIAL_CAT_LIMIT = 12
        has_more = len(cat_totals) > INITIAL_CAT_LIMIT
        if has_more and not st.session_state[show_all_key]:
            top_slice = cat_totals.head(INITIAL_CAT_LIMIT - 1)
            other_amt = cat_totals.iloc[INITIAL_CAT_LIMIT - 1:].sum()
            display_cats = pd.concat([top_slice, pd.Series([other_amt], index=["Everything else"])])
        else:
            display_cats = cat_totals

        # Assign colors — map each subcategory to its Sankey group color
        cat_colors = {}
        for cat in display_cats.index:
            if cat == "Everything else":
                cat_colors[cat] = GROUP_COLORS["Other"]
            else:
                grp = EXPENSE_GROUPS.get(cat, "Other")
                cat_colors[cat] = GROUP_COLORS.get(grp, GROUP_COLORS["Other"])

        col_donut, col_legend = st.columns([2, 3])

        with col_donut:
            fig_donut = go.Figure(go.Pie(
                labels=display_cats.index.tolist(),
                values=display_cats.values.tolist(),
                hole=0.6,
                marker=dict(
                    colors=[cat_colors[c] for c in display_cats.index],
                    line=dict(color=C["bg"], width=2),
                ),
                textinfo="none",
                hovertemplate="%{label}<br>$*** (**%)<extra></extra>" if st.session_state.get("privacy_mode") else "%{label}<br>$%{value:,.2f} (%{percent})<extra></extra>",
            ))
            dim = C["text_dim"]
            fig_donut.update_layout(
                showlegend=False,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=320,
                annotations=[dict(
                    text=f"{_mask(f'${total_spending:,.2f}')}<br><span style='font-size:11px;color:{dim}'>Total</span>",
                    x=0.5, y=0.5,
                    font=dict(size=16, color=C["text"], family="JetBrains Mono"),
                    showarrow=False,
                )],
            )
            st.plotly_chart(fig_donut, use_container_width=True, config=CHART_CONFIG, key="cf_donut")

        with col_legend:
            items_html = ""
            for cat, amt in display_cats.items():
                pct = (amt / total_spending * 100) if total_spending > 0 else 0
                clr = cat_colors[cat]
                items_html += (
                    f'<div style="display:flex;align-items:flex-start;gap:8px;padding:5px 0;">'
                    f'<div style="min-width:10px;width:10px;height:10px;border-radius:50%;'
                    f'background:{clr};margin-top:4px;flex-shrink:0;"></div>'
                    f'<div style="min-width:0;">'
                    f'<div style="font-size:13px;font-weight:500;color:{C["text"]};'
                    f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{cat}</div>'
                    f'<div class="mono" style="font-size:12px;color:{C["text_muted"]};">'
                    f'{_mask(f"${amt:,.2f}")} ({_mask(f"{pct:.1f}%", "pct")})</div>'
                    f'</div></div>'
                )
            st.markdown(
                f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:2px 16px;padding-top:8px;">'
                f'{items_html}</div>',
                unsafe_allow_html=True,
            )
            if has_more:
                toggle_label = "Show fewer categories" if st.session_state[show_all_key] else "Show all categories"
                if st.button(toggle_label, key="cf_toggle_cats"):
                    st.session_state[show_all_key] = not st.session_state[show_all_key]
                    st.rerun()

    st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)

    # ── 5. MONTHLY TREND (past 6 months) ──
    section_label("Monthly Trend")

    six_mo_start = (selected_period - 5).start_time
    trend_df = tdf[(tdf["Date"] >= six_mo_start) & (tdf["Date"] <= month_end)].copy()
    trend_df["Month"] = trend_df["Date"].dt.to_period("M").dt.to_timestamp()

    # Aggregate by expense group (matching Sankey) across 6-month window
    trend_expenses = trend_df[trend_df["Type"] == "Expense"].copy()
    trend_expenses["Group"] = trend_expenses["Category"].apply(lambda x: EXPENSE_GROUPS.get(x, "Other"))
    monthly_expense = trend_expenses.groupby(["Month", "Group"])["AbsAmount"].sum().reset_index()
    # Sort groups by total spend for consistent legend order
    group_order = monthly_expense.groupby("Group")["AbsAmount"].sum().sort_values(ascending=False).index.tolist()

    monthly_income = trend_df[trend_df["Type"] == "Income"].groupby("Month")["Amount"].sum().reset_index()

    if not monthly_expense.empty:
        fig_trend = px.bar(
            monthly_expense, x="Month", y="AbsAmount", color="Group",
            barmode="stack", color_discrete_map=GROUP_COLORS,
            category_orders={"Group": group_order},
            labels={"AbsAmount": "Spending", "Group": "Category"},
        )
        # Income overlay line
        if not monthly_income.empty:
            fig_trend.add_trace(go.Scatter(
                x=monthly_income["Month"], y=monthly_income["Amount"],
                name="Income", mode="lines+markers",
                line=dict(color=C["positive"], width=2, dash="dash"),
                marker=dict(size=6, color=C["positive"]),
            ))
        fig_trend = style_chart(fig_trend, height=350)
        fig_trend.update_layout(showlegend=True, legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(color=C["text_muted"], size=11), bgcolor="rgba(0,0,0,0)",
        ))
        fig_trend.update_xaxes(dtick="M1", tickformat="%b '%y")
        st.plotly_chart(fig_trend, use_container_width=True, config=CHART_CONFIG, key="cf_trend")

    st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)

    # ── 6. TRANSACTION FEED ──
    section_label("Recent Transactions")
    search_q = st.text_input("Search transactions", key="txn_search", placeholder="Filter by description or category...", label_visibility="collapsed")

    feed_df = ftdf.copy()
    if search_q:
        q = search_q.lower()
        feed_df = feed_df[
            feed_df["Description"].astype(str).str.lower().str.contains(q, na=False)
            | feed_df["Category"].astype(str).str.lower().str.contains(q, na=False)
        ]

    feed_df = feed_df.head(100)

    if not feed_df.empty:
        thead = (
            f'<tr style="border-bottom: 1px solid {C["border"]}; position: sticky; top: 0; background: {C["bg"]};">'
            f'<th style="text-align: left; padding: 6px 8px 6px 0; font-size: 10px; font-weight: 600; color: {C["text_dim"]}; text-transform: uppercase;">Date</th>'
            f'<th style="text-align: left; padding: 6px 8px; font-size: 10px; font-weight: 600; color: {C["text_dim"]}; text-transform: uppercase;">Description</th>'
            f'<th style="text-align: left; padding: 6px 8px; font-size: 10px; font-weight: 600; color: {C["text_dim"]}; text-transform: uppercase;">Category</th>'
            f'<th style="text-align: right; padding: 6px 0 6px 8px; font-size: 10px; font-weight: 600; color: {C["text_dim"]}; text-transform: uppercase;">Amount</th>'
            f'</tr>'
        )
        trows = ""
        for _, row in feed_df.iterrows():
            desc = str(row.get("Description", ""))[:40]
            amt = row["Amount"]
            amt_color = C["positive"] if amt > 0 else C["negative"]
            amt_str = _mask(f"${abs(amt):,.2f}" if amt >= 0 else f"-${abs(amt):,.2f}")
            dt_str = row["Date"].strftime("%b %d") if pd.notna(row["Date"]) else ""
            trows += (
                f'<tr style="border-bottom: 1px solid {C["border"]};">'
                f'<td class="mono" style="padding: 8px 8px 8px 0; font-size: 12px; color: {C["text_dim"]}; white-space: nowrap;">{dt_str}</td>'
                f'<td style="padding: 8px; font-size: 13px; color: {C["text"]}; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 200px;">{desc}</td>'
                f'<td style="padding: 8px; font-size: 12px; color: {C["text_muted"]};">{row["Category"]}</td>'
                f'<td class="mono" style="text-align: right; padding: 8px 0 8px 8px; font-size: 13px; color: {amt_color}; white-space: nowrap;">{amt_str}</td>'
                f'</tr>'
            )
        st.markdown(
            f'<div style="max-height: 500px; overflow-y: auto; border: 1px solid {C["border"]}; border-radius: 8px; padding: 0 12px;">'
            f'<table style="width: 100%; border-collapse: collapse;">'
            f'<thead>{thead}</thead><tbody>{trows}</tbody></table></div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("No transactions match your search.")

with tab4:
    render_cashflow_tab()

# ═══════════════════════════════════════════
# DATA LOADER: TRADING JOURNAL (SMART CUTOFF)
# ═══════════════════════════════════════════
@st.cache_data(ttl=DATA_CACHE_TTL)
def load_trading_journal():
    try:
        url = st.secrets.get("trading_journal_url")
        if not url:
            return pd.DataFrame()
        
        # Read the raw CSV
        df = pd.read_csv(url)
        
        # --- THE FIX: Stop reading at the first empty row ---
        if "Stock" in df.columns:
            # Find the index of the first row where 'Stock' is empty/NaN
            # This marks the end of the main table and the start of the "gap"
            empty_rows = df[df["Stock"].isna() | (df["Stock"].astype(str).str.strip() == "")].index
            
            if not empty_rows.empty:
                first_empty_idx = empty_rows[0]
                # Slice the DataFrame to keep only rows BEFORE the gap
                df = df.iloc[:first_empty_idx]
        
        # -----------------------------------------------------

        # 1. Clean Numeric Columns
        cols_to_clean = ["Shares", "Cost Basis", "Proceeds", "P/L", "R/R", "Risk (%)", "% Return"]
        for col in cols_to_clean:
            if col in df.columns:
                df[col] = (
                    df[col].astype(str)
                    .str.replace(r'[$,]', '', regex=True)
                    .str.replace('(', '-', regex=False)
                    .str.replace(')', '', regex=False)
                    .str.replace('%', '', regex=False)
                )
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # 2. Date Parsing
        for date_col in ["Entry Date", "Exit Date"]:
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

        # 3. Status Logic
        if "Status" not in df.columns:
            df["Status"] = df.apply(
                lambda x: "Closed" if pd.notnull(x.get("Exit Date")) else "Open", 
                axis=1
            )
            
        return df.reset_index(drop=True)

    except Exception as e:
        st.error(f"Error loading trading journal: {e}")
        return pd.DataFrame()

# ═══════════════════════════════════════════
# TAB 5: TRADING JOURNAL (HTML & CHART FIXES)
# ═══════════════════════════════════════════
with tab5:
    trades = load_trading_journal()
    
    if trades.empty:
        st.info("Connect your Google Sheet to track trading performance. Add `trading_journal_url` to secrets.")
    else:
        # Separate Open vs Closed
        closed = trades[trades["Status"] == "Closed"].copy()
        open_pos = trades[trades["Status"] == "Open"].copy()
        
        section_label("Performance Metrics")
        
        if not closed.empty:
            # --- CALCULATIONS ---
            total_pl = closed["P/L"].sum()
            count = len(closed)
            wins = closed[closed["P/L"] > 0]
            losses = closed[closed["P/L"] <= 0]
            
            n_wins = len(wins)
            win_rate = (n_wins / count * 100) if count > 0 else 0
            
            avg_win = wins["P/L"].mean() if not wins.empty else 0
            avg_loss = losses["P/L"].mean() if not losses.empty else 0
            largest_win = wins["P/L"].max() if not wins.empty else 0
            largest_loss = losses["P/L"].min() if not losses.empty else 0
            
            gross_win = wins["P/L"].sum()
            gross_loss = abs(losses["P/L"].sum())
            profit_factor = (gross_win / gross_loss) if gross_loss > 0 else 0
            payoff_ratio = (avg_win / abs(avg_loss)) if avg_loss != 0 else 0
            expectancy = total_pl / count
            
            # --- COLORS & FORMATTING ---
            pl_c = C["positive"] if total_pl >= 0 else C["negative"]
            pf_c = C["positive"] if profit_factor >= 2.0 else (C["warning"] if profit_factor >= 1.2 else C["negative"])

            # Use a simple lambda for safe string formatting
            def color_span(val, color=None):
                c = color if color else (C["positive"] if val >= 0 else C["negative"])
                return f'<span style="color:{c}">{_mask(f"${val:,.0f}")}</span>'

            # --- RENDER HTML (Flat String to prevent errors) ---
            st.markdown(f"""
<div class="metric-grid">
    <div class="metric-item">
        <div class="sub-label">TOTAL P/L</div>
        <div class="mono" style="font-size: 20px; color: {pl_c};">{_mask(f"${total_pl:,.0f}")}</div>
    </div>
    <div class="metric-item">
        <div class="sub-label">WIN RATE</div>
        <div class="mono" style="font-size: 20px; color: {C['text']};">{_mask(f"{win_rate:.0f}%", "pct")} <span style="font-size:12px;color:{C['text_dim']}">({_mask(f"{n_wins}", "num")}/{_mask(f"{count}", "num")})</span></div>
    </div>
    <div class="metric-item">
        <div class="sub-label">PROFIT FACTOR</div>
        <div class="mono" style="font-size: 20px; color: {pf_c};">{_mask(f"{profit_factor:.2f}", "num")}</div>
    </div>
    <div class="metric-item">
        <div class="sub-label">EXPECTANCY</div>
        <div class="mono" style="font-size: 20px; color: {C['text']};">{_mask(f"${expectancy:,.0f}")}<span style="font-size:12px;color:{C['text_dim']}">/trade</span></div>
    </div>
    <div class="metric-item">
        <div class="sub-label">AVG WIN / LOSS</div>
        <div class="mono" style="font-size: 16px;">{color_span(avg_win)} <span style="color:{C['text_dim']}">/</span> {color_span(avg_loss)}</div>
    </div>
    <div class="metric-item">
        <div class="sub-label">MAX WIN / LOSS</div>
        <div class="mono" style="font-size: 16px;">{color_span(largest_win)} <span style="color:{C['text_dim']}">/</span> {color_span(largest_loss)}</div>
    </div>
    <div class="metric-item">
        <div class="sub-label">PAYOFF RATIO</div>
        <div class="mono" style="font-size: 16px; color: {C['text']};">{_mask(f"1 : {payoff_ratio:.1f}", "num")}</div>
    </div>
        <div class="metric-item">
        <div class="sub-label">ACTIVE TRADES</div>
        <div class="mono" style="font-size: 16px; color: {C['text']};">{len(open_pos)}</div>
    </div>
</div>
""", unsafe_allow_html=True)

        st.markdown('<div style="height: 32px;"></div>', unsafe_allow_html=True)
        
        # --- CHARTS ---
        c1, c2 = st.columns([2, 1])
        
        with c1:
            section_label("Equity Curve")
            if not closed.empty:
                # Group by date to handle multiple trades in one day
                daily_pl = closed.dropna(subset=["Exit Date"]).groupby("Exit Date")["P/L"].sum().reset_index().sort_values("Exit Date")
                if daily_pl.empty:
                    st.caption("No closed trades to plot.")
                else:
                    daily_pl["Equity"] = daily_pl["P/L"].cumsum()

                    # Drawdown calculation
                    daily_pl["Peak"] = daily_pl["Equity"].cummax()
                    daily_pl["Drawdown"] = daily_pl["Equity"] - daily_pl["Peak"]
                    max_dd = daily_pl["Drawdown"].min()

                    fig_curve = go.Figure()
                    fig_curve.add_trace(go.Scatter(
                        x=daily_pl["Exit Date"], y=daily_pl["Equity"],
                        mode="lines",
                        line=dict(color=C["primary"], width=2),
                        fill="tozeroy",
                        fillcolor=_hex_to_rgba(C["primary"], 0.05),
                        name="Cumulative P/L"
                    ))

                    # Max DD Annotation
                    fig_curve.add_annotation(
                        x=daily_pl["Exit Date"].iloc[-1],
                        y=daily_pl["Equity"].iloc[-1],
                        text=_mask(f"Max DD: ${max_dd:,.0f}"),
                        showarrow=False, yshift=10,
                        font=dict(color=C["negative"], size=10)
                    )

                    fig_curve = style_chart(fig_curve, height=320)
                    fig_curve.update_yaxes(tickprefix="$", showgrid=True, gridcolor=C["grid"])
                    st.plotly_chart(fig_curve, use_container_width=True, config=CHART_CONFIG)
            else:
                st.caption("No closed trades to plot.")

        with c2:
            section_label("Win Rate by Setup")
            if not closed.empty and "Setup" in closed.columns:
                setup_stats = closed.groupby("Setup")["P/L"].sum().sort_values(ascending=True)
                colors = [C["positive"] if x > 0 else C["negative"] for x in setup_stats.values]
                
                # FIXED: Uses 'cornerradius' instead of 'rx'
                fig_setup = go.Figure(go.Bar(
                    y=setup_stats.index,
                    x=setup_stats.values,
                    orientation='h',
                    marker=dict(color=colors, cornerradius=4), 
                    text=setup_stats.apply(lambda x: _mask(f"${x:,.0f}")),
                    textposition="auto"
                ))
                fig_setup = style_chart(fig_setup, height=320)
                st.plotly_chart(fig_setup, use_container_width=True, config=CHART_CONFIG)

        st.markdown('<div style="height: 32px;"></div>', unsafe_allow_html=True)

        # --- LOG ---
        section_label("Trade Log")
        
        col_search, col_filter = st.columns([2, 1])
        with col_search:
            q_trade = st.text_input("Search Trades", placeholder="Symbol, Setup, Notes...", label_visibility="collapsed")
        with col_filter:
            status_filter = st.selectbox("Status", ["All", "Open", "Closed"], label_visibility="collapsed")

        display_df = trades.copy()
        if status_filter != "All":
            display_df = display_df[display_df["Status"] == status_filter]
        
        if q_trade:
            mask = display_df.astype(str).apply(lambda x: x.str.contains(q_trade, case=False, na=False)).any(axis=1)
            display_df = display_df[mask]

        if not display_df.empty:
            display_df = display_df.sort_values(["Exit Date", "Entry Date"], ascending=False)
            
            _pf = "***" if st.session_state.get("privacy_mode") else "$%d"
            _nf = "***" if st.session_state.get("privacy_mode") else "%d"
            st.dataframe(
                display_df,
                column_config={
                    "Entry Date": st.column_config.DateColumn("Entry", format="MMM DD"),
                    "Exit Date": st.column_config.DateColumn("Exit", format="MMM DD"),
                    "P/L": st.column_config.NumberColumn("P/L", format=_pf),
                    "Cost Basis": st.column_config.NumberColumn("Cost", format=_pf),
                    "Proceeds": st.column_config.NumberColumn("Proceeds", format=_pf),
                    "Shares": st.column_config.NumberColumn("Size", format=_nf),
                },
                hide_index=True,
                use_container_width=True,
                height=400
            )
        else:
            st.caption("No trades found.")
