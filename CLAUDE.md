# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Streamlit-based investment portfolio dashboard that pulls data from a Google Sheet (via public CSV URL) and displays interactive Plotly charts for portfolio tracking, performance analysis, allocation breakdown, and risk monitoring. Password-protected via Streamlit secrets.

## Running the App

```bash
streamlit run Portfolio.py
```

No build step. No tests. Dependencies: `pip install -r requirements.txt`

## Architecture

**Single-file app** — all logic lives in `Portfolio.py` (the active version, v3 with dark theme). `Portfolio_v1.py` and `Portfolio_v2.py` are earlier iterations kept for reference (light theme, fewer features).

### Data Flow
1. Google Sheet → `load_data()` fetches CSV via `st.secrets["public_sheet_url"]`
2. `clean_data()` normalizes currency strings, percentages, dates
3. `validate_data()` checks for required columns: Date, Bucket, Account, Total Value
4. Sidebar date filter → `fdf` (filtered DataFrame) used throughout
5. Benchmark data (SPY/QQQ) fetched via `yfinance` for YTD comparison

### Required Data Columns
- **Date** — datetime
- **Bucket** — category grouping (e.g., "Growth", "Income")
- **Account** — brokerage account name
- **Total Value** — currency (cleaned from `$1,234` format)
- **Cash**, **Margin Balance** — currency (optional but expected)
- **YTD** — percentage (cleaned from `12.3%` format)
- **W/D** — withdrawals/deposits (v3 only)

### Key Patterns
- `@st.cache_data(ttl=DATA_CACHE_TTL)` on data-loading functions (5-min TTL in v3, 60s in v2)
- `style_chart(fig)` applies consistent Plotly styling (dark theme in v3, light in v1/v2)
- `drawdown_chart()` renders risk monitoring with peak tracking and -7%/-15% threshold lines with colored fill zones
- Custom HTML/CSS for KPI cards, sparklines, and pill-style delta badges (v3)
- All charts use `config={"displayModeBar": False}` and `fixedrange=True` to disable zoom/pan

### Secrets (`.streamlit/secrets.toml`)
- `app_password` — dashboard login password
- `public_sheet_url` — Google Sheet CSV export URL

### Dashboard Tabs (v3 / Portfolio.py)
1. **Overview** — portfolio growth line chart + global risk monitor + per-account drawdown grid
2. **Performance** — per-account bar+line charts (value, cash, YTD%) with metrics
3. **Allocation** — sunburst chart + bucket data table + allocation-over-time area chart
