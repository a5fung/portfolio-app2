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

## Apollo tabs + theme system (added 2026-06)

`pages/Apollo_Trades.py` (paper-trade P&L) and `pages/Apollo_Themes.py` (RS theme
rank grid, ported from the `rs-theme-dash` sibling repo) are multipage pages. Both
read committed point-in-time JSON snapshots exported from Apollo's Postgres
(`apollo_trades_paper.json`, `apollo_themes_snapshot.json`) so the cloud app never
touches the private DB. Theme modules: `theme_data.py` (snapshot adapter),
`theme_grid.py` / `theme_detail.py` (views), `theme_palette.py` (grid colors),
`app_theme.py` (`is_dark()`).

**Theming — hard-won lessons, do not re-litigate:**
- The theme control is **Streamlit's NATIVE theme** (☰/⋮ → Settings → Theme), not an
  in-page toggle. It is the ONLY thing that flips `st.dataframe` tables — those
  render on an HTML **canvas**, so **CSS cannot recolor them**.
- A custom in-page `st.toggle` can flip injected CSS / custom HTML but NOT
  `st.dataframe`, and its widget state resets on page navigation. Don't reintroduce one.
- **`.streamlit/config.toml` `[theme]` HIDES the Settings→Theme menu** (Streamlit
  1.58). Keep config.toml free of a `[theme]` section or the user loses the switch.
- Custom server-side colors (`C_DARK/C_LIGHT`, the theme-grid palette) follow
  `app_theme.is_dark()` (reads `st.context.theme`). They lag one interaction on a
  theme flip — Streamlit doesn't re-run Python for a theme change (see Follow-ups).

## Follow-ups (dash)
- **Palette dup**: `Portfolio.C_DARK/C_LIGHT` == `Apollo_Trades.C_DARK/C_LIGHT`
  byte-for-byte → extract a shared `app_palette.py`. (NOT `theme_palette.DARK/LIGHT`
  — that's a different, grid-specific vocabulary; leave it.)
- **Theme repaint lag**: custom content (cards, theme grid) repaints on the next
  interaction, not instantly, on a native-theme flip. Fix = use Streamlit's live
  theme CSS variables instead of baked hex. Cosmetic, low priority.
- **Theme snapshot freshness**: `apollo_themes_snapshot.json` is regenerated
  manually (`Apollo_Assistant/scripts/export_theme_snapshot.sql`). Wire a daily
  auto-export after Apollo's 5PM data pull (Apollo-side; needs push creds to this repo).
- **Security**: `.streamlit/secrets.toml` was committed to public git history earlier
  (now gitignored). Rotate `app_password` + the Anthropic key — the old values
  remain recoverable from history.
