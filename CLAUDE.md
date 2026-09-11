# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Deployed at `https://alvin-portfolio-dashboard.streamlit.app`** — confirmed 2026-09-11 from the running app's own title bar. SSO-gated at the platform level, so an automated fetch cannot read the RENDERED page: dashboard changes are verified by recomputing against the same snapshot JSON the page reads, and the report must SAY the page was not opened. ⚠ **I RECORDED THE WRONG URL FIRST AND THE REASONING IS WORTH KEEPING.** I guessed `portfolio-app2.streamlit.app` from the repo name, saw it answer 303 to Streamlit's auth endpoint, and treated the signed login payload as proof the app existed. It is not — that endpoint answers for any subdomain under `*.streamlit.app`, so the handshake says nothing about whether an app is provisioned there. **A redirect is not an existence check.** The only proof would have been reaching the app, which SSO prevents; the real answer came from a screenshot.

A Streamlit-based investment portfolio dashboard that pulls data from a Google Sheet (via public CSV URL) and displays interactive Plotly charts for portfolio tracking, performance analysis, allocation breakdown, and risk monitoring. Password-protected via Streamlit secrets.


## Verifying a dashboard change (#641, 2026-09-11)

The app is **SSO-gated at the Streamlit platform level**, so no automated fetch can read the
rendered page — a password in `st.secrets` would not change that, because the gate sits in front of
the app rather than inside it. "Done = confirmed in production" therefore needs a defined substitute,
and this is it. Three steps, in order:

1. **Recompute the page's own function against the SAME snapshot JSON the page reads, and assert the
   numbers it will render.** Not a fixture — the committed snapshot IS the page's input, so this is
   genuinely strong. `test_theme_canon.py::TestRealSnapshot*` and `test_theme_movers.py` are the
   pattern: pin the real values, not a synthetic shape.
2. **Say in the report that the rendered page was NOT opened.** Every time. A verification that
   omits its own limit reads as a stronger claim than it is.
3. **Ask him to look ONLY when the change is visual** — a chart, a layout, a colour. That is the
   part step 1 cannot cover, and it is the only part worth his time. When his own words are the bar
   (*"he can tell in one look..."*), step 3 is not optional and no amount of step 1 replaces it.

⚠ **Two things that cost time on 2026-09-11 and will again:**
- **Streamlit Cloud served the PRE-push build for ~14 minutes.** A screenshot taken in that window
  shows the old code and looks like a failed fix. **The tell is a rendered string the new code
  computes** — after #640 the Rank Flow caption is built from the band list, so a live build reads
  `31+ · No rank`. Find that tell before concluding anything; a content push forces a rebuild.
- **A redirect is not an existence check.** `portfolio-app2.streamlit.app` answers 303 to
  Streamlit's auth endpoint and issues a signed login payload, and no app is deployed there —
  that endpoint answers for any `*.streamlit.app` subdomain.

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
- **Snapshot freshness (BOTH snapshots are manual until auto-export ships)**:
  `apollo_themes_snapshot.json` ← `Apollo_Assistant/scripts/export_theme_snapshot.sql`;
  `apollo_trades_paper.json` ← `Apollo_Assistant/scripts/export_trades_snapshot.sql`
  (saved 2026-06-10 after the trades snapshot silently sat at 6/03 for a week —
  the original export was ad-hoc/unsaved). One-liner in each SQL's header:
  ssh + `docker exec -i apollo-postgres psql … < script.sql > snapshot.json`,
  then commit+push here (Streamlit Cloud redeploys on push). Wire a daily
  auto-export after Apollo's 5PM data pull (Apollo-side #194; needs push creds
  to this repo).
- **Security**: `.streamlit/secrets.toml` was committed to public git history earlier
  (now gitignored). Rotate `app_password` + the Anthropic key — the old values
  remain recoverable from history.
