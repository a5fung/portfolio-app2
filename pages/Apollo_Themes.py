"""Apollo Themes — RS theme rank-evolution dashboard (portfolio-app2 page).

Ports the deployed rs-theme-dash V3 (grid + detail) into the portfolio app as a
sibling page to Apollo Trades. Same data-path pattern: reads a committed
point-in-time snapshot (apollo_themes_snapshot.json, exported read-only from
Hetzner mi_themes + mi_stock_scores) so the cloud app never touches private
Postgres. View logic lives in theme_grid.py / theme_detail.py; data in
theme_data.py; colors in theme_palette.py.

Source of truth: mi_themes (the live theme engine's output). Narrative/Lane-2
themes (#167) flow in automatically once they canonicalize into mi_themes.

Dark/light: follows the app's NATIVE Streamlit theme (☰ → Settings → Theme), the
single app-global toggle that persists across all tabs and flips everything
including the data grid. Custom colors follow it via theme_palette.active().
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Apollo Themes", layout="wide", page_icon="📈")


# No password gate on this page: theme data is low-sensitivity, and the grid's
# drill-down links do a full page reload (fresh session) — a gate here would
# re-prompt for the password on every theme click.

from theme_data import snapshot_meta  # noqa: E402 — after set_page_config
from theme_grid import render_grid  # noqa: E402
from theme_detail import render_detail  # noqa: E402
from theme_state import render_toggle  # noqa: E402
from theme_palette import active  # noqa: E402

# Shared Dark Mode toggle — one persisted state across all pages.
render_toggle()

# Page background + text follow the toggle (same approach as Apollo Trades), so
# the WHOLE Themes page flips — without this the page stayed dark and light-mode
# text was invisible (dark text on a dark page).
_P = active()
st.markdown(
    f"""<style>
    .stApp {{ background-color: {_P['page_bg']}; }}
    .stApp, .stMarkdown, p, span, label, li, h1, h2, h3, h4, h5, h6 {{ color: {_P['text']}; }}
    section[data-testid="stSidebar"] {{ background-color: {_P['sidebar_bg']}; }}
    /* Streamlit themes its input widgets statically (config base=dark), so the
       sidebar selectbox / multiselect / number inputs stayed dark in light mode.
       Re-color them to follow the toggle. */
    section[data-testid="stSidebar"] [data-baseweb="select"] > div,
    section[data-testid="stSidebar"] [data-baseweb="input"],
    section[data-testid="stSidebar"] [data-baseweb="base-input"],
    section[data-testid="stSidebar"] [data-testid="stNumberInputContainer"],
    section[data-testid="stSidebar"] input {{
        background-color: {_P['page_bg']} !important;
        color: {_P['text']} !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stNumberInput"] button {{
        background-color: {_P['sidebar_bg']} !important;
        color: {_P['text']} !important;
    }}
    </style>""",
    unsafe_allow_html=True,
)

st.title("📈 Apollo Themes")
st.caption("RS theme rank evolution · narrative arcs over weekly snapshots · source: mi_themes (live theme engine)")

_meta = snapshot_meta()
if _meta.get("generated_at"):
    _gen = str(_meta["generated_at"])[:16].replace("T", " ")
    st.caption(f"📸 Snapshot generated {_gen} UTC · RS as of {_meta.get('score_date')} · point-in-time, regenerate to refresh.")


# ── View router (ported from rs-theme-dash/ThemeDash.py) ─────────────────────
if "view" not in st.session_state:
    st.session_state["view"] = "Grid"

# Drill-down via query param: the grid renders ?drill=<theme> links. Catch it,
# set state, clear the param, rerun into the detail view.
if "drill" in st.query_params:
    st.session_state["selected_theme"] = st.query_params["drill"]
    st.session_state["view"] = "Detail"
    st.query_params.clear()
    st.rerun()

view = st.sidebar.radio(
    "View", ["Grid", "Detail"],
    index=["Grid", "Detail"].index(st.session_state["view"]),
)
st.session_state["view"] = view

if view == "Grid":
    render_grid()
else:
    render_detail()
