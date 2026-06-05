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
