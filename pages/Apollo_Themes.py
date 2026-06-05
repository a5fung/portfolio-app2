"""Apollo Themes — RS theme rank-evolution dashboard (portfolio-app2 page).

Ports the deployed rs-theme-dash V3 (grid + detail) into the portfolio app as a
sibling page to Apollo Trades. Same data-path pattern: reads a committed
point-in-time snapshot (apollo_themes_snapshot.json, exported read-only from
Hetzner mi_themes + mi_stock_scores) so the cloud app never touches private
Postgres. View logic lives in theme_grid.py / theme_detail.py; data in
theme_data.py; colors in theme_palette.py — all interface-parity with
rs-theme-dash so the views are near-verbatim ports.

Source of truth: mi_themes (the live theme engine's output). Narrative/Lane-2
themes (#167) flow in automatically once they canonicalize into mi_themes.

Dark/light: shares the `dark_mode` session key with Apollo Trades, so the
toggle on either tab drives both.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Apollo Themes", layout="wide", page_icon="📈")


# ── Auth gate ─────────────────────────────────────────────────────────────────
# Streamlit multipage pages are reachable by direct URL and do NOT inherit the
# main page's gate automatically. session_state IS shared within a browser
# session, so this passes through silently once you've logged in on the main
# Portfolio page, and prompts on direct access. Same mechanism as Portfolio.py
# (shared `password_correct` key + `app_password` secret).
def _check_password() -> bool:
    if st.session_state.get("password_correct"):
        return True
    st.title("📈 Apollo Themes")
    pw = st.text_input("Enter Password", type="password", key="pw_themes")
    if pw:
        try:
            if pw == st.secrets["app_password"]:
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("Incorrect password")
        except Exception:
            st.error("Password not configured in secrets")
    return False


if not _check_password():
    st.stop()


# ── Imports (after set_page_config) ───────────────────────────────────────────
from theme_palette import active  # noqa: E402
from theme_data import snapshot_meta  # noqa: E402
from theme_grid import render_grid  # noqa: E402
from theme_detail import render_detail  # noqa: E402


# ── Dark/light chrome ─────────────────────────────────────────────────────────
# Shares `dark_mode` with Apollo Trades (default dark). The ported views read
# the same palette via theme_palette.active() for their cell/chart colors.
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

P = active()
st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {P['page_bg']}; }}
    .stApp, .stMarkdown, p, span, label, li,
    h1, h2, h3, h4, h5, h6 {{ color: {P['text']}; }}
    [data-testid="stSidebar"] {{ background-color: {P['sidebar_bg']}; }}
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{ color: {P['text']}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Header + theme toggle ─────────────────────────────────────────────────────
col_title, col_toggle = st.columns([4, 1])
with col_title:
    st.title("📈 Apollo Themes")
    st.caption("RS theme rank evolution · narrative arcs over weekly snapshots · source: mi_themes (live theme engine)")
with col_toggle:
    if st.button("◐ Theme", use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

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
