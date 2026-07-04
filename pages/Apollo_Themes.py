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


# ── #398: two-way search — a ticker → its themes · a name → its stocks ───────
# Mirrors the Telegram `/themes <arg>` lanes (shipped 6/28): ticker-shaped input
# tries the ticker lane first, falls through to a theme-name search on a miss.
from theme_data import get_themes_for_ticker, get_theme_members, get_active_themes  # noqa: E402
from urllib.parse import quote  # noqa: E402 — drill URLs need %-encoding (spaces/'&' break markdown links)

_q = st.text_input(
    "🔎 Ticker or theme name", "",
    placeholder="e.g. NVDA — or: semiconductor",
    help="A 1-5 letter ticker lists its active themes; anything else searches theme names.",
).strip()
if _q:
    _hits = get_themes_for_ticker(_q) if (_q.isalpha() and len(_q) <= 5) else None
    if _hits is not None and not _hits.empty:
        st.markdown(f"**{_q.upper()}** is in **{len(_hits)}** active theme(s):")
        for _, _r in _hits.iterrows():
            st.markdown(
                f"- [{_r['name']}](?drill={quote(str(_r['name']), safe='')}) — {_r['stage']} · "
                f"RS {_r['rs_avg']:.0f} · {_r['members']} members"
            )
    else:
        _names = [n for n in get_active_themes() if _q.lower() in n.lower()]
        if not _names:
            st.info(
                f"No active theme contains **{_q.upper()}** and no theme name matches "
                f"“{_q}” (recency window 7d)."
            )
        elif len(_names) == 1:
            _members = get_theme_members(_names[0])
            st.markdown(
                f"**[{_names[0]}](?drill={quote(_names[0], safe='')})** — {len(_members)} members: "
                + " · ".join(_members)
            )
        else:
            st.markdown(f"**{len(_names)}** theme names match “{_q}”:")
            for _n in _names:
                st.markdown(f"- [{_n}](?drill={quote(_n, safe='')})")


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
