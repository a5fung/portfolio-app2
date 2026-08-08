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
# #472 ecosystem view is imported lazily in the render branch below, wrapped so a load
# failure degrades to Grid instead of crashing the whole Apollo Themes page.

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
# #472 (ADR 0032): "Ecosystems" is the new two-level board and the default
# landing view (parity with the Telegram /themes default), but the flat
# "Grid"/"Detail" views stay fully intact and one click away — nothing removed.
# #315: "Bump Chart" (R2) and "Forward Returns" (R4) are NEW views keyed on
# canonical identity (R3, theme_canon.py) — Grid/Ecosystems/Detail above are
# UNCHANGED and stay on raw theme `name` (they match the Telegram board).
# 2026-08-08 (operator: bump chart "not easiest to read... like Sankey"):
# "Rank Flow" (alluvial) is a NEW view on the same canonical data source as
# Bump Chart — Bump Chart itself stays (operator called it "ok", not wrong).
# A "Rank Heatmap" tab shipped alongside it and was PULLED the same evening:
# the operator asked "heatmap is the exact same as grid?" and he was right —
# measured, it rendered 294 rows against Grid's 306, with only 23 of 311
# cohorts merging anything at all. ~93% of it was Grid. Worse, the margin it
# did add was partly WRONG (a satellite theme absorbing the defense primes; a
# specialty-chemicals theme fused with an ad-tech one), so it bought almost no
# new information and misled where it differed. Removed rather than kept as
# near-duplicate. The identity defect underneath is tracked separately — it
# also affects Rank Flow, just less visibly at band level.
_VIEWS = ["Ecosystems", "Grid", "Detail", "Rank Flow", "Bump Chart", "Forward Returns"]
if "view" not in st.session_state:
    st.session_state["view"] = "Ecosystems"

# Drill-down via query param: the grid/ecosystem view renders ?drill=<theme>
# links. Catch it, set state, clear the param, rerun into the detail view.
if "drill" in st.query_params:
    st.session_state["selected_theme"] = st.query_params["drill"]
    st.session_state["view"] = "Detail"
    st.query_params.clear()
    st.rerun()

view = st.sidebar.radio(
    "View", _VIEWS,
    index=_VIEWS.index(st.session_state["view"]) if st.session_state["view"] in _VIEWS else 0,
)
st.session_state["view"] = view

if view == "Ecosystems":
    try:
        from theme_ecosystem_view import render_ecosystems  # lazy: isolate any load failure to this tab
        render_ecosystems()
    except Exception as _eco_err:  # never crash the whole page — Grid/Detail must stay usable
        st.error("⚠ Ecosystem view failed to load — the error is shown below; the Grid view still works.")
        st.exception(_eco_err)
        render_grid()
elif view == "Grid":
    render_grid()
elif view == "Detail":
    render_detail()
elif view == "Rank Flow":
    try:
        from theme_flow import render_flow  # lazy: isolate any load failure to this tab
        render_flow()
    except Exception as _flow_err:
        st.error("⚠ Rank flow failed to load — the error is shown below; the Grid view still works.")
        st.exception(_flow_err)
        render_grid()
elif view == "Bump Chart":
    try:
        from theme_bump import render_bump  # lazy: isolate any load failure to this tab
        render_bump()
    except Exception as _bump_err:
        st.error("⚠ Bump chart failed to load — the error is shown below; the Grid view still works.")
        st.exception(_bump_err)
        render_grid()
else:  # "Forward Returns"
    try:
        from theme_forward import render_forward  # lazy: isolate any load failure to this tab
        render_forward()
    except Exception as _fwd_err:
        st.error("⚠ Forward-returns view failed to load — the error is shown below; the Grid view still works.")
        st.exception(_fwd_err)
        render_grid()
