"""Single source of truth for the app's light/dark state.

Driven by Streamlit's NATIVE theme switch (☰ menu → Settings → Theme), which is
app-global and persists across every page automatically — so ONE control themes
all tabs (Portfolio, Apollo Trades, Apollo Themes), including the data grid that
a custom in-page toggle can't reach. Default is dark via .streamlit/config.toml
([theme] base="dark"); users override per-session via the menu. Every page
derives its palette from is_dark().
"""
from __future__ import annotations

import streamlit as st


def is_dark() -> bool:
    """True if the active Streamlit theme is dark. Defaults to dark if the
    theme API is unavailable (older Streamlit) — never raises."""
    try:
        return getattr(st.context.theme, "type", "dark") != "light"
    except Exception:
        return True
