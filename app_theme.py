"""Native Streamlit theme → is_dark().

The app follows Streamlit's OWN theme (☰ → Settings → Theme), because that is the
only thing that flips st.dataframe tables (canvas-rendered — CSS can't recolor
them). is_dark() reads the active theme; every page mirrors it into
st.session_state.dark_mode so the existing palette logic is unchanged.
"""
from __future__ import annotations

import streamlit as st


def is_dark() -> bool:
    """True if the active Streamlit theme is dark. Default dark; never raises."""
    try:
        return getattr(st.context.theme, "type", "dark") != "light"
    except Exception:
        return True
