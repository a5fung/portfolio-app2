"""Shared light/dark theme state for ALL pages.

One toggle, shared across every page, and it survives page navigation.

The bug this fixes: a plain `st.toggle(key="dark_mode")` makes `dark_mode` a
*widget* key, and Streamlit drops widget state when you switch pages — so the
theme reset to dark on every page change. Here `dark_mode` is set ONLY
programmatically (never via a widget `key=`), so Streamlit treats it as ordinary
session state that persists across pages. The visible toggle uses a separate
widget key and writes `dark_mode` via its on_change callback.

Every page keeps reading `st.session_state.dark_mode` as before; it just calls
`render_toggle()` instead of its own `st.toggle(...)`.
"""
from __future__ import annotations

import streamlit as st

_KEY = "dark_mode"                 # persisted truth — read by every page; NOT a widget key
_WIDGET = "_dark_mode_widget"      # the sidebar toggle widget (this is what gets GC'd on nav)


def is_dark() -> bool:
    """Current theme. Default dark. Persists across page navigation."""
    return bool(st.session_state.get(_KEY, True))


def _sync() -> None:
    st.session_state[_KEY] = st.session_state[_WIDGET]


def render_toggle() -> None:
    """Render the shared Dark Mode toggle in the sidebar. Call once per page."""
    if _KEY not in st.session_state:
        st.session_state[_KEY] = True
    st.sidebar.toggle(
        "Dark Mode",
        value=st.session_state[_KEY],
        key=_WIDGET,
        on_change=_sync,
    )
