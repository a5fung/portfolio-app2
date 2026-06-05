"""Dark/light palette for the Apollo Themes page.

Mirrors the Apollo Trades dark/light toggle and shares the SAME session key
(`dark_mode`), so toggling on either tab drives both. active() returns the
palette for the current mode; the ported views call it for the colors that
have to flip (page chrome, blank/out grid cells, chart bg/font). The green
rank gradient + its light text are mode-independent (dark-green cells read on
either page background), so they stay hardcoded in theme_grid.
"""
from __future__ import annotations

import streamlit as st

DARK = {
    "page_bg": "#0e1117", "text": "#e8e8e8", "sidebar_bg": "#161b22",
    "cell_blank_bg": "#0e1117", "cell_blank_txt": "#2c2e34",
    "cell_out_bg": "#1e1f24", "cell_out_txt": "#5a5d63",
    "chart_paper": "#0e1117", "chart_plot": "#161b22", "chart_font": "#e8e8e8",
}
LIGHT = {
    "page_bg": "#ffffff", "text": "#1a1a1a", "sidebar_bg": "#f0f2f6",
    "cell_blank_bg": "#ffffff", "cell_blank_txt": "#d6d6d6",
    "cell_out_bg": "#eceef1", "cell_out_txt": "#9aa0a6",
    "chart_paper": "#ffffff", "chart_plot": "#f4f4f5", "chart_font": "#1a1a1a",
}


def active() -> dict:
    """Palette for the current mode. Defaults to dark (matches Apollo Trades)."""
    return DARK if st.session_state.get("dark_mode", True) else LIGHT
