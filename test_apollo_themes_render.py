"""Render regression for pages/Apollo_Themes.py (#315).

A page can import cleanly and still throw on render — Streamlit executes
page code fresh on every session/interaction, so a clean `streamlit run`
boot log proves nothing about whether a specific VIEW renders without
raising. `st.testing.v1.AppTest` actually drives the script (real
Python execution against the real committed snapshot), which is the
load-bearing check #315 asked for: every sidebar `View` option must render
with zero exceptions, not just import.

Each view gets its own fresh AppTest run (session state from a prior view
selection shouldn't leak between cases, and it keeps failures isolated to
one view instead of one giant multi-step test).
"""
from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

_PAGE = "pages/Apollo_Themes.py"
_VIEWS = ["Ecosystems", "Grid", "Detail", "Weekly Movers", "Rank Flow", "Bump Chart", "Forward Returns"]


@pytest.mark.parametrize("view", _VIEWS)
def test_view_renders_without_exception(view):
    at = AppTest.from_file(_PAGE, default_timeout=90)
    at.run()
    assert not at.exception, f"Apollo_Themes.py raised on initial load: {list(at.exception)}"
    at.sidebar.radio[0].set_value(view).run()
    assert not at.exception, f"View {view!r} raised: {list(at.exception)}"


def test_bump_chart_plots_at_least_one_series():
    # A render with zero exceptions but zero content is the OTHER failure
    # mode (an empty panel silently shipped) — #315 explicitly asked not to
    # ship that. Confirm real chart output, not just a clean exception list.
    at = AppTest.from_file(_PAGE, default_timeout=90)
    at.run()
    at.sidebar.radio[0].set_value("Bump Chart").run()
    assert not at.exception
    assert at.get("plotly_chart"), "Bump Chart rendered no plotly_chart element"


def test_rank_flow_plots_a_sankey():
    # Same "no exceptions but no content" trap as the bump-chart check above —
    # confirm a real go.Sankey rendered, not just a clean run.
    at = AppTest.from_file(_PAGE, default_timeout=90)
    at.run()
    at.sidebar.radio[0].set_value("Rank Flow").run()
    assert not at.exception
    assert at.get("plotly_chart"), "Rank Flow rendered no plotly_chart element"


def test_rank_flow_opens_with_the_sentence_and_defaults_to_a_four_week_hop():
    # #640 rebuild (2026-09-19): the thing he reads is the sentence ABOVE the
    # chart — "N climbed · N fell · N held" — and the default hop is four weeks
    # with the picker still there. The band rename is the live-build tell
    # (CLAUDE.md): a pre-rebuild build reads "No rank" and has no sentence.
    at = AppTest.from_file(_PAGE, default_timeout=90)
    at.run()
    at.sidebar.radio[0].set_value("Rank Flow").run()
    assert not at.exception
    md = " ".join(m.value for m in at.markdown)
    assert "climbed" in md and "fell" in md and "held" in md, "the sentence above the chart did not render"
    assert "Biggest" in md, "the biggest move by name did not render"
    captions = " ".join(c.value for c in at.caption)
    assert "New / unranked" in captions and "No rank" not in captions, "band rename tell missing"
    hop = [s for s in at.sidebar.selectbox if s.key == "flow_hop"]
    assert hop and hop[0].value == 4, "the default hop is not four weeks"
    assert [s for s in at.sidebar.selectbox if s.key == "flow_week"], "the week picker is gone"



def test_weekly_movers_names_a_real_mover():
    # Same "no exceptions but no content" trap as the other views above —
    # confirm a real, named mover line rendered (the arrow format #561 asked
    # for: "Name prev -> curr"), not just the header and caption.
    at = AppTest.from_file(_PAGE, default_timeout=90)
    at.run()
    at.sidebar.radio[0].set_value("Weekly Movers").run()
    assert not at.exception
    assert len(at.subheader) > 0, "Weekly Movers rendered no 'Week of ...' header"
    md_text = " ".join(m.value for m in at.markdown)
    assert "→" in md_text, "Weekly Movers rendered no named mover/entrant line"


def test_forward_returns_shows_real_numbers():
    at = AppTest.from_file(_PAGE, default_timeout=90)
    at.run()
    at.sidebar.radio[0].set_value("Forward Returns").run()
    assert not at.exception
    # The "not a price return" disclosure must be up top, not buried.
    assert any("NOT a price return" in w.value for w in at.warning)
    assert len(at.metric) > 0, "Forward Returns rendered no summary metrics"
    assert len(at.dataframe) > 0, "Forward Returns rendered no event table"
