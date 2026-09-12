"""#648 (2026-09-12) — a cohort in its first week must not render like one held a month.

WHY THIS EXISTS, from #639's measurement over eleven post-launch weeks: the board's churn
is REAL ROTATION, not identity instability — NEW 107 (70%), REAPPEARED 44 (29%), RENAMED
**1**. So there is no matcher bug; what there is, is that about HALF of everything reaching
the board is gone a week later while a first-week cohort and a four-week one render
identically. Two remedies were measured and BOTH died: a basket-size floor cuts the
cohorts that persist better (43% of small ones gone after a week against 52% of larger),
and a two-week confirmation delay costs the cohorts that hold 4+ weeks exactly the five
sessions that lift remaining runway from 29% to 55%. Labelling spends nothing.

MUTATION-PROVEN, reported as the runs actually came back rather than as one-to-one (the
first draft of this docstring claimed a tidy pairing and two of its three claims were
wrong):
  - `_weeks_on_board` always returns 1 -> reddens THREE: not_called_first_week,
    stops_at_a_gap, below_the_board_cut_breaks_the_run;
  - counting THROUGH a break in the run (`continue` instead of `break`) -> reddens exactly
    one: stops_at_a_gap.

⚠ NOT COVERED, said plainly rather than implied: the RENDER itself. `_render_week` calls
`st.markdown`, and this module has no Streamlit stub, so a mutation that drops the tenure
from the rendered line would redden nothing here. What IS pinned is that both buckets
carry `weeks_held` — the renderer reads it with `.get(..., 1)` and has no special case, so
the failure mode that remains is a silent "everything is 1st week", not a crash. Worth a
render test the next time this file gets a Streamlit fake.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from theme_movers import compute_weekly_movers, _tenure


def _grid(rows):
    return pd.DataFrame(rows, columns=["canonical_id", "canonical_name", "week_start", "week_rank"])


WEEKS = [date(2026, 8, 3) + timedelta(days=7 * i) for i in range(5)]


def test_a_long_held_cohort_is_not_called_first_week():
    """THE POINT. A cohort on the board all five weeks, climbing in the last one,
    must read as established — not as a fresh arrival."""
    rows = [("c1", "Defense Primes", w, r) for w, r in zip(WEEKS, [9, 9, 9, 9, 4])]
    out = compute_weekly_movers(_grid(rows), WEEKS)
    latest = out[0]
    assert latest["gainers"], "a cohort that climbed 9 -> 4 must be reported"
    g = latest["gainers"][0]
    assert g["weeks_held"] == 5
    assert _tenure(g["weeks_held"]) == "5th week"
    assert _tenure(g["weeks_held"]) != "1st week"


def test_a_true_entrant_reads_as_first_week():
    rows = [("c2", "Neoclouds", WEEKS[-1], 14)]
    out = compute_weekly_movers(_grid(rows), WEEKS)
    e = out[0]["entrants"][0]
    assert e["weeks_held"] == 1 and _tenure(1) == "1st week"


def test_tenure_stops_at_a_gap():
    """A cohort that FELL OFF the board and came back is not credited for the
    weeks before the gap — it is provisional again, which is the whole point."""
    ranks = [5, None, 5, 5, 3]          # on, off, on, on, climbing
    rows = [("c3", "Uranium", w, r) for w, r in zip(WEEKS, ranks) if r is not None]
    out = compute_weekly_movers(_grid(rows), WEEKS)
    g = out[0]["gainers"][0]
    assert g["weeks_held"] == 3, "the run must restart after the week it was absent"


def test_a_cohort_below_the_board_cut_breaks_the_run():
    """Present in the grid but ranked past the board is NOT on the board."""
    rows = [("c4", "Shipping", w, r) for w, r in zip(WEEKS, [40, 40, 8, 8, 2])]
    out = compute_weekly_movers(_grid(rows), WEEKS)
    g = out[0]["gainers"][0]
    assert g["weeks_held"] == 3


def test_every_rendered_line_carries_its_tenure():
    """Both buckets must carry the field — the renderer has no special case, so a
    missing key would silently render every cohort as first-week."""
    rows = [("c5", "Held", w, r) for w, r in zip(WEEKS, [9, 9, 9, 9, 4])]
    rows += [("c6", "Fresh", WEEKS[-1], 11)]
    out = compute_weekly_movers(_grid(rows), WEEKS)[0]
    assert all("weeks_held" in m for m in out["gainers"] + out["entrants"])


def test_tenure_wording_reads_as_english():
    assert (_tenure(1), _tenure(2), _tenure(3), _tenure(4)) == (
        "1st week", "2nd week", "3rd week", "4th week")
