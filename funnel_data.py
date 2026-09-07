"""EP scan funnel data adapter — reads the committed mi_ep_scan_log /
mi_catalyst_tier_shadow / mi_ep_alerts / mi_live_trades / mi_ep_missed_outcomes
snapshot and turns it into the funnel-over-time + rejection-scorecard views.

Built for #589 (operator 2026-08-24): "/scanned answers one day well ... we can
probably build funnel views for more than a day but that likely belongs to the
dash." Two halves:
  HALF 1 — per-stage counts by day, drop-off rate at each stage, and the trend
           (which gate is cutting harder than it used to).
  HALF 2 — rejected names that subsequently RAN, broken down by the stage that
           rejected them (the cost of each gate, not just the count).

STAGE DEFINITIONS ARE PORTED, NOT REINVENTED. Every constant and resolution
function below (`_FUNNEL_STAGES`, `_CATEGORY_TO_STAGE`, `_categorize_skip_reason`,
`_resolve_tickers`, `_stage_for`, `_outcome_is_fresh`, `_outcome_text`) is a
line-for-line port of `agents/market_intelligence/scanned_report.py` +
`agents/market_intelligence/missed_outcomes.py` in the apollo_the_wise repo (read
2026-09-07, THE LINE repo — read-only, never modified). Cross-checked against the
REAL renderer output for 2026-09-03 (`render_scanned_day` run inside the
apollo-market container) — stage-for-stage, count-for-count identical. If
`/scanned`'s stage list ever changes, re-sync this file by diffing against the
source; a silent drift here would make the two surfaces contradict each other in
front of the operator, which is worse than not building this.

Data source: apollo_funnel_snapshot.json — a point-in-time export of the five
source tables (SELECT-only; ssh apollo@<host> + docker exec apollo-postgres psql,
see #589 commit). Mirrors the apollo_data.py / apollo_trades_paper.json and
theme_data.py / apollo_themes_snapshot.json snapshot pattern used elsewhere in
this repo — Streamlit Cloud can't reach the private DB, so the dashboard reads a
committed snapshot rather than opening a new connection path.

KNOWN DATA-COVERAGE CLIFFS (read before trusting any cross-period comparison):
  - mi_ep_scan_log's earliest row is 2026-04-13, not "back to March" as assumed
    going in — the funnel window is Apr 13 onward, five months, not six.
  - 2026-08-24 is a REGIME CHANGE, not a trend: the d1_universe_floor rows
    (filter:universe_prev_close_too_low / universe_prev_day_illiquid — the
    u_close/u_vol stages) do not exist before 2026-08-24, and daily ticker
    counts jump ~10x that day (mid-teens/day -> 150-300/day). This is a scan-log
    CAPTURE change (more of the pipeline started getting logged), not the
    universe suddenly admitting 10x more names. Every stage's share-of-total
    and the day's total-tickers count are NOT comparable across this boundary.
  - mi_catalyst_tier_shadow (the "graded" source; feeds `graded_cut` and part of
    `below_bar`) has data only from 2026-08-24 — `graded_cut` cannot be resolved
    at all before that date (there is no fallback path to it in `_stage_for`).
  - `below_floor` (#605 record-only capture band) starts 2026-08-29/31.
  - mi_ep_alerts starts 2026-05-11; before that, `alerted` resolves only via
    scan_row.score_tier (no independent alerts-table corroboration).
  - mi_ep_missed_outcomes.setup_at_open is well-backfilled (2 nulls out of 5024
    rows) but 73% of rows are False ("never gapped at the open" — see #595 in
    scanned_report.py) — those are correctly excluded from the "ran" ranking by
    `_outcome_text`, exactly as /scanned does.
Because of the above, `stage_moved_summary()` computes its "did this stage move"
verdict ONLY over the pre-2026-08-24 window (the one comparable-population
stretch with enough days), and reports 2026-08-24-onward separately, flagged as
too short and not yet comparable, rather than blending regimes into one trend.
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

import pandas as pd
import streamlit as st

_SNAPSHOT_PATH = os.path.join(os.path.dirname(__file__), "apollo_funnel_snapshot.json")

# The scan-log capture-coverage step (see module docstring). Everything from
# here on is a different (much larger) logged population than before it.
CAPTURE_REGIME_CHANGE = date(2026, 8, 24)

# ── ported verbatim from scanned_report.py (apollo_the_wise, read 2026-09-07) ─

DECLINED_NEVER_FILLED_STATUSES = ("skipped", "cancelled", "order_failed")
_NEVER_FILLED = frozenset(DECLINED_NEVER_FILLED_STATUSES) | {"expired"}

# (stage_key, label, always_print) — pipeline order, identical to /scanned.
FUNNEL_STAGES: list[tuple[str, str, bool]] = [
    ("u_close",      "prior close under the $5 universe floor",             True),
    ("u_vol",        "prior-day volume under the 50k-share universe floor", True),
    ("below_floor",  "gap under the admission floor (recorded only)",       False),
    ("adv_low",      "average daily dollar volume too thin",                True),
    ("mcap_low",     "market cap too small",                                True),
    ("atr_high",     "day-to-day swings too wild (ATR cap)",                True),
    ("extension",    "already ran too far before this gap",                 True),
    ("pm_rvol",      "pre-market volume below its usual pace",              True),
    ("session_rvol", "session volume below its usual pace",                 True),
    ("cooldown",     "alerted within the last 60 days (cooldown)",          False),
    ("ma_filter",    "merger/buyout news, not a momentum gap",              False),
    ("duplicate",    "already handled by an earlier scan today",            False),
    ("filter_other", "other filters",                                       False),
    ("top20",        "didn't make the top-20 grading shortlist",            True),
    ("routine",      "catalyst graded routine, not scoreable",              False),
    ("graded_cut",   "graded, then a mechanical filter cut it",             True),
    ("below_bar",    "graded and scored, but under the alert bar",          True),
    ("alerted",      "alerted, no entry attempted",                         True),
    ("blocked",      "alerted, then blocked or unfilled at entry",          True),
    ("traded",       "traded",                                              True),
]
STAGE_LABEL = {k: label for k, label, _ in FUNNEL_STAGES}
STAGE_ORDER = [k for k, _, _ in FUNNEL_STAGES]

_CATEGORY_TO_STAGE = {
    "below_gap_floor": "below_floor",
    "adv_low": "adv_low",
    "mcap_low": "mcap_low",
    "atr_high": "atr_high",
    "extension_gate": "extension",
    "pm_rvol_low": "pm_rvol",
    "session_rvol_low": "session_rvol",
    "cooldown": "cooldown",
    "ma_filter": "ma_filter",
    "duplicate_scan": "duplicate",
    "outside_top20": "top20",
    "score_below_50": "below_bar",
    "catalyst_downgrade": "routine",
}

# Stages a human could argue about — everything else is a bulk/mechanical cut.
GRADED_OR_BETTER = frozenset(
    {"routine", "graded_cut", "below_bar", "alerted", "blocked", "traded"})

# Same bar /scanned uses for "bulk cuts that ran anyway" — kept identical so a
# ticker counted as a "runner" here means the same thing it means in /scanned.
BULK_RUNNER_BAR = 0.20

# Stages whose resolution needs a source table with a coverage cliff (see module
# docstring) — excluded from the moved-stage ranking outside their comparable
# window, and flagged wherever they're shown.
_INSTRUMENTATION_LIMITED_STAGES = {"u_close", "u_vol", "below_floor", "graded_cut"}

# top20 is a CAP (fixed shortlist size), not a threshold gate — its drop-off
# rate tracks how many names reach the shortlist stage, not a rule change.
# Never headline it as "the gate loosened/tightened".
_CAP_STAGES = {"top20"}

# Verified 2026-09-07 against the clean (pre-2026-08-24) window's actual
# filter_reason text (see the #589 build notes) — hand-checked, not derived
# mechanically, so re-verify if the underlying wording changes again:
#   - filter_other's April-2026 rows are almost entirely "low rel volume Xx <
#     2.0x" / "low volume projected Xx < 2.0x" text that predates the
#     session_rvol category match (_categorize_skip_reason looks for the
#     substring "session_rvol"/"session rvol", which the April wording lacks).
#     From ~May 2026 the same gate's message became
#     "filter:session_rvol_too_low: ..." and IS matched. So filter_other's
#     13.5%->0% drop and session_rvol's 4.3%->9.9% rise are the SAME event
#     (a wording migration), not two independent moves. Pooled, the combined
#     rate moved 8.6% -> 5.6% (-3.0pp) across the same split — a real but much
#     smaller move than either individual row suggests.
#   - below_bar's filter_reason carries the literal bar it was compared
#     against ("score N < 50" everywhere in the clean window — the "< bar N"
#     separated-scale wording only appears after 2026-08-24). The bar itself
#     was CONSTANT at 50 the whole clean window, so below_bar's 34.3%->49.7%
#     rise is scored names increasingly landing under a fixed bar, not a
#     threshold change.
STAGE_CAVEATS = {
    "top20": ("cap, not a gate — tracks how many names reach the shortlist, "
              "not a rule change. In the clean window the daily scanned "
              "population roughly halved (46 days avg 42/day -> 47 days avg "
              "23/day); that is why fewer names got cut here, not a looser cap."),
    "filter_other": ("confounded with session_rvol — see funnel_data.py "
                      "STAGE_CAVEATS for the wording-migration explanation. "
                      "Pooled with session_rvol the real move is -3.0pp, not -13.5pp."),
    "session_rvol": ("confounded with filter_other — see funnel_data.py "
                      "STAGE_CAVEATS. Pooled the real move is -3.0pp, not +5.7pp."),
    "below_bar": ("the alert bar held constant at \"score < 50\" through the "
                  "whole clean window (no threshold change) — this rise is "
                  "real: scored names increasingly landing under that fixed bar."),
    "blocked": ("post-alert fill rate, not a gate — its denominator is "
                "alerted names, and the 2026-07-17 live-account cutover "
                "lands inside the second half of the clean window, so this "
                "can reflect execution/account changes rather than a filter."),
}

# Stages excluded from the HEADLINE "biggest mover" pick specifically —
# confounded (wording migration), a cap (tracks volume, not a rule), or
# post-alert (fill/execution outcome, not a rejection gate). All still show,
# with their note, in the full ranking table — this only affects which stage
# gets the single-line KPI summary. below_bar is deliberately NOT excluded:
# its move is real and unconfounded (see its STAGE_CAVEATS entry, which is
# context, not a discount).
_CONFOUNDED_STAGES = {"filter_other", "session_rvol"}
_POST_ALERT_STAGES = {"alerted", "blocked"}
HEADLINE_EXCLUDED_STAGES = _CONFOUNDED_STAGES | _CAP_STAGES | _POST_ALERT_STAGES


def _categorize_skip_reason(source: str, raw: Optional[str]) -> str:
    """Bucket the free-form reason into a stable category — ported verbatim
    from missed_outcomes.py."""
    if source == "moderate_alert":
        return "moderate_tier"
    s = (raw or "").lower()
    if s.startswith("block:max_positions"):
        return "cap_blocked"
    if s.startswith("block:circuit_breaker"):
        return "breaker_blocked"
    if s.startswith("block:"):
        return "block_other"
    if s.startswith("window:"):
        return "window_missed"
    if s.startswith("setup:stop_too_wide"):
        return "stop_too_wide"
    if s.startswith("setup:faded"):
        return "faded_from_orb"
    if s.startswith("setup:account_fetch") or s.startswith("infra:"):
        return "infra_skip"
    if s.startswith("setup:"):
        return "setup_other"
    if source == "high_unentered":
        return "high_unentered"
    if not raw:
        return "filter_other"
    if s.startswith("filter:universe_prev_close_too_low") or s.startswith("filter:universe_prev_day_illiquid"):
        return "d1_universe_floor"
    if s.startswith("filter:universe_below_gap_floor"):
        return "below_gap_floor"
    if "cooldown" in s:
        return "cooldown"
    if "m&a" in s or "buyout" in s or "merger" in s:
        return "ma_filter"
    if "already scored" in s or "duplicate" in s:
        return "duplicate_scan"
    if "outside top-20" in s or "top-20 gap cap" in s:
        return "outside_top20"
    if "score" in s and ("< 50" in s or "< bar" in s):
        return "score_below_50"
    if "pm_rvol" in s or "pre-market rvol" in s or "pre-mkt volume" in s:
        return "pm_rvol_low"
    if "session_rvol" in s or "session rvol" in s:
        return "session_rvol_low"
    if "adv" in s:
        return "adv_low"
    if "atr" in s:
        return "atr_high"
    if "mcap" in s or "market cap" in s:
        return "mcap_low"
    if "catalyst" in s and ("downgrade" in s or "routine" in s):
        return "catalyst_downgrade"
    if "extension" in s or "extended" in s:
        return "extension_gate"
    return "filter_other"


def _resolve_tickers(data: dict[str, list[dict[str, Any]]]) -> dict[str, dict]:
    """One day's five source lists -> one record per ticker with its resolved
    FINAL stage. Ported verbatim from scanned_report.py."""
    per: dict[str, dict] = {}

    def slot(t: str) -> dict:
        return per.setdefault(t, {"ticker": t})

    for r in data.get("scan") or []:
        slot(r["ticker"])["scan_row"] = r
    for r in data.get("graded") or []:
        slot(r["ticker"])["graded_row"] = r
    for r in data.get("alerts") or []:
        slot(r["ticker"])["alert_row"] = r
    for r in data.get("trades") or []:
        slot(r["ticker"]).setdefault("trade_rows", []).append(r)
    for r in data.get("outcomes") or []:
        slot(r["ticker"])["outcome_row"] = r

    for s in per.values():
        s["stage"] = _stage_for(s)
    return per


def _stage_for(s: dict) -> str:
    trade_rows = s.get("trade_rows") or []
    if any((tr.get("status") or "") not in _NEVER_FILLED for tr in trade_rows):
        return "traded"
    if trade_rows:
        return "blocked"
    sc = s.get("scan_row")
    if s.get("alert_row") or (sc and sc.get("score_tier")):
        return "alerted"
    reason = (sc or {}).get("filter_reason")
    cat = _categorize_skip_reason("scan_filter", reason) if reason else None
    if cat == "score_below_50":
        return "below_bar"
    g = s.get("graded_row")
    if g is not None:
        if g.get("live_ep_score") is not None:
            return "below_bar"
        return "graded_cut"
    if not reason:
        return "filter_other"
    if cat == "d1_universe_floor":
        return ("u_close" if reason.startswith("filter:universe_prev_close")
                else "u_vol")
    return _CATEGORY_TO_STAGE.get(cat, "filter_other")


def _outcome_is_fresh(o: Optional[dict], alert_d: date, now: datetime) -> bool:
    if not o:
        return False
    lr = o.get("last_refreshed_at")
    if lr is None:
        return False
    if isinstance(lr, str):
        lr = datetime.fromisoformat(lr)
    if lr.tzinfo is None:
        lr = lr.replace(tzinfo=timezone.utc)
    settled_by = datetime(alert_d.year, alert_d.month, alert_d.day,
                          tzinfo=timezone.utc) + timedelta(days=7)
    return lr >= now - timedelta(days=2) or lr >= settled_by


def _outcome_text(s: dict, alert_d: date, now: datetime) -> tuple[str, Optional[float]]:
    """-> (display text, rank key or None). Ported verbatim from
    scanned_report.py (the #595 setup_at_open gate included)."""
    o = s.get("outcome_row")
    if o is None:
        return "outcome pending", None
    if not _outcome_is_fresh(o, alert_d, now):
        return "outcome stale, not ranked", None
    mh, r5 = o.get("max_high_5d"), o.get("ret_5d")
    if mh is None and r5 is None:
        return "outcome pending", None
    parts = []
    if mh is not None:
        parts.append(f"ran {mh * 100:+.0f}% high")
    if r5 is not None:
        parts.append(f"settled {r5 * 100:+.0f}%")
    text = ", ".join(parts) + " in 5 sessions"
    if o.get("setup_at_open") is False:
        og = o.get("open_gap_pct")
        og_txt = f" (opened {og * 100:+.0f}%)" if og is not None else ""
        return text + f" — but no setup at the open{og_txt}, not ranked", None
    return text, mh


def _gap_of(s: dict) -> Optional[float]:
    for key, field in (("scan_row", "gap_pct"), ("alert_row", "gap_pct"),
                       ("graded_row", "gap_pct_last")):
        r = s.get(key)
        if r and r.get(field) is not None:
            return float(r[field])
    return None


# ── snapshot load ────────────────────────────────────────────────────────────

DATA_CACHE_TTL = 300  # 5 minutes — matches Portfolio.py's DATA_CACHE_TTL


@st.cache_data(ttl=DATA_CACHE_TTL)
def _load_raw() -> dict:
    with open(_SNAPSHOT_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    return raw


def snapshot_meta() -> dict:
    raw = _load_raw()
    return {"generated_at": raw.get("generated_at")}


def _parse_now() -> datetime:
    raw = _load_raw()
    gen = raw.get("generated_at")
    if gen:
        try:
            return datetime.fromisoformat(gen)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


@st.cache_data(ttl=DATA_CACHE_TTL)
def build_ticker_stage_table() -> pd.DataFrame:
    """One row per (date, ticker): the resolved stage plus what it did
    afterwards. This is the single computed table everything else derives from.
    """
    raw = _load_raw()
    now = _parse_now()

    by_date: dict[str, dict[str, list]] = {}

    def bucket(date_key: str, dd: str, kind: str, row: dict):
        by_date.setdefault(dd, {"scan": [], "graded": [], "alerts": [], "trades": [], "outcomes": []})
        by_date[dd][kind].append(row)

    for r in raw.get("scan") or []:
        bucket("scan_date", r["scan_date"], "scan", r)
    for r in raw.get("graded") or []:
        bucket("scan_date", r["scan_date"], "graded", r)
    for r in raw.get("alerts") or []:
        bucket("alert_date", r["alert_date"], "alerts", r)
    for r in raw.get("trades") or []:
        bucket("alert_date", r["alert_date"], "trades", r)
    for r in raw.get("outcomes") or []:
        bucket("alert_date", r["alert_date"], "outcomes", r)

    rows = []
    for dd in sorted(by_date):
        d = date.fromisoformat(dd)
        per = _resolve_tickers(by_date[dd])
        for s in per.values():
            otext, mh = _outcome_text(s, d, now)
            o = s.get("outcome_row") or {}
            rows.append({
                "date": d,
                "ticker": s["ticker"],
                "stage": s["stage"],
                "gap_pct": _gap_of(s),
                "outcome_text": otext,
                "max_high_5d": mh,  # only set when fresh + setup_at_open != False
                "ret_5d": o.get("ret_5d"),
                "has_outcome_row": bool(o),
                "fresh": _outcome_is_fresh(o if o else None, d, now),
                "setup_at_open": o.get("setup_at_open"),
                "ran_bulk": mh is not None and mh >= BULK_RUNNER_BAR,
                "filter_reason": (s.get("scan_row") or {}).get("filter_reason"),
                "is_graded_or_better": s["stage"] in GRADED_OR_BETTER,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df


# ── HALF 1: funnel over time ────────────────────────────────────────────────

MIN_STAGE_N = 10  # below this, a day's drop-off rate for a stage is not shown as a trend point


@st.cache_data(ttl=DATA_CACHE_TTL)
def funnel_by_day() -> pd.DataFrame:
    """One row per (date, stage): count, how many tickers were still 'alive'
    entering this stage (remaining_before, walking the pipeline in the order
    /scanned lists it), and the resulting drop-off rate. A day with zero scan
    rows simply has no rows here — never a fabricated zero."""
    st_df = build_ticker_stage_table()
    if st_df.empty:
        return pd.DataFrame(columns=["date", "stage", "label", "order", "count",
                                      "remaining_before", "drop_off_rate", "day_total"])

    counts = st_df.groupby(["date", "stage"]).size().rename("count").reset_index()
    day_totals = st_df.groupby("date").size().rename("day_total")

    out_rows = []
    for d, day_total in day_totals.items():
        day_counts = dict(zip(counts.loc[counts["date"] == d, "stage"],
                               counts.loc[counts["date"] == d, "count"]))
        remaining = int(day_total)
        for order, (key, label, _always) in enumerate(FUNNEL_STAGES):
            n = int(day_counts.get(key, 0))
            drop_off = (n / remaining) if remaining > 0 else float("nan")
            out_rows.append({
                "date": d, "stage": key, "label": label, "order": order,
                "count": n, "remaining_before": remaining,
                "drop_off_rate": drop_off, "day_total": int(day_total),
            })
            remaining -= n
    return pd.DataFrame(out_rows)


def _week_start(d: pd.Timestamp) -> pd.Timestamp:
    return d - pd.Timedelta(days=d.weekday())


@st.cache_data(ttl=DATA_CACHE_TTL)
def funnel_by_week() -> pd.DataFrame:
    """Weekly pooled rollup of funnel_by_day: count and remaining_before summed
    across the week's scan days, rate recomputed from the pooled sums (not a
    mean of daily rates) so a single noisy day can't dominate."""
    fd = funnel_by_day()
    if fd.empty:
        return fd
    fd = fd.copy()
    fd["week"] = fd["date"].apply(_week_start)
    grp = fd.groupby(["week", "stage", "label", "order"]).agg(
        count=("count", "sum"),
        remaining_before=("remaining_before", "sum"),
        n_days=("date", "nunique"),
    ).reset_index()
    grp["drop_off_rate"] = grp["count"] / grp["remaining_before"].replace(0, pd.NA)
    return grp.sort_values(["order", "week"])


@st.cache_data(ttl=DATA_CACHE_TTL)
def stage_moved_summary(min_n: int = MIN_STAGE_N) -> pd.DataFrame:
    """Rank every stage by how much its (pooled) drop-off rate moved between
    the first and second half of the CLEAN pre-capture-change window
    (2026-04-13 -> 2026-08-21) — the one stretch where the logged population is
    a stable, comparable definition. Stages that structurally cannot be
    resolved in that window (see _INSTRUMENTATION_LIMITED_STAGES) are reported
    with a flag instead of a delta, never silently blended in."""
    fd = funnel_by_day()
    if fd.empty:
        return pd.DataFrame()

    clean = fd[fd["date"] < pd.Timestamp(CAPTURE_REGIME_CHANGE)].copy()
    if clean.empty:
        return pd.DataFrame()
    dates = sorted(clean["date"].unique())
    mid = dates[len(dates) // 2]

    rows = []
    for key, label, _always in FUNNEL_STAGES:
        sub = clean[clean["stage"] == key]
        limited = key in _INSTRUMENTATION_LIMITED_STAGES
        first = sub[sub["date"] < mid]
        second = sub[sub["date"] >= mid]
        f_n, s_n = first["remaining_before"].sum(), second["remaining_before"].sum()
        f_rate = (first["count"].sum() / f_n) if f_n >= min_n else float("nan")
        s_rate = (second["count"].sum() / s_n) if s_n >= min_n else float("nan")
        delta = s_rate - f_rate if pd.notna(f_rate) and pd.notna(s_rate) else float("nan")
        rows.append({
            "stage": key, "label": label,
            "first_half_rate": f_rate, "first_half_n": int(f_n),
            "first_half_count": int(first["count"].sum()),
            "second_half_rate": s_rate, "second_half_n": int(s_n),
            "second_half_count": int(second["count"].sum()),
            "delta_pp": delta * 100 if pd.notna(delta) else float("nan"),
            "instrumentation_limited": limited,
            "insufficient_n": (f_n < min_n) or (s_n < min_n),
        })
    out = pd.DataFrame(rows)
    out["abs_delta_pp"] = out["delta_pp"].abs()
    return out.sort_values("abs_delta_pp", ascending=False, na_position="last")


@st.cache_data(ttl=DATA_CACHE_TTL)
def post_regime_change_snapshot() -> pd.DataFrame:
    """The 2026-08-24-onward window's stage counts, reported separately (too
    short + different population definition to call a trend) rather than
    blended into stage_moved_summary."""
    fd = funnel_by_day()
    if fd.empty:
        return fd
    recent = fd[fd["date"] >= pd.Timestamp(CAPTURE_REGIME_CHANGE)]
    agg = recent.groupby(["stage", "label", "order"]).agg(
        count=("count", "sum"),
        remaining_before=("remaining_before", "sum"),
        n_days=("date", "nunique"),
    ).reset_index()
    agg["drop_off_rate"] = agg["count"] / agg["remaining_before"].replace(0, pd.NA)
    return agg.sort_values("order")


# ── HALF 2: scorecard — rejected names that ran ─────────────────────────────

MIN_SCORECARD_N = 5


@st.cache_data(ttl=DATA_CACHE_TTL)
def scorecard() -> pd.DataFrame:
    """Per stage (every stage except 'traded' — a real position isn't a
    'miss'): how many names it rejected, how many of those have a settled,
    trustworthy outcome, and how many of THOSE ran >= BULK_RUNNER_BAR within 5
    sessions. Same bar and same freshness/setup-at-open gates /scanned uses, so
    'ran' means the same thing here as it does there."""
    st_df = build_ticker_stage_table()
    if st_df.empty:
        return pd.DataFrame()

    rows = []
    for key, label, _always in FUNNEL_STAGES:
        if key == "traded":
            continue
        sub = st_df[st_df["stage"] == key]
        n_rejected = len(sub)
        n_pending = int((sub["outcome_text"] == "outcome pending").sum())
        n_stale = int((sub["outcome_text"] == "outcome stale, not ranked").sum())
        n_no_setup = int(sub["outcome_text"].str.contains("no setup at the open", na=False).sum())
        measurable = sub[sub["max_high_5d"].notna()]
        n_measurable = len(measurable)
        n_ran = int((measurable["max_high_5d"] >= BULK_RUNNER_BAR).sum())
        rate = (n_ran / n_measurable) if n_measurable > 0 else float("nan")
        rows.append({
            "stage": key, "label": label,
            "is_gate": key not in ("alerted", "blocked"),
            "n_rejected": n_rejected,
            "n_measurable": n_measurable,
            "n_ran": n_ran,
            "ran_rate": rate,
            "n_pending": n_pending, "n_stale": n_stale, "n_no_setup_at_open": n_no_setup,
            "insufficient_n": n_measurable < MIN_SCORECARD_N,
        })
    out = pd.DataFrame(rows)
    return out.sort_values("n_ran", ascending=False)


@st.cache_data(ttl=DATA_CACHE_TTL)
def top_runners_by_stage(stage: str, limit: int = 8) -> pd.DataFrame:
    st_df = build_ticker_stage_table()
    if st_df.empty:
        return st_df
    sub = st_df[(st_df["stage"] == stage) & st_df["max_high_5d"].notna()].copy()
    sub = sub.sort_values("max_high_5d", ascending=False).head(limit)
    return sub[["date", "ticker", "gap_pct", "max_high_5d", "ret_5d",
                "outcome_text", "filter_reason"]]
