"""Apollo Themes data adapter — reads the committed mi_themes snapshot.

Interface-PARITY with rs-theme-dash/data.py: exposes the SAME public functions
(`get_weekly_grid`, `get_active_themes`, `get_theme_detail`,
`get_top_members_by_rs`, `get_correlated_themes`, `dedup_themes`) so the ported
grid/detail views call them unchanged. The only difference is the source: a
point-in-time JSON snapshot (`apollo_themes_snapshot.json`, exported read-only
from Hetzner `mi_themes` + `mi_stock_scores`) instead of live Postgres —
Streamlit Cloud can't reach the private DB. Mirrors the
`apollo_data.py` / `apollo_trades_paper.json` pattern used by Apollo Trades.

The SQL in rs-theme-dash/data.py is replicated here in pandas:
  - weekly bucketing  = date_trunc('week', ...)  → ISO-Monday via to_period('W')
  - DISTINCT ON (name, week)                     → drop_duplicates keep last by date
  - RANK() OVER (PARTITION BY week ORDER BY rs_avg DESC NULLS LAST)
                                                 → groupby.rank(method='min', desc)

Refresh: regenerate via scripts/export_theme_snapshot.sql on Apollo until the
daily auto-export job lands (#193). Themes recompute once/day at the 5PM data
pull, so a daily snapshot == the live dashboard's freshness.
"""
from __future__ import annotations

import json
import os
from datetime import date, timedelta

import pandas as pd
import streamlit as st

_SNAPSHOT_PATH = os.path.join(os.path.dirname(__file__), "apollo_themes_snapshot.json")


class TunnelDownError(RuntimeError):
    """Kept for import-parity with rs-theme-dash/data.py so the ported views
    import cleanly. Never raised in snapshot mode (there is no live connection)."""


# ── Snapshot load + normalize ────────────────────────────────────────────────


@st.cache_data(ttl=300)
def _load() -> dict:
    """Load + normalize the snapshot once. Returns
    {themes: DataFrame, scores: DataFrame, score_date, generated_at}."""
    with open(_SNAPSHOT_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    themes = pd.DataFrame(raw.get("themes", []))
    if not themes.empty:
        themes["theme_date"] = pd.to_datetime(themes["theme_date"]).dt.date
        # `tickers` arrives as a JSON array; normalize null -> [] so set ops are safe.
        themes["tickers"] = themes["tickers"].apply(lambda t: list(t) if t else [])
        # pct_above_20sma is stored 0-1 in mi_themes (the theme engine formats it
        # with ":.0%"). The ported views render it as "{:.0f}%" + chart range
        # [0,100], so normalize to 0-100 here once. This also fixes a latent
        # display bug in the original rs-theme-dash, which showed "1%" for a 0.66
        # breadth.
        if "pct_above_20sma" in themes.columns:
            themes["pct_above_20sma"] = (
                pd.to_numeric(themes["pct_above_20sma"], errors="coerce") * 100
            )
        # e_code (#194, ADR 0032) — the ecosystem this theme maps to. Normalize
        # missing/null/blank -> E-UNASSIGNED so every downstream consumer can
        # trust the column exists and is never null; older snapshots exported
        # before #194 simply won't have the field at all, which degrades
        # cleanly to "everything unassigned" (the flat view is unaffected).
        from ecosystem_score import E_UNASSIGNED
        if "e_code" not in themes.columns:
            themes["e_code"] = E_UNASSIGNED
        else:
            themes["e_code"] = themes["e_code"].fillna(E_UNASSIGNED)
            _blank = themes["e_code"].astype(str).str.strip() == ""
            themes.loc[_blank, "e_code"] = E_UNASSIGNED

    scores = pd.DataFrame(raw.get("stock_scores", []))

    return {
        "themes": themes,
        "scores": scores,
        "score_date": raw.get("score_date"),
        "generated_at": raw.get("generated_at"),
    }


def snapshot_meta() -> dict:
    """generated_at / score_date — for the freshness caption on the page."""
    d = _load()
    return {"generated_at": d["generated_at"], "score_date": d["score_date"]}


def _week_start(theme_dates: pd.Series) -> pd.Series:
    """Monday of the ISO week — matches Postgres date_trunc('week', ...)."""
    return pd.to_datetime(theme_dates).dt.to_period("W").dt.start_time.dt.date


def _weekly_universe(window: pd.DataFrame) -> pd.DataFrame:
    """One row per (theme, week) = the last snapshot in each ISO week, with
    week_rank computed over the FULL universe that week (rs_avg DESC, nulls
    unranked). Shared by the grid and the detail arc — both need the same
    per-week ranking universe.
    """
    w = window.copy()
    w["week_start"] = _week_start(w["theme_date"])
    # DISTINCT ON (name, week) ORDER BY theme_date DESC -> keep latest date per (name, week)
    w = w.sort_values(["name", "week_start", "theme_date"]).drop_duplicates(
        ["name", "week_start"], keep="last"
    )
    w["week_rank"] = w.groupby("week_start")["rs_avg"].rank(
        method="min", ascending=False, na_option="keep"
    )
    w["week_universe"] = w.groupby("week_start")["rs_avg"].transform(
        lambda s: int(s.notna().sum())
    )
    return w


# ── Public API (parity with rs-theme-dash/data.py) ───────────────────────────


@st.cache_data(ttl=300)
def get_weekly_grid(weeks: int = 12) -> pd.DataFrame:
    """One row per (theme, week) with rank by rs_avg — mirrors get_weekly_grid."""
    d = _load()
    df = d["themes"]
    if df.empty:
        return pd.DataFrame()
    cutoff = date.today() - timedelta(weeks=weeks)
    df = df[df["theme_date"] >= cutoff]
    if df.empty:
        return pd.DataFrame()
    grid = _weekly_universe(df).rename(columns={"theme_date": "as_of_date"})
    return grid.sort_values(
        ["week_start", "week_rank"], na_position="last"
    ).reset_index(drop=True)


@st.cache_data(ttl=300)
def get_active_themes(stale_after_days: int = 7) -> list[str]:
    """Theme names appearing within the recency window and not Retired."""
    d = _load()
    df = d["themes"]
    if df.empty:
        return []
    cutoff = date.today() - timedelta(days=stale_after_days)
    mask = (df["theme_date"] >= cutoff) & (df["stage"] != "Retired")
    return sorted(df.loc[mask, "name"].unique().tolist())


@st.cache_data(ttl=300)
def get_theme_detail(name: str, weeks: int = 12) -> dict:
    """Per-week arc + current snapshot (members, churn, breadth, thesis).

    Mirrors get_theme_detail. members_as_of = the snapshot's RS score_date (the
    snapshot carries a single latest score_date — see export_theme_snapshot.sql).
    """
    d = _load()
    df = d["themes"]
    if df.empty:
        return {}
    cutoff = date.today() - timedelta(weeks=weeks)
    window = df[df["theme_date"] >= cutoff]
    if window.empty:
        return {}

    uni = _weekly_universe(window)
    arc = uni[uni["name"] == name].copy()
    if arc.empty:
        return {}
    arc = arc.rename(columns={"theme_date": "as_of_date"}).sort_values("week_start")
    arc = arc[[
        "week_start", "as_of_date", "stage", "rs_avg", "pct_above_20sma",
        "tickers", "days_active", "score", "description", "week_rank",
    ]].reset_index(drop=True)

    latest = arc.iloc[-1]
    current_tickers: list[str] = list(latest["tickers"] or [])

    scores = d["scores"]
    members = pd.DataFrame()
    if not scores.empty and current_tickers:
        members = scores[scores["ticker"].isin(current_tickers)].copy()
        members = members.sort_values(
            "rs_composite", ascending=False, na_position="last"
        )[["ticker", "rs_composite", "rs_rank", "sector", "close", "sma_50"]]

    churn = _compute_churn(arc, weeks_back=4)

    return {
        "arc": arc,
        "current_tickers": current_tickers,
        "members": members,
        "members_as_of": d["score_date"],
        "thesis": latest.get("description") or "",
        "stage": latest["stage"],
        "current_rank": int(latest["week_rank"]) if pd.notna(latest["week_rank"]) else None,
        "rs_avg": latest["rs_avg"],
        "pct_above_20sma": latest["pct_above_20sma"],
        "joined": churn["joined"],
        "exited": churn["exited"],
    }


@st.cache_data(ttl=300)
def get_top_members_by_rs(theme_tickers: dict[str, tuple[str, ...]], n: int = 4) -> dict[str, str]:
    """Compact "TICKER · TICKER · ... +K" preview ordered by current RS.

    Verbatim from rs-theme-dash/data.py but RS comes from the snapshot's scores
    frame. Callers must pass tuples (hashable) so the cache key is stable.
    """
    if not theme_tickers:
        return {}
    d = _load()
    scores = d["scores"]
    rs_by_ticker: dict[str, float] = {}
    if not scores.empty:
        rs_by_ticker = dict(zip(scores["ticker"], scores["rs_composite"]))

    out: dict[str, str] = {}
    for theme, tickers in theme_tickers.items():
        if not tickers:
            out[theme] = ""
            continue
        ranked = sorted(
            tickers,
            key=lambda t: (-(rs_by_ticker[t] if rs_by_ticker.get(t) is not None else -1), t),
        )
        head = ranked[:n]
        overflow = len(tickers) - len(head)
        text = " · ".join(head)
        if overflow > 0:
            text += f"  +{overflow}"
        out[theme] = text
    return out


@st.cache_data(ttl=300)
def get_ecosystem_board(stale_after_days: int = 7, movement_weeks: int = 6) -> dict:
    """ADR 0032 two-level board data: ecosystems ranked by the D3 boosted
    score, each with its member sub-themes carrying a global theme rank —
    the SAME grouping/scoring path as the Telegram `/themes` v2 board
    (`ecosystem_score._group_and_rank_ecosystems`, ported verbatim from
    Apollo's theme_ecosystems.py; see that module's docstring).

    Mirrors agent.py's `_handle_theme_query` (briefing._compute_scored_themes
    + theme_ecosystems.format_ecosystem_board): each theme's "comp" is
    recomputed as trimmed_mean(rs_composite) over its CURRENT members from
    the snapshot's single stock_scores cross-section — never the stored
    `score` column (mixed-scale landmine, see CLAUDE.md). Day-over-day
    `delta` compares against the most recent PRIOR theme_date's stored
    `rs_avg` — one global prior date, mirroring Apollo's
    `get_prior_theme_scores` (not a per-theme lookback).

    Each active theme also carries a "movement" signal (2026-07-19 operator
    follow-up — the Ecosystems view otherwise loses the Grid's W/W
    visibility): the last `movement_weeks` ISO weeks of that theme's stored
    weekly rs_avg, REUSING get_weekly_grid (the exact series theme_grid.py's
    full heatmap already computes — not recomputed/reinvented here) via
    `ecosystem_score.compute_theme_movement`. One extra get_weekly_grid call
    total, joined by name — themes with <2 usable weekly points (too new)
    get "movement": {"delta": None, "n": <2, ...}, which the Streamlit view
    degrades to "—" rather than crashing.

    Returns {} when there's no theme with at least one scored member in the
    window (degenerate/empty-snapshot case — caller falls back to the flat
    view, same contract as get_active_themes returning []).

    Return shape:
      {
        "ordered_codes": [e_code, ...],           # ecosystem display order
        "active_by_eco": {e_code: [scored_theme_dict, ...]},
        "fading_by_eco": {e_code: [{"name", "tickers"}, ...]},
        "scores": {e_code: {"raw","strong","depth","boost","boosted",
                             "member_union","n_scored","n_active_themes"}},
        "global_rank": {theme_name: int},         # rank across ALL themes by comp
        "eco_display": {e_code: {"e_code","name"}},  # taxonomy display names
        "latest_date": date,
      }
    scored_theme_dict = {"name","stage","comp","delta","tickers","n_scored",
    "movement"} (delta is None when no prior-day rs_avg exists for that
    theme; movement is the dict described above).
    """
    from ecosystem_score import (
        E_UNASSIGNED, _group_and_rank_ecosystems, compute_theme_movement,
        get_ecosystem_map, trimmed_mean,
    )

    d = _load()
    df = d["themes"]
    if df.empty:
        return {}

    cutoff = date.today() - timedelta(days=stale_after_days)
    window = df[(df["theme_date"] >= cutoff) & (df["stage"] != "Retired")]
    if window.empty:
        return {}

    latest = window.sort_values("theme_date").groupby("name").tail(1)

    scores_df = d["scores"]
    rs_by_ticker: dict[str, dict] = {}
    if not scores_df.empty:
        rs_by_ticker = {
            row["ticker"]: {"rs_composite": row["rs_composite"]}
            for _, row in scores_df.iterrows()
        }

    # One global prior date — the most recent theme_date strictly before the
    # latest date seen in this window (mirrors get_prior_theme_scores: a
    # single snapshot-day-back, not a per-theme lookback).
    latest_date = latest["theme_date"].max()
    prior_dates = df.loc[df["theme_date"] < latest_date, "theme_date"]
    prior_rs_avg: dict[str, float] = {}
    if not prior_dates.empty:
        prior_date = prior_dates.max()
        prior_rows = df[df["theme_date"] == prior_date]
        prior_rs_avg = dict(zip(prior_rows["name"], prior_rows["rs_avg"]))

    # Movement signal — REUSE get_weekly_grid (the same weekly series
    # theme_grid.py's heatmap renders), not a second recomputation. A small
    # buffer beyond movement_weeks absorbs partial/boundary ISO weeks;
    # compute_theme_movement caps to the last `movement_weeks` usable points.
    weekly = get_weekly_grid(weeks=movement_weeks + 2)
    weekly_points_by_name: dict[str, list[tuple]] = {}
    if not weekly.empty:
        for wname, grp in weekly.groupby("name"):
            weekly_points_by_name[wname] = list(zip(grp["week_start"], grp["rs_avg"]))

    eco_map: dict[str, str] = {}
    scored_themes: list[dict] = []
    fading: list[dict] = []
    for _, row in latest.iterrows():
        name = row["name"]
        tickers = list(row["tickers"] or [])
        eco_map[name] = row["e_code"]

        if row["stage"] == "Fading":
            fading.append({"name": name, "tickers": tickers})
            continue

        comps = [
            rs_by_ticker[tk]["rs_composite"] for tk in tickers
            if tk in rs_by_ticker and rs_by_ticker[tk]["rs_composite"] is not None
            and pd.notna(rs_by_ticker[tk]["rs_composite"])
        ]
        if not comps:
            continue
        comp = trimmed_mean(comps)
        prior = prior_rs_avg.get(name)
        delta = (comp - prior) if (prior is not None and pd.notna(prior)) else None
        movement = compute_theme_movement(
            weekly_points_by_name.get(name, []), max_weeks=movement_weeks)
        scored_themes.append({
            "name": name, "stage": row["stage"], "comp": comp, "delta": delta,
            "tickers": tickers, "n_scored": len(comps), "movement": movement,
        })

    if not scored_themes and not fading:
        return {}

    scored_themes.sort(key=lambda x: -x["comp"])
    global_rank = {st["name"]: i for i, st in enumerate(scored_themes, 1)}

    ordered, active_by_eco, fading_by_eco, eco_scores = _group_and_rank_ecosystems(
        scored_themes, fading, rs_by_ticker, eco_map)

    return {
        "ordered_codes": ordered,
        "active_by_eco": active_by_eco,
        "fading_by_eco": fading_by_eco,
        "scores": eco_scores,
        "global_rank": global_rank,
        "eco_display": get_ecosystem_map(),
        "latest_date": latest_date,
    }


@st.cache_data(ttl=300)
def get_correlated_themes(name: str, min_overlap: float = 0.30) -> pd.DataFrame:
    """Themes with >= min_overlap member-share with `name` (latest snapshot).

    Returns [name, stage, rs_avg, other_size, shared_count, shared_tickers,
    overlap_pct]. Mirrors get_correlated_themes (14-day latest-per-theme).
    """
    d = _load()
    df = d["themes"]
    if df.empty:
        return pd.DataFrame()
    cutoff = date.today() - timedelta(days=14)
    recent = df[df["theme_date"] >= cutoff]
    if recent.empty:
        return pd.DataFrame()
    latest = recent.sort_values(["name", "theme_date"]).drop_duplicates("name", keep="last")

    tgt = latest[latest["name"] == name]
    if tgt.empty:
        return pd.DataFrame()
    t_set = set(tgt.iloc[0]["tickers"] or [])
    if not t_set:
        return pd.DataFrame()

    rows = []
    for _, r in latest.iterrows():
        if r["name"] == name:
            continue
        other = set(r["tickers"] or [])
        if not other:
            continue
        shared = t_set & other
        rows.append({
            "name": r["name"],
            "stage": r["stage"],
            "rs_avg": r["rs_avg"],
            "other_size": len(other),
            "shared_count": len(shared),
            "shared_tickers": sorted(shared),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["overlap_pct"] = out["shared_count"] / out["other_size"]
    out = out[out["overlap_pct"] >= min_overlap]
    return out.sort_values("overlap_pct", ascending=False).reset_index(drop=True)


def dedup_themes(
    theme_tickers: dict[str, tuple[str, ...]],
    threshold: float = 0.50,
    min_shared: int = 3,
    jaccard_floor: float | None = None,
) -> dict[str, str]:
    """Subset-aware dedup. Returns parent_of: alias_name -> representative.

    Verbatim from rs-theme-dash/data.py (pure function, no DB). Sort by size DESC;
    a smaller theme S merges into a larger un-merged L when
    |S ∩ L| / |S| >= threshold AND |S ∩ L| >= min_shared (tie-break by Jaccard).

    `jaccard_floor` (#553 fix, theme_canon.py Suspect #1): default None preserves
    the ORIGINAL containment-only behavior exactly — theme_grid.py's Grid view
    never passes it, so Grid's rendered output is byte-identical before/after
    this change (proven by test_grid_output_unchanged in test_theme_canon.py).
    When a caller passes a float, a symmetric Jaccard floor becomes a GATE
    (not just a tie-break): a small theme wholly CONTAINED in a much larger,
    unrelated one scores containment 1.0 regardless of true relatedness (real
    example: "Oil & Gas" contains 3 distinct sub-themes at containment 1.0
    each, Jaccard ~0.5) — same-day dedup needs the option to also require real
    mutual overlap, not just one-sided containment. theme_canon.py's cross-day
    canonicalizer passes its own overlap_threshold here so a small theme fully
    swallowed by an unrelated large theme on the SAME day can't seed a bad
    cross-day identity chain before Tier 2 ever runs.
    """
    if not theme_tickers:
        return {}

    by_size = sorted(theme_tickers.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    parent_of: dict[str, str] = {name: name for name, _ in by_size}
    sets: dict[str, set[str]] = {name: set(tickers) for name, tickers in by_size}

    for i, (s_name, _s_tickers) in enumerate(by_size):
        s_set = sets[s_name]
        if not s_set:
            continue
        candidates: list[tuple[str, float]] = []
        for j in range(i):
            l_name = by_size[j][0]
            if parent_of[l_name] != l_name:  # only roots can absorb
                continue
            l_set = sets[l_name]
            shared = len(s_set & l_set)
            if shared < min_shared:
                continue
            if shared / len(s_set) >= threshold:
                jaccard = shared / len(s_set | l_set)
                if jaccard_floor is not None and jaccard < jaccard_floor:
                    continue
                candidates.append((l_name, jaccard))
        if candidates:
            parent_of[s_name] = max(candidates, key=lambda kv: kv[1])[0]
    return parent_of


def _compute_churn(arc: pd.DataFrame, weeks_back: int = 4) -> dict:
    """Tickers that joined / exited over the trailing window. Verbatim."""
    if len(arc) < 2:
        return {"joined": [], "exited": []}
    window = arc.tail(weeks_back + 1)
    baseline = set(window.iloc[0]["tickers"] or [])
    latest = set(window.iloc[-1]["tickers"] or [])
    return {
        "joined": sorted(latest - baseline),
        "exited": sorted(baseline - latest),
    }


@st.cache_data(ttl=300)
def get_themes_for_ticker(ticker: str, stale_after_days: int = 7) -> pd.DataFrame:
    """#398 two-way lookup, ticker→themes half: active themes (recency window,
    not Retired) whose LATEST snapshot row contains the ticker. Membership is
    judged on each theme's latest row only — a ticker that exited last week
    doesn't count (mirrors the Telegram `/themes TICKER` lane semantics).
    Columns: name, stage, theme_date, rs_avg, members."""
    d = _load()
    df = d["themes"]
    if df.empty:
        return pd.DataFrame()
    cutoff = date.today() - timedelta(days=stale_after_days)
    live = df[(df["theme_date"] >= cutoff) & (df["stage"] != "Retired")]
    if live.empty:
        return pd.DataFrame()
    latest = live.sort_values("theme_date").groupby("name").tail(1)
    tk = ticker.strip().upper()
    hits = latest[latest["tickers"].apply(lambda ts: tk in (ts or []))].copy()
    if hits.empty:
        return pd.DataFrame()
    hits["members"] = hits["tickers"].apply(len)
    return hits.sort_values("rs_avg", ascending=False, na_position="last")[
        ["name", "stage", "theme_date", "rs_avg", "members"]
    ].reset_index(drop=True)


@st.cache_data(ttl=300)
def get_theme_members(name: str, stale_after_days: int = 7) -> list[str]:
    """#398 name→stocks half: the LATEST active row's ticker list for a theme
    (empty list when the theme is outside the recency window / Retired)."""
    d = _load()
    df = d["themes"]
    if df.empty:
        return []
    cutoff = date.today() - timedelta(days=stale_after_days)
    live = df[(df["theme_date"] >= cutoff) & (df["stage"] != "Retired")
              & (df["name"] == name)]
    if live.empty:
        return []
    return list(live.sort_values("theme_date").iloc[-1]["tickers"] or [])


# ── Canonical identity (#315 R3/R2/R4) ───────────────────────────────────────
# The theme engine re-mints each theme's NAME nightly (an LLM description), so
# the SAME underlying ticker cohort appears under a different string day to
# day. Everything above this line is UNCHANGED and stays keyed on raw `name`
# (Grid, Ecosystems, Detail — these intentionally match the Telegram board's
# grouping; #315 does not touch them). The functions below attach a stable
# CANONICAL identity (theme_canon.py — ticker-set-driven, see that module's
# docstring for the algorithm and the real counter-examples that set its
# thresholds) and are consumed only by the new bump-chart / forward-study
# views.


@st.cache_data(ttl=300)
def get_canonical_themes() -> pd.DataFrame:
    """Full theme history with `canonical_id` / `canonical_name` attached.

    Always canonicalizes the FULL history, not a windowed slice — identity
    threading needs the whole timeline (windowing first would sever a cohort
    right at the window edge). Callers window the RESULT
    (get_canonical_weekly_grid does this)."""
    d = _load()
    df = d["themes"]
    if df.empty:
        return df
    from theme_canon import canonicalize_themes   # lazy: mirrors ecosystem_score's pattern above
    return canonicalize_themes(df)


def _canonical_weekly_universe(window: pd.DataFrame) -> pd.DataFrame:
    """Same shape as `_weekly_universe` above, but DISTINCT ON
    (canonical_id, week) instead of (name, week), and week_rank recomputed
    over the CANONICAL population. A canonical rank will NOT always match
    the raw Grid's rank for the same calendar week — merging same-cohort
    name-variants shrinks that week's row count by design (mirrors
    theme_grid.py's own "recompute over survivors when dedup is on"
    precedent). The Grid/Ecosystems views are unaffected; this is a separate
    read."""
    w = window.copy()
    w["week_start"] = _week_start(w["theme_date"])
    w = w.sort_values(["canonical_id", "week_start", "theme_date"]).drop_duplicates(
        ["canonical_id", "week_start"], keep="last"
    )
    w["week_rank"] = w.groupby("week_start")["rs_avg"].rank(
        method="min", ascending=False, na_option="keep"
    )
    w["week_universe"] = w.groupby("week_start")["rs_avg"].transform(
        lambda s: int(s.notna().sum())
    )
    return w


@st.cache_data(ttl=300)
def get_canonical_weekly_grid(weeks: int = 24) -> pd.DataFrame:
    """Weekly rank grid keyed by CANONICAL identity — the R2 bump chart's
    data source (and R4's forward-study source, via a wide `weeks`).

    `weeks` windows the OUTPUT only (see get_canonical_themes). Anchored on
    the snapshot's own latest `theme_date`, NOT wall-clock `date.today()` —
    this snapshot is a nightly export and is routinely a day (or a weekend)
    behind; anchoring on today() would silently return an empty window on
    export lag rather than the most recent data actually on file."""
    canon = get_canonical_themes()
    if canon.empty:
        return pd.DataFrame()
    latest_date = canon["theme_date"].max()
    cutoff = latest_date - timedelta(weeks=weeks)
    window = canon[canon["theme_date"] >= cutoff]
    if window.empty:
        return pd.DataFrame()
    grid = _canonical_weekly_universe(window).rename(columns={"theme_date": "as_of_date"})
    return grid.sort_values(
        ["week_start", "week_rank"], na_position="last"
    ).reset_index(drop=True)


@st.cache_data(ttl=300)
def get_forward_rs_movement(
    surface_rank: int = 10,
    baseline_lo: int = 11,
    baseline_hi: int = 30,
    forward_weeks: tuple[int, ...] = (1, 4),
) -> dict:
    """#315 R4 — "what happened after a theme surfaced", in RS-score terms.

    NOT a price return, and this function's caller MUST say so wherever it
    renders: `stock_scores` is a single cross-section for one `score_date`
    (no historical per-ticker close series — see theme_detail.py's V5
    placeholder, which real forward $/％ returns still require via a future
    `mi_signal_outcomes` join). `rs_avg` — each theme's own trimmed-mean
    relative-strength composite, tracked weekly — is the closest honest
    proxy available in this snapshot. This reports the CHANGE in that score
    (and in rank) after a cohort first enters a rank band; it is a
    directional read on the RS engine's own output, not a substitute for a
    realized return.

    Selection-effect guard: `rs_avg` is a 0-100 percentile composite, so
    conditioning on "just entered the top N" selects on an already-high
    score, which mean-reverts by construction — a bare median forward delta
    for that group alone would misread as "the board picks badly" when it is
    partly a statistical artifact of the selection itself. Every event is
    therefore computed for BOTH the "surfaced" band (rank <= surface_rank)
    and a "baseline" band (baseline_lo <= rank <= baseline_hi), so callers
    compare the two instead of reading either bare.

    **The two bands are a genuine control group, not the same cohorts
    measured twice.** A cohort that EVER reaches the surfaced band (at any
    point in its recorded history, before or after) is excluded from the
    baseline band entirely — baseline is "reached rank 11-30 and, as far as
    this snapshot has seen, never made the top N", surfaced is "made the top
    N". Measured on the real snapshot: without this exclusion, 40% of
    baseline-band cohorts were also surfaced-band cohorts (same theme,
    counted in both groups from two different weeks of its own life) —
    that's a within-cohort comparison wearing a between-group label, and the
    fix is this exclusion, not different math.

    Per canonical cohort: "surfaced" = the FIRST week (chronologically) its
    week_rank fell into a given band. The forward point at horizon N is that
    cohort's Nth SUBSEQUENT recorded weekly point in ITS OWN sequence (not a
    calendar lookup) — a cohort with fewer than N subsequent points has no
    observation at that horizon yet and is excluded. Never interpolated or
    fabricated (#315 instruction). Note some "surfaced" events are a brand
    new cohort's FIRST-ever recorded week already landing in the top N (no
    "climbing" to observe, just a strong debut) rather than a promotion from
    lower down — `debut_share` in the return reports how common that is so
    callers can disclose it rather than imply every event is a climb.

    Returns {"events": DataFrame, "summary": DataFrame, "weeks_available": int,
      "debut_share": float | None}.
      events: canonical_id, canonical_name, band, surfaced_week,
        rank_at_surfaced, rs_avg_at_surfaced, is_debut (bool — surfaced_week
        is the cohort's first recorded week at all), and per N in
        forward_weeks: rank_fwd_{N}w, rs_avg_fwd_{N}w, delta_rank_{N}w
        (negative = rank improved — a SMALLER rank number), delta_rs_{N}w.
      summary: band, horizon_weeks, n, median_delta_rank, median_delta_rs,
        pct_rank_improved (share of `n` with delta_rank < 0).
      weeks_available: distinct ISO weeks in the canonical grid — the honest
      depth ceiling; callers must show this, not silently truncate.
      debut_share: of surfaced_top events, the share where is_debut is True
      (None if there are no surfaced_top events).
    """
    # weeks=520 (10 years) is a "give me everything on file" sentinel — the
    # real ceiling is however much history the snapshot actually holds.
    grid = get_canonical_weekly_grid(weeks=520)
    empty = {
        "events": pd.DataFrame(), "summary": pd.DataFrame(),
        "weeks_available": 0, "debut_share": None,
    }
    if grid.empty:
        return empty

    weeks_available = int(grid["week_start"].nunique())
    names_of = grid.drop_duplicates("canonical_id").set_index("canonical_id")["canonical_name"]

    def _first_entry(seq: pd.DataFrame, lo: int, hi: int):
        mask = seq["week_rank"].between(lo, hi)
        return mask.idxmax() if mask.any() else None

    def _ever_in_band(seq: pd.DataFrame, lo: int, hi: int) -> bool:
        return bool(seq["week_rank"].between(lo, hi).any())

    rows = []
    for cid, g in grid.groupby("canonical_id"):
        seq = g.sort_values("week_start")[["week_start", "week_rank", "rs_avg"]].reset_index(drop=True)

        # Control-group exclusion (see docstring): a cohort that EVER reaches
        # the surfaced band anywhere in its history is not eligible for the
        # baseline band, at any of its baseline-band weeks — otherwise the
        # same cohort contributes to both groups and "baseline" stops being
        # a comparison population.
        ever_surfaced = _ever_in_band(seq, 1, surface_rank)
        band_defs = [("surfaced_top", 1, surface_rank)]
        if not ever_surfaced:
            band_defs.append(("baseline", baseline_lo, baseline_hi))

        for band_label, lo, hi in band_defs:
            idx = _first_entry(seq, lo, hi)
            if idx is None:
                continue
            row = {
                "canonical_id": cid,
                "canonical_name": names_of.get(cid, cid),
                "band": band_label,
                "surfaced_week": seq.at[idx, "week_start"],
                "rank_at_surfaced": seq.at[idx, "week_rank"],
                "rs_avg_at_surfaced": seq.at[idx, "rs_avg"],
                "is_debut": idx == 0,   # first-ever recorded week already in-band — a strong debut, not a climb
            }
            for n in forward_weeks:
                fwd_idx = idx + n
                have_fwd = fwd_idx < len(seq)
                row[f"rank_fwd_{n}w"] = seq.at[fwd_idx, "week_rank"] if have_fwd else None
                row[f"rs_avg_fwd_{n}w"] = seq.at[fwd_idx, "rs_avg"] if have_fwd else None
                row[f"delta_rank_{n}w"] = (
                    seq.at[fwd_idx, "week_rank"] - seq.at[idx, "week_rank"] if have_fwd else None
                )
                row[f"delta_rs_{n}w"] = (
                    seq.at[fwd_idx, "rs_avg"] - seq.at[idx, "rs_avg"] if have_fwd else None
                )
            rows.append(row)

    events = pd.DataFrame(rows)
    if events.empty:
        return {
            "events": events, "summary": pd.DataFrame(),
            "weeks_available": weeks_available, "debut_share": None,
        }

    surfaced_events = events[events["band"] == "surfaced_top"]
    debut_share = (
        float(surfaced_events["is_debut"].mean()) if not surfaced_events.empty else None
    )

    bands = [("surfaced_top", 1, surface_rank), ("baseline", baseline_lo, baseline_hi)]

    summary_rows = []
    for band_label, _, _ in bands:
        band_events = events[events["band"] == band_label]
        for n in forward_weeks:
            usable = band_events.dropna(subset=[f"delta_rank_{n}w"])
            if usable.empty:
                summary_rows.append({
                    "band": band_label, "horizon_weeks": n, "n": 0,
                    "median_delta_rank": None, "median_delta_rs": None,
                    "pct_rank_improved": None,
                })
                continue
            summary_rows.append({
                "band": band_label,
                "horizon_weeks": n,
                "n": len(usable),
                "median_delta_rank": float(usable[f"delta_rank_{n}w"].median()),
                "median_delta_rs": float(usable[f"delta_rs_{n}w"].median()),
                "pct_rank_improved": float((usable[f"delta_rank_{n}w"] < 0).mean() * 100),
            })

    return {
        "events": events, "summary": pd.DataFrame(summary_rows),
        "weeks_available": weeks_available, "debut_share": debut_share,
    }
