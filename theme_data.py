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
) -> dict[str, str]:
    """Subset-aware dedup. Returns parent_of: alias_name -> representative.

    Verbatim from rs-theme-dash/data.py (pure function, no DB). Sort by size DESC;
    a smaller theme S merges into a larger un-merged L when
    |S ∩ L| / |S| >= threshold AND |S ∩ L| >= min_shared (tie-break by Jaccard).
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
