"""
Apollo Trades data adapter.

Single source of truth for "where do trades come from" — switches between
methodology-realistic mock data (scaffold phase, pre-live-cutover) and
direct DB read (post-cutover, ≥30 closed live trades).

Mock generator produces a DataFrame that matches the eventual real schema
EXACTLY — same columns, same dtypes, same null patterns. UI written
against this shape will work unchanged when live data lands.

Toggle via env var:
    APOLLO_DATA_MODE=mock   (default; methodology-realistic generator)
    APOLLO_DATA_MODE=db     (read from Apollo Postgres — needs Tailscale)

Schema (matches mi_live_trades joined with mi_ep_alerts + mi_market_regime):

    ticker             TEXT
    entry_strategy     TEXT       'magna53' | '9m_day2'
    alert_date         DATE
    filled_at          TIMESTAMP
    closed_at          TIMESTAMP  null if still open
    status             TEXT       'closed' | 'stopped' | 'open'
    entry_price        FLOAT
    stop_price         FLOAT
    exit_price         FLOAT      null if still open
    entry_shares       INT
    total_pnl          FLOAT      0 if still open
    regime             TEXT       'Bull' | 'Choppy' | 'Correcting' | 'Crisis'
    gap_pct            FLOAT
    catalyst_quality   TEXT       'game_changer' | 'strong' | 'routine'
    ep_score           FLOAT      0-115 (post-uncap)
    account_mode       TEXT       'live' | 'paper'
    lowest_price_seen  FLOAT      Worst price during trade's open life
    highest_price_seen FLOAT      Best price during trade's open life

Derived (computed in this module, never queried):
    r_multiple        FLOAT      (exit - entry) / (entry - stop)
    holding_days      INT        (closed_at - filled_at).days
    pnl_per_share     FLOAT      total_pnl / entry_shares
    worst_r           FLOAT      (lowest_price - entry) / risk_per_share  ≤ 0
    best_r            FLOAT      (highest_price - entry) / risk_per_share  ≥ 0
    capture_pct       FLOAT      r_multiple / best_r (open trades: NaN; best_r=0: NaN)
                                 "how much of the available upside did we capture?"
"""
from __future__ import annotations

import os
from datetime import date, datetime, time, timedelta
from random import Random
from typing import Optional

import pandas as pd

# Methodology constants — Pradeep / Qullamaggie shape
_WIN_RATE = 0.28           # ~25-30% across both strategies
_WIN_RATE_MAGNA53 = 0.32   # MAGNA53 slightly higher (catalyst-confirmed)
_WIN_RATE_9M_DAY2 = 0.24   # 9M Day 2 lower (Day 2 ORB has gap-through risk)

_AVG_R_WINNER = 4.5        # Big winners drive returns
_AVG_R_LOSER = -1.0        # Tight stops, fast losers
_GAP_THROUGH_RATE = 0.05   # ~5% of losers gap through stop, exit at -1.5 to -2.5R
_AVG_HOLD_WINNER_DAYS = 9  # Pradeep: hold winners
_AVG_HOLD_LOSER_DAYS = 1   # Stop fast

_REGIMES = ["Bull", "Choppy", "Correcting", "Crisis"]
_REGIME_WEIGHTS = [0.55, 0.25, 0.15, 0.05]  # Bull-leaning recent period
_CATALYSTS = ["game_changer", "strong", "routine"]
_CATALYST_WEIGHTS = [0.20, 0.55, 0.25]
_STRATEGIES = ["magna53", "9m_day2"]
_STRATEGY_WEIGHTS = [0.65, 0.35]  # MAGNA53 more frequent (broader universe)

# Reasonable mid-cap universe for mock — matches Apollo's typical EP names
_MOCK_TICKERS = [
    "FTNT", "FLNC", "JMIA", "SITM", "PCT", "MTSI", "SEZL", "ARRY",
    "PGNY", "WEST", "WLDN", "INOD", "AKAM", "BILL", "LASR", "CALY",
    "FROG", "DDOG", "BLMN", "HUT", "OSS", "WOLF", "GDYN", "FLEX",
    "DOC", "OSCR", "FLYW", "BRBR", "SMCI", "AMD", "GEO", "CRI",
    "ARM", "GLW", "LIVN", "COMP", "POET", "STRL", "CYRX", "FTRE",
    "TEAM", "SOUN", "TWLO", "TTMI", "EVER",
]


def _generate_one_trade(
    rng: Random,
    seed_date: date,
    today: date,
    account_mode: str,
) -> dict:
    """Methodology-realistic single trade. Strategy chosen by weighted draw,
    win/loss by per-strategy rate, R-multiple by skewed normal.
    """
    strategy = rng.choices(_STRATEGIES, weights=_STRATEGY_WEIGHTS, k=1)[0]
    win_rate = _WIN_RATE_MAGNA53 if strategy == "magna53" else _WIN_RATE_9M_DAY2
    is_winner = rng.random() < win_rate

    if is_winner:
        # Winners: skewed positive — most are 2-5R, occasional 8-12R outlier
        if rng.random() < 0.15:
            r = rng.uniform(7.0, 12.0)
        else:
            r = rng.uniform(1.5, 5.5)
        hold_days = max(1, int(rng.gauss(_AVG_HOLD_WINNER_DAYS, 4.0)))
        hold_days = min(hold_days, 25)
    else:
        # Losers: most -1R clean stop, ~5% gap-through to -1.5 to -2.5R
        if rng.random() < _GAP_THROUGH_RATE:
            r = rng.uniform(-2.5, -1.5)
        else:
            r = rng.uniform(-1.1, -0.85)
        hold_days = max(0, int(rng.gauss(_AVG_HOLD_LOSER_DAYS, 1.5)))
        hold_days = min(hold_days, 5)

    # Price + stop based on a typical EP setup
    ticker = rng.choice(_MOCK_TICKERS)
    entry_price = round(rng.uniform(15.0, 250.0), 2)
    # Stop at 3-7% below entry depending on volatility
    stop_pct = rng.uniform(0.03, 0.07)
    stop_price = round(entry_price * (1 - stop_pct), 2)
    risk_per_share = entry_price - stop_price
    # Account-aware sizing: live $10K (1% risk = $100), paper $100K ($1000)
    account_risk = 100 if account_mode == "live" else 1000
    entry_shares = max(1, int(account_risk / risk_per_share))
    exit_price = round(entry_price + r * risk_per_share, 2)
    total_pnl = round((exit_price - entry_price) * entry_shares, 2)

    # Worst-price / best-price during open life — methodology-realistic
    # shapes for the excursion analytics panel.
    if is_winner:
        # Winners often pull back slightly before running (worst-r small negative).
        worst_r_sim = rng.uniform(-0.7, 0.0)
        # Best-r ≥ actual r. ~30% of winners exit too early (best > r * 1.3);
        # ~70% exit near peak (best ~ r * 1.0–1.2).
        if rng.random() < 0.30:
            best_r_sim = max(r, r * rng.uniform(1.3, 2.2))
        else:
            best_r_sim = max(r, r * rng.uniform(1.0, 1.2))
    else:
        # Losers usually run straight to stop; ~20% have a false breakout first.
        worst_r_sim = min(r, -0.95)  # roughly the stop level
        if rng.random() < 0.20:
            best_r_sim = rng.uniform(0.3, 0.9)
        else:
            best_r_sim = rng.uniform(0.0, 0.25)
    lowest_price_seen = round(entry_price + worst_r_sim * risk_per_share, 2)
    highest_price_seen = round(entry_price + best_r_sim * risk_per_share, 2)

    # Time stamps — entry at market open, exit at close of holding-period day
    filled_at = datetime.combine(seed_date, time(9, 31))
    closed_at = datetime.combine(seed_date + timedelta(days=hold_days), time(15, 55))
    # If close would be in the future, mark as still open
    if closed_at.date() > today:
        status = "open"
        closed_at = None
        exit_price = None
        total_pnl = 0.0
    else:
        status = "stopped" if r < 0 else "closed"

    regime = rng.choices(_REGIMES, weights=_REGIME_WEIGHTS, k=1)[0]
    catalyst = rng.choices(_CATALYSTS, weights=_CATALYST_WEIGHTS, k=1)[0]
    gap_pct = round(rng.uniform(8.0, 35.0), 2)
    # ep_score: post-uncap range 70-115 in Bull, 70-95 elsewhere
    ep_score = round(rng.uniform(70.0, 115.0 if regime == "Bull" else 95.0), 1)

    return {
        "ticker": ticker,
        "entry_strategy": strategy,
        "alert_date": seed_date,
        "filled_at": filled_at,
        "closed_at": closed_at,
        "status": status,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "exit_price": exit_price,
        "entry_shares": entry_shares,
        "total_pnl": total_pnl,
        "regime": regime,
        "gap_pct": gap_pct,
        "catalyst_quality": catalyst,
        "ep_score": ep_score,
        "account_mode": account_mode,
        "lowest_price_seen": lowest_price_seen,
        "highest_price_seen": highest_price_seen,
    }


def generate_mock_trades(
    n_trades: int = 90,
    days_back: int = 90,
    account_mode: str = "paper",
    seed: int = 42,
) -> pd.DataFrame:
    """Methodology-realistic mock trade history.

    `n_trades` over the trailing `days_back` calendar days. Trades cluster
    on weekdays only (no weekend entries). Random.choices used throughout
    so seed is deterministic for layout-debugging stability.
    """
    rng = Random(seed)
    today = date.today()

    rows: list[dict] = []
    placed = 0
    attempts = 0
    while placed < n_trades and attempts < n_trades * 3:
        attempts += 1
        offset = rng.randint(0, days_back - 1)
        seed_date = today - timedelta(days=offset)
        # Skip weekends
        if seed_date.weekday() >= 5:
            continue
        rows.append(_generate_one_trade(rng, seed_date, today, account_mode))
        placed += 1

    df = pd.DataFrame(rows)
    df = df.sort_values("filled_at").reset_index(drop=True)

    # Derived analytics columns via the shared helper — so mock and the
    # db/snapshot reader yield identical derived shapes, incl. the corrupt-stop
    # risk>0 guard (a clean generator won't trip it, but the path stays single).
    return _add_derived(df)


def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the derived analytics columns on a base-schema DataFrame — shared
    by the mock generator and the db/snapshot reader so both yield identical
    derived shapes. Null-safe for open trades + missing excursion/fill data."""
    # Guard: a valid long has stop < entry (risk > 0). A recorded stop >= entry
    # — corrupt stop data (e.g. CRMD $8.45 vs entry $8.36) or trailed-to-breakeven
    # (== entry) — makes risk <= 0, which would flip a LOSS into a positive R or
    # divide by zero. Treat risk <= 0 as invalid -> NaN so the trade drops out of
    # the R-rankings instead of showing a bogus +R (its real P&L still shows).
    risk = df["entry_price"] - df["stop_price"]
    risk = risk.where(risk > 0)
    df["r_multiple"] = (df["exit_price"] - df["entry_price"]) / risk
    df["holding_days"] = ((df["closed_at"] - df["filled_at"]).dt.days
                          .where(df["closed_at"].notna() & df["filled_at"].notna()))
    df["pnl_per_share"] = ((df["total_pnl"] / df["entry_shares"])
                           .where(df["entry_shares"] > 0))
    df["worst_r"] = (df["lowest_price_seen"] - df["entry_price"]) / risk
    df["best_r"] = (df["highest_price_seen"] - df["entry_price"]) / risk
    is_winner = df["r_multiple"] > 0
    has_upside = df["best_r"] > 0
    df["capture_pct"] = (df["r_multiple"] / df["best_r"]).where(is_winner & has_upside)
    return df


# Point-in-time snapshot of real paper trades (exported from mi_live_trades);
# its presence is also what resolve_data_mode() uses to auto-detect db mode.
_SNAPSHOT_PATH = os.path.join(os.path.dirname(__file__), "apollo_trades_paper.json")


def _load_from_snapshot(account_mode: str = "paper") -> pd.DataFrame:
    """Read the real-Apollo-trade snapshot (exported from mi_live_trades) and
    return the documented schema, account-mode-filtered.

    The snapshot (apollo_trades_paper.json) is a point-in-time export from the
    Hetzner DB — lets the dashboard be reviewed/iterated on REAL data pre-cutover
    (operator 2026-06-03). The LIVE direct-Postgres reader is the seamless
    end-state (same query + schema, swaps in once Tailscale exposes the DB);
    regenerate the snapshot to refresh until then.
    """
    import json
    with open(_SNAPSHOT_PATH) as f:
        rows = json.load(f)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[df["account_mode"] == account_mode].copy()
    # Parse to naive UTC. Intraday ET fills/closes (≈13:30–20:00 UTC) don't cross
    # midnight, so the calendar DATE already equals the ET trading day — no tz
    # conversion needed, which avoids a tz-database dependency on the hosted
    # (Streamlit Cloud) runtime. (Times render in UTC; cosmetic, fixable later.)
    for col in ("filled_at", "closed_at"):
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce").dt.tz_localize(None)
    df["alert_date"] = pd.to_datetime(df["alert_date"], errors="coerce").dt.date
    df = df.sort_values("filled_at", na_position="first").reset_index(drop=True)
    return _add_derived(df)


def resolve_data_mode() -> str:
    """The mode load_trades() will actually use: explicit APOLLO_DATA_MODE if
    set, else auto-detect (db when the real-trade snapshot is present, else
    mock). Exposed so the UI banner reflects the REAL data source, not a guess."""
    mode = os.environ.get("APOLLO_DATA_MODE", "").strip().lower()
    if mode:
        return mode
    return "db" if os.path.exists(_SNAPSHOT_PATH) else "mock"


def load_trades(account_mode: str = "paper") -> pd.DataFrame:
    """Adapter — returns a DataFrame matching the documented schema.

    Mode selected by APOLLO_DATA_MODE env var:
        'mock' (default)  → methodology-realistic generator
        'db'              → direct Postgres read (requires APOLLO_DB_URL +
                            Tailscale; raises NotImplementedError until
                            live cutover lands and ≥30 trades exist)
    """
    mode = resolve_data_mode()
    if mode == "mock":
        return generate_mock_trades(account_mode=account_mode)
    elif mode == "db":
        # Reads the real-Apollo-trade snapshot (paper trades from mi_live_trades).
        # Operator 2026-06-03: paper-now so the dash is reviewed/iterated on REAL
        # data; account_mode-parameterized so the LIVE direct-Postgres reader
        # (Tailscale end-state) swaps in with zero schema change. Snapshot is a
        # point-in-time export — regenerate to refresh until the live reader lands.
        return _load_from_snapshot(account_mode)
    else:
        raise ValueError(f"Unknown APOLLO_DATA_MODE={mode!r}; expected 'mock' or 'db'")


# ── Aggregations used by the UI ─────────────────────────────────────────────


def daily_pnl(df: pd.DataFrame) -> pd.DataFrame:
    """One row per calendar date with realized P&L from trades closed that
    day. Used by the calendar heatmap. Open trades excluded (no P&L yet).
    """
    closed = df[df["status"].isin(["closed", "stopped"])].copy()
    if closed.empty:
        return pd.DataFrame(columns=["date", "pnl", "n_trades", "tickers"])
    closed["close_date"] = closed["closed_at"].dt.date
    agg = closed.groupby("close_date").agg(
        pnl=("total_pnl", "sum"),
        n_trades=("ticker", "size"),
        tickers=("ticker", lambda s: ", ".join(sorted(s.unique()))),
    ).reset_index().rename(columns={"close_date": "date"})
    return agg


SCRATCH_R = 0.25  # |R| below this = a breakeven scratch, not a win or loss


def classify_outcome(df: pd.DataFrame) -> list:
    """Classify each closed trade as 'win' | 'scratch' | 'loss' by R-multiple.

    A near-breakeven trade (|R| < SCRATCH_R) is a SCRATCH — not a win or loss —
    so win rate reflects real wins, not flat exits. Corrupt-stop trades (NaN R,
    e.g. a stored stop above entry) fall back to P&L sign. This is the ONE
    outcome definition; every win-rate surface (setup_stats, the KPI strip, the
    digest via n_win) consumes it so they cannot drift apart.
    """
    def _b(r, pnl):
        if pd.notna(r):
            return "win" if r > SCRATCH_R else ("loss" if r < -SCRATCH_R else "scratch")
        return "win" if pnl > 0 else ("loss" if pnl < 0 else "scratch")
    return [_b(r, p) for r, p in zip(df["r_multiple"], df["total_pnl"])]


def setup_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-strategy stats table — win rate / profit factor / expectancy /
    avg R / avg holding days. Open trades excluded.
    """
    closed = df[df["status"].isin(["closed", "stopped"])].copy()
    if closed.empty:
        return pd.DataFrame(columns=[
            "strategy", "n_trades", "win_rate", "n_win", "n_scratch", "n_loss",
            "avg_r", "avg_r_win", "avg_r_loss", "profit_factor", "expectancy",
            "avg_hold_days", "largest_win", "largest_loss",
        ])

    # One scratch-aware outcome definition (see classify_outcome): scratches
    # (|R| < SCRATCH_R) stay in the WR denominator but aren't wins (conservative).
    closed["_bucket"] = classify_outcome(closed)

    rows = []
    for strat, group in closed.groupby("entry_strategy"):
        wins = group[group["_bucket"] == "win"]
        losses = group[group["_bucket"] == "loss"]
        scratches = group[group["_bucket"] == "scratch"]
        win_pnl = wins["total_pnl"].sum()
        loss_pnl_abs = abs(losses["total_pnl"].sum()) or 1.0  # avoid div-by-zero
        rows.append({
            "strategy": strat,
            "n_trades": len(group),
            "win_rate": len(wins) / max(1, len(group)),  # scratches in denom (conservative)
            "n_win": len(wins),
            "n_scratch": len(scratches),
            "n_loss": len(losses),
            "avg_r": group["r_multiple"].mean(),
            "avg_r_win": wins["r_multiple"].mean() if len(wins) else 0.0,
            "avg_r_loss": losses["r_multiple"].mean() if len(losses) else 0.0,
            "profit_factor": win_pnl / loss_pnl_abs,
            "expectancy": group["total_pnl"].mean(),
            "avg_hold_days": group["holding_days"].mean(),
            "largest_win": group["total_pnl"].max(),
            "largest_loss": group["total_pnl"].min(),
        })
    return pd.DataFrame(rows).sort_values("strategy").reset_index(drop=True)


def excursion_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-strategy worst/best R-excursion stats — surfaces setup-quality
    signals invisible in P&L-only views.

    For each strategy:
      - median worst_r: how far did trades go underwater typically?
        (closer to 0 = clean entries; near -1.0 = often touch the stop)
      - median best_r: how far did trades run in your favor at peak?
        (high values = wide opportunity; setup worth keeping)
      - median capture_pct: how much of the available upside did the exit
        capture? (<0.6 = exiting too early; >0.85 = trail discipline good)
      - n: closed-trade count (excludes open)

    Open trades excluded — their excursion data isn't final (peak could
    still extend).
    """
    closed = df[df["status"].isin(["closed", "stopped"])].copy()
    if closed.empty or "worst_r" not in closed.columns:
        return pd.DataFrame(columns=[
            "strategy", "n", "n_capture", "median_worst_r", "median_best_r",
            "median_capture_pct", "median_r",
        ])

    rows = []
    for strat, group in closed.groupby("entry_strategy"):
        valid = group.dropna(subset=["worst_r", "best_r"])
        if valid.empty:
            continue
        # capture_pct median computed over WINNERS only — that's the only
        # context where "how much of the upside did we get" makes sense.
        # "Winner" here = raw r>0, deliberately NOT the scratch-aware
        # classify_outcome: a +0.1R scratch that ran to +3R is a real
        # under-capture, so it stays in the capture denominator.
        winners = valid[valid["r_multiple"] > 0]
        rows.append({
            "strategy": strat,
            "n": len(valid),
            # winners with a real capture value (the N feeding the median below);
            # consumers gate the capture recommendation on this small-sample count.
            "n_capture": int(winners["capture_pct"].notna().sum()),
            "median_worst_r": valid["worst_r"].median(),
            "median_best_r": valid["best_r"].median(),
            "median_capture_pct": (
                winners["capture_pct"].median() if len(winners) else float("nan")
            ),
            "median_r": valid["r_multiple"].median(),
        })
    return pd.DataFrame(rows).sort_values("strategy").reset_index(drop=True)


if __name__ == "__main__":
    # Smoke check
    df = generate_mock_trades(n_trades=90, days_back=90)
    print(f"Generated {len(df)} trades")
    print(df[["ticker", "entry_strategy", "filled_at", "status", "total_pnl", "r_multiple"]].head(10))
    print()
    print("Daily P&L (head):")
    print(daily_pnl(df).head(8))
    print()
    print("Setup stats:")
    print(setup_stats(df))
