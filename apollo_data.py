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

    ticker            TEXT
    entry_strategy    TEXT       'magna53' | '9m_day2'
    alert_date        DATE
    filled_at         TIMESTAMP
    closed_at         TIMESTAMP  null if still open
    status            TEXT       'closed' | 'stopped' | 'open'
    entry_price       FLOAT
    stop_price        FLOAT
    exit_price        FLOAT      null if still open
    entry_shares      INT
    total_pnl         FLOAT      0 if still open
    regime            TEXT       'Bull' | 'Choppy' | 'Correcting' | 'Crisis'
    gap_pct           FLOAT
    catalyst_quality  TEXT       'game_changer' | 'strong' | 'routine'
    ep_score          FLOAT      0-115 (post-uncap)
    account_mode      TEXT       'live' | 'paper'

Derived (computed in this module, never queried):
    r_multiple        FLOAT      (exit - entry) / (entry - stop)
    holding_days      INT        (closed_at - filled_at).days
    pnl_per_share     FLOAT      total_pnl / entry_shares
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

    # Derived columns — never queried; computed for UI convenience
    df["r_multiple"] = ((df["exit_price"] - df["entry_price"])
                        / (df["entry_price"] - df["stop_price"]))
    df["holding_days"] = ((df["closed_at"] - df["filled_at"]).dt.days
                          .where(df["closed_at"].notna()))
    df["pnl_per_share"] = ((df["total_pnl"] / df["entry_shares"])
                           .where(df["entry_shares"] > 0))

    return df


def load_trades(account_mode: str = "paper") -> pd.DataFrame:
    """Adapter — returns a DataFrame matching the documented schema.

    Mode selected by APOLLO_DATA_MODE env var:
        'mock' (default)  → methodology-realistic generator
        'db'              → direct Postgres read (requires APOLLO_DB_URL +
                            Tailscale; raises NotImplementedError until
                            live cutover lands and ≥30 trades exist)
    """
    mode = os.environ.get("APOLLO_DATA_MODE", "mock").lower()
    if mode == "mock":
        return generate_mock_trades(account_mode=account_mode)
    elif mode == "db":
        # Deferred until ≥30 closed live trades exist (~July 2026 earliest).
        # See memory/project_apollo_trades_dashboard.md for the architectural
        # fork (direct Postgres + Tailscale OR nightly CSV → Sheets).
        raise NotImplementedError(
            "APOLLO_DATA_MODE=db not yet wired. Mock mode covers Phase 1 UI "
            "scaffold; flip to db after live-trading flip + 30 closed trades."
        )
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


def setup_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-strategy stats table — win rate / profit factor / expectancy /
    avg R / avg holding days. Open trades excluded.
    """
    closed = df[df["status"].isin(["closed", "stopped"])].copy()
    if closed.empty:
        return pd.DataFrame(columns=[
            "strategy", "n_trades", "win_rate", "avg_r", "avg_r_win",
            "avg_r_loss", "profit_factor", "expectancy", "avg_hold_days",
            "largest_win", "largest_loss",
        ])

    rows = []
    for strat, group in closed.groupby("entry_strategy"):
        wins = group[group["total_pnl"] > 0]
        losses = group[group["total_pnl"] <= 0]
        win_pnl = wins["total_pnl"].sum()
        loss_pnl_abs = abs(losses["total_pnl"].sum()) or 1.0  # avoid div-by-zero
        rows.append({
            "strategy": strat,
            "n_trades": len(group),
            "win_rate": len(wins) / max(1, len(group)),
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
