"""
Apollo Trades — AI digest.

Generates 4-6 bullet executive summary at the top of the dashboard.
Pulls together state + methodology pulse + setup-specific signals +
observational recommendations.

Engine: Claude Haiku via Anthropic API. Falls back to deterministic
rule-based bullets if API key missing or call fails — digest always
renders.

Caching: caller wraps with @st.cache_data(ttl=3600) so the API gets
called at most once per hour per (period, account_mode, fingerprint).
Manual refresh button in the UI clears the cache.

Tone: observational — flag patterns and tradeoffs, avoid prescriptive
edicts. Pradeep methodology accepts low win rates with big winners; an
N<10 sample is treated as variance, not signal.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Capture % is a median over WINNERS; below this many winners it's variance, not
# signal, so the rule-based path reports it as a fact + caveat rather than
# recommending a trail review (matches the system prompt's N<10 = variance rule).
_CAPTURE_MIN_WINNERS = 10


_SYSTEM_PROMPT = """You are a trading analytics assistant for Apollo, a momentum/EP trading system following Pradeep Bonde / Qullamaggie methodology.

Methodology context (NON-NEGOTIABLE — do not contradict):
- Low win rates are ACCEPTED if winners are large multiples of risk. Median ~25-30% win rate is normal.
- "Lose small, win big" — winners hold days/weeks, losers stop within hours.
- Per-trade R-multiple expectancy is the metric, NOT win rate.
- Two active strategies: MAGNA53 EP (event-driven gappers + Sonnet+Perplexity validation), 9M Day 2 (virgin 9M-share volume day → next-day ORB entry).
- N<10 = sample variance, not signal. Do not recommend changes based on small samples.
- "Capture %" = realized exit R / peak R during hold. >85% = strong exit discipline; <60% = exiting too early.

Output 4-6 bullets organized in roughly this order:
1. STATE (1-2 bullets): Net P&L, activity level, win rate.
2. METHODOLOGY PULSE (1-2 bullets): Capture %, drawdown patterns, hold lengths vs Pradeep targets.
3. SETUP-SPECIFIC (1 bullet): Which strategies are outperforming/underperforming and by how much.
4. RECOMMENDATIONS (1-2 bullets): Observational only — flag patterns, surface tradeoffs, never edict.

Tone rules:
- Observational. "MAGNA53 capture is 88% — exit discipline is solid." Not "Tighten the stops."
- Surface uncertainty. If N<10 say so. If sample is mixed across regimes, note it.
- Flag tradeoffs, not directives. "Consider whether..." is fine; "Do X" is not.
- Brief: each bullet should be one sentence, no more than ~25 words.

Return ONLY a JSON object with this exact shape:
{"bullets": ["bullet 1", "bullet 2", ...]}

No prose outside the JSON. No markdown formatting inside bullets (no ** or backticks)."""


def _format_context(
    df: pd.DataFrame,
    setup_stats_df: pd.DataFrame,
    excursion_stats_df: pd.DataFrame,
    period_label: str,
    account_mode: str,
) -> str:
    """Build the stats summary string the LLM consumes.

    Includes only what's needed for the bullets — keeps prompt cheap.
    """
    closed = df[df["status"].isin(["closed", "stopped"])]
    open_n = (df["status"] == "open").sum()

    if closed.empty:
        return (
            f"Period: {period_label}\nAccount mode: {account_mode}\n"
            f"Closed trades: 0\nOpen trades: {open_n}\n"
            "No closed-trade data available."
        )

    winners = closed[closed["r_multiple"] > 0]
    losers = closed[closed["r_multiple"] <= 0]
    total_pnl = closed["total_pnl"].sum()
    avg_r = closed["r_multiple"].mean()
    win_rate = len(winners) / len(closed) if len(closed) else 0.0
    avg_hold_winner = winners["holding_days"].mean() if len(winners) else 0
    avg_hold_loser = losers["holding_days"].mean() if len(losers) else 0
    largest_winner = closed["total_pnl"].max()
    largest_loser = closed["total_pnl"].min()

    lines = [
        f"Period: {period_label}",
        f"Account: {account_mode}",
        "",
        "OVERALL STATE",
        f"  Closed trades: {len(closed)} ({len(winners)} winners, {len(losers)} losers)",
        f"  Open trades: {open_n}",
        f"  Win rate: {win_rate:.1%}",
        f"  Net P&L: ${total_pnl:+,.0f}",
        f"  Avg R-multiple: {avg_r:+.2f}R",
        f"  Largest winner: ${largest_winner:+,.0f}",
        f"  Largest loser: ${largest_loser:+,.0f}",
        f"  Avg hold (winners): {avg_hold_winner:.1f} days",
        f"  Avg hold (losers): {avg_hold_loser:.1f} days",
    ]

    if not setup_stats_df.empty:
        lines.extend(["", "PER-STRATEGY STATS"])
        for _, row in setup_stats_df.iterrows():
            lines.append(
                f"  {row['strategy']}: n={int(row['n_trades'])} "
                f"winrate={row['win_rate']:.1%} avg_r={row['avg_r']:+.2f}R "
                f"pf={row['profit_factor']:.2f} expectancy=${row['expectancy']:+,.0f}"
            )

    if not excursion_stats_df.empty:
        lines.extend(["", "EXCURSION (drawdown/peak during hold)"])
        for _, row in excursion_stats_df.iterrows():
            cap = row.get("median_capture_pct")
            cap_str = f"{cap*100:.0f}%" if pd.notna(cap) else "n/a"
            lines.append(
                f"  {row['strategy']}: median_drawdown={row['median_worst_r']:+.2f}R "
                f"median_peak={row['median_best_r']:+.2f}R capture={cap_str}"
            )

    return "\n".join(lines)


def _rule_based_bullets(
    df: pd.DataFrame,
    setup_stats_df: pd.DataFrame,
    excursion_stats_df: pd.DataFrame,
    period_label: str,
) -> list[str]:
    """Deterministic fallback bullets when LLM call fails or is disabled."""
    closed = df[df["status"].isin(["closed", "stopped"])]
    if closed.empty:
        return [
            f"No closed trades in {period_label}. Open trades pending: "
            f"{(df['status'] == 'open').sum()}."
        ]

    total_pnl = closed["total_pnl"].sum()
    avg_r = closed["r_multiple"].mean()
    # Win rate consumes the scratch-aware setup_stats (n_win excludes |R|<0.25
    # breakeven scratches) so the pulse matches the scorecards — rather than
    # re-rolling r>0 here, which counted scratches like PURR/KURA as wins.
    if not setup_stats_df.empty and "n_win" in setup_stats_df.columns:
        n_win = int(setup_stats_df["n_win"].sum())
        n_scratch = int(setup_stats_df["n_scratch"].sum())
        n_loss = int(setup_stats_df["n_loss"].sum())
        win_rate = n_win / len(closed)
        wsl = f" ({n_win}W/{n_scratch}S/{n_loss}L)"
    else:
        win_rate = len(closed[closed["r_multiple"] > 0]) / len(closed)
        wsl = ""

    bullets = [
        f"Closed {len(closed)} trades in {period_label} — net P&L "
        f"${total_pnl:+,.0f}, win rate {win_rate:.0%}{wsl}, avg {avg_r:+.2f}R per trade."
    ]

    if len(closed) < 10:
        bullets.append(
            f"Sample size N={len(closed)} is below the methodology noise floor "
            "(N≥10) — treat patterns as variance until cohort grows."
        )

    if not excursion_stats_df.empty:
        for _, row in excursion_stats_df.iterrows():
            cap = row.get("median_capture_pct")
            if pd.notna(cap):
                n_cap = int(row.get("n_capture", 0) or 0)
                if n_cap < _CAPTURE_MIN_WINNERS:
                    # Too few winners to read the capture median as signal — state
                    # the fact + sample caveat, don't recommend a trail change.
                    note = f"only {n_cap} winner{'' if n_cap == 1 else 's'} — variance, not signal"
                elif cap >= 0.85:
                    note = "exit discipline solid"
                elif cap < 0.60:
                    note = "exiting earlier than peak — trail review worth considering"
                else:
                    note = "exits in normal range"
                bullets.append(
                    f"{row['strategy']} capture {cap*100:.0f}% — {note}."
                )

    if not setup_stats_df.empty and len(setup_stats_df) >= 2:
        sorted_stats = setup_stats_df.sort_values("expectancy", ascending=False)
        top = sorted_stats.iloc[0]
        bot = sorted_stats.iloc[-1]
        if top["expectancy"] != bot["expectancy"]:
            bullets.append(
                f"{top['strategy']} leading on expectancy "
                f"(${top['expectancy']:+,.0f}/trade vs ${bot['expectancy']:+,.0f} "
                f"for {bot['strategy']}); compare N to weigh."
            )

    return bullets[:6]  # cap at 6


def _call_haiku(context: str, api_key: str) -> Optional[list[str]]:
    """Call Claude Haiku, parse JSON response, return bullets list or None."""
    try:
        from anthropic import Anthropic
    except ImportError:
        logger.warning("anthropic package not installed; skipping LLM digest")
        return None

    try:
        client = Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=800,
            system=_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": (
                    "Generate the digest. Stats:\n\n"
                    f"{context}"
                ),
            }],
        )
        raw = resp.content[0].text.strip()
        # Tolerate optional fenced JSON
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        parsed = json.loads(raw)
        bullets = parsed.get("bullets")
        if not isinstance(bullets, list) or not bullets:
            logger.warning(f"LLM returned unexpected shape: {raw[:200]}")
            return None
        return [str(b) for b in bullets][:6]
    except Exception as e:
        logger.warning(f"Haiku digest call failed: {e}")
        return None


def generate_digest(
    df: pd.DataFrame,
    setup_stats_df: pd.DataFrame,
    excursion_stats_df: pd.DataFrame,
    period_label: str,
    account_mode: str,
    api_key: Optional[str] = None,
) -> dict:
    """Generate the digest. Returns:
        {"bullets": [str, ...], "source": "llm"|"rule_based", "generated_at": iso}
    """
    context = _format_context(
        df, setup_stats_df, excursion_stats_df, period_label, account_mode,
    )

    bullets = None
    source = "rule_based"
    if api_key:
        bullets = _call_haiku(context, api_key)
        if bullets:
            source = "llm"

    if not bullets:
        bullets = _rule_based_bullets(df, setup_stats_df, excursion_stats_df, period_label)

    return {
        "bullets": bullets,
        "source": source,
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
