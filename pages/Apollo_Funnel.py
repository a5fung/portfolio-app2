"""
Apollo Funnel — the EP scan funnel over time (#589).

Operator (2026-08-24), after /scanned shipped: "we can probably build funnel
views for more than a day but that likely belongs to the dash vs a command so
we can actually have richer views & stats." /scanned answers ONE day well; it
cannot say whether a stage is cutting more than it used to, or what a rejected
name actually cost. This page answers both, read-only:

  HALF 1 — the funnel over time: per-stage counts by day, the drop-off rate at
           each stage, and which stage's rate has actually MOVED (ranked by
           the data, not a guessed subset of "stages he cares about" — see
           funnel_data.py docstring).
  HALF 2 — the scorecard: rejected names that subsequently ran, broken down by
           the stage that rejected them, so the cost of each gate is
           measurable instead of assumed.

THE LINE: read-only reporting. Changes no rule, threshold, filter, or score.
Stage definitions are ported verbatim from apollo_the_wise's
agents/market_intelligence/scanned_report.py (see funnel_data.py header) and
cross-checked against the real /scanned renderer for 2026-09-03 — identical,
stage for stage. If that ever drifts, funnel_data.py's docstring says how to
re-sync.

Data source: apollo_funnel_snapshot.json, a point-in-time export of
mi_ep_scan_log / mi_catalyst_tier_shadow / mi_ep_alerts / mi_live_trades /
mi_ep_missed_outcomes (SELECT-only). Same snapshot pattern as apollo_data.py /
apollo_trades_paper.json and theme_data.py / apollo_themes_snapshot.json.
"""
from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import funnel_data as fdat
from app_theme import is_dark

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Apollo Funnel", layout="wide", page_icon="◆")

# Mirror Portfolio.py / Apollo_Trades.py palette so this page feels native.
C_DARK = {
    "bg": "#000000", "surface": "#111113", "surface2": "#18181B",
    "text": "#FFFFFF", "text_sec": "#D4D4D8", "text_muted": "#A1A1AA",
    "text_dim": "#52525B",
    "primary": "#00D26A", "primary_dim": "#00331B",
    "positive": "#00D26A", "positive_dim": "#004D26",
    "negative": "#F82C2C", "negative_dim": "#450A0A",
    "warning": "#F59E0B",
    "border": "#27272A", "grid": "#18181B",
}
C_LIGHT = {
    "bg": "#FFFFFF", "surface": "#F4F4F5", "surface2": "#E4E4E7",
    "text": "#09090B", "text_sec": "#27272A", "text_muted": "#52525B",
    "text_dim": "#71717A",
    "primary": "#00B85E", "primary_dim": "#DCFCE7",
    "positive": "#00B85E", "positive_dim": "#DCFCE7",
    "negative": "#DC2626", "negative_dim": "#FEE2E2",
    "warning": "#D97706",
    "border": "#D4D4D8", "grid": "#D4D4D8",
}

st.session_state.dark_mode = is_dark()
C = C_DARK if st.session_state.dark_mode else C_LIGHT

CHART_CONFIG = {"displayModeBar": False, "staticPlot": False, "scrollZoom": False}


def style_chart(fig, height=None):
    """Ported from Portfolio.py's style_chart() (each page in this repo keeps
    its own copy — importing Portfolio.py would re-run its whole top level)."""
    fig.update_layout(
        template="plotly_dark" if st.session_state.dark_mode else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C["text_dim"], size=10, family="JetBrains Mono"),
        margin=dict(l=0, r=0, t=10, b=0),
        dragmode=False,
        hovermode="x unified",
        hoverlabel=dict(bgcolor=C["surface2"], font_size=12, font_family="JetBrains Mono", bordercolor=C["border"]),
        xaxis=dict(showgrid=False, showline=False, fixedrange=True, tickfont=dict(color=C["text_muted"])),
        yaxis=dict(showgrid=True, gridcolor=C["grid"], gridwidth=1, showline=False,
                    fixedrange=True, tickfont=dict(color=C["text_muted"])),
        showlegend=False,
    )
    fig.update_layout(modebar_remove=["zoom", "pan", "select", "lasso2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"])
    if height:
        fig.update_layout(height=height)
    return fig


st.markdown(
    f"""
    <style>
    .stApp {{ background: {C['bg']}; }}
    .callout {{
        background: {C['surface']}; border-left: 3px solid {C['warning']};
        padding: 12px 16px; border-radius: 4px; color: {C['text_sec']};
        font-size: 12.5px; margin-bottom: 16px; line-height: 1.5;
    }}
    .kpi-card {{
        background: {C['surface']}; border: 1px solid {C['border']};
        border-radius: 8px; padding: 14px 16px; height: 100%;
    }}
    .kpi-label {{ color: {C['text_muted']}; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }}
    .kpi-value {{ color: {C['text']}; font-size: 22px; font-weight: 600; margin-top: 4px; }}
    .kpi-sub {{ color: {C['text_muted']}; font-size: 11px; margin-top: 2px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("◆ Apollo Funnel")
st.caption("EP scan funnel over time · per-stage drop-off trend · the cost of each gate (#589)")

meta = fdat.snapshot_meta()
gen_at = meta.get("generated_at")
if gen_at:
    try:
        gen_dt = datetime.fromisoformat(gen_at)
        st.caption(f"Snapshot generated {gen_dt.strftime('%Y-%m-%d %H:%M UTC')} · point-in-time export, regenerate to refresh")
    except ValueError:
        pass

# ── Load computed tables ─────────────────────────────────────────────────────
stage_table = fdat.build_ticker_stage_table()

if stage_table.empty:
    st.warning("No scan rows in the snapshot. Either the export is empty or scan logging broke — "
               "this is worth a look, not a chart with zeros.")
    st.stop()

funnel_day = fdat.funnel_by_day()
funnel_week = fdat.funnel_by_week()
moved = fdat.stage_moved_summary()
post_change = fdat.post_regime_change_snapshot()
score = fdat.scorecard()

min_date = stage_table["date"].min().date()
max_date = stage_table["date"].max().date()
n_scan_days = stage_table["date"].nunique()

st.markdown(
    f"""<div class="callout">
    <b>Data coverage — read before comparing across dates.</b><br>
    Funnel data runs <b>{min_date} → {max_date}</b> ({n_scan_days} scan days) — mi_ep_scan_log's
    earliest row is 2026-04-13, not "back to March" as assumed going in.<br>
    <b>2026-08-24 is a capture-instrumentation change, not a trend</b> — the universe-floor stages
    (u_close / u_vol) and the graded-then-cut stage (graded_cut) were not being logged before that
    date, and daily ticker counts jump roughly 10× that day (mid-teens/day → 150-300/day). The
    "stages that moved" ranking below is computed only over the clean {min_date} → 2026-08-21
    window for exactly this reason; 2026-08-24 onward is shown separately, flagged as too short and
    not yet comparable rather than blended into a trend.
    </div>""",
    unsafe_allow_html=True,
)

# ── KPI strip ────────────────────────────────────────────────────────────────
n_tickers_logged = len(stage_table)
n_alert_or_better = int(stage_table["is_graded_or_better"].sum())
alert_rate = n_alert_or_better / n_tickers_logged if n_tickers_logged else float("nan")
n_bulk_runners = int((~stage_table["is_graded_or_better"] & stage_table["ran_bulk"]).sum())
# Exclude instrumentation-limited stages, small-N stages, and stages whose
# clean-window move needs a caveat (a cap, a wording-migration confound, or a
# post-alert fill-rate effect — see funnel_data.HEADLINE_EXCLUDED_STAGES) from
# the headline "biggest mover" so the first thing the operator sees is a real,
# unconfounded signal.
_movable = moved[
    ~moved["instrumentation_limited"] & ~moved["insufficient_n"]
    & ~moved["stage"].isin(fdat.HEADLINE_EXCLUDED_STAGES)
]
top_mover = _movable.iloc[0] if len(_movable) else None

kpi_cols = st.columns(4)
with kpi_cols[0]:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Scan days covered</div>
        <div class="kpi-value">{n_scan_days}</div>
        <div class="kpi-sub">{min_date} → {max_date}</div></div>""", unsafe_allow_html=True)
with kpi_cols[1]:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Tickers logged (all-time)</div>
        <div class="kpi-value">{n_tickers_logged:,}</div>
        <div class="kpi-sub">{n_alert_or_better} reached grading or better ({alert_rate*100:.1f}%)</div></div>""", unsafe_allow_html=True)
with kpi_cols[2]:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Bulk-cut names that ran ≥20%</div>
        <div class="kpi-value">{n_bulk_runners}</div>
        <div class="kpi-sub">within 5 sessions of the gap day, of names with a settled outcome</div></div>""", unsafe_allow_html=True)
with kpi_cols[3]:
    if top_mover is not None:
        arrow = "▲" if top_mover["delta_pp"] > 0 else "▼"
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Biggest mover (clean window)</div>
            <div class="kpi-value">{top_mover['label'][:22]}</div>
            <div class="kpi-sub">{arrow} {abs(top_mover['delta_pp']):.1f}pp ({top_mover['first_half_rate']*100:.0f}% → {top_mover['second_half_rate']*100:.0f}%)</div></div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="kpi-card"><div class="kpi-label">Biggest mover</div>
            <div class="kpi-value">—</div><div class="kpi-sub">not enough clean-window data</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Chart: tickers scanned per day (shows the capture-regime jump) ─────────
st.subheader("Tickers scanned per day")
daily_total = stage_table.groupby("date").size().reset_index(name="n")
fig_pop = go.Figure()
fig_pop.add_trace(go.Scatter(
    x=daily_total["date"], y=daily_total["n"], mode="lines",
    line=dict(color=C["primary"], width=1.5),
    fill="tozeroy", fillcolor=C["primary_dim"],
    hovertemplate="%{x|%Y-%m-%d}<br>%{y} tickers<extra></extra>",
))
fig_pop.add_vline(
    x=pd.Timestamp(fdat.CAPTURE_REGIME_CHANGE), line_width=1, line_dash="dash",
    line_color=C["warning"],
    annotation_text="capture expanded 08-24", annotation_position="top",
    annotation_font=dict(color=C["warning"], size=10),
)
style_chart(fig_pop, height=220)
fig_pop.update_layout(yaxis=dict(title=None, showgrid=True, gridcolor=C["grid"], fixedrange=True, tickfont=dict(color=C["text_muted"])))
st.plotly_chart(fig_pop, width='stretch', config=CHART_CONFIG)
st.caption("The 08-24 jump is scan-log CAPTURE expanding (more of the pipeline started getting logged), "
           "not the universe suddenly admitting 10× more names — see the coverage note above.")

# ── HALF 1: funnel over time ─────────────────────────────────────────────────
st.markdown("---")
st.subheader("Half 1 — the funnel over time")
st.caption("Every stage, every week. Color = drop-off rate (share of names still alive entering that "
           "stage that got cut there). Each cell also prints its sample size (N = names alive entering "
           "the stage that week) — a rate with too few names to trust is shown blank, not as a misleading zero.")

if not funnel_week.empty:
    weeks = sorted(funnel_week["week"].unique())
    # 'traded' is excluded here: remaining_before for the LAST pipeline stage
    # always equals its own count, so its rate is a trivial 1.0 whenever
    # nonzero — a solid red row that carries no information. Still in the
    # raw counts table below.
    stages_present = [k for k, _, _ in fdat.FUNNEL_STAGES
                       if k in set(funnel_week["stage"]) and k != "traded"]

    def _short(label: str, n: int = 34) -> str:
        return label if len(label) <= n else label[: n - 1].rstrip() + "…"

    z, text, hover = [], [], []
    for key in stages_present:
        sub = funnel_week[funnel_week["stage"] == key].set_index("week")
        z_row, t_row, h_row = [], [], []
        for w in weeks:
            if w not in sub.index:
                z_row.append(None); t_row.append(""); h_row.append(f"{w.date()} · no data")
                continue
            row = sub.loc[w]
            n = int(row["remaining_before"])
            if n < fdat.MIN_STAGE_N:
                z_row.append(None)
                t_row.append(f"N={n}")
                h_row.append(f"week of {w.date()}<br>{fdat.STAGE_LABEL[key]}<br>N={n} — too few to trust")
            else:
                rate = row["drop_off_rate"]
                z_row.append(rate)
                t_row.append(f"{rate*100:.0f}%<br>N={n}")
                h_row.append(f"week of {w.date()}<br>{fdat.STAGE_LABEL[key]}<br>cut {int(row['count'])} of {n} ({rate*100:.0f}%)")
        z.append(z_row); text.append(t_row); hover.append(h_row)

    colorscale = [
        [0.0, C["surface2"]],
        [1.0, C["negative"]],
    ]
    fig_hm = go.Figure(go.Heatmap(
        z=z, text=text, texttemplate="%{text}",
        textfont={"size": 8.5, "color": C["text"]},
        customdata=hover, hovertemplate="%{customdata}<extra></extra>",
        colorscale=colorscale, zmin=0, zmax=1, showscale=True,
        colorbar=dict(title="drop-off", tickformat=".0%", tickfont=dict(color=C["text_muted"]), title_font=dict(color=C["text_muted"])),
        xgap=2, ygap=2,
    ))
    is_limited = [k in fdat._INSTRUMENTATION_LIMITED_STAGES for k in stages_present]
    yticktext = [
        (f"⚠ {_short(fdat.STAGE_LABEL[k])}" if lim else _short(fdat.STAGE_LABEL[k]))
        for k, lim in zip(stages_present, is_limited)
    ]
    fig_hm.update_layout(
        paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
        font=dict(color=C["text_dim"], size=10, family="JetBrains Mono"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=max(360, 26 * len(stages_present) + 60),
        xaxis=dict(
            tickmode="array", tickvals=list(range(len(weeks))),
            ticktext=[w.strftime("%m-%d") for w in weeks],
            showgrid=False, zeroline=False, fixedrange=True, side="top",
            tickfont=dict(color=C["text_muted"], size=9),
        ),
        yaxis=dict(
            tickmode="array", tickvals=list(range(len(stages_present))),
            ticktext=yticktext, autorange="reversed",
            showgrid=False, zeroline=False, fixedrange=True,
            tickfont=dict(color=C["text_muted"], size=10),
        ),
    )
    st.plotly_chart(fig_hm, width='stretch', config=CHART_CONFIG)
    st.caption("⚠ = stage depends on a source table with a data-coverage cliff (see the note above) — "
               "early weeks are blank/thin by construction, not a real trend.")
else:
    st.info("No weekly funnel data available.")

# ── Which gates moved ────────────────────────────────────────────────────────
st.markdown("#### Which gates actually moved")
st.caption(f"First half vs second half of the clean window ({min_date} → 2026-08-21) — the one stretch "
           "with a stable, comparable logged population. Ranked by |change|, largest first.")

if not moved.empty:
    clean_moved = moved[~moved["insufficient_n"]].copy()
    clean_moved["Stage"] = clean_moved["label"]
    clean_moved["First half"] = (clean_moved["first_half_rate"] * 100).map(lambda v: f"{v:.1f}%" if pd.notna(v) else "—")
    clean_moved["Second half"] = (clean_moved["second_half_rate"] * 100).map(lambda v: f"{v:.1f}%" if pd.notna(v) else "—")
    clean_moved["Change"] = clean_moved["delta_pp"].map(lambda v: f"{v:+.1f}pp" if pd.notna(v) else "—")
    clean_moved["N (1st / 2nd half)"] = clean_moved.apply(lambda r: f"{r['first_half_n']:,} / {r['second_half_n']:,}", axis=1)

    def _note(row):
        if row["instrumentation_limited"]:
            return "capture-limited — see coverage note"
        return fdat.STAGE_CAVEATS.get(row["stage"], "")
    clean_moved["Note"] = clean_moved.apply(_note, axis=1)
    st.dataframe(
        clean_moved[["Stage", "First half", "Second half", "Change", "N (1st / 2nd half)", "Note"]]
        .reset_index(drop=True),
        width='stretch', hide_index=True,
    )
    excluded = moved[moved["insufficient_n"]]
    if len(excluded):
        st.caption(f"{len(excluded)} stage(s) excluded from the ranking above — fewer than "
                   f"{fdat.MIN_STAGE_N} tickers reached them in one or both halves of the window "
                   f"({', '.join(excluded['label'].str[:24])}).")
    st.markdown(
        """<div class="callout">
        <b>The gate that actually moved: below_bar.</b> 138 → 166 graded/scored names landed under
        the alert bar between the two halves of the clean window (34.3% → 49.7% of everyone who got
        that far) — and the bar itself was a constant "score &lt; 50" the entire time, no threshold
        changed. Half the names that reach scoring now come in under the bar, versus a third before.<br><br>
        <b>adv_low</b> (+6.4pp, 114 → 133 names) moved too, but its rate roughly doubling is mostly
        the shrinking daily population in the denominator (see below), not a stricter dollar-volume
        floor — the raw count barely grew.<br><br>
        <b>Three rows on the table above are NOT independent gate moves</b> — excluded from the
        headline for this reason:<br>
        • <b>filter_other</b> (-13.5pp) / <b>session_rvol</b> (+5.7pp) are the SAME gate: April 2026's
        low-relative-volume rejections used wording that predated the session_rvol text match, so
        they landed in filter_other; from ~May the message changed and they were correctly bucketed.
        Pooled, the real combined move is <b>8.6% → 5.6% (-3.0pp)</b>.<br>
        • <b>top20</b> (-28.8pp) is a cap, not a gate — the daily scanned population roughly halved
        across the window (46 days averaging 42/day → 47 days averaging 23/day starting mid-June),
        which is the real story there.<br>
        • <b>blocked</b> (-6.7pp, 167 → 97 names) is a post-alert fill rate, not a rejection gate —
        and the 2026-07-17 live-account cutover sits inside the second half, a plausible confound of
        its own.<br><br>
        <b>Open question, not answered by this data:</b> the scanned population halving mid-June
        (42/day → 23/day) drives several of the rate moves above — summer seasonality or the scan
        itself admitting fewer names is not something this log can tell apart; worth a look.
        </div>""",
        unsafe_allow_html=True,
    )
else:
    st.info("Not enough history yet for a first-half/second-half comparison.")

with st.expander("Since 2026-08-24 (new capture regime — too short to call a trend yet)"):
    if not post_change.empty:
        pc = post_change.copy()
        pc["Stage"] = pc["label"]
        pc["Drop-off rate"] = (pc["drop_off_rate"] * 100).map(lambda v: f"{v:.1f}%" if pd.notna(v) else "—")
        pc["N"] = pc["remaining_before"]
        pc["Cut"] = pc["count"]
        pc["Days"] = pc["n_days"]
        st.dataframe(pc[["Stage", "Cut", "N", "Drop-off rate", "Days"]].reset_index(drop=True),
                     width='stretch', hide_index=True)
        st.caption("Only 10 scan days in this window as of the snapshot — revisit once it has enough "
                   "history to split into halves like the clean window above.")
    else:
        st.info("No data in this window.")

with st.expander("Raw counts by day (stage × day) — download"):
    pivot = funnel_day.pivot_table(index="date", columns="label", values="count", aggfunc="sum").fillna(0).astype(int)
    pivot = pivot[[fdat.STAGE_LABEL[k] for k, _, _ in fdat.FUNNEL_STAGES if fdat.STAGE_LABEL[k] in pivot.columns]]
    st.dataframe(pivot, width='stretch')
    st.download_button(
        "Download CSV", pivot.to_csv().encode("utf-8"),
        file_name="ep_funnel_by_day.csv", mime="text/csv",
    )

# ── HALF 2: scorecard ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Half 2 — the scorecard: what each gate cost")
st.caption("For every rejecting stage: how many names it rejected, how many of those have a settled, "
           "trustworthy outcome (fresh + a real setup at the open — same gates /scanned uses), and how "
           "many ran ≥20% high within 5 sessions anyway. 'traded' is excluded — a filled position isn't a miss.")

if not score.empty:
    sc = score.copy()
    sc["Stage"] = sc["label"]
    sc["Rejected"] = sc["n_rejected"]
    sc["Measurable"] = sc["n_measurable"]
    sc["Ran ≥20%"] = sc["n_ran"]
    sc["Cost rate"] = sc.apply(
        lambda r: (f"{r['ran_rate']*100:.0f}%" if pd.notna(r["ran_rate"]) and not r["insufficient_n"]
                    else f"n/a (N={int(r['n_measurable'])})"),
        axis=1,
    )
    sc["Kind"] = sc["is_gate"].map(lambda g: "gate" if g else "post-alert")

    base_measurable = int(sc["n_measurable"].sum())
    base_ran = int(sc["n_ran"].sum())
    base_rate = base_ran / base_measurable if base_measurable else float("nan")
    st.markdown(
        f"""<div class="callout">
        <b>Baseline: across every rejecting stage, {base_ran} of {base_measurable} measurable rejects
        ({base_rate*100:.0f}%) ran ≥20% high within 5 sessions.</b> Judge each stage against that, not
        zero — a gate whose job is "already extended" or "too volatile" is partly selecting FOR names
        that move, so a high rate there is expected, not a leak.
        </div>""",
        unsafe_allow_html=True,
    )

    st.dataframe(
        sc[["Stage", "Kind", "Rejected", "Measurable", "Ran ≥20%", "Cost rate"]]
        .sort_values("Ran ≥20%", ascending=False).reset_index(drop=True),
        width='stretch', hide_index=True,
    )
    st.caption(f"'Kind' = gate (a mechanical/grading cut) vs post-alert (alerted but never filled) — "
               f"both are shown per the brief (every stage, not a favoured subset); they cost differently "
               f"(a gate never gave the name a chance, a post-alert miss is an execution gap). "
               f"extension (55%) and atr_high (59%) sit well above the {base_rate*100:.0f}% baseline; "
               f"below_bar (12%) sits well below it — the scoring gate is cutting names that mostly don't run.")

    top_by_rate = sc[sc["n_measurable"] >= fdat.MIN_SCORECARD_N].sort_values("ran_rate", ascending=False).head(5)
    if len(top_by_rate):
        st.markdown("##### Highest-cost gates (by rate, N ≥ %d)" % fdat.MIN_SCORECARD_N)
        for _, r in top_by_rate.iterrows():
            with st.expander(f"{r['label']} — {r['ran_rate']*100:.0f}% of measurable names ran ≥20% (N={int(r['n_measurable'])})"):
                runners = fdat.top_runners_by_stage(r["stage"])
                if runners.empty:
                    st.caption("No runners recorded for this stage.")
                else:
                    disp = runners.copy()
                    disp["date"] = disp["date"].dt.date
                    disp["gap_pct"] = disp["gap_pct"].map(lambda v: f"{v:+.0f}%" if pd.notna(v) else "?")
                    disp["max_high_5d"] = (disp["max_high_5d"] * 100).map(lambda v: f"{v:+.0f}%")
                    disp["ret_5d"] = disp["ret_5d"].map(lambda v: f"{v*100:+.0f}%" if pd.notna(v) else "—")
                    disp = disp.rename(columns={
                        "date": "Date", "ticker": "Ticker", "gap_pct": "Gap",
                        "max_high_5d": "High (5d)", "ret_5d": "Settled (5d)",
                        "outcome_text": "Outcome", "filter_reason": "Cut reason",
                    })
                    st.dataframe(disp[["Date", "Ticker", "Gap", "High (5d)", "Settled (5d)", "Cut reason"]],
                                 width='stretch', hide_index=True)
else:
    st.info("No scorecard data available.")

st.markdown("---")
st.caption(
    "Read-only reporting — changes no gate, threshold, or trading behavior (#589). "
    "Stage list ported from agents/market_intelligence/scanned_report.py in apollo_the_wise; "
    "cross-checked against the live /scanned renderer for 2026-09-03, identical stage-for-stage. "
    "Regenerate apollo_funnel_snapshot.json (SELECT-only export) to refresh."
)
