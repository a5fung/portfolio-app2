"""Performance attribution rows for the Performance tab (pure pandas, no Streamlit).

CONTRIBUTION is each account's market gain as a share of the WHOLE portfolio's starting capital, in
percentage points, so the column adds up to the portfolio's own return for the period.
"""
from __future__ import annotations

import pandas as pd


def attribution_rows(fdf: pd.DataFrame, account_order) -> list[dict]:
    """One row per account: start/end value, market $ change (net of deposits/withdrawals), the
    account's own Modified-Dietz % change, and its contribution in points of the portfolio's return.
    `fdf` has Date, Account, Total Value and W/D columns (the filtered frame the page already uses)."""
    first_snap = fdf[fdf["Date"] == fdf["Date"].min()]
    last_snap = fdf[fdf["Date"] == fdf["Date"].max()]
    total_net_wd = fdf["W/D"].sum()
    # Modified Dietz for the WHOLE portfolio: cash flows land at the period midpoint (0.5 weight),
    # the same convention each account's own % change uses below.
    total_md_denom = first_snap["Total Value"].sum() + 0.5 * total_net_wd

    rows = []
    for acct in account_order:
        s_val = first_snap.loc[first_snap["Account"] == acct, "Total Value"].sum()
        e_val = last_snap.loc[last_snap["Account"] == acct, "Total Value"].sum()
        net_wd = fdf.loc[fdf["Account"] == acct, "W/D"].sum()
        d_change = e_val - s_val - net_wd  # market gains only, excludes deposits
        md_denom = s_val + 0.5 * net_wd
        rows.append({
            "Account": acct,
            "Start Value": s_val,
            "Current Value": e_val,
            "$ Change": d_change,
            "% Change": d_change / md_denom * 100 if md_denom else 0,
            "Contribution": d_change / total_md_denom * 100 if total_md_denom else 0,
        })
    return rows
