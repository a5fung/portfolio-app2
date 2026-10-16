"""Pin tests for the Performance-tab contribution column (2026-10-14, "why is the contribution %
in the thousands, looks wrong").

The fixture is shaped like his YTD screenshot: IBKR's loss nearly cancels the other accounts' gains,
so the portfolio's total gain is close to zero. The OLD column divided each account's gain by that
total and blew up into the thousands. The NEW column is points of the portfolio's own return: it is
bounded, and it adds up to the portfolio's return.
"""
import pandas as pd
import pytest

from attribution import attribution_rows

# end value, YTD change, from his 2026-10-14 screenshot; start = end / (1 + change), no cash flows
_SCREEN = {
    "IBKR": (1_200_000, -0.115), "Brokeragelink": (880_000, 0.096), "Fidelity": (610_000, 0.083),
    "401k": (320_000, 0.147), "Robinhood": (230_000, 0.075), "Schwab": (150_000, 0.208),
    "Crypto": (60_000, -0.507), "Savings": (25_000, 0.0),
}
ORDER = list(_SCREEN)


def _frame(flows: dict | None = None) -> pd.DataFrame:
    flows = flows or {}
    rows = []
    for acct, (end, chg) in _SCREEN.items():
        start = end / (1 + chg)
        rows.append({"Date": pd.Timestamp("2026-01-02"), "Account": acct, "Total Value": start, "W/D": 0.0})
        rows.append({"Date": pd.Timestamp("2026-10-13"), "Account": acct, "Total Value": end,
                     "W/D": flows.get(acct, 0.0)})
    return pd.DataFrame(rows)


def _by_acct(rows):
    return {r["Account"]: r for r in rows}


def test_contribution_is_points_of_the_portfolio_return_and_sums_to_it():
    df = _frame()
    rows = attribution_rows(df, ORDER)
    start_total = df[df["Date"] == df["Date"].min()]["Total Value"].sum()
    portfolio_return = sum(r["$ Change"] for r in rows) / start_total * 100
    assert sum(r["Contribution"] for r in rows) == pytest.approx(portfolio_return, abs=1e-9)
    assert all(abs(r["Contribution"]) < 10 for r in rows)          # bounded: no account moves it >10 pts
    by = _by_acct(rows)
    assert by["IBKR"]["Contribution"] == pytest.approx(-4.5, abs=0.1)   # -$156K on ~$3.49M
    assert by["Crypto"]["Contribution"] == pytest.approx(-1.8, abs=0.1)
    assert by["Savings"]["Contribution"] == 0


def test_signs_follow_the_account_gain_not_the_portfolio_sign():
    """The old form flipped every sign when the total gain was negative (gainers read -4891%)."""
    rows = _by_acct(attribution_rows(_frame(), ORDER))
    assert rows["Brokeragelink"]["Contribution"] > 0 and rows["IBKR"]["Contribution"] < 0


def test_deposits_are_excluded_from_the_gain_and_half_weighted_in_the_base():
    """A $100K deposit into Fidelity adds no 'gain' and its own % change keeps the Modified Dietz
    convention; the contributions still sum to the portfolio's Modified Dietz return."""
    df = _frame({"Fidelity": 100_000.0})
    df.loc[(df["Account"] == "Fidelity") & (df["Date"] == df["Date"].max()), "Total Value"] += 100_000
    rows = attribution_rows(df, ORDER)
    plain = _by_acct(attribution_rows(_frame(), ORDER))
    assert _by_acct(rows)["Fidelity"]["$ Change"] == pytest.approx(plain["Fidelity"]["$ Change"])
    start_total = df[df["Date"] == df["Date"].min()]["Total Value"].sum()
    md_return = sum(r["$ Change"] for r in rows) / (start_total + 0.5 * 100_000) * 100
    assert sum(r["Contribution"] for r in rows) == pytest.approx(md_return, abs=1e-9)


def test_an_empty_portfolio_base_reads_zero_not_a_division_error():
    df = pd.DataFrame([{"Date": pd.Timestamp("2026-01-02"), "Account": "IBKR", "Total Value": 0.0, "W/D": 0.0}])
    assert attribution_rows(df, ["IBKR"])[0]["Contribution"] == 0
