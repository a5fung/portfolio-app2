"""#639 — is the top-30's weekly churn REAL ROTATION, or identity instability?

Two thirds of a 30-slot board is a name that was not there seven days earlier, every week for two
months. The DoD is the SPLIT, not the percentage: for each week's entrants, is the cohort genuinely
new — its tickers were not together on the board under any name — or is it an existing cohort
wearing a changed name or membership?

$0: reads the canonical weekly grid the dashboard already computes. Read-only.

Four classes, most-instability-first:
  RENAMED      the same ticker basket was on the board last week under a DIFFERENT canonical id
  RETURNED     this canonical id was on the board before, left, and came back
  PROMOTED     this canonical id existed below the cut and climbed in — real movement, known cohort
  NEW          neither the id nor its basket has been on the board before
Only NEW is rotation in the sense the board implies. RENAMED is the engine losing identity.
"""
import sys
from collections import defaultdict

import theme_data

BOARD = 30
JACCARD = 0.60          # basket-overlap floor for "the same cohort under another name"


def _tickers(cell) -> frozenset:
    if isinstance(cell, (list, tuple, set, frozenset)):
        return frozenset(str(t).strip().upper() for t in cell if str(t).strip())
    if isinstance(cell, str):
        return frozenset(t.strip().upper() for t in cell.replace(";", ",").split(",") if t.strip())
    return frozenset()


def main() -> int:
    g = theme_data.get_canonical_weekly_grid()
    g = g[g["week_rank"].notna()]
    weeks = sorted(g["week_start"].unique())
    board, baskets, seen_on_board, ever_seen = {}, {}, set(), set()
    for w in weeks:
        wk = g[g["week_start"] == w]
        board[w] = set(wk[wk["week_rank"] <= BOARD]["canonical_id"])
        baskets[w] = {r["canonical_id"]: _tickers(r["tickers"]) for _, r in wk.iterrows()}

    tally = defaultdict(int)
    rows = []
    for i, w in enumerate(weeks):
        if i == 0:
            seen_on_board |= board[w]; ever_seen |= set(baskets[w]); continue
        prev = weeks[i - 1]
        entrants = board[w] - board[prev]
        per = defaultdict(list)
        for cid in entrants:
            mine = baskets[w].get(cid, frozenset())
            twin = None
            if mine:
                for other in board[prev]:
                    theirs = baskets[prev].get(other, frozenset())
                    if theirs and len(mine & theirs) / len(mine | theirs) >= JACCARD:
                        twin = other
                        break
            # ⚠ LAST WEEK IS NOT ENOUGH. The first run scored 2026-06-22 as 24 entrants, ALL "new",
            # and RENAMED fell to zero from that week onward — the signature of a canonical_id
            # RESET, not of rotation. A cohort that existed in April under one id and reappears in
            # July under another reads NEW to an id-only test. So a basket is also checked against
            # EVERY basket ever seen, at any rank, in any earlier week.
            hist_twin = None
            if mine and not twin:
                for pw in weeks[:i]:
                    for other, theirs in baskets[pw].items():
                        if theirs and len(mine & theirs) / len(mine | theirs) >= JACCARD:
                            hist_twin = (pw, other)
                            break
                    if hist_twin:
                        break
            if twin:
                klass = "RENAMED"
            elif hist_twin:
                klass = "REAPPEARED"
            elif cid in seen_on_board:
                klass = "RETURNED"
            elif cid in ever_seen:
                klass = "PROMOTED"
            else:
                klass = "NEW"
            tally[klass] += 1
            per[klass].append(cid)
        rows.append((w, len(entrants), {k: len(v) for k, v in per.items()}))
        seen_on_board |= board[w]; ever_seen |= set(baskets[w])

    print(f"weeks {weeks[0]} → {weeks[-1]}  ·  board size {BOARD}  ·  basket-twin Jaccard ≥ {JACCARD}\n")
    print(f"{'week':<12}{'entrants':>9}  {'RENAMED':>8}{'REAPPEARED':>11}{'RETURNED':>9}{'PROMOTED':>9}{'NEW':>5}")
    for w, n, per in rows:
        print(f"{str(w):<12}{n:>9}  {per.get('RENAMED',0):>8}{per.get('REAPPEARED',0):>11}"
              f"{per.get('RETURNED',0):>9}{per.get('PROMOTED',0):>9}{per.get('NEW',0):>5}")
    total = sum(tally.values())
    print(f"\nTOTAL entrants over {len(rows)} transitions: {total}")
    for k in ("RENAMED", "REAPPEARED", "RETURNED", "PROMOTED", "NEW"):
        print(f"  {k:<9} {tally[k]:>4}  {tally[k]/total*100:5.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
