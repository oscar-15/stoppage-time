"""Turn Pinnacle's match odds + its own totals line into a full scoreline model,
so we can get a fair price for a totals line Pinnacle doesn't show.

Example: US books offer Over/Under 2.5 but Pinnacle is dealing 2.75 (an Asian
line: half the stake on 2.5, half on 3.0). We find the expected goals for each
team that reproduce Pinnacle's 1X2 fair odds and make its 2.75 price a fair bet,
then read P(over 2.5) off that model.
"""
import numpy as np

from . import model

RHO = -0.05


def _totals_dist(M):
    n = M.shape[0]
    g = np.arange(n)
    tot = (g[:, None] + g[None, :]).ravel()
    return np.bincount(tot, M.ravel(), minlength=2 * n - 1)


def over_ev(dist, line, odds):
    """Expected profit per unit on Over `line` at decimal `odds`, Asian rules."""
    frac = round(line * 4) % 4
    if frac in (1, 3):                      # quarter line: split stake
        return 0.5 * over_ev(dist, line - 0.25, odds) + 0.5 * over_ev(dist, line + 0.25, odds)
    k = np.arange(len(dist))
    win = dist[k > line].sum()
    lose = dist[k < line].sum()             # exact hits on a whole line push
    return win * (odds - 1) - lose


def fit(h2h_fair, line, over_odds_fair):
    """Find (lambda_home, lambda_away) matching Pinnacle's fair 1X2 and totals price."""
    target_sup = h2h_fair[0] - h2h_fair[2]
    T, s = 2.6, 0.0
    for _ in range(8):
        lo, hi = -T + 0.05, T - 0.05        # supremacy, given total
        for _ in range(40):
            mid = (lo + hi) / 2
            p = model.outcome_probs(model.score_matrix((T + mid) / 2, (T - mid) / 2, RHO))
            if p["Home"] - p["Away"] < target_sup:
                lo = mid
            else:
                hi = mid
        s = (lo + hi) / 2
        lo, hi = max(abs(s) + 0.1, 0.6), 6.0   # total, given supremacy
        for _ in range(40):
            mid = (lo + hi) / 2
            M = model.score_matrix((mid + s) / 2, (mid - s) / 2, RHO)
            if over_ev(_totals_dist(M), line, over_odds_fair) < 0:
                lo = mid                     # Over is a bad bet -> need more goals
            else:
                hi = mid
        T = (lo + hi) / 2
    lh, la = (T + s) / 2, (T - s) / 2
    M = model.score_matrix(lh, la, RHO)
    p = model.outcome_probs(M)
    if abs(p["Home"] - h2h_fair[0]) > 0.03 or abs(p["Away"] - h2h_fair[2]) > 0.03:
        return None                          # model can't reproduce the market; don't trust it
    return M


def from_sharp(h2h_prices, totals):
    """h2h_prices: [home, draw, away] decimal. totals: {line: (over, under)}.
    Uses the most balanced totals line. Returns a score matrix or None."""
    if not h2h_prices or not totals:
        return None
    from .odds import devig_power
    line, (o, u) = min(totals.items(), key=lambda kv: abs(kv[1][0] - kv[1][1]))
    q = 1 / o / (1 / o + 1 / u)              # proportional de-vig for a two-way line
    M = fit(devig_power(h2h_prices), line, 1 / q)
    return (M, line) if M is not None else None
