"""Dixon-Coles goal model, plus an Elo bridge for cross-league cup games."""
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

from . import config

MAX_GOALS = 10


class DixonColes:
    """log(home goals rate) = mu + home + att[home] + dfn[away]
       log(away goals rate) = mu        + att[away] + dfn[home]
    att > 0 scores more than average, dfn > 0 concedes more than average."""

    def __init__(self, xi=None, reg=None):
        self.xi = config.XI if xi is None else xi
        self.reg = config.REG if reg is None else reg
        self.fitted = False

    def fit(self, df, ref_date):
        df = df[df["date"] < ref_date]
        if len(df) < 50:
            return self
        days = (ref_date - df["date"]).dt.days.to_numpy(float)
        w = np.exp(-self.xi * days)
        keep = w > 1e-3
        df, w = df[keep], w[keep]

        # Teams in the league this season (or most recent season) plus any history.
        self.teams = sorted(set(df["home"]) | set(df["away"]))
        ix = {t: i for i, t in enumerate(self.teams)}
        hi = df["home"].map(ix).to_numpy()
        ai = df["away"].map(ix).to_numpy()
        hg = df["hg"].to_numpy(float)
        ag = df["ag"].to_numpy(float)
        n = len(self.teams)

        # Thin-history teams (usually promoted) get a below-average prior.
        exposure = np.bincount(hi, w, n) + np.bincount(ai, w, n)
        shrink = np.clip(1 - exposure / 12.0, 0, 1)
        prior_att = config.NEW_TEAM_PRIOR * shrink
        prior_dfn = -config.NEW_TEAM_PRIOR * shrink

        def nll(theta):
            mu, home = theta[0], theta[1]
            att, dfn = theta[2:2 + n], theta[2 + n:]
            llh = mu + home + att[hi] + dfn[ai]
            lla = mu + att[ai] + dfn[hi]
            lh, la = np.exp(llh), np.exp(lla)
            ll = np.sum(w * (hg * llh - lh + ag * lla - la))
            pen = 0.5 * self.reg * (np.sum((att - prior_att) ** 2) + np.sum((dfn - prior_dfn) ** 2))
            rh, ra = w * (hg - lh), w * (ag - la)
            g = np.empty_like(theta)
            g[0] = rh.sum() + ra.sum()
            g[1] = rh.sum()
            g[2:2 + n] = np.bincount(hi, rh, n) + np.bincount(ai, ra, n) - self.reg * (att - prior_att)
            g[2 + n:] = np.bincount(ai, rh, n) + np.bincount(hi, ra, n) - self.reg * (dfn - prior_dfn)
            return -(ll - pen), -g

        x0 = np.zeros(2 + 2 * n)
        x0[0] = np.log(max(ag.mean(), 0.5))
        x0[1] = 0.2
        res = minimize(nll, x0, jac=True, method="L-BFGS-B")
        th = res.x
        self.mu, self.home = th[0], th[1]
        self.att = dict(zip(self.teams, th[2:2 + n]))
        self.dfn = dict(zip(self.teams, th[2 + n:]))

        # Low-score correlation (rho), fitted given the rates.
        lh = np.exp(self.mu + self.home + th[2:2 + n][hi] + th[2 + n:][ai])
        la = np.exp(self.mu + th[2:2 + n][ai] + th[2 + n:][hi])

        def rho_nll(r):
            t = _tau(hg, ag, lh, la, r)
            return -np.sum(w * np.log(np.clip(t, 1e-9, None)))

        self.rho = minimize_scalar(rho_nll, bounds=(-0.2, 0.2), method="bounded").x
        self.n_matches = len(df)
        self.fitted = True
        return self

    def knows(self, team):
        return self.fitted and team in self.att

    def rates(self, home, away):
        lh = np.exp(self.mu + self.home + self.att[home] + self.dfn[away])
        la = np.exp(self.mu + self.att[away] + self.dfn[home])
        return lh, la

    def profile(self, team):
        """Average-opponent goal rates for one team: (scored, conceded) per game."""
        base = np.exp(self.mu + self.home / 2)
        return base * np.exp(self.att[team]), base * np.exp(self.dfn[team])


def _tau(x, y, lh, la, rho):
    t = np.ones_like(lh, dtype=float)
    m = (x == 0) & (y == 0); t[m] = 1 - lh[m] * la[m] * rho
    m = (x == 0) & (y == 1); t[m] = 1 + lh[m] * rho
    m = (x == 1) & (y == 0); t[m] = 1 + la[m] * rho
    m = (x == 1) & (y == 1); t[m] = 1 - rho
    return t


def score_matrix(lh, la, rho=0.0):
    g = np.arange(MAX_GOALS + 1)
    M = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    M[0, 0] *= 1 - lh * la * rho
    M[0, 1] *= 1 + lh * rho
    M[1, 0] *= 1 + la * rho
    M[1, 1] *= 1 - rho
    return M / M.sum()


def outcome_probs(M):
    return {"Home": float(np.tril(M, -1).sum()),
            "Draw": float(np.trace(M)),
            "Away": float(np.triu(M, 1).sum())}


def over_prob(M, line):
    """P(total goals > line). Only meaningful for x.5 lines."""
    g = np.arange(MAX_GOALS + 1)
    tot = g[:, None] + g[None, :]
    return float(M[tot > line].sum())


def elo_expectancy(diff):
    return 1.0 / (1.0 + 10 ** (-diff / 400.0))


def bridge_rates(total, elo_diff, rho=-0.05):
    """Split an expected total into home/away rates so that the model's
    win expectancy (W + D/2) matches what the Elo gap implies."""
    target = elo_expectancy(elo_diff)

    def we(s):
        lh = max((total + s) / 2, 0.05)
        la = max((total - s) / 2, 0.05)
        p = outcome_probs(score_matrix(lh, la, rho))
        return p["Home"] + 0.5 * p["Draw"]

    lo, hi = -total + 0.1, total - 0.1
    for _ in range(40):
        mid = (lo + hi) / 2
        if we(mid) < target:
            lo = mid
        else:
            hi = mid
    s = (lo + hi) / 2
    return max((total + s) / 2, 0.05), max((total - s) / 2, 0.05)
