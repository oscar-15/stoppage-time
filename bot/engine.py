"""Turn an Odds API event into model probabilities."""
import math
import traceback

import pandas as pd

from . import config, history, model, names


class Engine:
    def __init__(self, today, log):
        self.today = today
        self.log = log
        self.models = {}       # league code -> DixonColes
        self.team_league = {}  # football-data team -> league code (latest season wins)
        self.elo = None
        self.unmatched = set()

    def build(self, need_cups):
        ref = pd.Timestamp(self.today)
        leagues = dict(config.LEAGUES)
        if need_cups:
            leagues.update(config.SUPPORT_LEAGUES)
        for code, lg in leagues.items():
            try:
                df = history.load_league(lg["fd"], self.today, xg_slug=lg.get("us"))
                m = model.DixonColes().fit(df, ref)
            except Exception as e:     # one bad league shouldn't sink the run
                self.log(f"ERROR model {code}: {type(e).__name__}: {e}")
                traceback.print_exc()
                continue
            if not m.fitted:
                self.log(f"model: not enough data for {code}")
                continue
            self.models[code] = m
            latest = df[df["season"] == df["season"].max()]
            for t in set(latest["home"]) | set(latest["away"]):
                self.team_league[t] = code
            self.log(f"model: {code} fit on {m.n_matches} matches ({m.xg_share:.0%} with xG), home adv {m.home:.2f}, rho {m.rho:.3f}")
        if need_cups:
            try:
                self.elo = history.load_elo(self.today)
            except Exception as e:
                self.log(f"ERROR elo: {type(e).__name__}: {e}")
                self.elo = None
                return
            self.log(f"elo: {len(self.elo)} clubs")

    # --- domestic --------------------------------------------------------
    def _league_team(self, code, name):
        m = self.models.get(code)
        if not m:
            return None
        current = [t for t, c in self.team_league.items() if c == code]
        t = names.match(name, current) or names.match(name, m.teams)
        if t is None:
            self.unmatched.add(f"{code}: {name}")
        return t

    def rates(self, comp, event):
        """(lambda_home, lambda_away, rho) or None if we can't price it."""
        home, away = event["home_team"], event["away_team"]
        if comp in config.LEAGUES:
            h, a = self._league_team(comp, home), self._league_team(comp, away)
            if not h or not a:
                return None
            m = self.models[comp]
            lh, la = m.rates(h, a)
            return lh, la, m.rho
        return self._cup_rates(home, away)

    # --- European cups ---------------------------------------------------
    def _cup_team(self, name):
        fd = names.match(name, list(self.team_league))
        code = self.team_league.get(fd) if fd else None
        country = None
        if code:
            country = (config.LEAGUES.get(code) or config.SUPPORT_LEAGUES.get(code))["elo"]
        club = names.match_elo(fd, self.elo, country) if fd else None
        if club is None:   # e.g. a team from a league we don't load
            club = names.match_elo(name, self.elo, None)
        elo = None
        if club is not None:
            row = self.elo[self.elo["Club"] == club]
            elo = float(row["Elo"].iloc[0]) if len(row) else None
        prof = self.models[code].profile(fd) if code and self.models.get(code) and self.models[code].knows(fd) else None
        return elo, prof

    def _cup_rates(self, home, away):
        if self.elo is None or self.elo.empty:
            return None
        eh, ph = self._cup_team(home)
        ea, pa = self._cup_team(away)
        if eh is None or ea is None:
            if eh is None: self.unmatched.add(f"CUP elo: {home}")
            if ea is None: self.unmatched.add(f"CUP elo: {away}")
            return None
        if ph and pa:
            base = 1.35  # typical goals per team per game in top leagues
            lh0 = ph[0] * pa[1] / base * 1.10
            la0 = pa[0] * ph[1] / base * 0.90
            total = min(max(lh0 + la0, 2.0), 4.0)
        else:
            total = config.CUP_DEFAULT_TOTAL
        lh, la = model.bridge_rates(total, eh - ea + config.ELO_HOME_ADV)
        return lh, la, -0.05

    def probabilities(self, comp, event):
        r = self.rates(comp, event)
        if r is None:
            return None
        lh, la, rho = r
        if not (math.isfinite(lh) and math.isfinite(la)):
            return None
        M = model.score_matrix(lh, la, rho)
        p = model.outcome_probs(M)
        out = {("h2h", None, k): v for k, v in p.items()}
        out["_M"] = M
        out["_xg"] = (float(lh), float(la))
        return out
