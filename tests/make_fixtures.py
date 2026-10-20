"""Generate realistic fake data in the exact formats of football-data.co.uk,
ClubElo and The Odds API, so the whole pipeline can be tested offline.
Usage: python tests/make_fixtures.py OUT_DIR"""
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bot import model  # noqa: E402
from bot.history import season_codes  # noqa: E402

rng = np.random.default_rng(7)
TODAY = date(2026, 10, 14)

# football-data name -> (Odds API name, ClubElo name)
TEAMS = {
    "E0": {"Arsenal": ("Arsenal", "Arsenal"), "Man City": ("Manchester City", "Man City"),
           "Liverpool": ("Liverpool", "Liverpool"), "Chelsea": ("Chelsea", "Chelsea"),
           "Tottenham": ("Tottenham Hotspur", "Tottenham"), "Man United": ("Manchester United", "Man United"),
           "Newcastle": ("Newcastle United", "Newcastle"), "Aston Villa": ("Aston Villa", "Aston Villa"),
           "Brighton": ("Brighton and Hove Albion", "Brighton"), "Nott'm Forest": ("Nottingham Forest", "Forest"),
           "West Ham": ("West Ham United", "West Ham"), "Wolves": ("Wolverhampton Wanderers", "Wolves"),
           "Everton": ("Everton", "Everton"), "Fulham": ("Fulham", "Fulham"), "Brentford": ("Brentford", "Brentford"),
           "Crystal Palace": ("Crystal Palace", "Crystal Palace"), "Bournemouth": ("Bournemouth", "Bournemouth"),
           "Leeds": ("Leeds United", "Leeds"), "Sunderland": ("Sunderland", "Sunderland"), "Burnley": ("Burnley", "Burnley")},
    "SP1": {"Real Madrid": ("Real Madrid", "Real Madrid"), "Barcelona": ("Barcelona", "Barcelona"),
            "Ath Madrid": ("Atlético Madrid", "Atletico"), "Ath Bilbao": ("Athletic Bilbao", "Bilbao"),
            "Betis": ("Real Betis", "Betis"), "Sociedad": ("Real Sociedad", "Sociedad"),
            "Villarreal": ("Villarreal", "Villarreal"), "Sevilla": ("Sevilla", "Sevilla"),
            "Valencia": ("Valencia", "Valencia"), "Celta": ("Celta Vigo", "Celta"), "Girona": ("Girona", "Girona"),
            "Osasuna": ("CA Osasuna", "Osasuna"), "Getafe": ("Getafe", "Getafe"), "Mallorca": ("Mallorca", "Mallorca"),
            "Vallecano": ("Rayo Vallecano", "Rayo Vallecano"), "Espanol": ("Espanyol", "Espanyol"),
            "Alaves": ("Alavés", "Alaves"), "Levante": ("Levante", "Levante"), "Elche": ("Elche CF", "Elche"),
            "Oviedo": ("Real Oviedo", "Oviedo")},
    "D1": {"Bayern Munich": ("Bayern Munich", "Bayern"), "Leverkusen": ("Bayer Leverkusen", "Leverkusen"),
           "Dortmund": ("Borussia Dortmund", "Dortmund"), "RB Leipzig": ("RB Leipzig", "RB Leipzig"),
           "Stuttgart": ("VfB Stuttgart", "Stuttgart"), "Ein Frankfurt": ("Eintracht Frankfurt", "Frankfurt"),
           "Freiburg": ("SC Freiburg", "Freiburg"), "Wolfsburg": ("VfL Wolfsburg", "Wolfsburg"),
           "M'gladbach": ("Borussia Monchengladbach", "Gladbach"), "Hoffenheim": ("TSG Hoffenheim", "Hoffenheim"),
           "Mainz": ("FSV Mainz 05", "Mainz"), "Werder Bremen": ("Werder Bremen", "Werder"),
           "Union Berlin": ("Union Berlin", "Union Berlin"), "Augsburg": ("FC Augsburg", "Augsburg"),
           "Heidenheim": ("1. FC Heidenheim", "Heidenheim"), "St Pauli": ("FC St. Pauli", "St Pauli"),
           "FC Koln": ("1. FC Köln", "Koeln"), "Hamburg": ("Hamburger SV", "Hamburg")},
    "I1": {"Inter": ("Inter Milan", "Inter"), "Milan": ("AC Milan", "Milan"), "Napoli": ("Napoli", "Napoli"),
           "Juventus": ("Juventus", "Juventus"), "Atalanta": ("Atalanta BC", "Atalanta"), "Roma": ("AS Roma", "Roma"),
           "Lazio": ("Lazio", "Lazio"), "Fiorentina": ("Fiorentina", "Fiorentina"), "Bologna": ("Bologna", "Bologna"),
           "Torino": ("Torino", "Torino"), "Como": ("Como", "Como"), "Udinese": ("Udinese", "Udinese"),
           "Genoa": ("Genoa", "Genoa"), "Cagliari": ("Cagliari", "Cagliari"), "Verona": ("Hellas Verona", "Verona"),
           "Lecce": ("Lecce", "Lecce"), "Parma": ("Parma", "Parma"), "Sassuolo": ("Sassuolo", "Sassuolo"),
           "Pisa": ("Pisa", "Pisa"), "Cremonese": ("Cremonese", "Cremonese")},
    "F1": {"Paris SG": ("Paris Saint Germain", "Paris SG"), "Marseille": ("Marseille", "Marseille"),
           "Monaco": ("AS Monaco", "Monaco"), "Lille": ("Lille", "Lille"), "Lyon": ("Lyon", "Lyon"),
           "Nice": ("Nice", "Nice"), "Lens": ("RC Lens", "Lens"), "Rennes": ("Rennes", "Rennes"),
           "Strasbourg": ("Strasbourg", "Strasbourg"), "Brest": ("Brest", "Brest"), "Toulouse": ("Toulouse", "Toulouse"),
           "Nantes": ("Nantes", "Nantes"), "Auxerre": ("Auxerre", "Auxerre"), "Angers": ("Angers", "Angers"),
           "Le Havre": ("Le Havre", "Le Havre"), "Lorient": ("Lorient", "Lorient"), "Metz": ("Metz", "Metz"),
           "Paris FC": ("Paris FC", "Paris FC")},
    "N1": {"PSV Eindhoven": ("PSV Eindhoven", "PSV"), "Ajax": ("Ajax", "Ajax"), "Feyenoord": ("Feyenoord", "Feyenoord"),
           "AZ Alkmaar": ("AZ Alkmaar", "Alkmaar"), "Twente": ("FC Twente", "Twente"), "Utrecht": ("FC Utrecht", "Utrecht")},
    "P1": {"Benfica": ("Benfica", "Benfica"), "Porto": ("FC Porto", "Porto"), "Sp Lisbon": ("Sporting Lisbon", "Sporting"),
           "Sp Braga": ("Braga", "Braga"), "Guimaraes": ("Vitoria Guimaraes", "Guimaraes"), "Famalicao": ("Famalicao", "Famalicao")},
    "B1": {"Club Brugge": ("Club Brugge", "Brugge"), "Anderlecht": ("Anderlecht", "Anderlecht"),
           "Genk": ("Genk", "Genk"), "Gent": ("Gent", "Gent")},
    "SC0": {"Celtic": ("Celtic", "Celtic"), "Rangers": ("Rangers", "Rangers"), "Hearts": ("Hearts", "Hearts"), "Aberdeen": ("Aberdeen", "Aberdeen")},
    "T1": {"Galatasaray": ("Galatasaray", "Galatasaray"), "Fenerbahce": ("Fenerbahçe", "Fenerbahce"),
           "Besiktas": ("Beşiktaş", "Besiktas"), "Trabzonspor": ("Trabzonspor", "Trabzonspor")},
    "G1": {"Olympiakos": ("Olympiacos", "Olympiakos"), "PAOK": ("PAOK", "PAOK"), "AEK": ("AEK Athens", "AEK"), "Panathinaikos": ("Panathinaikos", "Panathinaikos")},
}
COUNTRY = {"E0": "ENG", "SP1": "ESP", "D1": "GER", "I1": "ITA", "F1": "FRA", "N1": "NED",
           "P1": "POR", "B1": "BEL", "SC0": "SCO", "T1": "TUR", "G1": "GRE"}
SPORT = {"E0": "soccer_epl", "SP1": "soccer_spain_la_liga", "D1": "soccer_germany_bundesliga",
         "I1": "soccer_italy_serie_a", "F1": "soccer_france_ligue_one"}

TRUE = {}   # (code, team) -> [att, dfn]
US_SLUG = {"E0": "EPL", "SP1": "La_liga", "D1": "Bundesliga", "I1": "Serie_A", "F1": "Ligue_1"}


def odds_from(probs, margin, noise):
    p = np.array(probs) * np.exp(rng.normal(0, noise, len(probs)))
    p = p / p.sum() * (1 + margin)
    return [round(float(1 / x), 2) for x in p]


def make_history(out):
    codes = season_codes(TODAY, 6)
    for code, teams in TEAMS.items():
        names = list(teams)
        n = len(names)
        att = rng.normal(0, 0.28, n)
        dfn = -0.6 * att + rng.normal(0, 0.15, n)
        mu, home, rho = 0.25, 0.25, -0.06
        for si, season in enumerate(codes):
            y = 2000 + int(season[:2])
            start = date(y, 8, 16)
            att = att * 0.9 + rng.normal(0, 0.08, n)
            dfn = dfn * 0.9 + rng.normal(0, 0.06, n)
            pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
            rng.shuffle(pairs)
            per_week = n // 2
            us_dates = []
            rows = ["Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,PSH,PSD,PSA,PSCH,PSCD,PSCA,P>2.5,P<2.5,PC>2.5,PC<2.5"]
            for k, (i, j) in enumerate(pairs):
                d = start + timedelta(days=7 * (k // per_week) + int(rng.integers(0, 3)))
                if d >= TODAY:
                    continue
                lh = np.exp(mu + home + att[i] + dfn[j])
                la = np.exp(mu + att[j] + dfn[i])
                M = model.score_matrix(lh, la, rho)
                p = model.outcome_probs(M)
                pov = model.over_prob(M, 2.5)
                hg, ag = divmod(int(rng.choice(M.size, p=M.ravel())), M.shape[1])
                o = odds_from([p["Home"], p["Draw"], p["Away"]], 0.025, 0.07)
                c = odds_from([p["Home"], p["Draw"], p["Away"]], 0.022, 0.03)
                ou = odds_from([pov, 1 - pov], 0.03, 0.07)
                cu = odds_from([pov, 1 - pov], 0.025, 0.03)
                us_dates.append({"id": str(k), "isResult": True, "datetime": f"{d:%Y-%m-%d} 19:00:00",
                                 "h": {"title": teams[names[i]][0]}, "a": {"title": teams[names[j]][0]},
                                 "goals": {"h": str(hg), "a": str(ag)},
                                 "xG": {"h": f"{lh * rng.lognormal(0, 0.25):.4f}", "a": f"{la * rng.lognormal(0, 0.25):.4f}"}})
                ftr = "H" if hg > ag else "A" if ag > hg else "D"
                rows.append(f"{code},{d:%d/%m/%Y},{names[i]},{names[j]},{hg},{ag},{ftr},"
                            f"{o[0]},{o[1]},{o[2]},{c[0]},{c[1]},{c[2]},{ou[0]},{ou[1]},{cu[0]},{cu[1]}")
            if len(rows) > 1:
                (out / f"fd_{season}_{code}.csv").write_text("\n".join(rows) + "\n", encoding="latin-1")
            if code in US_SLUG and us_dates:
                us_dates.append({"id": "future", "isResult": False, "datetime": "2026-12-01 19:00:00",
                                 "h": {"title": "x"}, "a": {"title": "y"}, "goals": {"h": None, "a": None},
                                 "xG": {"h": None, "a": None}})
                (out / f"us_{US_SLUG[code]}_{y}.json").write_text(json.dumps({"dates": us_dates, "teams": {}, "players": []}))
        for t, a, d in zip(names, att, dfn):
            TRUE[(code, t)] = (a, d)


def make_elo(out):
    rows = ["Rank,Club,Country,Level,Elo,From,To"]
    for code, teams in TEAMS.items():
        for fd, (_, elo_name) in teams.items():
            a, d = TRUE[(code, fd)]
            elo = 1600 + 420 * (a - d) / 2 + (80 if code in SPORT else -40)
            rows.append(f"None,{elo_name},{COUNTRY[code]},1,{elo:.1f},2026-10-10,2026-10-15")
    (out / "clubelo.csv").write_text("\n".join(rows) + "\n")


def book_market(home, away, p, pov, margin, noise):
    h = odds_from([p["Home"], p["Draw"], p["Away"]], margin, noise)
    t = odds_from([pov, 1 - pov], margin + 0.01, noise)
    return [{"key": "h2h", "outcomes": [{"name": home, "price": h[0]}, {"name": away, "price": h[2]},
                                        {"name": "Draw", "price": h[1]}]},
            {"key": "totals", "outcomes": [{"name": "Over", "price": t[0], "point": 2.5},
                                           {"name": "Under", "price": t[1], "point": 2.5},
                                           {"name": "Over", "price": 2.6, "point": 2.25}]}]


def make_event(sport, eid, home, away, ko, lh, la):
    M = model.score_matrix(lh, la, -0.06)
    p = model.outcome_probs(M)
    pov = model.over_prob(M, 2.5)
    books = [("pinnacle", 0.025, 0.04), ("betfair_ex_eu", 0.02, 0.05), ("draftkings", 0.05, 0.07),
             ("fanduel", 0.05, 0.07), ("betmgm", 0.055, 0.08)]
    return {"id": eid, "sport_key": sport, "commence_time": ko, "home_team": home, "away_team": away,
            "bookmakers": [{"key": k, "title": k, "markets": book_market(home, away, p, pov, m, nz)}
                           for k, m, nz in books]}


def make_odds(out):
    base = datetime(2026, 10, 17, 11, 30, tzinfo=timezone.utc)
    scores = {}
    for code, sport in SPORT.items():
        teams = list(TEAMS[code])
        evs = []
        for g in range(4):
            h, a = teams[2 * g], teams[2 * g + 1 + 6]
            ko = (base + timedelta(hours=3 * g)).strftime("%Y-%m-%dT%H:%M:%SZ")
            ah, dh = TRUE[(code, h)]; aa, da = TRUE[(code, a)]
            lh, la = np.exp(0.5 + ah + da), np.exp(0.25 + aa + dh)
            evs.append(make_event(sport, f"{code}-{g}", TEAMS[code][h][0], TEAMS[code][a][0], ko, lh, la))
        # one game outside the 36h window, one with an unknown team
        evs.append(make_event(sport, f"{code}-far", TEAMS[code][teams[0]][0], TEAMS[code][teams[1]][0],
                              (base + timedelta(days=5)).strftime("%Y-%m-%dT%H:%M:%SZ"), 1.4, 1.1))
        evs.append(make_event(sport, f"{code}-x", "Mystery FC", TEAMS[code][teams[3]][0],
                              (base + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ"), 1.4, 1.1))
        write_sport(out, sport, evs)
        scores[sport] = evs
    # Champions League + Europa League
    cups = {"soccer_uefa_champs_league": [("Benfica", "Manchester City"), ("PSV Eindhoven", "Inter Milan"),
                                          ("Real Madrid", "Club Brugge"), ("Bayern Munich", "Celtic")],
            "soccer_uefa_europa_league": [("Galatasaray", "AS Roma"), ("Olympiacos", "Real Betis"),
                                          ("Lyon", "FC Porto"), ("Tottenham Hotspur", "Sporting Lisbon")]}
    cup_base = datetime(2026, 10, 15, 19, 0, tzinfo=timezone.utc)
    for sport, games in cups.items():
        evs = [make_event(sport, f"{sport[-6:]}-{i}", h, a,
                          (cup_base + timedelta(hours=i % 2 * 2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                          float(rng.uniform(1.1, 2.0)), float(rng.uniform(0.8, 1.6)))
               for i, (h, a) in enumerate(games)]
        write_sport(out, sport, evs)
        scores[sport] = evs
    # completed scores for everything (used on a later run)
    for sport, evs in scores.items():
        res = []
        for e in evs:
            hg, ag = int(rng.integers(0, 4)), int(rng.integers(0, 3))
            res.append({"id": e["id"], "sport_key": sport, "commence_time": e["commence_time"], "completed": True,
                        "home_team": e["home_team"], "away_team": e["away_team"],
                        "scores": [{"name": e["home_team"], "score": str(hg)}, {"name": e["away_team"], "score": str(ag)}]})
        (out / f"scores_{sport}.json").write_text(json.dumps(res))


def write_sport(out, sport, evs):
    (out / f"odds_{sport}.json").write_text(json.dumps(evs))
    slim = [{k: e[k] for k in ("id", "sport_key", "commence_time", "home_team", "away_team")} for e in evs]
    (out / f"events_{sport}.json").write_text(json.dumps(slim))


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    make_history(out)
    make_elo(out)
    make_odds(out)
    print("fixtures written to", out)
