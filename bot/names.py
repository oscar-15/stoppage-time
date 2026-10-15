"""Match team names across sources (Odds API vs football-data vs ClubElo)."""
import difflib
import json
import re
import unicodedata

from . import config

DROP = {"fc", "cf", "afc", "ac", "as", "sc", "ss", "ssc", "us", "sv", "vfb", "vfl", "tsg",
        "rc", "rcd", "cd", "ud", "sd", "ca", "fk", "sk", "bk", "club", "de", "calcio",
        "the", "1", "05", "04", "1899", "1900", "1909", "1846", "29", "osc", "ogc", "bc",
        "hsc", "aj", "ec", "sl", "fsv", "if", "kv", "krc", "rsc", "psv"}

# normalized source name -> football-data name. Add to data/aliases.json to extend.
BUILTIN = {
    "manchester city": "Man City", "manchester united": "Man United", "man utd": "Man United",
    "newcastle united": "Newcastle", "nottingham forest": "Nott'm Forest",
    "tottenham hotspur": "Tottenham", "west ham united": "West Ham",
    "wolverhampton wanderers": "Wolves", "brighton and hove albion": "Brighton",
    "brighton hove albion": "Brighton", "leeds united": "Leeds", "leicester city": "Leicester",
    "ipswich town": "Ipswich", "sheffield united": "Sheffield United", "sheffield utd": "Sheffield United",
    "athletic bilbao": "Ath Bilbao", "athletic": "Ath Bilbao", "atletico madrid": "Ath Madrid",
    "real betis": "Betis", "celta vigo": "Celta", "espanyol": "Espanol",
    "real sociedad": "Sociedad", "rayo vallecano": "Vallecano", "real oviedo": "Oviedo",
    "deportivo alaves": "Alaves", "alaves": "Alaves", "las palmas": "Las Palmas",
    "bayern munich": "Bayern Munich", "bayern munchen": "Bayern Munich",
    "borussia dortmund": "Dortmund", "eintracht frankfurt": "Ein Frankfurt",
    "koln": "FC Koln", "cologne": "FC Koln", "bayer leverkusen": "Leverkusen",
    "borussia monchengladbach": "M'gladbach", "monchengladbach": "M'gladbach",
    "mainz": "Mainz", "hamburger": "Hamburg", "hamburg": "Hamburg", "st pauli": "St Pauli",
    "union berlin": "Union Berlin", "werder bremen": "Werder Bremen", "hoffenheim": "Hoffenheim",
    "inter milan": "Inter", "internazionale": "Inter", "inter": "Inter", "milan": "Milan",
    "hellas verona": "Verona", "roma": "Roma", "napoli": "Napoli", "atalanta": "Atalanta",
    "paris saint germain": "Paris SG", "paris sg": "Paris SG", "psg": "Paris SG",
    "paris": "Paris FC", "olympique lyonnais": "Lyon", "lyon": "Lyon",
    "olympique marseille": "Marseille", "marseille": "Marseille", "stade rennais": "Rennes",
    "stade brestois": "Brest", "brest": "Brest", "lens": "Lens", "monaco": "Monaco",
    "nice": "Nice", "lille": "Lille", "le havre": "Le Havre", "saint etienne": "St Etienne",
    "psv eindhoven": "PSV Eindhoven", "eindhoven": "PSV Eindhoven", "ajax": "Ajax",
    "feyenoord": "Feyenoord", "az alkmaar": "AZ Alkmaar", "twente": "Twente",
    "sporting lisbon": "Sp Lisbon", "sporting cp": "Sp Lisbon", "sporting": "Sp Lisbon",
    "benfica": "Benfica", "porto": "Porto", "braga": "Sp Braga", "sporting braga": "Sp Braga",
    "club brugge": "Club Brugge", "brugge": "Club Brugge", "celtic": "Celtic", "rangers": "Rangers",
    "galatasaray": "Galatasaray", "fenerbahce": "Fenerbahce", "olympiacos": "Olympiakos",
    "olympiakos piraeus": "Olympiakos", "paok": "PAOK",
}


def norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = s.lower().replace("&", " and ").replace("'", "")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    toks = [t for t in s.split() if t not in DROP]
    return " ".join(toks) or s.strip()


def _load_user_aliases():
    p = config.PRIVATE / "aliases.json"
    if p.exists():
        try:
            data = json.loads(p.read_text())
            return {norm(k): v for k, v in data.get("teams", {}).items()}, data.get("elo", {})
        except (ValueError, AttributeError):
            pass
    return {}, {}


USER_TEAMS, USER_ELO = _load_user_aliases()


def match(name: str, candidates, cutoff=0.78):
    """Best candidate name for `name`, or None if not confident."""
    cands = list(candidates)
    if not cands:
        return None
    if name in cands:
        return name
    key = norm(name)
    by_norm = {}
    for c in cands:
        by_norm.setdefault(norm(c), []).append(c)
    for table in (USER_TEAMS, BUILTIN):
        if key in table and table[key] in cands:
            return table[key]
    if key in by_norm and len(by_norm[key]) == 1:
        return by_norm[key][0]
    # token containment, must be unique
    kt = set(key.split())
    hits = [c for n, cs in by_norm.items() for c in cs
            if n and (set(n.split()) <= kt or kt <= set(n.split()))]
    if len(hits) == 1:
        return hits[0]
    scored = sorted(((difflib.SequenceMatcher(None, key, n).ratio(), c)
                     for n, cs in by_norm.items() for c in cs), reverse=True)
    if scored and scored[0][0] >= cutoff and (len(scored) == 1 or scored[0][0] - scored[1][0] > 0.05):
        return scored[0][1]
    return None


def match_elo(fd_name: str, elo_df, country=None):
    """Find the ClubElo club for a football-data team name."""
    if fd_name in USER_ELO:
        return USER_ELO[fd_name]
    pool = elo_df if country is None else elo_df[elo_df["Country"] == country]
    pool = pool[pool["Level"] <= 2] if "Level" in pool else pool
    clubs = pool["Club"].tolist()
    if ELO_BUILTIN.get(fd_name) in clubs:
        return ELO_BUILTIN[fd_name]
    return match(fd_name, clubs, cutoff=0.8 if country else 0.88)


# football-data name -> ClubElo name, where fuzzy matching can't bridge the gap.
ELO_BUILTIN = {
    "Ath Madrid": "Atletico", "Ath Bilbao": "Bilbao", "Sp Lisbon": "Sporting",
    "PSV Eindhoven": "PSV", "Nott'm Forest": "Forest", "Ein Frankfurt": "Frankfurt",
    "M'gladbach": "Gladbach", "FC Koln": "Koeln", "Bayern Munich": "Bayern",
    "Espanol": "Espanyol", "Vallecano": "Rayo Vallecano", "Sp Braga": "Braga",
}
