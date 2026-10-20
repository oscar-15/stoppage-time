"""All the knobs in one place. Environment variables override the defaults."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PUBLIC = Path(os.getenv("PUBLIC_DIR", ROOT / "docs" / "data"))   # what the website reads
PRIVATE = Path(os.getenv("PRIVATE_DIR", ROOT / "data"))          # bot state + your aliases
CACHE = Path(os.getenv("CACHE_DIR", ROOT / "cache"))             # downloaded history

# --- Competitions -----------------------------------------------------------
# fd = football-data.co.uk file code, odds_key = The Odds API sport key,
# elo = ClubElo country code
LEAGUES = {
    "EPL":        {"name": "Premier League", "us": "EPL",        "fd": "E0",  "odds_key": "soccer_epl",                 "elo": "ENG"},
    "LALIGA":     {"name": "La Liga",        "us": "La_liga",    "fd": "SP1", "odds_key": "soccer_spain_la_liga",       "elo": "ESP"},
    "BUNDESLIGA": {"name": "Bundesliga",     "us": "Bundesliga", "fd": "D1",  "odds_key": "soccer_germany_bundesliga",  "elo": "GER"},
    "SERIEA":     {"name": "Serie A",        "us": "Serie_A",    "fd": "I1",  "odds_key": "soccer_italy_serie_a",       "elo": "ITA"},
    "LIGUE1":     {"name": "Ligue 1",        "us": "Ligue_1",    "fd": "F1",  "odds_key": "soccer_france_ligue_one",    "elo": "FRA"},
}
CUPS = {
    "UCL": {"name": "Champions League", "odds_key": "soccer_uefa_champs_league"},
    "UEL": {"name": "Europa League",    "odds_key": "soccer_uefa_europa_league"},
}
# History-only leagues, so cup opponents like Benfica or PSV have a goal profile.
SUPPORT_LEAGUES = {
    "NED": {"name": "Eredivisie",       "fd": "N1",  "elo": "NED"},
    "POR": {"name": "Primeira Liga",    "fd": "P1",  "elo": "POR"},
    "BEL": {"name": "Pro League",       "fd": "B1",  "elo": "BEL"},
    "SCO": {"name": "Scottish Prem",    "fd": "SC0", "elo": "SCO"},
    "TUR": {"name": "Super Lig",        "fd": "T1",  "elo": "TUR"},
    "GRE": {"name": "Super League",     "fd": "G1",  "elo": "GRE"},
}

# Rough log goal-rate edge of an average team in each league vs an average
# Premier League side, used for cup games when ClubElo is unreachable.
LEAGUE_STRENGTH = {"EPL": 0.0, "LALIGA": -0.08, "BUNDESLIGA": -0.10, "SERIEA": -0.10,
                   "LIGUE1": -0.22, "POR": -0.35, "NED": -0.38, "BEL": -0.50,
                   "TUR": -0.50, "GRE": -0.55, "SCO": -0.62}

# --- Model ------------------------------------------------------------------
SEASONS_BACK = int(os.getenv("SEASONS_BACK", 3))   # past seasons used to fit
XI = float(os.getenv("XI", 0.0019))                # time decay per day (~1yr half-life)
REG = float(os.getenv("REG", 1.5))                 # pull toward prior for thin data
XG_WEIGHT = float(os.getenv("XG_WEIGHT", 0.7))      # 0 = goals only
NEW_TEAM_PRIOR = -0.22                             # promoted teams start below average
ELO_HOME_ADV = 65.0                                # ClubElo-scale home advantage
CUP_DEFAULT_TOTAL = 2.85                           # cup total goals if no profile

# --- Betting rules ----------------------------------------------------------
# "shop": bet a US book only when it pays more than Pinnacle's no-vig price.
# "model": the original goal-model picks (backtested at a loss; kept for reference).
STRATEGY = os.getenv("STRATEGY", "shop")
SHOP_MIN_EV = float(os.getenv("SHOP_MIN_EV", 0.02))   # beat Pinnacle fair by 2%+
SHARP_BOOKS = ["pinnacle", "betfair_ex_eu"]           # fair price source, in order
MODEL_WEIGHT = float(os.getenv("MODEL_WEIGHT", 0.5))   # blend model with the market
MIN_EV = float(os.getenv("MIN_EV", 0.03))              # need >= 3% expected value
MAX_DISAGREE = 0.12        # skip if model and market differ by > 12 pts (likely bad data)
MIN_PRICE, MAX_PRICE = 1.45, 5.0   # decimal odds range we'll bet
MIN_BOOKS = 3
PICKS_PER_DAY = int(os.getenv("PICKS_PER_DAY", 10))
SLATE_TZ = os.getenv("SLATE_TZ", "America/Los_Angeles")

# --- Odds API ---------------------------------------------------------------
ODDS_API = "https://api.the-odds-api.com/v4"
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
# Listing up to 10 bookmakers costs the same as one region.
BOOKMAKERS = os.getenv(
    "BOOKMAKERS",
    "pinnacle,betfair_ex_eu,draftkings,fanduel,betmgm,williamhill_us,betrivers,espnbet,bet365,unibet_eu",
)
# Books whose prices we actually "bet" at (US books). Others only inform fair value.
BET_BOOKS = os.getenv("BET_BOOKS", "draftkings,fanduel,betmgm,williamhill_us,betrivers,espnbet").split(",")
ODDS_WINDOW_HOURS = float(os.getenv("ODDS_WINDOW_HOURS", 36))   # look this far ahead
REFRESH_HOURS = float(os.getenv("REFRESH_HOURS", 8))            # re-pull a sport this often
PREKICK_REFRESH_MIN = 120      # ...and once more inside 2h of a kickoff
CREDIT_RESERVE = int(os.getenv("CREDIT_RESERVE", 15))

BOOK_NAMES = {
    "draftkings": "DK", "fanduel": "FanDuel", "betmgm": "MGM", "williamhill_us": "Caesars",
    "betrivers": "BetRivers", "espnbet": "ESPN BET", "pinnacle": "Pinnacle",
    "bet365": "bet365", "unibet_eu": "Unibet", "betfair_ex_eu": "Betfair",
}
