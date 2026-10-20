"""Historical results (football-data.co.uk) and club strength (ClubElo)."""
import io
import time
from datetime import date, datetime

import pandas as pd

from . import config, net

FD_URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
ELO_URL = "http://api.clubelo.com/{day}"

BOOKS_1X2 = ["PS", "B365", "BW", "IW", "WH", "VC", "Max", "Avg"]
KEEP = (["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
        + [f"{b}{s}" for b in BOOKS_1X2 for s in "HDA"]           # pre-closing
        + [f"{b}C{s}" for b in BOOKS_1X2 for s in "HDA"]          # closing
        + [f"{b}{s}2.5" for b in ("P", "B365", "Max", "Avg") for s in "><"]
        + [f"{b}C{s}2.5" for b in ("P", "B365", "Max", "Avg") for s in "><"])


def season_start(d: date) -> int:
    return d.year if d.month >= 7 else d.year - 1


def season_codes(d: date, back: int):
    s = season_start(d)
    return [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(s - back, s + 1)]


def _parse(text: str, season: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), on_bad_lines="skip")
    df = df[[c for c in KEEP if c in df.columns]].copy()
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.rename(columns={"HomeTeam": "home", "AwayTeam": "away", "FTHG": "hg", "FTAG": "ag"})
    df["home"] = df["home"].astype(str).str.strip()
    df["away"] = df["away"].astype(str).str.strip()
    df["hg"] = df["hg"].astype(int)
    df["ag"] = df["ag"].astype(int)
    df["season"] = season
    return df.drop(columns=["Date"])


def load_league(fd_code: str, today: date, back: int = None, xg_slug: str = None) -> pd.DataFrame:
    back = config.SEASONS_BACK if back is None else back
    config.CACHE.mkdir(parents=True, exist_ok=True)
    codes = season_codes(today, back)
    frames = []
    for i, season in enumerate(codes):
        path = config.CACHE / f"fd_{season}_{fd_code}.csv"
        current = i == len(codes) - 1
        stale = current and (not path.exists() or time.time() - path.stat().st_mtime > 3 * 3600)
        if not path.exists() or stale:
            raw = net.get(FD_URL.format(season=season, code=fd_code),
                          fixture=f"fd_{season}_{fd_code}.csv", as_json=False)
            if raw is None:
                continue
            path.write_bytes(raw)
        frames.append(_parse(net.decode(path.read_bytes()), season))
    if not frames:
        return pd.DataFrame(columns=["date", "home", "away", "hg", "ag", "season"])
    df = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    if xg_slug:
        from . import xg
        df = xg.attach(df, xg_slug, codes)
        df = df.sort_values("date").reset_index(drop=True)
    return df


def load_elo(day: date) -> pd.DataFrame:
    """ClubElo ratings. Keeps the last good copy in data/ so an outage doesn't matter."""
    keep = config.PRIVATE / "clubelo_last.csv"
    try:
        raw = net.get(ELO_URL.format(day=day.isoformat()), fixture="clubelo.csv", as_json=False)
        if raw:
            df = pd.read_csv(io.StringIO(net.decode(raw)))[["Club", "Country", "Level", "Elo"]]
            if len(df) > 100:
                keep.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(keep, index=False)
                return df
    except Exception as e:
        print(f"elo: ClubElo unavailable ({e})")
    if keep.exists():
        print("elo: using last saved ratings")
        return pd.read_csv(keep)
    return pd.DataFrame(columns=["Club", "Country", "Level", "Elo"])
