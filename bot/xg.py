"""Expected goals (xG) per match from Understat, joined onto football-data results.

Understat names teams differently ("Wolverhampton Wanderers" vs "Wolves"), so
rather than keep a name list we learn the mapping from the data itself: the
same match appears in both sources on the same day with the same score.
"""
import json
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from . import config, net

URL = "https://understat.com/getLeagueData/{slug}/{year}"


def fetch_season(slug: str, year: int, current: bool) -> pd.DataFrame:
    config.CACHE.mkdir(parents=True, exist_ok=True)
    path = config.CACHE / f"us_{slug}_{year}.json"
    stale = current and (not path.exists() or time.time() - path.stat().st_mtime > 3 * 3600)
    if not path.exists() or stale:
        try:
            data = net.get(URL.format(slug=slug, year=year), fixture=f"us_{slug}_{year}.json",
                           headers={"X-Requested-With": "XMLHttpRequest",
                                    "Referer": f"https://understat.com/league/{slug}/{year}"})
        except Exception as e:      # xG is a bonus: never let it break a run
            print(f"xg: fetch failed {slug} {year}: {e}")
            data = None
        if data:
            path.write_text(json.dumps(data))
    if not path.exists():
        return pd.DataFrame()
    data = json.loads(path.read_text())
    rows = []
    for m in data.get("dates", []):
        if not m.get("isResult"):
            continue
        try:
            rows.append({"date": pd.Timestamp(m["datetime"]).normalize(),
                         "uh": m["h"]["title"], "ua": m["a"]["title"],
                         "ugh": int(m["goals"]["h"]), "uga": int(m["goals"]["a"]),
                         "hxg": float(m["xG"]["h"]), "axg": float(m["xG"]["a"])})
        except (KeyError, TypeError, ValueError):
            continue
    return pd.DataFrame(rows)


def learn_names(fd: pd.DataFrame, us: pd.DataFrame) -> dict:
    """understat name -> football-data name, from matches with same date (+/-1 day) and score."""
    votes = defaultdict(Counter)
    by_day = defaultdict(list)
    for r in fd.itertuples():
        by_day[r.date.normalize()].append(r)
    for u in us.itertuples():
        for d in (u.date, u.date - pd.Timedelta(days=1), u.date + pd.Timedelta(days=1)):
            for f in by_day.get(d, []):
                if f.hg == u.ugh and f.ag == u.uga:
                    votes[u.uh][f.home] += 1
                    votes[u.ua][f.away] += 1
    mapping = {}
    for u, c in votes.items():
        (best, n), *rest = c.most_common(2) + [(None, 0)]
        if n >= 3 and n >= 2 * rest[0][1]:
            mapping[u] = best
    return mapping


def attach(fd: pd.DataFrame, slug: str, seasons: list) -> pd.DataFrame:
    """Add hxg/axg columns to a football-data frame. Missing xG stays NaN."""
    fd = fd.copy()
    fd["hxg"] = np.nan
    fd["axg"] = np.nan
    if fd.empty or not slug:
        return fd
    frames = []
    for i, code in enumerate(seasons):
        year = 2000 + int(code[:2])
        u = fetch_season(slug, year, current=(i == len(seasons) - 1))
        if not u.empty:
            u["season"] = code
            frames.append(u)
    if not frames:
        return fd
    us = pd.concat(frames, ignore_index=True)
    names = learn_names(fd, us)
    us["home"] = us["uh"].map(names)
    us["away"] = us["ua"].map(names)
    us = us.dropna(subset=["home", "away"]).drop_duplicates(["season", "home", "away"])
    key = ["season", "home", "away"]
    merged = fd.drop(columns=["hxg", "axg"]).merge(us[key + ["hxg", "axg"]], on=key, how="left")
    unmapped = sorted(set(frames[-1]["uh"]) - set(names))
    if unmapped:
        print(f"xg: {slug} teams not matched: {unmapped}")
    return merged
