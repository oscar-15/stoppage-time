"""One-off check: which markets do US books (BetMGM especially) offer for soccer
through The Odds API? Costs about 1 credit per game checked.
Usage: python -m bot.market_check"""
import json
from datetime import datetime, timezone

from . import config, net, odds

SPORTS = ["soccer_uefa_champs_league", "soccer_epl"]


def main():
    out = {"checked": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "games": []}
    for sport in SPORTS:
        evs = sorted(odds.events(sport), key=lambda e: e["commence_time"])
        if not evs:
            continue
        ev = evs[0]
        data = net.get(f"{config.ODDS_API}/sports/{sport}/events/{ev['id']}/markets",
                       params={"apiKey": config.ODDS_API_KEY, "regions": "us,us2,eu", "dateFormat": "iso"},
                       fixture=f"markets_{sport}.json")
        books = {}
        for bk in (data or {}).get("bookmakers", []):
            books[bk["key"]] = sorted(m["key"] for m in bk.get("markets", []))
        out["games"].append({"sport": sport, "game": f"{ev['home_team']} v {ev['away_team']}",
                             "kickoff": ev["commence_time"], "books": books})
        print(sport, ev["home_team"], "v", ev["away_team"])
        for b, ms in sorted(books.items()):
            print(f"  {b}: {', '.join(ms)}")
    out["credits_remaining"] = odds.credits_remaining()
    config.PUBLIC.mkdir(parents=True, exist_ok=True)
    (config.PUBLIC / "market_check.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
