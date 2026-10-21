"""Arbitrage scanning and phone alerts. Both use data the bot already has: no credits."""
import os

import requests

from . import config, odds

SITE = os.getenv("SITE_URL", "https://oscar-15.github.io/stoppage-time/")


# ---------------------------------------------------------------- arbitrage
def find_arbs(events, comp, comp_name, now_iso, parse_ts, now):
    """A market is an arb when the best US prices across books, one per outcome,
    add up to less than 100% implied. Betting every outcome in proportion then
    returns the same amount whatever happens."""
    out = []
    for ev in events:
        if parse_ts(ev["commence_time"]) <= now:
            continue
        summary = odds.summarize(ev)
        groups = {}
        for (market, point, sel), mk in summary.items():
            groups.setdefault((market, point), {})[sel] = mk
        for (market, point), sides in groups.items():
            need = {"h2h": {"Home", "Draw", "Away"}, "totals": {"Over", "Under"}}.get(market)
            if need is None or set(sides) != need:
                continue
            implied = sum(1 / s["price"] for s in sides.values())
            profit = 1 / implied - 1
            if profit < config.ARB_MIN_PROFIT:
                continue
            legs = []
            for sel, s in sides.items():
                share = (1 / s["price"]) / implied
                names = {"Home": ev["home_team"], "Away": ev["away_team"], "Draw": "Draw"}
                label = names[sel] if sel in names else f"{sel} {point:g}"
                legs.append({"selection": label, "american": odds.american(s["price"]), "price": round(s["price"], 3),
                             "book": config.BOOK_NAMES.get(s["book"], s["book"]),
                             "stake_per_100": round(100 * share, 2)})
            out.append({"id": f"{ev['id']}:{market}:{point if point is not None else ''}",
                        "sport_key": ev["sport_key"], "comp": comp, "comp_name": comp_name,
                        "home": ev["home_team"], "away": ev["away_team"], "kickoff": ev["commence_time"],
                        "market": "Match result" if market == "h2h" else f"Total goals {point:g}",
                        "profit": round(profit, 4), "legs": legs, "found": now_iso})
    return out


# ---------------------------------------------------------------- alerts
def notify(title, body, tags="soccer"):
    """Push a notification through ntfy.sh. Set the NTFY_TOPIC secret to turn it on."""
    topic = config.NTFY_TOPIC
    if not topic or os.environ.get("FIXTURES"):
        return False
    try:
        r = requests.post(f"https://ntfy.sh/{topic}", data=body.encode("utf-8"), timeout=15,
                          headers={"Title": title.encode("utf-8"), "Tags": tags, "Click": SITE})
        return r.ok
    except requests.RequestException as e:
        print(f"notify failed: {e}")
        return False


def pick_message(p):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    ko = datetime.fromisoformat(p["kickoff"].replace("Z", "+00:00")).astimezone(ZoneInfo(config.SLATE_TZ))
    return (f"{p['label']} {p['american']} at {p['book']}\n"
            f"{p['home']} v {p['away']} ({p['comp_name']})\n"
            f"Edge +{p['ev'] * 100:.1f}% · kicks off {ko:%a %-I:%M %p}\n"
            f"Check the price is still {p['american']} or better before betting.")


def arb_message(a):
    legs = "\n".join(f"• {l['selection']} {l['american']} at {l['book']}: ${l['stake_per_100']:.2f}" for l in a["legs"])
    return (f"{a['home']} v {a['away']}, {a['market']}\n{legs}\n"
            f"Locks in about +{a['profit'] * 100:.1f}% (per $100 total). Prices move fast: confirm every leg first.")
