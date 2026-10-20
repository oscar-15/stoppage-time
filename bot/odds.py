"""The Odds API client and price math."""
import statistics

from . import config, net


def _get(path, params, fixture):
    params = dict(params, apiKey=config.ODDS_API_KEY)
    return net.get(f"{config.ODDS_API}{path}", params=params, fixture=fixture)


def credits_remaining():
    v = net.last_headers.get("x-requests-remaining")
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def events(sport):
    """Upcoming fixtures. Free: doesn't use quota."""
    return _get(f"/sports/{sport}/events", {}, f"events_{sport}.json") or []


def odds(sport):
    return _get(f"/sports/{sport}/odds",
                {"bookmakers": config.BOOKMAKERS, "markets": "h2h,totals",
                 "oddsFormat": "decimal", "dateFormat": "iso"},
                f"odds_{sport}.json") or []


def scores(sport, days_from=3):
    return _get(f"/sports/{sport}/scores", {"daysFrom": days_from, "dateFormat": "iso"},
                f"scores_{sport}.json") or []


def american(dec):
    if dec >= 2:
        return f"+{round((dec - 1) * 100)}"
    return f"{round(-100 / (dec - 1))}"


def devig(prices):
    inv = [1 / p for p in prices]
    s = sum(inv)
    return [x / s for x in inv]


def devig_power(prices):
    """Remove the margin with the power method: find k so that sum((1/p)^k) = 1.
    Unlike proportional de-vigging, this accounts for books loading more margin
    onto long shots, so it doesn't invent fake value on underdogs."""
    inv = [1 / p for p in prices]
    if sum(inv) <= 1:
        return [x / sum(inv) for x in inv]
    lo, hi = 1.0, 3.0
    for _ in range(60):
        k = (lo + hi) / 2
        if sum(x ** k for x in inv) > 1:
            lo = k
        else:
            hi = k
    k = (lo + hi) / 2
    out = [x ** k for x in inv]
    s = sum(out)
    return [x / s for x in out]


def summarize(event):
    """Collapse all bookmakers into, per outcome: fair prob (median of de-vigged
    books), best bettable price and which book has it.
    Returns {("h2h", None, "Home"): {...}, ("totals", 2.5, "Over"): {...}, ...}"""
    home, away = event["home_team"], event["away_team"]
    name_map = {home: "Home", away: "Away", "Draw": "Draw"}
    fair, best, sharp = {}, {}, {}
    bet_books = set(b for b in config.BET_BOOKS if b)

    for bk in event.get("bookmakers", []):
        for mk in bk.get("markets", []):
            if mk["key"] == "h2h":
                outs = {name_map.get(o["name"]): o["price"] for o in mk["outcomes"]}
                if set(outs) != {"Home", "Draw", "Away"} or min(outs.values()) <= 1:
                    continue
                keys = [("h2h", None, k) for k in ("Home", "Draw", "Away")]
                prices = [outs[k[2]] for k in keys]
            elif mk["key"] == "totals":
                by_point = {}
                for o in mk["outcomes"]:
                    by_point.setdefault(o.get("point"), {})[o["name"]] = o["price"]
                keys, prices = [], []
                for pt, d in by_point.items():
                    if pt is None or "Over" not in d or "Under" not in d:
                        continue
                    if abs(pt * 2 - round(pt * 2)) > 1e-9 or float(pt).is_integer():
                        continue          # only x.5 lines: no pushes, clean grading
                    ks = [("totals", float(pt), "Over"), ("totals", float(pt), "Under")]
                    ps = [d["Over"], d["Under"]]
                    for k, p in zip(ks, devig(ps)):
                        fair.setdefault(k, []).append(p)
                    if bk["key"] in config.SHARP_BOOKS:
                        for k, p in zip(ks, devig_power(ps)):
                            sharp.setdefault(k, {})[bk["key"]] = p
                    for k, pr in zip(ks, ps):
                        _best(best, k, pr, bk["key"], bet_books)
                continue
            else:
                continue
            for k, p in zip(keys, devig(prices)):
                fair.setdefault(k, []).append(p)
            if bk["key"] in config.SHARP_BOOKS:
                for k, p in zip(keys, devig_power(prices)):
                    sharp.setdefault(k, {})[bk["key"]] = p
            for k, pr in zip(keys, prices):
                _best(best, k, pr, bk["key"], bet_books)

    out = {}
    for k, ps in fair.items():
        if k in best:
            sh = sharp.get(k, {})
            src = next((b for b in config.SHARP_BOOKS if b in sh), None)
            out[k] = {"fair": statistics.median(ps), "n_books": len(ps),
                      "price": best[k][0], "book": best[k][1],
                      "sharp": sh[src] if src else None, "sharp_src": src}
    return out


def _best(best, key, price, book, bet_books):
    if bet_books and book not in bet_books:
        return
    if key not in best or price > best[key][0]:
        best[key] = (price, book)
