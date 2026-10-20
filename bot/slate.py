"""Full breakdown of every upcoming game: fair odds for match result, totals,
both teams to score and handicaps (from Pinnacle), next to the best US price
where one exists. Uses data the bot already pulled, so it costs no credits."""
import numpy as np

from . import config, model, odds

TOTAL_LINES = (1.5, 2.5, 3.5)
HANDICAPS = (-1.5, -0.5, 0.5, 1.5)        # from the home team's side


def _us(summary, key):
    mk = summary.get(key)
    if not mk:
        return None
    return {"price": round(mk["price"], 3), "american": odds.american(mk["price"]),
            "book": config.BOOK_NAMES.get(mk["book"], mk["book"])}


def _row(fair, us=None):
    r = {"p": round(fair, 4), "fair": odds.american(1 / fair) if 0 < fair < 1 else None}
    if us:
        r["us"] = us
        r["edge"] = round(fair * us["price"] - 1, 4)
    return r


def entry(event, comp, comp_name):
    summary = odds.summarize(event)
    raw = odds.sharp_raw(event)
    src = next((b for b in config.SHARP_BOOKS if raw.get(b, {}).get("h2h")), None)
    if src is None:
        return None
    M, _, line = odds.sharp_matrix(event)
    h, d, a = odds.devig_power(raw[src]["h2h"])
    e = {"id": event["id"], "comp": comp, "comp_name": comp_name, "kickoff": event["commence_time"],
         "home": event["home_team"], "away": event["away_team"],
         "source": config.BOOK_NAMES.get(src, src),
         "result": {"home": _row(h, _us(summary, ("h2h", None, "Home"))),
                    "draw": _row(d, _us(summary, ("h2h", None, "Draw"))),
                    "away": _row(a, _us(summary, ("h2h", None, "Away")))}}
    if M is None:
        return e
    g = np.arange(M.shape[0])
    lh = float((M.sum(axis=1) * g).sum())
    la = float((M.sum(axis=0) * g).sum())
    diff = g[:, None] - g[None, :]
    btts = float(M[1:, 1:].sum())
    flat = [(float(M[i, j]), f"{i}-{j}") for i in range(6) for j in range(6)]
    e.update({
        "xg": [round(lh, 2), round(la, 2)],
        "pinnacle_total_line": line,
        "totals": [{"line": L,
                    "over": _row(model.over_prob(M, L), _us(summary, ("totals", L, "Over"))),
                    "under": _row(1 - model.over_prob(M, L), _us(summary, ("totals", L, "Under")))}
                   for L in TOTAL_LINES],
        "btts": {"yes": _row(btts), "no": _row(1 - btts)},
        "handicaps": [{"line": L,
                       "home": _row(float(M[diff + L > 0].sum())),
                       "away": _row(float(M[-diff - L > 0].sum()))}
                      for L in HANDICAPS],
        "scores": [{"score": s, "p": round(p, 4)} for p, s in sorted(flat, reverse=True)[:4]],
    })
    return e


def update(slate, events, comp, comp_name, now, parse_ts, horizon_h):
    """Refresh this competition's games in the slate dict (keyed by event id)."""
    from datetime import timedelta
    for ev in events:
        ko = parse_ts(ev["commence_time"])
        if not (now < ko <= now + timedelta(hours=horizon_h)):
            continue
        try:
            row = entry(ev, comp, comp_name)
        except Exception as ex:      # one odd game shouldn't break the slate
            print(f"slate: skipped {ev.get('home_team')} v {ev.get('away_team')}: {ex}")
            continue
        if row:
            row["updated"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
            slate[ev["id"]] = row


def prune(slate, now, parse_ts):
    from datetime import timedelta
    return {k: v for k, v in slate.items() if parse_ts(v["kickoff"]) > now - timedelta(hours=3)}
