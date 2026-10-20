"""One bot run: lock started picks, grade finished ones, refresh odds, choose the card.
Usage: python -m bot.run"""
import json
import os
import traceback
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from . import config, model, odds
from .engine import Engine

COMPS = {**{k: v for k, v in config.LEAGUES.items()}, **config.CUPS}
LOG = []


def log(msg):
    print(msg)
    LOG.append(msg)


def now_utc():
    v = os.environ.get("NOW")
    return datetime.fromisoformat(v.replace("Z", "+00:00")) if v else datetime.now(timezone.utc)


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def iso(dt):
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load(path, default):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return default


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def slate_day(kickoff_iso):
    return parse_ts(kickoff_iso).astimezone(ZoneInfo(config.SLATE_TZ)).date().isoformat()


# ---------------------------------------------------------------- grading
def grade_pick(p, hg, ag):
    if p["market"] == "h2h":
        res = "Home" if hg > ag else "Away" if ag > hg else "Draw"
        won = res == p["selection"]
    else:
        tot = hg + ag
        won = tot > p["point"] if p["selection"] == "Over" else tot < p["point"]
    p["status"] = "won" if won else "lost"
    p["units"] = round(p["price"] - 1, 3) if won else -1.0
    p["final"] = f"{hg}-{ag}"


def grade(picks, now):
    todo = [p for p in picks if p["status"] == "locked"
            and parse_ts(p["kickoff"]) < now - timedelta(hours=2)]
    for sport in sorted({p["sport_key"] for p in todo}):
        try:
            results = odds.scores(sport)
        except Exception as e:     # keep going; we'll retry next run
            log(f"grade: scores failed for {sport}: {e}")
            continue
        done = {r["id"]: r for r in results if r.get("completed") and r.get("scores")}
        for p in todo:
            r = done.get(p["event_id"]) if p["sport_key"] == sport else None
            if not r:
                continue
            sc = {s["name"]: s["score"] for s in r["scores"]}
            try:
                hg, ag = int(sc[r["home_team"]]), int(sc[r["away_team"]])
            except (KeyError, ValueError, TypeError):
                continue
            grade_pick(p, hg, ag)
            log(f"grade: {p['label']} ({p['home']} v {p['away']} {hg}-{ag}) -> {p['status']}")
    for p in picks:   # postponed / abandoned / never reported
        if p["status"] == "locked" and parse_ts(p["kickoff"]) < now - timedelta(days=4):
            p["status"], p["units"] = "void", 0.0
            log(f"grade: voided {p['label']} ({p['home']} v {p['away']}) - no result after 4 days")


# ---------------------------------------------------------------- candidates
def label_for(market, sel, point, home, away):
    if market == "h2h":
        return {"Home": f"{home} to win", "Away": f"{away} to win", "Draw": "Draw"}[sel]
    return f"{sel} {point:g} goals"


def _pick(event, comp, key, mk, now, **extra):
    ev = event
    market, point, sel = key
    return {
        "id": f"{ev['id']}:{market}:{sel}:{point if point is not None else ''}",
        "event_id": ev["id"], "sport_key": ev["sport_key"], "comp": comp,
        "comp_name": COMPS[comp]["name"],
        "home": ev["home_team"], "away": ev["away_team"], "kickoff": ev["commence_time"],
        "market": market, "selection": sel, "point": point,
        "label": label_for(market, sel, point, ev["home_team"], ev["away_team"]),
        "price": round(mk["price"], 3), "american": odds.american(mk["price"]),
        "book": config.BOOK_NAMES.get(mk["book"], mk["book"]),
        "status": "pending", "created": iso(now), "strategy": config.STRATEGY, **extra,
    }


def candidates_for(comp, events, eng, now):
    out = []
    horizon = now + timedelta(hours=config.ODDS_WINDOW_HOURS)
    for ev in events:
        ko = parse_ts(ev["commence_time"])
        if not (now + timedelta(minutes=10) < ko <= horizon):
            continue
        summary = odds.summarize(ev)
        if config.STRATEGY == "shop":
            for key, mk in summary.items():
                fair = mk.get("sharp")
                if fair is None or not (config.MIN_PRICE <= mk["price"] <= config.MAX_PRICE):
                    continue
                ev_ = fair * mk["price"] - 1
                if ev_ < config.SHOP_MIN_EV:
                    continue
                src = config.BOOK_NAMES.get(mk["sharp_src"], mk["sharp_src"])
                out.append(_pick(ev, comp, key, mk, now, p=round(fair, 4), model_p=round(fair, 4),
                                 market_p=round(1 / mk["price"], 4), ev=round(ev_, 4),
                                 fair_src=src, open_fair=round(fair, 4)))
            continue

        probs = eng.probabilities(comp, ev) if eng else None
        if probs is None:
            continue
        M = probs["_M"]
        for key, mk in summary.items():
            market, point, sel = key
            if market == "totals":
                po = model.over_prob(M, point)
                mp = po if sel == "Over" else 1 - po
            else:
                mp = probs[key]
            if mk["n_books"] < config.MIN_BOOKS:
                continue
            if abs(mp - mk["fair"]) > config.MAX_DISAGREE:
                continue
            if not (config.MIN_PRICE <= mk["price"] <= config.MAX_PRICE):
                continue
            p = config.MODEL_WEIGHT * mp + (1 - config.MODEL_WEIGHT) * mk["fair"]
            ev_ = p * mk["price"] - 1
            if ev_ < config.MIN_EV:
                continue
            out.append(_pick(ev, comp, key, mk, now, model_p=round(mp, 4), market_p=round(mk["fair"], 4),
                             p=round(p, 4), ev=round(ev_, 4), xg=[round(x, 2) for x in probs["_xg"]]))
    return out


def update_clv(picks, events, now):
    """Before kickoff, keep re-reading Pinnacle's fair price for each posted pick.
    The last reading before kickoff is the closing line; beating it is the
    best evidence an edge is real."""
    by_id = {e["id"]: e for e in events}
    for p in picks:
        ev = by_id.get(p["event_id"])
        if not ev or p["status"] != "pending" or parse_ts(p["kickoff"]) <= now:
            continue
        mk = odds.summarize(ev).get((p["market"], p["point"], p["selection"]))
        if mk and mk.get("sharp"):
            p["close_fair"] = round(mk["sharp"], 4)
            p["clv"] = round(p["price"] * mk["sharp"] - 1, 4)
            p["close_checked"] = iso(now)


def rank_key(c):
    """Edge per unit of risk (the Kelly fraction): favors solid edges at sane prices
    over thin edges on long shots."""
    return c["ev"] / (c["price"] - 1)


def build_card(picks, new_cands, refreshed_sports):
    """Posted picks stay posted, so the record grades exactly what was shown.
    New candidates only fill open slots (up to PICKS_PER_DAY per slate day).
    Pending picks from a different strategy are dropped when you switch."""
    fixed = [p for p in picks if p["status"] != "pending"
             or p.get("strategy", "model") == config.STRATEGY]
    taken = {p["event_id"] for p in fixed}
    by_day = {}
    for c in new_cands:
        if c["event_id"] not in taken:
            by_day.setdefault(slate_day(c["kickoff"]), []).append(c)
    chosen = []
    for day, cs in by_day.items():
        slots = config.PICKS_PER_DAY - sum(1 for p in fixed if slate_day(p["kickoff"]) == day)
        seen = set()
        for c in sorted(cs, key=rank_key, reverse=True):
            if slots <= 0:
                break
            if c["event_id"] in seen:
                continue
            seen.add(c["event_id"])
            chosen.append(c)
            slots -= 1
    return fixed + chosen


# ---------------------------------------------------------------- record
def record(picks):
    graded = [p for p in picks if p["status"] in ("won", "lost")]
    w = sum(p["status"] == "won" for p in graded)
    l = len(graded) - w
    v = sum(p["status"] == "void" for p in picks)
    units = round(sum(p.get("units", 0) for p in graded), 2)
    clv = [p["clv"] for p in picks if p["status"] != "pending" and p.get("clv") is not None]
    return {"wins": w, "losses": l, "voids": v, "units": units,
            "roi": round(units / len(graded), 4) if graded else None, "graded": len(graded),
            "avg_clv": round(sum(clv) / len(clv), 4) if clv else None,
            "beat_close": round(sum(c > 0 for c in clv) / len(clv), 3) if clv else None,
            "clv_n": len(clv)}


# ---------------------------------------------------------------- main
def needs_odds(sport, evs, state, now, picks=()):
    last = state.get("last_odds", {}).get(sport)
    last = parse_ts(last) if last else None
    upcoming = [parse_ts(e["commence_time"]) for e in evs
                if now < parse_ts(e["commence_time"]) <= now + timedelta(hours=config.ODDS_WINDOW_HOURS)]
    if not upcoming:
        return False
    if last is None or now - last >= timedelta(hours=config.REFRESH_HOURS):
        return True
    # Re-check close to kickoff of a posted pick, to measure the closing line.
    pick_ko = [parse_ts(p["kickoff"]) for p in picks if p["sport_key"] == sport and p["status"] == "pending"]
    soon = any(timedelta(0) < k - now <= timedelta(minutes=config.PREKICK_REFRESH_MIN) for k in pick_ko)
    return soon and now - last >= timedelta(minutes=50)


def main():
    now = now_utc()
    picks_path = config.PUBLIC / "picks.json"
    state_path = config.PRIVATE / "state.json"
    picks = load(picks_path, [])
    state = load(state_path, {})

    if not config.ODDS_API_KEY and not os.environ.get("FIXTURES"):
        log("ODDS_API_KEY is not set: add it under Settings > Secrets > Actions.")

    # 1. lock picks whose match has started
    for p in picks:
        if p["status"] == "pending" and parse_ts(p["kickoff"]) <= now:
            p["status"], p["locked"] = "locked", iso(now)
            log(f"lock: {p['label']} ({p['home']} v {p['away']}) @ {p['american']}")

    # 2. grade finished matches
    grade(picks, now)

    # 3. which competitions have games coming up and need fresh odds?
    sched = {}
    for comp, c in COMPS.items():
        try:
            evs = odds.events(c["odds_key"])
        except Exception as e:
            log(f"events failed for {comp}: {e}")
            continue
        if needs_odds(c["odds_key"], evs, state, now, picks):
            sched[comp] = c["odds_key"]

    remaining = odds.credits_remaining() or state.get("credits")
    new_cands, refreshed, eng = [], set(), None
    if sched and remaining is not None and remaining < config.CREDIT_RESERVE:
        log(f"credits low ({remaining}); skipping odds pull")
        sched = {}
    if sched:
        if config.STRATEGY == "model":
            eng = Engine(now.date(), log)
            eng.build(need_cups=any(k in config.CUPS for k in sched))
        for comp, sport in sched.items():
            try:
                evs = odds.odds(sport)
                update_clv(picks, evs, now)
                cs = candidates_for(comp, evs, eng, now)
            except Exception as e:
                log(f"ERROR {comp}: {type(e).__name__}: {e}")
                traceback.print_exc()
                continue
            refreshed.add(sport)
            state.setdefault("last_odds", {})[sport] = iso(now)
            log(f"odds: {comp} {len(evs)} games, {len(cs)} bets clear the bar")
            new_cands += cs
        remaining = odds.credits_remaining() or remaining

    # 4. pick the card
    picks = build_card(picks, new_cands, refreshed)
    picks.sort(key=lambda p: p["kickoff"], reverse=True)

    state["credits"] = remaining
    state["last_run"] = iso(now)
    save(picks_path, picks)
    save(state_path, state)
    save(config.PUBLIC / "status.json", {
        "updated": iso(now),
        "record": record(picks),
        "credits_remaining": remaining,
        "refreshed": sorted(k for k, v in COMPS.items() if v["odds_key"] in refreshed),
        "unmatched_teams": sorted(eng.unmatched) if eng else state.get("unmatched", []),
        "log": LOG[-40:],
    })
    if eng:
        state["unmatched"] = sorted(eng.unmatched)
        save(state_path, state)


if __name__ == "__main__":
    main()
