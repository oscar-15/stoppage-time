"""Backtest of line shopping: treat Pinnacle's no-vig price as the truth and bet
whenever a softer book offers better than that.

football-data.co.uk snapshots every book at the same moment (shortly before
kickoff) plus again at the close, so for each match we can ask: was Bet365 /
William Hill / etc. paying more than Pinnacle's fair price, and did those bets
win over time?

Usage: python -m bot.shop_backtest [--seasons 6]
"""
import argparse
import json
import math
from datetime import date

import numpy as np
import pandas as pd

from . import config, history
from .odds import devig_power

SOFT = ["B365", "BW", "IW", "WH", "VC"]            # real, retail books
REF = {"Max": "best price anywhere (not all bettable)", "Avg": "market average"}
THRESHOLDS = [0.0, 0.01, 0.02, 0.03, 0.05]
NAMES = {"B365": "Bet365", "BW": "Bet&Win", "IW": "Interwetten", "WH": "William Hill",
         "VC": "BetVictor", "Max": "Best anywhere", "Avg": "Market average", "SOFT": "Best soft book"}


def _get(row, cols):
    vals = []
    for c in cols:
        v = row.get(c)
        if v is None or pd.isna(v) or v <= 1.0:
            return None
        vals.append(float(v))
    return vals


def candidates(df, league):
    """One row per (match, snapshot, book, outcome) where a price beats Pinnacle fair."""
    out = []
    for g in df.to_dict("records"):
        res = "H" if g["hg"] > g["ag"] else "A" if g["ag"] > g["hg"] else "D"
        over = g["hg"] + g["ag"] > 2.5
        close_fair = {}
        pc = _get(g, ["PSCH", "PSCD", "PSCA"])
        if pc:
            close_fair.update(zip(("1x2H", "1x2D", "1x2A"), devig_power(pc)))
        pco = _get(g, ["PC>2.5", "PC<2.5"])
        if pco:
            close_fair.update(zip(("ouO", "ouU"), devig_power(pco)))

        for snap, tag in (("pre", ""), ("close", "C")):
            # 1X2
            sharp = _get(g, [f"PS{tag}{s}" for s in "HDA"])
            if sharp:
                fair = devig_power(sharp)
                for b in SOFT + list(REF):
                    pr = _get(g, [f"{b}{tag}{s}" for s in "HDA"])
                    if not pr:
                        continue
                    for i, s in enumerate("HDA"):
                        out.append((league, g["season"], snap, b, "1x2", pr[i], pr[i] * fair[i] - 1,
                                    res == s, close_fair.get(f"1x2{s}"), g["date"], g["home"], s))
            # Over/under 2.5
            sharp = _get(g, [f"P{tag}>2.5", f"P{tag}<2.5"])
            if sharp:
                fair = devig_power(sharp)
                for b in ("B365", "Max", "Avg"):
                    pr = _get(g, [f"{b}{tag}>2.5", f"{b}{tag}<2.5"])
                    if not pr:
                        continue
                    for i, (s, won) in enumerate((("O", over), ("U", not over))):
                        out.append((league, g["season"], snap, b, "ou", pr[i], pr[i] * fair[i] - 1,
                                    won, close_fair.get(f"ou{s}"), g["date"], g["home"], s))
    cols = ["league", "season", "snap", "book", "market", "price", "ev", "won", "cfair", "date", "home", "sel"]
    return pd.DataFrame(out, columns=cols)


def stats(b):
    n = len(b)
    if n == 0:
        return {"bets": 0}
    units = np.where(b["won"], b["price"] - 1, -1.0)
    roi = units.mean()
    sd = units.std() if n > 1 else 0
    clv = (b["price"] * b["cfair"] - 1).dropna()
    return {"bets": int(n), "wins": int(b["won"].sum()), "units": round(float(units.sum()), 1),
            "roi": round(float(roi), 4),
            "roi_ci95": [round(float(roi - 1.96 * sd / math.sqrt(n)), 4), round(float(roi + 1.96 * sd / math.sqrt(n)), 4)],
            "avg_price": round(float(b["price"].mean()), 2), "avg_ev": round(float(b["ev"].mean()), 4),
            "avg_clv": round(float(clv.mean()), 4) if len(clv) else None,
            "beat_close": round(float((clv > 0).mean()), 3) if len(clv) else None}


def pick_best(c):
    """At most one bet per match, snapshot and book group: the highest-EV outcome."""
    return c.sort_values("ev", ascending=False).drop_duplicates(["date", "home", "snap", "book"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, default=6)
    args = ap.parse_args()
    today = date.today()

    frames = []
    for code, lg in config.LEAGUES.items():
        df = history.load_league(lg["fd"], today, back=args.seasons)
        c = candidates(df, code)
        print(f"{code}: {len(df)} matches, {len(c)} priced outcomes")
        frames.append(c)
    allc = pd.concat(frames, ignore_index=True)
    allc = allc[(allc["price"] >= config.MIN_PRICE) & (allc["price"] <= config.MAX_PRICE)]

    # "Best soft book": best price across the retail books for each outcome.
    soft = allc[allc["book"].isin(SOFT)]
    soft_best = soft.loc[soft.groupby(["date", "home", "snap", "market", "sel"])["price"].idxmax()].copy()
    soft_best["book"] = "SOFT"
    allc = pd.concat([allc, soft_best], ignore_index=True)

    grid = []
    for snap in ("pre", "close"):
        for book in ["SOFT"] + SOFT + list(REF):
            for t in THRESHOLDS:
                c = allc[(allc["snap"] == snap) & (allc["book"] == book) & (allc["ev"] >= t)]
                c = pick_best(c)
                if len(c) == 0:
                    continue
                grid.append({"snap": snap, "book": book, "book_name": NAMES[book], "min_ev": t,
                             **stats(c),
                             "by_market": {m: stats(c[c["market"] == m]) for m in ("1x2", "ou")}})

    # Headline: pre-close snapshot, best soft book, 2% threshold, chosen in advance.
    head = pick_best(allc[(allc["snap"] == "pre") & (allc["book"] == "SOFT") & (allc["ev"] >= 0.02)])
    by_season = {s: stats(head[head["season"] == s]) for s in sorted(head["season"].unique())}
    by_league = {l: stats(head[head["league"] == l]) for l in config.LEAGUES}
    curve, run = [], 0.0
    for r in head.sort_values("date").itertuples():
        run += (r.price - 1) if r.won else -1
        curve.append([str(r.date.date()), round(run, 2)])
    if len(curve) > 300:
        step = len(curve) / 300
        curve = [curve[int(i * step)] for i in range(300)] + [curve[-1]]

    out = {"generated": today.isoformat(),
           "seasons": sorted(allc["season"].unique().tolist()),
           "headline_rule": "Bet the best of Bet365/Bet&Win/Interwetten/William Hill/BetVictor when its price "
                            "beats Pinnacle's no-vig price by 2%+, at the pre-closing snapshot",
           "headline": stats(head), "by_season": by_season, "by_league": by_league,
           "curve": curve, "grid": grid}
    config.PUBLIC.mkdir(parents=True, exist_ok=True)
    (config.PUBLIC / "shop_backtest.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"headline": out["headline"], "by_season": by_season}, indent=1))
    for r in grid:
        if r["book"] in ("SOFT", "B365", "Max") and r["min_ev"] in (0.0, 0.02, 0.05):
            print(r["snap"], r["book"], r["min_ev"], {k: r[k] for k in ("bets", "roi", "roi_ci95", "avg_clv", "beat_close")})


if __name__ == "__main__":
    main()
