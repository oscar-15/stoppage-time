"""Walk-forward backtest on past seasons with real closing odds.

For each week, the model is refit on matches *before* that week only, then it
prices that week's games. We "bet" at Pinnacle's early price (football-data's
PSH/PSD/PSA columns) using the same rules as the live bot, and measure:
  - profit at that price
  - CLV: did we beat Pinnacle's closing line? (the honest test of an edge)

Usage: python -m bot.backtest  [--test-seasons 3]
"""
import argparse
import json
import math
from datetime import date

import numpy as np
import pandas as pd

from . import config, history, model
from .odds import devig


def _price_cols(row, cols_list):
    for cols in cols_list:
        if all(c in row and pd.notna(row[c]) and row[c] > 1 for c in cols):
            return [float(row[c]) for c in cols]
    return None


H2H_OPEN = [("PSH", "PSD", "PSA"), ("B365H", "B365D", "B365A"), ("AvgH", "AvgD", "AvgA")]
H2H_CLOSE = [("PSCH", "PSCD", "PSCA")]
OU_OPEN = [("P>2.5", "P<2.5"), ("B365>2.5", "B365<2.5"), ("Avg>2.5", "Avg<2.5")]
OU_CLOSE = [("PC>2.5", "PC<2.5")]


def bets_for_league(code, lg, today, test_seasons):
    df = history.load_league(lg["fd"], today, back=config.SEASONS_BACK + test_seasons)
    if df.empty:
        return [], []
    seasons = sorted(df["season"].unique())
    test = seasons[-test_seasons:] if len(seasons) > test_seasons else seasons[1:]
    bets, preds = [], []
    for season in test:
        sdf = df[df["season"] == season]
        start = sdf["date"].min().normalize()
        for wk in pd.date_range(start, sdf["date"].max() + pd.Timedelta(days=1), freq="7D"):
            games = sdf[(sdf["date"] >= wk) & (sdf["date"] < wk + pd.Timedelta(days=7))]
            if games.empty:
                continue
            m = model.DixonColes().fit(df[df["date"] < wk], wk)
            if not m.fitted:
                continue
            for _, g in games.iterrows():
                if not (m.knows(g["home"]) and m.knows(g["away"])):
                    continue
                lh, la = m.rates(g["home"], g["away"])
                M = model.score_matrix(lh, la, m.rho)
                po = model.outcome_probs(M)
                pov = model.over_prob(M, 2.5)
                res = "Home" if g["hg"] > g["ag"] else "Away" if g["ag"] > g["hg"] else "Draw"
                over = g["hg"] + g["ag"] > 2.5
                cand = []

                op = _price_cols(g, H2H_OPEN)
                cp = _price_cols(g, H2H_CLOSE)
                if op:
                    fair = devig(op)
                    cfair = devig(cp) if cp else None
                    preds.append({"model": [po["Home"], po["Draw"], po["Away"]], "open": fair,
                                  "close": cfair, "res": ["Home", "Draw", "Away"].index(res)})
                    for i, sel in enumerate(("Home", "Draw", "Away")):
                        cand.append(("h2h", sel, po[sel], fair[i], op[i],
                                     cfair[i] if cfair else None, res == sel))
                op = _price_cols(g, OU_OPEN)
                cp = _price_cols(g, OU_CLOSE)
                if op:
                    fair = devig(op)
                    cfair = devig(cp) if cp else None
                    for i, (sel, mp, won) in enumerate((("Over", pov, over), ("Under", 1 - pov, not over))):
                        cand.append(("totals", sel, mp, fair[i], op[i], cfair[i] if cfair else None, won))

                best = None
                for market, sel, mp, fair, price, cfair, won in cand:
                    if abs(mp - fair) > config.MAX_DISAGREE:
                        continue
                    if not (config.MIN_PRICE <= price <= config.MAX_PRICE):
                        continue
                    p = config.MODEL_WEIGHT * mp + (1 - config.MODEL_WEIGHT) * fair
                    ev = p * price - 1
                    if ev >= config.MIN_EV and (best is None or ev / (price - 1) > best["ev"] / (best["price"] - 1)):
                        best = {"league": code, "season": season, "date": g["date"].date().isoformat(),
                                "match": f"{g['home']} v {g['away']}", "market": market, "sel": sel,
                                "price": price, "ev": ev, "won": bool(won),
                                "units": price - 1 if won else -1.0,
                                "clv": price * cfair - 1 if cfair else None}
                if best:
                    bets.append(best)
    return bets, preds


def summarize(bets):
    if not bets:
        return {"bets": 0}
    n = len(bets)
    w = sum(b["won"] for b in bets)
    u = sum(b["units"] for b in bets)
    clv = [b["clv"] for b in bets if b["clv"] is not None]
    # rough 95% interval on ROI
    sd = float(np.std([b["units"] for b in bets])) if n > 1 else 0
    return {"bets": n, "wins": w, "losses": n - w, "units": round(u, 2), "roi": round(u / n, 4),
            "roi_ci95": [round(u / n - 1.96 * sd / math.sqrt(n), 4), round(u / n + 1.96 * sd / math.sqrt(n), 4)],
            "avg_price": round(float(np.mean([b["price"] for b in bets])), 3),
            "avg_clv": round(float(np.mean(clv)), 4) if clv else None,
            "beat_close": round(sum(c > 0 for c in clv) / len(clv), 3) if clv else None}


def logloss(preds, key):
    xs = [-math.log(max(p[key][p["res"]], 1e-9)) for p in preds if p[key] is not None]
    return round(float(np.mean(xs)), 4) if xs else None


def top_per_day(bets, k):
    out = []
    by = {}
    for b in bets:
        by.setdefault(b["date"], []).append(b)
    for d in sorted(by):
        out += sorted(by[d], key=lambda b: b["ev"] / (b["price"] - 1), reverse=True)[:k]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-seasons", type=int, default=3)
    args = ap.parse_args()
    today = date.today()

    all_bets, all_preds, per_league = [], [], {}
    for code, lg in config.LEAGUES.items():
        bets, preds = bets_for_league(code, lg, today, args.test_seasons)
        print(f"{code}: {len(bets)} bets, {len(preds)} matches priced")
        per_league[code] = {"name": lg["name"], **summarize(bets),
                            "model_logloss": logloss(preds, "model"),
                            "open_logloss": logloss(preds, "open"),
                            "close_logloss": logloss(preds, "close")}
        all_bets += bets
        all_preds += preds

    card = top_per_day(all_bets, config.PICKS_PER_DAY)
    curve, run = [], 0.0
    for b in sorted(card, key=lambda b: b["date"]):
        run += b["units"]
        curve.append([b["date"], round(run, 2)])
    if len(curve) > 300:
        step = len(curve) / 300
        curve = [curve[int(i * step)] for i in range(300)] + [curve[-1]]

    out = {
        "generated": today.isoformat(),
        "rules": {"model_weight": config.MODEL_WEIGHT, "min_ev": config.MIN_EV,
                  "picks_per_day": config.PICKS_PER_DAY, "price_range": [config.MIN_PRICE, config.MAX_PRICE],
                  "bet_price": "Pinnacle early line (football-data PSH/PSD/PSA, P>2.5/P<2.5)"},
        "seasons": sorted({b["season"] for b in all_bets}),
        "card": summarize(card),
        "all_qualifying": summarize(all_bets),
        "by_market": {m: summarize([b for b in card if b["market"] == m]) for m in ("h2h", "totals")},
        "by_league": per_league,
        "logloss": {"model": logloss(all_preds, "model"), "open": logloss(all_preds, "open"),
                    "close": logloss(all_preds, "close")},
        "curve": curve,
        "note": "Big 5 leagues only. Cup games have no free history with closing odds, so UCL/UEL picks are unbacktested.",
    }
    config.PUBLIC.mkdir(parents=True, exist_ok=True)
    (config.PUBLIC / "backtest.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("card", "all_qualifying", "logloss")}, indent=2))


if __name__ == "__main__":
    main()
