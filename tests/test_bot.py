"""Run with: python -m pytest -q tests"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bot import model, names, odds  # noqa: E402
from bot.run import grade_pick  # noqa: E402


def test_score_matrix_sums_and_markets():
    M = model.score_matrix(1.6, 1.1, -0.06)
    assert abs(M.sum() - 1) < 1e-9
    p = model.outcome_probs(M)
    assert abs(sum(p.values()) - 1) < 1e-9
    assert p["Home"] > p["Away"]
    assert 0.4 < model.over_prob(M, 2.5) < 0.65


def test_dixon_coles_recovers_strengths():
    rng = np.random.default_rng(1)
    teams = [f"T{i}" for i in range(12)]
    att = np.linspace(-0.4, 0.4, 12)
    rows, d0 = [], pd.Timestamp("2024-08-01")
    for k in range(6):
        for i in range(12):
            for j in range(12):
                if i != j:
                    lh, la = np.exp(0.25 + 0.25 + att[i] - att[j] * 0.5), np.exp(0.25 + att[j] - att[i] * 0.5)
                    rows.append({"date": d0 + pd.Timedelta(days=k * 60 + i), "home": teams[i], "away": teams[j],
                                 "hg": rng.poisson(lh), "ag": rng.poisson(la)})
    m = model.DixonColes(xi=0.0005).fit(pd.DataFrame(rows), d0 + pd.Timedelta(days=400))
    assert m.att["T11"] > m.att["T0"] + 0.4
    assert 0.1 < m.home < 0.4


def test_bridge_matches_elo_expectancy():
    lh, la = model.bridge_rates(2.8, 150)
    p = model.outcome_probs(model.score_matrix(lh, la, -0.05))
    assert abs(p["Home"] + p["Draw"] / 2 - model.elo_expectancy(150)) < 1e-3
    assert abs(lh + la - 2.8) < 1e-6


def test_name_matching():
    epl = ["Man City", "Man United", "Nott'm Forest", "Brighton", "Tottenham", "Wolves", "Arsenal"]
    assert names.match("Manchester City", epl) == "Man City"
    assert names.match("Nottingham Forest", epl) == "Nott'm Forest"
    assert names.match("Brighton and Hove Albion", epl) == "Brighton"
    assert names.match("Mystery FC", epl) is None
    seriea = ["Inter", "Milan", "Roma", "Verona"]
    assert names.match("Inter Milan", seriea) == "Inter"
    assert names.match("AC Milan", seriea) == "Milan"
    assert names.match("Hellas Verona", seriea) == "Verona"
    l1 = ["Paris SG", "Paris FC", "Lens"]
    assert names.match("Paris Saint Germain", l1) == "Paris SG"
    assert names.match("Paris FC", l1) == "Paris FC"
    assert names.match("1. FC Köln", ["FC Koln", "Mainz"]) == "FC Koln"


def test_devig_power():
    f = odds.devig_power([1.5, 4.2, 7.0])
    assert abs(sum(f) - 1) < 1e-9
    prop = odds.devig([1.5, 4.2, 7.0])
    assert f[2] < prop[2] and f[0] > prop[0]     # trims long shots, not favorites


def test_devig_and_american():
    f = odds.devig([2.0, 3.5, 4.0])
    assert abs(sum(f) - 1) < 1e-9
    assert odds.american(2.05) == "+105"
    assert odds.american(1.5) == "-200"


def test_grading():
    p = {"market": "totals", "selection": "Over", "point": 2.5, "price": 1.9}
    grade_pick(p, 2, 1)
    assert p["status"] == "won" and abs(p["units"] - 0.9) < 1e-9
    p = {"market": "h2h", "selection": "Draw", "point": None, "price": 3.4}
    grade_pick(p, 1, 2)
    assert p["status"] == "lost" and p["units"] == -1


def test_full_pipeline_model_mode(tmp_path):
    _pipeline(tmp_path, "model")


def test_full_pipeline(tmp_path):
    status = _pipeline(tmp_path, "shop")
    assert status["record"]["clv_n"] == status["record"]["graded"]   # closing line captured


def _pipeline(tmp_path, strategy):
    fx = tmp_path / "fx"
    subprocess.run([sys.executable, str(ROOT / "tests" / "make_fixtures.py"), str(fx)], check=True)
    env = dict(os.environ, FIXTURES=str(fx), PUBLIC_DIR=str(tmp_path / "pub"),
               PRIVATE_DIR=str(tmp_path / "priv"), CACHE_DIR=str(tmp_path / "cache"), STRATEGY=strategy)
    run = lambda now: subprocess.run([sys.executable, "-m", "bot.run"], cwd=ROOT, check=True,
                                     env=dict(env, NOW=now), capture_output=True, text=True)
    run("2026-10-15T08:00:00Z")      # cup day: picks made
    picks = json.loads((tmp_path / "pub" / "picks.json").read_text())
    assert picks and all(p["status"] == "pending" for p in picks)
    assert len(picks) <= 2
    first = len(picks)
    assert all(p["strategy"] == strategy for p in picks)
    run("2026-10-15T17:30:00Z")      # inside 2h of kickoff: re-read the closing line
    picks2 = json.loads((tmp_path / "pub" / "picks.json").read_text())
    assert [p["id"] for p in picks2] == [p["id"] for p in picks]      # posted picks don't change
    assert [p["price"] for p in picks2] == [p["price"] for p in picks]
    run("2026-10-17T08:00:00Z")      # cup picks lock + grade, league picks made
    picks = json.loads((tmp_path / "pub" / "picks.json").read_text())
    status = json.loads((tmp_path / "pub" / "status.json").read_text())
    graded = [p for p in picks if p["status"] in ("won", "lost")]
    assert len(graded) == first
    assert status["record"]["graded"] == first
    assert all("Mystery" in u for u in status["unmatched_teams"])
    # pick count per slate day never exceeds the limit
    from collections import Counter
    from bot.run import slate_day
    assert max(Counter(slate_day(p["kickoff"]) for p in picks).values()) <= 2
    return status


def test_xg_join(tmp_path):
    fx = tmp_path / "fx"
    subprocess.run([sys.executable, str(ROOT / "tests" / "make_fixtures.py"), str(fx)], check=True)
    os.environ.update(FIXTURES=str(fx), CACHE_DIR=str(tmp_path / "cache"))
    try:
        import importlib
        from bot import config, history
        importlib.reload(config)
        importlib.reload(history)
        from datetime import date
        df = history.load_league("E0", date(2026, 10, 14), back=2, xg_slug="EPL")
        assert df["hxg"].notna().mean() > 0.95          # names learned without any alias list
        m = model.DixonColes(xg_weight=0.7).fit(df, pd.Timestamp("2026-10-14"))
        assert m.xg_share > 0.95 and m.fitted
    finally:
        for k in ("FIXTURES", "CACHE_DIR"):
            os.environ.pop(k, None)
