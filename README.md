# Stoppage Time

> **Current strategy: line shopping.** The bot treats Pinnacle's no-vig price as
> the true odds and posts a bet only when a US book pays at least 2% more than
> that. Posted picks never change, and the bot re-reads Pinnacle just before
> kickoff to measure closing-line value (CLV), the fastest honest test of an edge.
> The original goal model backtested at a loss, so it's off by default
> (`STRATEGY=model` turns it back on). See `bot/shop_backtest.py`.

A soccer betting-signals bot for the Premier League, La Liga, Bundesliga, Serie A,
Ligue 1, Champions League and Europa League. Match markets only: result (1X2) and
over/under goals.

It runs free on GitHub Actions and publishes a phone-friendly page on GitHub Pages.
Picks lock at kickoff and grade themselves from the final score.

## How it works

1. **History.** Pulls a few seasons of results for each league from
   [football-data.co.uk](https://www.football-data.co.uk) (free, includes closing odds).
2. **Model.** Fits a Dixon-Coles goal model per league: each team gets an attack and
   defense rating, weighted toward recent games, with a home-advantage term. Ratings
   are fit to a 70/30 blend of expected goals (xG, from [Understat](https://understat.com))
   and actual goals, since xG is far less noisy. Set `XG_WEIGHT=0` for goals only. That
   produces a full scoreline grid, so it prices both 1X2 and totals.
3. **European cups.** Teams from different leagues can't be compared directly, so cup
   games use [ClubElo](http://clubelo.com) ratings to set who's stronger and the
   domestic models to set how many goals to expect. (Unbacktested; see below.)
4. **Odds.** Pulls prices from [The Odds API](https://the-odds-api.com). The fair
   market probability is the median of de-vigged prices across books, including
   Pinnacle (the sharpest book).
5. **Picks.** Blends model and market 50/50 (markets are smart; the model only gets
   half a vote), and keeps bets with at least 3% expected value at the best US-book
   price. Ranks by edge relative to risk, so long shots don't crowd the card. Up to 2
   picks per day.
6. **Grading.** After full time, results come from The Odds API scores feed.

## Setup (about 10 minutes)

1. **Create a repo** on GitHub (public is simplest for free Pages) and upload all of
   these files, keeping the folder structure. `.github/workflows` must be included.
2. **Get an Odds API key** at the-odds-api.com.
3. **Add the key:** repo **Settings → Secrets and variables → Actions → New
   repository secret**. Name: `ODDS_API_KEY`.
4. **Turn on Pages:** **Settings → Pages → Build and deployment → Deploy from a
   branch**, branch `main`, folder `/docs`.
5. **Allow the bot to commit:** **Settings → Actions → General → Workflow
   permissions → Read and write permissions**.
6. **Run the backtest:** **Actions → Backtest → Run workflow**. Takes a few minutes.
7. **Run the bot once:** **Actions → Picks → Run workflow**. After that it runs every
   2 hours by itself.

Your site will be at `https://<your-username>.github.io/<repo-name>/`.

## Reading the backtest

The backtest replays past seasons week by week, fitting the model only on games
before each week, and "bets" at Pinnacle's early line. Two numbers matter most:

- **Closing-line value (CLV):** did our price beat where the market closed? Positive
  average CLV is the best evidence of a real edge, much better than win-loss.
- **ROI 95% range:** if it includes zero, the profit could easily be luck.

If CLV is negative, the model isn't beating the market yet, and the page says so.
Try adjusting `MODEL_WEIGHT`, `MIN_EV`, or the time decay `XI` in `bot/config.py`,
then rerun the backtest. Change one thing at a time to avoid curve-fitting.

## Odds API credits

Each odds pull costs 3 credits (match result, totals and handicap; listing up to 10
bookmakers counts as one region). The schedule check (`/events`) is free. The bot
only pulls a competition when it has games in the next 36 hours, at most every 8
hours, plus once inside 2 hours of kickoff for any posted pick (to read the
closing line).

On a typical week that's roughly 1,000–1,500 credits a month, more than the free
500. When credits drop below 15 the bot stops pulling odds until the month resets.
Options:
- Add a repo variable (**Settings → Secrets and variables → Actions → Variables**)
  `REFRESH_HOURS` = `16` for fewer pulls, or `MARKETS` = `h2h,totals` to drop handicaps.
- Or use the paid tier (20,000 credits). Then also add `williamhill_us` (Caesars)
  and `fanatics` to `BOOKMAKERS`, and consider `REFRESH_HOURS` = `2`.

## Fixing team names

Every source spells teams differently ("Nott'm Forest" vs "Nottingham Forest").
Most are matched automatically. If a game is skipped, it appears under
`unmatched_teams` in `docs/data/status.json`. Add it to `data/aliases.json`:

```json
{
  "teams": { "Name As The Odds API Writes It": "Name As football-data Writes It" },
  "elo":   { "football-data Name": "ClubElo Name" }
}
```

## Settings worth knowing (`bot/config.py`)

| Setting | Default | What it does |
|---|---|---|
| `PICKS_PER_DAY` | 10 | Max picks per day (Pacific time), one per match |
| `MIN_EV` | 0.03 | Minimum expected value to bet |
| `MODEL_WEIGHT` | 0.5 | How much to trust the model vs the market |
| `MIN_PRICE`/`MAX_PRICE` | 1.45 / 5.0 | Decimal odds range (about -220 to +400) |
| `BET_BOOKS` | US books | Where the listed price comes from |
| `SLATE_TZ` | America/Los_Angeles | Which day a pick counts toward |

## Running locally

```bash
pip install -r requirements.txt
ODDS_API_KEY=... python -m bot.run
python -m bot.backtest
python -m pytest -q tests      # uses simulated data, no network needed
```

## Honest limits

- Cup picks (UCL/UEL) have no backtest, because there's no free cup history with
  closing odds. Watch their live record separately.
- Lineups, injuries and rotation aren't modeled; markets know about them, which is
  part of why the bot blends with the market.
- The live price is checked every few hours, so a book may have moved by the time
  you look.

Bet small, and only where it's legal for you.
