# Premier League prediction bot

The bot predicts every Premier League match twice and posts the result to Discord or Telegram:

1. **Preview**, up to 36 hours before kickoff, using each team's expected XI (Fantasy Premier League injury and suspension flags included).
2. **Final**, once the confirmed lineups are out (about 20 to 40 minutes before kickoff). If no lineups have appeared 20 minutes before kickoff, it posts the final prediction with the expected XI instead.

Every prediction is saved to `predictions_log.csv`. With a free odds key, the bot also shows the market's view and the best price on offer next to its own, flags value bets, and records the closing odds just before kickoff, so you can paper trade and see whether it's really beating the market.

Under the match result, each post also gives chances for the other bets on a betting site's menu: total goals, both teams to score, likely scores, the half-time result, corners, cards, shots on target and the likeliest scorers. Those go to `markets_log.csv`.

## What goes into a prediction

The model has two layers. Every setting was chosen in the Simulation Lab, which replayed 184,000 real matches season by season.

**Layer 1: team strength, learned from 22 leagues.** Four ratings, rebuilt from every result since 2000 and updated live from this season's results:

| Rating | What it captures |
| --- | --- |
| Elo | One strength number per club |
| Pi-ratings | Separate home and away strength, in goals |
| Attack and defence (goals) | Expected goals for each side, and how open the game should be |
| Attack and defence (shots on target) | The same from shots on target, which show who's really playing well sooner than goals do |

Plus each league's usual home-win and draw rates. Promoted clubs start from the relegated clubs' average, nudged by how dominant they were in the Championship. This layer learns from about 114,000 matches in 22 European leagues, because the Premier League alone isn't enough data (it scored worse in testing).

Fantasy Premier League doesn't list shots, but every shot on target is either saved or scored, so a side's shots on target are the other keeper's saves plus its own goals. That matches the official match stats to within one shot 99% of the time.

**Layer 2: Premier League news.** Two factors only Fantasy Premier League has (from 2022-23) nudge the result:

| Factor | What it is |
| --- | --- |
| xG form | Each team's xG for minus xG against over its last 6 matches |
| Star absence | How much cheaper today's starting XI is than the usual XI, counting drops bigger than 3%, up to about 7% (past that there's too little data to say it keeps growing) |

**How good is it?** Trained only on earlier seasons and tested on 2023-24, 2024-25 and 2025-26 (1,140 matches):

| | Log loss (lower is better) | Favourite won |
| --- | --- | --- |
| This bot | 0.975 | 54.2% |
| Previous version of the bot | 0.983 | |
| Elo alone | 0.985 | |
| Bookie (Bet365, margin removed) | 0.966 | 54.0% |

It now picks winners about as well as the bookie, but the bookie still prices the chances better. (The star-absence cap was picked by testing on these same seasons, so the gap to the previous version is a little flattering.) In the Simulation Lab the bot trailed the bookie in every situation tested, and its value picks lost money. Treat every post as a learning tool, not betting advice.

## A post looks like this

```
⚽ Liverpool vs Everton
Sat 20 Sep, 12:30 UK (Sat 21:30 Sydney) · Final, lineups confirmed
Bot: Liverpool 66% · Draw 19% · Everton 15%
Market: Liverpool 63% · Draw 21% · Everton 16% (24 bookies)
Fair odds: 1.52 · 5.26 · 6.67
Best odds: 1.60 Unibet · 4.80 Betfair · 6.50 Sportsbet
Value: none at these prices
Goals: Over 2.5 61% · Both score 49% · Likely scores 2-0 11% · 2-1 10% · 1-0 10%
Half-time: Liverpool 47% · Draw 37% · Everton 16%
Corners: Over 9.5 58% · Cards: Over 3.5 50% · Shots on target: Over 8.5 57%
Scorers: Salah 41% · Ekitike 36% · Gakpo 28% · Wirtz 24%
Why: Elo gap 176 in Liverpool's favour · expected goals 1.9 v 0.8 · xG form Liverpool +0.51, Everton +0.21 · Liverpool XI 4% cheaper than usual (Wirtz out)
```

- **Bot** is the bot's own view. **Market** is every bookie's odds with their margin taken out, averaged.
- **Fair odds** are 1 divided by the bot's probability. **Best odds** are the highest price any bookie offers on each result.
- **Value**: the edge is the bot's chance times the best odds, minus 1. +13% means a $1 bet would return $1.13 on average if the bot were right. Only edges of at least 5% get flagged (change this with `VALUE_EDGE`).
- **Heads-up** appears when the bot is 10+ points away from the market. In the Simulation Lab, when that happened the bookie's pick won 54% of the time and the bot's 50%.
- **Goals, Half-time, Corners, Cards, Shots on target and Scorers** are the side markets (next section).

## Side markets

These come from one model of the score (every scoreline from 0-0 to 10-10, with each side's expected goals from the ratings) and separate models for corners, cards and shots on target. Every goals bet is read off the same score grid, so two bets from one game (say Liverpool to win and over 2.5 goals) come out with the right combined chance. Never multiply two legs from the same game: in testing that was off by 3 to 4 points.

Scorers are each starter's chance of scoring at least once. It uses his goals and xG per 90 minutes (pulled toward his position's average when he has little history), how long he usually plays, and how many goals his team is expected to score tonight. Goalkeepers are left out.

**How good are they?** Tested in the Markets Lab on up to 184,000 matches (2005-06 to 2023-24), against each league's usual rate for that bet. Skill is how much closer the bot gets than that rate: 0% means no better.

| Bet | Skill | Honest? |
| --- | --- | --- |
| Match result (home or away win) | about 8% | Yes |
| Shots on target, total and team | 6 to 12% | Yes |
| Corners: which side gets more | about 7% | Yes |
| Home win and over 2.5, same game | about 5.5% | Yes |
| Half-time result | about 4% | Yes |
| Cards, total | about 3% | A little low: says 51% for over 3.5 cards when 53% happened (bookings are rising, and there's no referee data) |
| Anytime goalscorer | about 3% | Yes (said 11.3%, happened 11.0%) |
| Corners, total | about 1.5% | Yes |
| Over 2.5 goals, both teams score | about 1% or less | Yes, but close to the league average |
| Draw, odd/even goals | about 0% | |

So the bot knows the most about who wins and which side dominates (corners, shots), and the least about how many goals a game will have. Odds for corners, cards, scorers and both teams to score aren't in any free history, so those bets were checked for accuracy only, not paper traded. Over/under 2.5 goals was paper traded at real odds: it lost about 8% at Bet365 and about 2% at the best price.

**Corners lag a bit.** Fantasy Premier League has no corners, so the bot's corners ratings come from the results archive when it retrains each Monday. Corners from the last few days may not be counted yet.

`python -m bot.scoreboard` also shows how the side markets have done: what the bot said for over 2.5 goals, both teams to score and over 3.5 cards against how often they happened.

## Set it up (about 15 minutes)

1. Create a new GitHub repository and upload everything in this folder. A public repository gets unlimited free GitHub Actions minutes. A private one gets a monthly free allowance, which this bot should stay inside.
2. Make somewhere for posts to go. For Discord, create a webhook for a channel and copy its URL. For Telegram, create a bot with BotFather, then get your chat ID.
3. In the repository's settings, add these as GitHub Actions secrets. You only need the ones you use.
    - `DISCORD_WEBHOOK_URL`
    - `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`
    - `API_FOOTBALL_KEY` (optional): a free key from api-football.com. Without it the bot still works, but final posts use the expected XI instead of confirmed lineups.
    - `ODDS_API_KEY` (optional): a free key from the-odds-api.com. Without it the bot still predicts, but it can't show the market, flag value bets or paper trade.
4. Open the Actions tab, pick **Prediction bot**, and run it by hand once. If no match is within 36 hours it will just say "Nothing to post this run."

After that it runs by itself every 15 minutes between 10:00 and 21:45 UTC, and retrains every Monday (which picks up each finished season automatically).

## Run it on your own computer

```
pip install -r requirements.txt
python -m bot.run --dry-run   # prints predictions, posts nothing, logs nothing
python -m bot.run             # the real thing
python -m bot.scoreboard      # how the final predictions and paper bets have done so far
python -m bot.train           # retrain from the public datasets (about a minute)
```

Set `BOT_TIMEZONE` (for example `Australia/Sydney`) to see kickoff times in your own timezone too.

## Paper trading

Every final prediction with a value pick counts as a one-unit paper bet at the best price shown. About 10 minutes before kickoff the bot quietly records the closing odds. `python -m bot.scoreboard` then tells you four things, fastest signal first:

1. **Price movement.** How often the odds the bot took were bigger than the closing odds. The closing odds are the market's final, best-informed guess, so beating them most of the time (say 55% or more over a few hundred picks) is the quickest sign of a real edge.
2. **Bot vs market.** The same matches scored for the bot and for the market's view.
3. **Accuracy.** How often the bot's favourite won, and how much chance it gave to the result that actually happened.
4. **Profit.** Flat one-unit bets on every value pick. This is the noisiest of the four: even a genuinely good bettor can be down after 200 bets.

**What to expect, from the Simulation Lab** (128,000 matches, 2007-08 to 2023-24, odds from a few days before kickoff): the bot's value picks lost about 8% at Bet365 prices and about 3% at the best price across 17 bookies. The only bets that made paper money were ones where the best price beat Bet365's own fair odds, and that worked just as well with no model at all (about +3.6% on 14,000 bets, before counting how hard those prices are to get in practice). If the bot ever has an edge, the likeliest place is the hour before kickoff, when confirmed lineups land and some bookies are slow to move. The closing-odds tracking is there to find out.

## How it was tested

- `python tests/replay_test.py` trains the bot as if it were the end of 2024-25, then replays a sample of 2025-26 matchdays. It sets the clock to 50 minutes before kickoff, feeds the real starting XIs in as if they came from API-Football, and checks the live path gives the same features as the training path (every rating, xG form and star absence).
- `python tests/live_shape_test.py` feeds the live client API-shaped data built from the archive and checks it parses everything the same way, including shots on target from saves.
- `python tests/odds_test.py` (run after the replay test) uses historical prices in place of The Odds API and checks the final post shows the market and the best odds, value picks get logged, and closing odds get recorded.

None of the tests touch the live APIs, so the first real run is the final check. Run `python -m bot.run --dry-run` once before switching posting on.

## Things to know

- The Fantasy Premier League API is free and needs no key, but it is unofficial, so its field names can change without warning.
- API-Football's free plan allows 100 requests a day. Each final prediction uses two, so a full matchday of ten games uses about twenty.
- The Odds API's free plan gives 500 credits a month. Each run that needs prices makes one call for every Premier League match at once, costing one credit per region (three with the default `au,uk,eu`). That works out to roughly 150 to 300 credits a month. If you run low, set `ODDS_REGIONS` to `uk,eu` or just `au`. The bot prints the credits left after each run.
- GitHub can start scheduled runs a few minutes late, which is why the bot looks for lineups from 75 minutes before kickoff.
- Training downloads one 45 MB file of results from 22 leagues, plus the Fantasy Premier League archive.
- The model only knows what's in its factors. A manager benching a star for a cup final, or a player carrying a knock that isn't flagged, will catch it out.

## Files

| File | What it does |
| --- | --- |
| `bot/run.py` | One run: update ratings, find fixtures, build features, predict, post, log |
| `bot/train.py` | Builds the training data, fits both layers and writes `model/model.json` and `model/state.json` |
| `bot/ratings.py` | Elo, pi-ratings and the attack-and-defence ratings, plus how clubs move between seasons and leagues |
| `bot/players.py` | xG form, the usual XI, star absence and the expected XI |
| `bot/sources.py` | The live Fantasy Premier League, API-Football and The Odds API clients, plus the replay used in tests |
| `bot/history.py` | Downloads the historical datasets from GitHub and works out shots on target from saves |
| `bot/model.py` | The two model layers, stored as plain JSON |
| `bot/markets.py` | The side markets: score grid, goals and half-time bets, corners, cards, shots on target and scorers |
| `bot/post.py` | Sends posts to Discord and Telegram |
| `bot/scoreboard.py` | Scores the logged predictions and paper bets against real results, the market and closing odds |

## Data

- Historical results, match stats and odds since 2000: [xgabora/Club-Football-Match-Data-2000-2025](https://github.com/xgabora/Club-Football-Match-Data-2000-2025)
- Fantasy Premier League history: [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League)
- Live data: the Fantasy Premier League API and, optionally, API-Football and The Odds API
