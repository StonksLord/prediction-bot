"""Settings for the prediction bot.

Secrets (API keys, webhook URLs) come from environment variables, never from this file.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.environ.get("BOT_MODEL_PATH", ROOT / "model" / "model.json"))
STATE_PATH = Path(os.environ.get("BOT_STATE_PATH", ROOT / "model" / "state.json"))
LOG_PATH = Path(os.environ.get("BOT_LOG_PATH", ROOT / "predictions_log.csv"))
MARKETS_LOG_PATH = Path(os.environ.get("BOT_MARKETS_LOG_PATH", ROOT / "markets_log.csv"))
DATA_DIR = Path(os.environ.get("BOT_DATA_DIR", ROOT / ".data"))

# Live sources
FPL_API = "https://fantasy.premierleague.com/api"
API_FOOTBALL = "https://v3.football.api-sports.io"
API_FOOTBALL_KEY = os.environ.get("API_FOOTBALL_KEY", "")
API_FOOTBALL_LEAGUE = 39  # Premier League
ODDS_API = "https://api.the-odds-api.com/v4"
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
ODDS_REGIONS = os.environ.get("ODDS_REGIONS", "au,uk,eu")  # each region costs one credit per call

# Paper trading
VALUE_EDGE = float(os.environ.get("VALUE_EDGE", "0.05"))  # only flag a bet when the bot's price beats the best odds by this much
MIN_VALUE_CHANCE = float(os.environ.get("MIN_VALUE_CHANCE", "0.5"))  # ...and the bot gives that result at least this chance
MAX_VALUE_ODDS = float(os.environ.get("MAX_VALUE_ODDS", "3.0"))      # ...at odds no longer than this (long shots lost 14-18%)
# Line shopping: back the market's own view when one bookie's price beats it. No model involved. The only rule that
# made money walk-forward (tests/papertrade.py): chosen on 2005-16, then +8.6% on 942 bets in 2016-26 at the best price.
SHOP_EDGE, SHOP_MIN_CHANCE, SHOP_MAX_ODDS = 0.05, 0.4, 2.5
CLOSING_WINDOW_MINUTES = 15  # record the closing odds in the last run before kickoff

# Where predictions get posted (set whichever you use)
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# Historical data for training
HISTORY_RESULTS_URL = "https://raw.githubusercontent.com/xgabora/Club-Football-Match-Data-2000-2025/main/data/Matches.csv"
FPL_HISTORY_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
FIRST_FPL_SEASON = "2019-20"     # from here on, Premier League results come from Fantasy Premier League
FIRST_TRAIN_SEASON = "2003-04"   # the base model learns from every league from here (earlier seasons warm the ratings up)
FIRST_XG_SEASON = "2022-23"      # xG and "starts" exist from here on, so the Premier League layer learns from here

# The 22 leagues the base model learns from (code: country, tier). Promotion links leagues within a country.
LEAGUES = {
    "E0": ("England", 1), "E1": ("England", 2), "E2": ("England", 3), "E3": ("England", 4), "EC": ("England", 5),
    "SC0": ("Scotland", 1), "SC1": ("Scotland", 2), "SC2": ("Scotland", 3), "SC3": ("Scotland", 4),
    "D1": ("Germany", 1), "D2": ("Germany", 2), "I1": ("Italy", 1), "I2": ("Italy", 2),
    "SP1": ("Spain", 1), "SP2": ("Spain", 2), "F1": ("France", 1), "F2": ("France", 2),
    "N1": ("Netherlands", 1), "B1": ("Belgium", 1), "P1": ("Portugal", 1), "T1": ("Turkey", 1), "G1": ("Greece", 1),
}

# The model. Every setting below won its place in the Simulation Lab (184,000 matches, walk-forward).
BASE_FEATURES = ["elo_diff", "pi_diff", "goal_ratio", "goal_total", "shot_gap", "lg_home", "lg_draw"]
EPL_FEATURES = ["xg_form_diff", "star_absence_diff"]  # the Premier League layer
ELO = {"k": 20, "home_adv": 40, "carry": 0.8, "mov": "log"}
PI = {"lam": 0.05, "gamma": 0.9, "c": 3.0, "carry": 0.8}
GOALS = {"eta": 0.02, "carry": 0.8, "eta_league": 0.002}  # attack and defence ratings from goals
SHOTS = {"eta": 0.02, "carry": 0.8, "eta_league": 0.002}  # the same, from shots on target
CORNERS = {"eta": 0.02, "carry": 0.8, "eta_league": 0.002}  # side markets (Markets Lab): corners...
CARDS = {"eta": 0.02, "carry": 0.8, "eta_league": 0.002}    # ...and cards (yellow plus red)
PROMOTED_BETA = 0.75   # promoted clubs start from the relegated clubs' average, plus 75% of how dominant they were below
CONTEXT_SEASONS = 3    # league home-win and draw rates come from the last three seasons
XG_WINDOW = 6          # matches of xG form
LINEUP_LOOKBACK = 6    # matches used to work out the usual XI
MIN_MATCHES_FOR_XI = 3  # fewer matches than this in the season and star absence is set to zero
STAR_THRESHOLD = 0.03  # only XI value drops bigger than 3% count
STAR_CAP = 0.04        # ...and the effect stops growing past an XI about 7% cheaper than usual (tested season by season)
FIT_CHANCE = 75        # FPL "chance of playing" at or above this counts as available

# When to post
PREVIEW_HOURS = 36           # preview every fixture this far ahead, using the expected XI
LINEUP_WINDOW_MINUTES = 75   # start looking for confirmed lineups this long before kickoff
LAST_CALL_MINUTES = 20       # no lineups by now: post the final prediction with the expected XI

# Fantasy Premier League spells some clubs differently from the historical data
TEAM_NAMES = {
    "Man Utd": "Man United", "Spurs": "Tottenham", "Sheffield Utd": "Sheffield United",
    "Coventry City": "Coventry", "Hull City": "Hull", "Ipswich Town": "Ipswich", "Nottm Forest": "Nott'm Forest",
}
POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def canon(name):
    name = str(name).strip()
    return TEAM_NAMES.get(name, name)
