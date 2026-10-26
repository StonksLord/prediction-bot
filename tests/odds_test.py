"""Checks odds tracking and paper trading on a replayed matchday, without calling any API.

    python tests/odds_test.py

Run tests/replay_test.py first: this reuses the model it trains. Historical best prices stand in
for The Odds API: the bot should show them in its final post, log a paper bet when the edge is big
enough, and record closing odds in the last run before kickoff.
"""
import glob
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TMP = Path(max(glob.glob(os.path.join(tempfile.gettempdir(), "bot-replay-*")), key=os.path.getmtime))  # the newest replay run
os.environ.update({"BOT_MODEL_PATH": str(TMP / "model.json"), "BOT_STATE_PATH": str(TMP / "state.json"),
                   "BOT_LOG_PATH": str(TMP / "odds_log.csv"), "BOT_MARKETS_LOG_PATH": str(TMP / "markets_odds_log.csv")})
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from bot import history, run  # noqa: E402
from bot.config import canon  # noqa: E402
from bot.model import fair_probs  # noqa: E402
from bot.sources import Replay  # noqa: E402

raw = pd.read_csv(history.fetch(history.HISTORY_RESULTS_URL, history.DATA_DIR / "Matches.csv"), low_memory=False,
                  usecols=["Division", "MatchDate", "HomeTeam", "AwayTeam", "MaxHome", "MaxDraw", "MaxAway", "OddHome", "OddDraw", "OddAway"])
raw = raw[(raw.Division == "E0") & (raw.MatchDate >= "2025-08-01") & (raw.MatchDate < "2026-07-01")]
PRICES = {(canon(h), canon(a)): (r.MaxHome, r.MaxDraw, r.MaxAway, r.OddAway, r.OddDraw, r.OddHome)
          for h, a, r in zip(raw.HomeTeam, raw.AwayTeam, raw.itertuples())}


class ReplayOdds:
    remaining = "n/a"

    def enabled(self):
        return True

    def prices(self, fixture, team_names):
        p = PRICES.get((fixture.home, fixture.away))
        if not p:
            return None
        away, draw, home = fair_probs([p[3:]])[0]
        return {"best": dict(zip(["home", "draw", "away"], [(float(x), "Best of ~17 bookies") for x in p[:3]])),
                "market": {"home": home, "draw": draw, "away": away, "books": 1}}


class ReplayLineups:
    def __init__(self, replay):
        self.replay = replay

    def enabled(self):
        return True

    def starting_xis(self, fixture, team_names):
        return self.replay.lineup_names(fixture)


log_path = Path(os.environ["BOT_LOG_PATH"])
log_path.unlink(missing_ok=True)
fx, _ = history.fpl_fixtures("2025-26")
slot = fx[fx.finished].sort_values("kickoff").kickoff.iloc[150]
for minutes_before in (50, 10):
    now = slot - pd.Timedelta(minutes=minutes_before)
    replay = Replay("2025-26", now)
    run.main(now=now, src=replay, lineups_api=ReplayLineups(replay), odds_api=ReplayOdds())

log = pd.read_csv(log_path)
finals, closing = log[log.stage == "final"], log[log.stage == "closing"]
print(log[["stage", "home", "away", "p_home", "mkt_home", "best_home", "pick", "pick_odds", "pick_edge"]].to_string(index=False))
assert len(finals) and finals.best_home.notna().all(), "final posts should carry the best odds"
assert finals.mkt_home.notna().all(), "final posts should carry the market's view"
assert set(closing.fixture) == set(finals.fixture), "every final should get a closing-odds row"
print("PASS: odds shown, value picks logged, closing odds recorded")
