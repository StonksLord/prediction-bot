"""Checks the live Fantasy Premier League parsing without calling the API.

    python tests/live_shape_test.py

It builds API-shaped JSON (bootstrap-static, fixtures, element-summary) from the archive for a past
moment, feeds it to the live client, and checks the bot gives the same prediction as the replay path.
Run tests/replay_test.py first: this reuses the model it trains.
"""
import glob
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TMP = Path(max(glob.glob(os.path.join(tempfile.gettempdir(), "bot-replay-*")), key=os.path.getmtime))  # the newest replay run
os.environ.update({"BOT_MODEL_PATH": str(TMP / "model.json"), "BOT_STATE_PATH": str(TMP / "state.json"),
                   "BOT_LOG_PATH": str(TMP / "live_log.csv"), "BOT_MARKETS_LOG_PATH": str(TMP / "markets_live_log.csv")})
sys.path.insert(0, str(ROOT))

import ast  # noqa: E402

import pandas as pd  # noqa: E402

from bot import history, run  # noqa: E402
from bot.sources import FplLive, Replay  # noqa: E402

SEASON = "2025-26"
fx, names = history.fpl_fixtures(SEASON)
target = fx[fx.finished].sort_values("kickoff").iloc[200]
now = target.kickoff - pd.Timedelta(hours=3)
hist = history.fpl_player_history(SEASON)
raw_gw = pd.read_csv(history.fpl_path(SEASON, "merged_gw.csv"), low_memory=False)
raw_fx = pd.read_csv(history.fpl_path(SEASON, "fixtures.csv")).set_index("id")
seen = hist[hist.kickoff < now].sort_values("kickoff").groupby("element").tail(1)
ids = {v: k for k, v in names.items()}
types = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}

payload = {
    "bootstrap-static/": {
        "events": [{"id": 1, "deadline_time": fx.kickoff.min().isoformat()}],
        "teams": [{"id": i, "name": n} for i, n in names.items()],
        "elements": [{"id": int(r.element), "first_name": r["name"].split()[0], "second_name": " ".join(r["name"].split()[1:]) or r["name"],
                      "web_name": r["name"].split()[-1], "team": ids[r.team], "element_type": types[r.position],
                      "now_cost": int(round(r.price * 10)), "status": "a", "chance_of_playing_next_round": None,
                      "news": "", "minutes": 90} for _, r in seen.iterrows()],
    },
    "fixtures/": [{"id": int(f.fixture_id), "kickoff_time": f.kickoff.isoformat() if pd.notna(f.kickoff) else None,
                   "team_h": int(f.home_id), "team_a": int(f.away_id), "finished": bool(f.finished and f.kickoff < now),
                   "team_h_score": int(f.hg) if f.finished and f.kickoff < now else None,
                   "team_a_score": int(f.ag) if f.finished and f.kickoff < now else None,
                   "stats": ast.literal_eval(raw_fx.loc[f.fixture_id, "stats"]) if f.finished and f.kickoff < now else []}
                  for f in fx.itertuples()],
}
for element, rows in raw_gw.groupby("element"):
    rows = rows[pd.to_datetime(rows.kickoff_time, utc=True) < now]
    payload[f"element-summary/{int(element)}/"] = {"history": [
        {"fixture": int(r.fixture), "was_home": bool(r.was_home), "kickoff_time": r.kickoff_time, "minutes": int(r.minutes),
         "starts": int(r.starts), "expected_goals": str(r.expected_goals), "expected_goals_conceded": str(r.expected_goals_conceded),
         "expected_assists": str(r.expected_assists), "goals_scored": int(r.goals_scored), "assists": int(r.assists),
         "yellow_cards": int(r.yellow_cards), "red_cards": int(r.red_cards), "value": int(r.value)} for r in rows.itertuples()]}


class FakeFpl(FplLive):
    def _get(self, path):
        return payload[path]


class NoLineups:
    def enabled(self):
        return False


live = run.main(dry_run=True, now=now, src=FakeFpl(), lineups_api=NoLineups())
replay = run.main(dry_run=True, now=now, src=Replay(SEASON, now), lineups_api=NoLineups())
cols = ["p_home", "p_draw", "p_away", "elo_diff", "goal_ratio", "shot_gap", "xg_form_diff"]
a = pd.DataFrame(live).set_index("fixture")[cols]
b = pd.DataFrame(replay).set_index("fixture")[cols]
gap = (a - b.reindex(a.index)).abs().max().max()
print(f"{len(a)} previews from the live client, largest gap vs replay: {gap:.5f}")
assert len(a) and gap < 1e-3
print("PASS: the live client parses API-shaped data the same way")
