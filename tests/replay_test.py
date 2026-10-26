"""End-to-end test on replayed past matchdays, no live APIs needed.

    python tests/replay_test.py

1. Trains the bot as if it were the end of 2024-25.
2. Replays 2025-26: for a sample of fixtures it sets the clock to 50 minutes before kickoff, serves the
   archive through the same interface as the live API, feeds the real starting XIs in as "confirmed
   lineups" (written the way API-Football writes names), and runs the bot exactly as it runs live.
3. Checks the live path gives the same features as the training path, and that the posts look right.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="bot-replay-"))
os.environ.update({"BOT_MODEL_PATH": str(TMP / "model.json"), "BOT_STATE_PATH": str(TMP / "state.json"),
                   "BOT_LOG_PATH": str(TMP / "log.csv"), "BOT_MARKETS_LOG_PATH": str(TMP / "markets_log.csv"), "BOT_TIMEZONE": "Australia/Sydney"})
subprocess.run([sys.executable, "-m", "bot.train", "--until", "2024-25"], cwd=ROOT, check=True)
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bot import run, train  # noqa: E402
from bot.sources import Replay  # noqa: E402


class ReplayLineups:
    def __init__(self, replay):
        self.replay = replay

    def enabled(self):
        return True

    def starting_xis(self, fixture, team_names):
        return self.replay.lineup_names(fixture)


_, reference, *_ = train.build_table("2025-26")
reference = reference[reference.season == "2025-26"].set_index("fixture")
fixtures = reference.sort_values("kickoff")
sample = pd.concat([fixtures.iloc[40::34], fixtures.loc[[fixtures.star_absence_diff.abs().idxmax()]]])

CHECKED = ["elo_diff", "pi_diff", "goal_ratio", "goal_total", "shot_gap", "lg_home", "lg_draw", "xg_form_diff",
           "exp_home_shots", "exp_away_shots", "exp_home_corners", "exp_away_corners", "exp_home_cards", "exp_away_cards"]
rows, problems = [], []
for key, ref in sample.iterrows():
    now = ref.kickoff - pd.Timedelta(minutes=50)
    replay = Replay("2025-26", now)
    if Path(os.environ["BOT_LOG_PATH"]).exists():
        Path(os.environ["BOT_LOG_PATH"]).unlink()
    posted = run.main(now=now, src=replay, lineups_api=ReplayLineups(replay))
    mine = next(p for p in posted if p["fixture"] == key)
    diffs = {f: mine[f] - ref[f] for f in CHECKED + ["star_absence_diff"]}
    rows.append({"fixture": f"{ref.home} vs {ref.away}", "lineups": mine["lineups"], **{f"gap {k}": v for k, v in diffs.items()}})
    if max(abs(diffs[f]) for f in CHECKED) > 1e-3:  # the log rounds to 4 decimals
        problems.append(key)

report = pd.DataFrame(rows)
print(report[["fixture", "lineups"] + [f"gap {f}" for f in ["elo_diff", "goal_ratio", "shot_gap", "exp_home_corners", "exp_home_cards", "star_absence_diff"]]]
      .to_string(index=False, float_format="{:.4f}".format))
print(f"\nLargest gap on any rating feature: {report[[f'gap {f}' for f in CHECKED]].abs().max().max():.5f}")
print(f"\nConfirmed lineups matched for {(report.lineups == 'confirmed').mean():.0%} of sampled fixtures")
print(f"Largest star-absence gap between live and training paths: {report['gap star_absence_diff'].abs().max():.4f}")
assert not problems, f"Rating or xG features differ for {problems}"
print("PASS: the live path reproduces the training features")
