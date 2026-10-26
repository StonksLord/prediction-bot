"""How has the bot done so far? Your paper-trading scoreboard.

    python -m bot.scoreboard

Four things to watch, in order of how fast they tell you something:
1. Price movement: did the odds on the bot's picks shrink before kickoff? Beating the closing odds
   most of the time is the quickest sign of a real edge.
2. Bot vs market: the same matches scored for the bot and for the bookies' own view. The Simulation Lab says
   to expect the market to come out slightly ahead; the bot catching it would be big news.
3. Accuracy: how often the favourite won, and how much chance the bot gave to what actually happened.
4. Profit: flat one-unit paper bets on every value pick. This needs the most matches to mean anything.
"""
import numpy as np
import pandas as pd

from .config import MARKETS_LOG_PATH
from .run import read_log
from .sources import FplLive

SIDES = ["away", "draw", "home"]


def markets_report(fx):
    """Side markets: how often each one landed against what the bot said, for finished matches."""
    if not MARKETS_LOG_PATH.exists():
        return
    mk = pd.read_csv(MARKETS_LOG_PATH)
    mk = mk[mk.stage == "final"].drop_duplicates("fixture", keep="last")
    cols = ["fixture", "hg", "ag"] + [c for c in ["hcd", "acd"] if c in fx]
    done = mk.merge(fx[fx.finished][cols], on="fixture")
    if done.empty:
        return
    tests = {"Over 2.5 goals": ("goals_over_2.5", done.hg + done.ag > 2.5), "Both teams score": ("btts", (done.hg > 0) & (done.ag > 0))}
    if "hcd" in done:
        tests["Over 3.5 cards"] = ("cards_over_3.5", done.hcd + done.acd > 3.5)
    print(f"Side markets on {len(done)} matches (said vs happened, and Brier score: lower is better, 0.25 is a coin flip):")
    for name, (col, y) in tests.items():
        p, yy = done[col].astype(float), y.astype(float)
        print(f"  {name}: said {p.mean():.0%}, happened {yy.mean():.0%}, Brier {((p - yy) ** 2).mean():.3f}")


def main():
    log = read_log()
    finals = log[log.stage == "final"].drop_duplicates("fixture", keep="last")
    closing = log[log.stage == "closing"].drop_duplicates("fixture", keep="last").set_index("fixture")
    fx = FplLive().fixtures()
    done = finals.merge(fx[fx.finished][["fixture", "hg", "ag"]], on="fixture")

    bets = finals[finals.pick.notna()].copy()
    if len(bets):
        bets["close_odds"] = [closing.loc[f, f"best_{p}"] if f in closing.index else np.nan for f, p in zip(bets.fixture, bets.pick)]
        moved = bets.dropna(subset=["close_odds"])
        if len(moved):
            beat = (moved.pick_odds > moved.close_odds).mean()
            gain = (moved.pick_odds / moved.close_odds - 1).mean()
            print(f"Price movement: the bot's odds beat the closing odds on {beat:.0%} of {len(moved)} picks "
                  f"(average {gain:+.1%}). Over about 55% for a few hundred picks is a good sign.")

    if done.empty:
        print("No finished matches with a final prediction yet.")
        return
    result = np.select([done.hg > done.ag, done.hg == done.ag], ["home", "draw"], default="away")
    rows = np.arange(len(done))
    pos = np.array([SIDES.index(r) for r in result])
    bot = done[["p_away", "p_draw", "p_home"]].to_numpy(float)
    print(f"Accuracy: the bot's favourite won {(bot.argmax(axis=1) == pos).mean():.0%} of {len(done)} matches, "
          f"and it gave the real result an average chance of {bot[rows, pos].mean():.0%}.")

    mkt = done[["mkt_away", "mkt_draw", "mkt_home"]].to_numpy(float)
    both = np.isfinite(mkt).all(axis=1)
    if both.sum():
        ll = lambda p: -np.log(np.clip(p[rows[both], pos[both]], 1e-12, 1)).mean()
        print(f"Bot vs market on {both.sum()} matches: favourite won {(bot[both].argmax(axis=1) == pos[both]).mean():.0%} (bot) vs "
              f"{(mkt[both].argmax(axis=1) == pos[both]).mean():.0%} (market); log loss {ll(bot):.3f} vs {ll(mkt):.3f} (lower is better).")

    markets_report(fx)

    settled = bets.merge(done[["fixture", "hg", "ag"]], on="fixture") if len(bets) else pd.DataFrame()
    if len(settled):
        res = np.select([settled.hg > settled.ag, settled.hg == settled.ag], ["home", "draw"], default="away")
        profit = np.where(res == settled.pick, settled.pick_odds - 1, -1.0)
        print(f"Paper bets: {len(settled)} settled, {(profit > 0).sum()} won, profit {profit.sum():+.2f} units "
              f"({profit.mean():+.1%} per bet).")
    recent = done.tail(10).assign(score=[f"{h}-{a}" for h, a in zip(done.tail(10).hg, done.tail(10).ag)])
    print("\n" + recent[["home", "away", "p_home", "p_draw", "p_away", "mkt_home", "pick", "pick_odds", "score"]].to_string(index=False))


if __name__ == "__main__":
    main()
