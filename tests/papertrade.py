"""Paper-trade the live bot's betting rules on every past match that has odds, walk-forward season by season.

    python tests/papertrade.py

For each season the model trains only on the seasons before, then bets on every match of that season with the
live bot's rules (bot/run.py), at the best price across bookies, as the live bot takes:
- Bot value bets: the bot's chance times the best odds beats 1 by VALUE_EDGE, with the bot giving that result at
  least MIN_VALUE_CHANCE, at odds up to MAX_VALUE_ODDS. The old rule (edge only) is shown for comparison.
- Line shopping: the same test with the market's own chances in place of the bot's (SHOP_* settings), no model.
  The live bot averages every bookie's fair chances; the archive only has Bet365's, so those stand in for the market.
Rules were chosen on 2005-06 to 2015-16; 2016-17 on is the honest test. Prices are from a few days before kickoff.
1. Every league: the base layer, 2005-06 on (every league-season with shots on target recorded).
2. Premier League: the full model with the xG layer, 2023-24 on (the first season it has a season to learn from).
Every bet is saved to .data/papertrade_bets.csv.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bot import history, train  # noqa: E402
from bot.config import (BASE_FEATURES, DATA_DIR, FIRST_XG_SEASON, MAX_VALUE_ODDS, MIN_VALUE_CHANCE, SHOP_EDGE,  # noqa: E402
                        SHOP_MAX_ODDS, SHOP_MIN_CHANCE, VALUE_EDGE)
from bot.model import CLASSES, fair_probs, fit_model, predict, predict_full  # noqa: E402

B365 = ["OddAway", "OddDraw", "OddHome"]  # the same order as the model's classes
BEST = ["MaxAway", "MaxDraw", "MaxHome"]
FIRST_TEST_SEASON = "2005-06"
HOLDOUT = "2016-17"  # rules were picked on the seasons before this
RULES = {"Bot value bets (new rule)": ("bot", VALUE_EDGE, MIN_VALUE_CHANCE, MAX_VALUE_ODDS),
         "Bot value bets (old rule, edge only)": ("bot", VALUE_EDGE, 0.0, np.inf),
         "Line shopping (no model)": ("market", SHOP_EDGE, SHOP_MIN_CHANCE, SHOP_MAX_ODDS)}


def add_odds(table, epl_odds):
    """Bet365's and the best 1X2 prices: archive rows by kickoff, Fantasy Premier League rows by season and clubs."""
    arch = history.all_leagues()[["league", "kickoff", "home", "away"] + B365 + BEST].drop_duplicates(["league", "kickoff", "home", "away"])
    fpl = table.fixture.notna()
    a = table[~fpl].merge(arch, on=["league", "kickoff", "home", "away"], how="left")
    b = table[fpl].merge(epl_odds[["season", "home", "away"] + B365 + BEST], on=["season", "home", "away"], how="left")
    return pd.concat([a, b], ignore_index=True)


def every_bet(rows, probs):
    """Every bet each rule makes, the same way as bot.run.value_pick: the result with the biggest edge, if it passes."""
    b365, best = rows[B365].to_numpy(float), rows[BEST].to_numpy(float)
    ok = (np.isfinite(b365) & (b365 > 1) & np.isfinite(best) & (best >= b365 * 0.999)).all(axis=1)  # drop broken best prices
    rows, probs, b365, best = rows[ok], probs[ok], b365[ok], best[ok]
    chances = {"bot": probs, "market": fair_probs(b365)}
    i = np.arange(len(rows))
    out = []
    for rule, (who, edge, min_chance, max_odds) in RULES.items():
        p = chances[who]
        edges = p * best - 1
        pick = edges.argmax(axis=1)
        take = (edges[i, pick] >= edge) & (p[i, pick] >= min_chance) & (best[i, pick] <= max_odds)
        out.append(pd.DataFrame({"rule": rule, "league": rows.league.to_numpy(), "season": rows.season.to_numpy(),
                                 "kickoff": rows.kickoff.to_numpy(), "home": rows.home.to_numpy(), "away": rows.away.to_numpy(),
                                 "result": rows.result.to_numpy(), "pick": np.array(CLASSES)[pick], "chance": p[i, pick],
                                 "odds": best[i, pick], "edge": edges[i, pick]})[take])
    fav = b365.argmin(axis=1)  # for comparison: the bookie's favourite every match, at Bet365's price
    out.append(pd.DataFrame({"rule": "Bookie's favourite every match (Bet365)", "league": rows.league.to_numpy(),
                             "season": rows.season.to_numpy(), "kickoff": rows.kickoff.to_numpy(), "home": rows.home.to_numpy(),
                             "away": rows.away.to_numpy(), "result": rows.result.to_numpy(), "pick": np.array(CLASSES)[fav],
                             "chance": fair_probs(b365)[i, fav], "odds": b365[i, fav], "edge": np.nan}))
    bets = pd.concat(out, ignore_index=True)
    bets["won"] = bets.pick == bets.result
    bets["profit"] = np.where(bets.won, bets.odds - 1, -1.0)
    return bets


def summary(b):
    n = len(b)
    if n == 0:
        return {"bets": 0}
    se = b.profit.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return {"bets": n, "won": int(b.won.sum()), "lost": int(n - b.won.sum()), "win rate": f"{b.won.mean():.1%}",
            "avg odds": round(float(b.odds.mean()), 2), "profit (units)": round(float(b.profit.sum()), 1),
            "ROI": f"{b.profit.mean():+.1%}", "ROI 95% range": f"{b.profit.mean() - 1.96 * se:+.1%} to {b.profit.mean() + 1.96 * se:+.1%}"}


def report(bets, title, by_season=True):
    print(f"\n=== {title} ===")
    for period, sel in [(f"Rules chosen on: to {HOLDOUT[:4]}", bets.season < HOLDOUT), (f"Honest test: {HOLDOUT} on", bets.season >= HOLDOUT),
                        ("All seasons", bets.season.notna())]:
        rows = {rule: summary(bets[sel & (bets.rule == rule)]) for rule in bets.rule.unique()}
        print(f"\n{period}")
        print(pd.DataFrame(rows).T.fillna("").to_string())
    if by_season:
        for rule in ["Bot value bets (new rule)", "Line shopping (no model)"]:
            s = bets[bets.rule == rule].groupby("season").apply(lambda b: pd.Series(summary(b)))
            print(f"\n{rule}, season by season: {(s['profit (units)'] > 0).sum()} of {len(s)} in profit")
            print("  " + " · ".join(f"{k} {r.ROI} ({r.bets})" for k, r in s.iterrows()))


def main():
    complete = [s for s in history.fpl_seasons() if history.fpl_fixtures(s)[0].finished.all()]
    last = complete[-1]
    print(f"Building every match up to {last} (downloads about 45 MB the first time)...")
    base, epl, *_ = train.build_table(last)
    _, epl_odds = history.all_matches(last)
    base = add_odds(base, epl_odds)
    epl = epl.drop(columns=[c for c in B365 + BEST + ["Over25", "Under25"] if c in epl]).merge(
        epl_odds[["season", "home", "away"] + B365 + BEST], on=["season", "home", "away"], how="left")

    leagues = []
    for s in sorted(s for s in base.season.unique() if FIRST_TEST_SEASON <= s <= last):  # walk forward
        test = base[base.season == s]
        leagues.append(every_bet(test, predict(fit_model(base[base.season < s], BASE_FEATURES), test)))
    leagues = pd.concat(leagues, ignore_index=True)

    epl_bets = []
    for s in sorted(x for x in epl.season.unique() if x > FIRST_XG_SEASON):
        test = epl[epl.season == s]
        epl_bets.append(every_bet(test, predict_full(train.fit_all(base[base.season < s], epl[epl.season < s]), test)[0]))
    epl_bets = pd.concat(epl_bets, ignore_index=True)

    report(leagues, "All 22 leagues, base layer")
    report(leagues[leagues.league == "E0"], "Premier League only, base layer", by_season=False)
    print("\n=== Premier League, full model (what the live bot runs), 2023-24 on ===")
    print(pd.DataFrame({rule: summary(epl_bets[epl_bets.rule == rule]) for rule in epl_bets.rule.unique()}).T.fillna("").to_string())

    out = DATA_DIR / "papertrade_bets.csv"
    pd.concat([leagues.assign(model="base layer"), epl_bets.assign(model="full model")]).to_csv(out, index=False)
    print(f"\nEvery bet saved to {out}")


if __name__ == "__main__":
    main()
