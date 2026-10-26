"""Train the model and save the starting state for the live bot.

    python -m bot.train                 # train on every complete season
    python -m bot.train --until 2024-25 # pretend it's the end of 2024-25 (used by the replay test)

Writes model/model.json (both model layers, the side-market models and honest backtests) and model/state.json
(Premier League and Championship ratings, xG form and player records at the end of the last complete season,
which the live bot picks up from).
"""
import argparse
import datetime as dt

import numpy as np
import pandas as pd

from . import history, markets, players
from .config import (BASE_FEATURES, EPL_FEATURES, FIRST_TRAIN_SEASON, FIRST_XG_SEASON, MODEL_PATH, STAR_CAP, STATE_PATH,
                     XG_WINDOW)
from .model import accuracy, fair_probs, fit_model, fit_shift, log_loss, predict, predict_full, save_json
from .ratings import replay_all

CLASSES = ["A", "D", "H"]


def build_table(last):
    """Every match up to `last` with its rating features (the base model's training data), and the Premier League
    matches from 2022-23 with xG form and star absence added (the Premier League layer's training data)."""
    results, odds = history.all_matches(last)
    feats, states = replay_all(results)
    table = results.join(feats)
    table["result"] = np.select([table.hg > table.ag, table.hg == table.ag], ["H", "D"], default="A")
    known = table[["hst", "ast"]].notna().all(axis=1).groupby([table.league, table.season]).transform("mean")
    base = table[(table.season >= FIRST_TRAIN_SEASON) & (known > 0.9)]

    seasons = [s for s in history.fpl_seasons() if FIRST_XG_SEASON <= s <= last]
    hist = pd.concat([history.fpl_player_history(s) for s in seasons], ignore_index=True)
    xg_table = players.add_xg_form(players.team_match_table(hist))
    xg_form = xg_table.set_index(["fixture", "team"]).xg_form
    star = players.add_star_absence(hist)
    epl = table[(table.league == "E0") & (table.season >= FIRST_XG_SEASON)].copy()
    for side in ["home", "away"]:
        key = pd.MultiIndex.from_arrays([epl.fixture, epl[side]])
        epl[f"xg_form_{side}"] = xg_form.reindex(key).fillna(0).to_numpy()
        epl[f"star_absence_{side}"] = star.reindex(key).fillna(0).to_numpy()
    epl["xg_form_diff"] = epl.xg_form_home - epl.xg_form_away
    epl["star_absence_diff"] = epl.star_absence_home - epl.star_absence_away
    epl = epl.merge(odds, on=["season", "home", "away"], how="left")
    return base, epl, states, xg_table, table, hist


def fit_all(base, epl):
    stage1 = fit_model(base, BASE_FEATURES)
    return {"stage1": stage1, "stage2": fit_shift(predict(stage1, epl), epl, EPL_FEATURES, caps={"star_absence_diff": STAR_CAP})}


def backtest(base, epl, last):
    """Train on the seasons before `last`, predict every Premier League match of `last`."""
    model = fit_all(base[base.season < last], epl[epl.season < last])
    test = epl[epl.season == last]
    full, first = predict_full(model, test)
    elo = predict(fit_model(base[base.season < last], ["elo_diff", "lg_home", "lg_draw"]), test)
    out = {"season": last, "matches": len(test),
           "model log loss": log_loss(full, test.result), "base layer log loss": log_loss(first, test.result),
           "Elo-only log loss": log_loss(elo, test.result), "model accuracy": accuracy(full, test.result)}
    if test[["OddAway", "OddDraw", "OddHome"]].notna().all(axis=None):
        bookie = fair_probs(test[["OddAway", "OddDraw", "OddHome"]])
        out.update({"bookie log loss": log_loss(bookie, test.result), "bookie accuracy": accuracy(bookie, test.result)})
    return out


def stat_known(table, a, b):
    """League-seasons where a stat is recorded for at least 90% of matches."""
    return table[[a, b]].notna().all(axis=1).groupby([table.league, table.season]).transform("mean") > 0.9


def fit_markets(table):
    """The side-market models: goals (home, away, the low-score tweak, half-time shares), then shots on target, corners and cards."""
    t = markets.add_inputs(table[table.season >= FIRST_TRAIN_SEASON])
    shots = t[stat_known(t, "hst", "ast")]
    goals = {"home": markets.fit_poisson(shots, markets.GOAL_FEATURES, "hg"),
             "away": markets.fit_poisson(shots, markets.GOAL_FEATURES, "ag")}
    sample = shots.sample(min(len(shots), 40000), random_state=1)
    goals["rho"] = markets.fit_rho(markets.predict_mean(goals["home"], sample), markets.predict_mean(goals["away"], sample),
                                   sample.hg.to_numpy(), sample.ag.to_numpy())
    ht = t[(t.league == "E0") & t.hhg.notna()]
    ht = ht[ht.season >= sorted(ht.season.unique())[-10]]
    goals["share_h"], goals["share_a"] = float(ht.hhg.sum() / ht.hg.sum()), float(ht.hag.sum() / ht.ag.sum())
    out = {"goals": goals}
    for name, (a, b, _) in markets.COUNTS.items():
        out[name] = markets.fit_counts(t[stat_known(t, a, b)], name)
    return out


def player_priors(hist):
    """Per-90 rates of goals, assists and cards by position, the starting point for every player's own rate."""
    h = hist[hist.minutes > 0]
    n90 = h.groupby("position").minutes.sum() / 90
    return {c: (h.groupby("position")[c].sum() / n90).to_dict() for c in ["goals", "assists", "cards"]}


def team_mu_table(model, epl):
    """Each Premier League match's expected goals for both sides, from the goals model."""
    t = markets.add_inputs(epl)
    return pd.DataFrame({"season": epl.season.to_numpy(), "kickoff": epl.kickoff.to_numpy(), "fixture": epl.fixture.to_numpy(),
                         "home": epl.home.to_numpy(), "away": epl.away.to_numpy(),
                         "mu_home": markets.predict_mean(model["goals"]["home"], t),
                         "mu_away": markets.predict_mean(model["goals"]["away"], t)})


def team_scale(mu, season, team, kickoff, tonight, last_means):
    """Tonight's expected goals relative to the club's usual: this season's earlier matches (3+), else last season's average."""
    rows = mu[(mu.season == season) & (mu.kickoff < kickoff) & ((mu.home == team) | (mu.away == team))]
    vals = np.where(rows.home == team, rows.mu_home, rows.mu_away)
    usual = vals.mean() if len(vals) >= 3 else last_means.get(team)
    return float(np.clip(tonight / usual, 0.4, 2.5)) if usual else 1.0


def team_means(mu, season):
    rows = mu[mu.season == season]
    long = pd.concat([pd.DataFrame({"team": rows.home, "mu": rows.mu_home}), pd.DataFrame({"team": rows.away, "mu": rows.mu_away})])
    return long.groupby("team").mu.mean().to_dict()


def backtest_markets(table, epl, hist, last):
    """Train the side-market models on the seasons before `last`, then score every Premier League match of `last`
    against the old rate in each market (what you'd guess knowing only the league's usual numbers)."""
    model = fit_markets(table[table.season < last])
    e0 = table[(table.league == "E0") & (table.season <= last)]
    test = e0[e0.season == last]
    probs, _ = markets.all_markets(model, test)
    hg, ag = test.hg.to_numpy(), test.ag.to_numpy()
    outcomes = {"goals_over_2.5": hg + ag > 2.5, "btts": (hg > 0) & (ag > 0), "home_win_to_nil": (hg > ag) & (ag == 0),
                "home_win_and_over_2.5": (hg > ag) & (hg + ag > 2), "shots_over_8.5": test.hst + test.ast > 8.5,
                "corners_over_9.5": test.hc + test.ac > 9.5, "corners_over_10.5": test.hc + test.ac > 10.5,
                "cards_over_3.5": test.hcd + test.acd > 3.5, "cards_over_4.5": test.hcd + test.acd > 4.5,
                "corners_home_more": test.hc > test.ac, "ht_draw": test.hhg == test.hag}
    prev = e0[e0.season < last]
    old_rates = {"goals_over_2.5": (prev.hg + prev.ag > 2.5), "btts": (prev.hg > 0) & (prev.ag > 0),
                 "home_win_to_nil": (prev.hg > prev.ag) & (prev.ag == 0), "home_win_and_over_2.5": (prev.hg > prev.ag) & (prev.hg + prev.ag > 2),
                 "shots_over_8.5": prev.hst + prev.ast > 8.5, "corners_over_9.5": prev.hc + prev.ac > 9.5,
                 "corners_over_10.5": prev.hc + prev.ac > 10.5, "cards_over_3.5": prev.hcd + prev.acd > 3.5,
                 "cards_over_4.5": prev.hcd + prev.acd > 4.5, "corners_home_more": prev.hc > prev.ac, "ht_draw": prev.hhg == prev.hag}
    report = {}
    for k, y in outcomes.items():
        y = pd.Series(np.asarray(y), index=test.index)
        ok = y.notna() & ~test[["hst", "hc", "hcd", "hhg"]].isna().all(axis=1)
        ok &= np.isfinite(probs[k])
        if ok.sum() < 100:
            continue
        p, yy = probs[k][ok.to_numpy()], y[ok].astype(float).to_numpy()
        base = float(old_rates[k][prev.index[old_rates[k].notna()]].mean())
        report[k] = {"happened": float(yy.mean()), "bot said": float(p.mean()),
                     "Brier skill vs old rate": float(1 - ((p - yy) ** 2).mean() / ((base - yy) ** 2).mean())}
    # players: every starter of season `last`, with last season's totals as the carry-in
    report_players = {}
    seasons = sorted(hist.season.unique())
    if last in seasons and seasons.index(last) > 0:
        prior_season = seasons[seasons.index(last) - 1]
        mu = team_mu_table(model, e0[e0.season >= prior_season])
        ev = markets.player_evidence(hist[hist.season == last], markets.season_totals(hist[hist.season == prior_season]))
        means = team_means(mu, prior_season)
        ev = ev[(ev.starts == 1) & ev.position.isin(["DEF", "MID", "FWD"])]
        fx_mu = mu.set_index("fixture")
        tonight = [np.nan if f not in fx_mu.index else fx_mu.loc[f, "mu_home"] if fx_mu.loc[f, "home"] == t else fx_mu.loc[f, "mu_away"]
                   for f, t in zip(ev.fixture, ev.team)]
        ev["scale"] = [team_scale(mu, last, t, k, x, means) if x == x else 1.0 for t, k, x in zip(ev.team, ev.kickoff, tonight)]
        pp = markets.player_probs(ev, player_priors(hist[hist.season < last]), ev.scale)
        pri = player_priors(hist[hist.season < last])
        for key, col in [("p_score", "goals"), ("p_assist", "assists"), ("p_card", "cards")]:
            y = (ev[col] > 0).astype(float).to_numpy()
            p = np.asarray(pp[key])
            base = 1 - np.exp(-ev.position.map(pri[col]).to_numpy() * np.clip(ev.exp_minutes / 90, 0.3, 1.0).to_numpy())
            report_players[key] = {"starts": int(len(y)), "happened": float(y.mean()), "bot said": float(p.mean()),
                                   "Brier skill vs position average": float(1 - ((p - y) ** 2).mean() / ((base - y) ** 2).mean())}
    return {"season": last, "matches": int(len(test)), "markets": report, "players": report_players}


def carry_in(table, last):
    """Each club's final xG differences of the season, so xG form carries into the next one."""
    end = table[table.season == last].sort_values("kickoff")
    return {team: g.xgd.tail(XG_WINDOW).tolist() for team, g in end.groupby("team")}


def main(until=None):
    complete = [s for s in history.fpl_seasons() if history.fpl_fixtures(s)[0].finished.all()]
    last = until or complete[-1]
    base, epl, states, xg_table, table, hist = build_table(last)
    model = fit_all(base, epl)
    model["markets"] = fit_markets(table)
    model["player_priors"] = player_priors(hist)
    model["trained_on"] = (f"base layer: {len(base):,} matches in {base.league.nunique()} leagues, {FIRST_TRAIN_SEASON} to {last}; "
                           f"Premier League layer: {len(epl):,} matches, {FIRST_XG_SEASON} to {last}")
    model["trained_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    model["backtest"] = backtest(base, epl, last)
    model["markets_backtest"] = backtest_markets(table, epl, hist, last)
    save_json(model, MODEL_PATH)
    leagues = {code: states[code].to_dict() for code in ["E0", "E1"] if code in states}
    mu = team_mu_table(model["markets"], table[(table.league == "E0") & (table.season == last)])
    nxt = history.label(int(last[:4]) + 1)
    arch = history.all_leagues()
    now = arch[(arch.league == "E0") & (arch.season == nxt)].dropna(subset=["hc", "ac"])
    save_json({"season": last, "leagues": leagues, "xg_carry_in": carry_in(xg_table, last),
               "players_last_season": markets.season_totals(hist[hist.season == last]),
               "team_mu_last_season": team_means(mu, last),
               # Fantasy Premier League has no corners: this season's corners come from the archive, refreshed each retrain
               "corners_this_season": {f"{h}|{a}": [float(c1), float(c2)] for h, a, c1, c2 in zip(now.home, now.away, now.hc, now.ac)}},
              STATE_PATH)
    print(f"Trained on {model['trained_on']}")
    print("Backtest:", {k: round(v, 4) if isinstance(v, float) else v for k, v in model["backtest"].items()})
    mb = model["markets_backtest"]
    for k, v in {**mb["markets"], **mb["players"]}.items():
        print(f"  {k}: happened {v['happened']:.1%}, bot said {v['bot said']:.1%}, skill {list(v.values())[-1]:+.1%}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--until", help="last season to train on, e.g. 2024-25")
    main(parser.parse_args().until)
