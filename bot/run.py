"""Run the bot once. Schedule it every 15 minutes (see .github/workflows/bot.yml).

Each run:
1. Brings team ratings up to date with this season's results.
2. Posts a preview for every fixture in the next 36 hours, using the expected XI (injury flags included).
3. From 75 minutes before kickoff, looks for confirmed lineups and posts the final prediction.
   No lineups 20 minutes before kickoff: posts the final prediction with the expected XI instead.
   The final prediction is also the paper bet: if the bot's price beats the best odds by VALUE_EDGE, it's logged.
4. In the last run before kickoff, records the closing odds (no post), so you can see whether the
   market moved towards the bot's picks.

    python -m bot.run            # live
    python -m bot.run --dry-run  # print only: nothing posted, nothing logged
"""
import argparse
import os
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import markets, players
from .config import (CLOSING_WINDOW_MINUTES, LAST_CALL_MINUTES, LINEUP_LOOKBACK, LINEUP_WINDOW_MINUTES, LOG_PATH,
                     MARKETS_LOG_PATH, MODEL_PATH, PREVIEW_HOURS, STAR_THRESHOLD, STATE_PATH, VALUE_EDGE)
from .model import load_json, predict_full
from .post import post
from .ratings import LeagueState
from .sources import ApiFootballLineups, FplLive, OddsApi, match_players

FEATURE_COLUMNS = ["elo_diff", "pi_diff", "goal_ratio", "goal_total", "shot_gap", "lg_home", "lg_draw",
                   "exp_home_goals", "exp_away_goals", "exp_home_shots", "exp_away_shots", "exp_home_corners",
                   "exp_away_corners", "exp_home_cards", "exp_away_cards", "xg_form_diff", "star_absence_diff"]
LOG_COLUMNS = (["logged_at", "stage", "lineups", "fixture", "kickoff", "home", "away", "p_home", "p_draw", "p_away"]
               + FEATURE_COLUMNS + ["missing_home", "missing_away", "mkt_home", "mkt_draw", "mkt_away",
                                    "best_home", "best_draw", "best_away", "book_home", "book_draw", "book_away",
                                    "pick", "pick_odds", "pick_edge"])
SIDES = ["home", "draw", "away"]
DISAGREE = 0.10  # in the Simulation Lab, when the bot and the bookie differed by 10+ points the bookie's pick won more often


def read_log():
    log = pd.read_csv(LOG_PATH) if LOG_PATH.exists() else pd.DataFrame(columns=LOG_COLUMNS)
    return log.reindex(columns=LOG_COLUMNS)


def current_ratings(state, fixtures, season, now):
    """The Premier League's ratings right now: last season's end state, moved into this season, plus every result so far.

    Also returns each finished match's pre-match features (players' scoring rates are scaled by them)."""
    epl = LeagueState.from_dict(state["leagues"]["E0"])
    if epl.season >= season:
        return epl, []  # the saved state already includes this season
    below = state["leagues"].get("E1")
    neighbours = []
    if below and below["season"] == epl.season:
        e1 = LeagueState.from_dict(below)
        neighbours = [(e1.teams, e1.snapshot(), None)]
    epl.start_season(season, set(fixtures.home) | set(fixtures.away), neighbours)
    corners = state.get("corners_this_season", {})
    done = fixtures[fixtures.finished & (fixtures.kickoff < now)].sort_values(["kickoff", "home"], kind="stable")
    past, nan = [], float("nan")
    for _, group in done.groupby("kickoff", sort=True):  # matches kicking off together are rated together
        for r in group.itertuples():
            past.append({"fixture": r.fixture, "kickoff": r.kickoff, "home": r.home, "away": r.away, **epl.features(r.home, r.away)})
        for r in group.itertuples():
            hc, ac = corners.get(f"{r.home}|{r.away}", [nan, nan])
            epl.update(r.home, r.away, r.hg, r.ag, r.hst, r.ast, hc, ac, getattr(r, "hcd", nan), getattr(r, "acd", nan))
    return epl, past


def likely_scorers(model, state, hist, squads, f, sides, mu_tonight, past_mu, n=4):
    """Starters' chances to score (and be carded), scaled to tonight's expected goals. Outfield players only."""
    rows = []
    last_means = state.get("team_mu_last_season", {})
    for team, side in sides.items():
        starters = [p for p in side["starters"] if p in squads.index and squads.loc[p, "position"] != "GK"]
        if not starters:
            continue
        names = squads.loc[starters, "name"].tolist()
        own = hist[hist.player.isin(starters) & (hist.kickoff < f.kickoff)]
        tonight = pd.DataFrame({"name": names, "position": squads.loc[starters, "position"].to_numpy(), "kickoff": f.kickoff,
                                "minutes": 0, "starts": 1, "goals": 0, "assists": 0, "cards": 0, "xg": np.nan, "xa": np.nan,
                                "tonight": True, "web_name": squads.loc[starters, "web_name"].to_numpy()})
        both = pd.concat([own.assign(tonight=False), tonight], ignore_index=True)
        ev = markets.player_evidence(both, state.get("players_last_season", {}))
        ev = ev[ev.tonight.astype(bool)]
        mine = past_mu[(past_mu.home == team) | (past_mu.away == team)]
        vals = np.where(mine.home == team, mine.mu_home, mine.mu_away)
        usual = vals.mean() if len(vals) >= 3 else last_means.get(team)
        scale = float(np.clip(mu_tonight[team] / usual, 0.4, 2.5)) if usual else 1.0
        probs = markets.player_probs(ev, model["player_priors"], scale)
        rows.append(pd.DataFrame({"team": team, "name": ev.web_name.to_numpy(), **{k: np.asarray(v) for k, v in probs.items()}}))
    if not rows:
        return pd.DataFrame(columns=["team", "name", "p_score", "p_card"])
    return pd.concat(rows).sort_values("p_score", ascending=False)


def side_features(team, squad, hist, table, kickoff, carry_in, lineup_names=None):
    """xG form and star absence for one team, plus who is missing."""
    usual, n_recent = players.usual_xi(hist, team, kickoff, squad.index)
    games = hist[(hist.team == team) & (hist.kickoff < kickoff)]
    recent = games[["fixture", "kickoff"]].drop_duplicates().sort_values("kickoff").fixture.tail(LINEUP_LOOKBACK)
    info = squad.assign(recent_minutes=games[games.fixture.isin(recent)].groupby("player").minutes.sum()
                        .reindex(squad.index).fillna(0))
    starters, source = None, "expected"
    if lineup_names:
        matched, unmatched = match_players(lineup_names, squad)
        if len(matched) >= 10:
            starters, source = matched, "confirmed"
    if starters is None:
        starters, _ = players.expected_xi(usual, info)
    missing = [p for p in usual if p not in starters]
    missing = squad.loc[missing].sort_values("price", ascending=False)
    return {"xg_form": players.xg_form_now(table, team, kickoff, carry_in.get(team, [])),
            "star_absence": players.star_absence(starters, usual, squad.price, n_recent),
            "source": source, "missing": missing, "starters": list(starters)}


def value_pick(probs, best):
    """The outcome where the bot's chance times the best odds beats 1 by the most, if by at least VALUE_EDGE."""
    if not best:
        return None
    edges = {s: probs[s] * best[s][0] - 1 for s in SIDES}
    side = max(edges, key=edges.get)
    return (side, best[side][0], best[side][1], edges[side]) if edges[side] >= VALUE_EDGE else None


def when(kickoff):
    uk = kickoff.tz_convert(ZoneInfo("Europe/London"))
    text = f"{uk:%a %d %b, %H:%M} UK"
    local_tz = os.environ.get("BOT_TIMEZONE")
    if local_tz and local_tz != "Europe/London":
        local = kickoff.tz_convert(ZoneInfo(local_tz))
        text += f" ({local:%a %H:%M} {local_tz.split('/')[-1].replace('_', ' ')})"
    return text


def market_lines(f, mk, scores, scorers):
    """The side markets in four short lines."""
    pct = lambda x: f"{x:.0%}"
    lines = [f"Goals: Over 2.5 {pct(mk['goals_over_2.5'])} · Both score {pct(mk['btts'])} · Likely scores "
             + " · ".join(f"{h}-{a} {pct(p)}" for h, a, p in scores[:3]),
             f"Half-time: {f.home} {pct(mk['ht_home'])} · Draw {pct(mk['ht_draw'])} · {f.away} {pct(mk['ht_away'])}",
             f"Corners: Over 9.5 {pct(mk['corners_over_9.5'])} · Cards: Over 3.5 {pct(mk['cards_over_3.5'])} · "
             f"Shots on target: Over 8.5 {pct(mk['shots_over_8.5'])}"]
    if len(scorers):
        lines.append("Scorers: " + " · ".join(f"{r.name} {pct(r.p_score)}" for r in scorers.head(4).itertuples()))
    return lines


def describe(f, probs, feats, sides, stage, odds=None, pick=None, extra=None):
    label = {"preview": "Preview, expected XIs",
             "final": "Final, lineups confirmed" if all(s["source"] == "confirmed" for s in sides.values())
             else "Final, expected XIs (no confirmed lineups yet)"}[stage]
    why = [f"Elo gap {abs(feats['elo_diff']):.0f} in {f.home if feats['elo_diff'] >= 0 else f.away}'s favour",
           f"expected goals {feats['exp_home_goals']:.1f} v {feats['exp_away_goals']:.1f}",
           f"xG form {f.home} {sides[f.home]['xg_form']:+.2f}, {f.away} {sides[f.away]['xg_form']:+.2f}"]
    for team, s in sides.items():
        if s["star_absence"] > 0:
            names = ", ".join(s["missing"].web_name.head(3)) if "web_name" in s["missing"] else ""
            weaker = s["star_absence"] + STAR_THRESHOLD
            why.append(f"{team} XI {weaker:.0%} cheaper than usual" + (f" ({names} out)" if names else ""))
    lines = [
        f"⚽ {f.home} vs {f.away}",
        f"{when(f.kickoff)} · {label}",
        f"Bot: {f.home} {probs['home']:.0%} · Draw {probs['draw']:.0%} · {f.away} {probs['away']:.0%}",
    ]
    market, best = (odds or {}).get("market"), (odds or {}).get("best")
    if market:
        lines.append(f"Market: {f.home} {market['home']:.0%} · Draw {market['draw']:.0%} · {f.away} {market['away']:.0%} "
                     f"({market['books']} bookies)")
    lines.append(f"Fair odds: {1 / probs['home']:.2f} · {1 / probs['draw']:.2f} · {1 / probs['away']:.2f}")
    if best:
        lines.append("Best odds: " + " · ".join(f"{best[s][0]:.2f} {best[s][1]}" for s in SIDES))
        names = {"home": f.home, "draw": "the draw", "away": f.away}
        lines.append(f"Value: {names[pick[0]]} at {pick[1]:.2f} with {pick[2]} (edge {pick[3]:+.0%})" if pick
                     else "Value: none at these prices")
    if market:
        gap = max(abs(probs[s] - market[s]) for s in SIDES)
        if gap >= DISAGREE:
            lines.append(f"Heads-up: {gap * 100:.0f} points away from the market. In past matches like this the market was right more often.")
    lines += extra or []
    lines.append("Why: " + " · ".join(why))
    return "\n".join(lines)


def odds_fields(odds):
    out = {}
    if odds and odds.get("best"):
        best = odds["best"]
        out.update({**{f"best_{s}": best[s][0] for s in SIDES}, **{f"book_{s}": best[s][1] for s in SIDES}})
    if odds and odds.get("market"):
        out.update({f"mkt_{s}": round(odds["market"][s], 4) for s in SIDES})
    return out


def main(dry_run=False, now=None, src=None, lineups_api=None, odds_api=None):
    now = now or pd.Timestamp.now(tz="UTC")
    src = src or FplLive()
    lineups_api = lineups_api if lineups_api is not None else ApiFootballLineups()
    odds_api = odds_api if odds_api is not None else OddsApi()
    model, state = load_json(MODEL_PATH), load_json(STATE_PATH)
    season, fx = src.season(), src.fixtures()
    team_names = sorted(set(fx.home))
    ratings, past = current_ratings(state, fx, season, now)
    past = pd.DataFrame(past)
    if len(past):
        mk_past, _ = markets.all_markets(model["markets"], past)
        past_mu = past[["fixture", "kickoff", "home", "away"]].assign(mu_home=mk_past["goals_exp_home"], mu_away=mk_past["goals_exp_away"])
    else:
        past_mu = pd.DataFrame(columns=["fixture", "kickoff", "home", "away", "mu_home", "mu_away"])

    log = read_log()
    done = set(zip(log.fixture.astype(str), log.stage))
    jobs = []
    for f in fx[~fx.finished & fx.kickoff.notna() & (fx.kickoff > now)].sort_values("kickoff").itertuples():
        to_go = (f.kickoff - now).total_seconds() / 60
        if to_go <= CLOSING_WINDOW_MINUTES and (f.fixture, "final") in done and (f.fixture, "closing") not in done:
            jobs.append((f, "closing", to_go))
        elif to_go <= LINEUP_WINDOW_MINUTES and (f.fixture, "final") not in done:
            jobs.append((f, "final", to_go))
        elif to_go <= PREVIEW_HOURS * 60 and not {(f.fixture, "preview"), (f.fixture, "final")} & done:
            jobs.append((f, "preview", to_go))
    if not jobs:
        print("Nothing to post this run.")
        return []

    def get_odds(f):
        if not odds_api.enabled():
            return None
        try:
            return odds_api.prices(f, team_names)
        except Exception as err:  # odds are a bonus; never let them stop a prediction
            print(f"Odds unavailable for {f.home} vs {f.away}: {err}")
            return None

    rows, market_rows = [], []
    for f, stage, _ in [j for j in jobs if j[1] == "closing"]:
        odds = get_odds(f)
        if odds:
            rows.append({"logged_at": now.isoformat(timespec="seconds"), "stage": "closing", "fixture": f.fixture,
                         "kickoff": f.kickoff.isoformat(), "home": f.home, "away": f.away, **odds_fields(odds)})
    jobs = [j for j in jobs if j[1] != "closing"]

    if jobs:
        squads = src.players()
        squads = squads[squads.status != "u"]
        needed = {f.home for f, _, _ in jobs} | {f.away for f, _, _ in jobs}
        has_played = squads.minutes > 0 if "minutes" in squads.columns else pd.Series(True, index=squads.index)
        hist = src.histories(list(squads.index[squads.team.isin(needed) & has_played]), squads)
        table = players.team_match_table(hist) if len(hist) else pd.DataFrame(columns=["team", "kickoff", "xgd"])
        consecutive = int(season[:4]) - int(state["season"][:4]) == 1
        carry_in = state["xg_carry_in"] if consecutive else {}

    for f, stage, to_go in jobs:
        names = None
        if stage == "final" and lineups_api.enabled():
            try:
                names = lineups_api.starting_xis(f, team_names)
            except Exception as err:  # a lineup hiccup should never stop the prediction
                print(f"Lineups unavailable for {f.home} vs {f.away}: {err}")
        if stage == "final" and names is None and to_go > LAST_CALL_MINUTES:
            continue  # lineups not out yet, try again next run
        sides = {team: side_features(team, squads[squads.team == team], hist, table, f.kickoff, carry_in,
                                     (names or {}).get(team)) for team in (f.home, f.away)}
        feats = {**ratings.features(f.home, f.away),
                 "xg_form_diff": sides[f.home]["xg_form"] - sides[f.away]["xg_form"],
                 "star_absence_diff": sides[f.home]["star_absence"] - sides[f.away]["star_absence"]}
        final, _ = predict_full(model, [feats])
        away_p, draw_p, home_p = final[0]
        probs = {"home": home_p, "draw": draw_p, "away": away_p}
        mk_all, grid = markets.all_markets(model["markets"], pd.DataFrame([feats]))
        mk = {k: float(v[0]) for k, v in mk_all.items()}
        mu_tonight = {f.home: mk["goals_exp_home"], f.away: mk["goals_exp_away"]}
        scorers = likely_scorers(model, state, hist, squads, f, sides, mu_tonight, past_mu[past_mu.kickoff < f.kickoff])
        odds = get_odds(f)
        pick = value_pick(probs, (odds or {}).get("best"))
        extra = market_lines(f, mk, markets.top_scores(grid[0]), scorers)
        post(describe(f, probs, feats, sides, stage, odds, pick, extra), dry_run=dry_run)
        market_rows.append({"logged_at": now.isoformat(timespec="seconds"), "stage": stage, "fixture": f.fixture,
                            "kickoff": f.kickoff.isoformat(), "home": f.home, "away": f.away,
                            **{k: round(v, 4) for k, v in mk.items()},
                            "likely_scores": "; ".join(f"{h}-{a} {p:.3f}" for h, a, p in markets.top_scores(grid[0], 6)),
                            "scorers": "; ".join(f"{r.team}|{r.name}|{r.p_score:.3f}|{r.p_assist:.3f}|{r.p_card:.3f}" for r in scorers.itertuples())})
        rows.append({"logged_at": now.isoformat(timespec="seconds"), "stage": stage,
                     "lineups": "+".join(sorted({s["source"] for s in sides.values()})), "fixture": f.fixture,
                     "kickoff": f.kickoff.isoformat(), "home": f.home, "away": f.away,
                     "p_home": round(home_p, 4), "p_draw": round(draw_p, 4), "p_away": round(away_p, 4),
                     **{k: round(float(feats[k]), 4) for k in FEATURE_COLUMNS},
                     "missing_home": "; ".join(sides[f.home]["missing"].get("web_name", pd.Series(dtype=str)).head(5)),
                     "missing_away": "; ".join(sides[f.away]["missing"].get("web_name", pd.Series(dtype=str)).head(5)),
                     **odds_fields(odds),
                     **({"pick": pick[0], "pick_odds": pick[1], "pick_edge": round(pick[3], 4)} if pick and stage == "final" else {})})
    if odds_api.enabled() and odds_api.remaining is not None:
        print(f"The Odds API credits left this month: {odds_api.remaining}")
    if rows and not dry_run:
        pd.concat([log, pd.DataFrame(rows)], ignore_index=True)[LOG_COLUMNS].to_csv(LOG_PATH, index=False)
    if market_rows and not dry_run:
        old = pd.read_csv(MARKETS_LOG_PATH) if MARKETS_LOG_PATH.exists() else pd.DataFrame()
        pd.concat([old, pd.DataFrame(market_rows)], ignore_index=True).to_csv(MARKETS_LOG_PATH, index=False)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="print predictions without posting or logging")
    main(dry_run=parser.parse_args().dry_run)
