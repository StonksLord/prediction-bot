"""Player-level factors: xG form, the usual XI, star absence and the expected XI.

Everything here works on a "player history" table with one row per player per match:
    player, name, team, position, season, fixture, kickoff, minutes, starts, xg, xgc, price
`fixture` is a match key that is unique across seasons. `xgc` is the xG conceded while the
player was on the pitch, which Fantasy Premier League records for every player.
"""
import numpy as np
import pandas as pd

from .config import FIT_CHANCE, LINEUP_LOOKBACK, MIN_MATCHES_FOR_XI, STAR_THRESHOLD, XG_WINDOW


def team_match_table(hist):
    """One row per team per match: xG for, xG against, and the difference."""
    key = ["team", "fixture"]
    g = hist.groupby(key)
    table = pd.DataFrame({"kickoff": g.kickoff.first(), "season": g.season.first(), "xg": g.xg.sum(min_count=1)})
    table["xga"] = hist[hist.minutes > 0].groupby(key).xgc.max()
    table["xgd"] = table.xg - table.xga
    return table.reset_index().sort_values(["kickoff", "team"]).reset_index(drop=True)


def _spells(table):
    """Number each unbroken run of seasons per team, so a club back from relegation starts fresh."""
    seasons = table[["team", "season"]].drop_duplicates()
    seasons = seasons.assign(year=seasons.season.str[:4].astype(int)).sort_values(["team", "year"])
    seasons["spell"] = seasons.groupby("team").year.transform(lambda y: (y.diff() != 1).cumsum())
    return table.merge(seasons[["team", "season", "spell"]], on=["team", "season"], how="left").spell.to_numpy()


def add_xg_form(table, window=XG_WINDOW):
    """xG difference averaged over each team's previous `window` matches (crossing into last season if the club was there)."""
    table = table.sort_values("kickoff").copy()
    table["spell"] = _spells(table)
    table["xg_form"] = table.groupby(["team", "spell"]).xgd.transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    return table


def xg_form_now(table, team, before, carry_in=(), window=XG_WINDOW):
    """Live version of add_xg_form for one team. `carry_in` = last season's final xG differences."""
    this_season = table[(table.team == team) & (table.kickoff < before)].sort_values("kickoff").xgd.tolist()
    recent = [v for v in (list(carry_in) + this_season)[-window:] if pd.notna(v)]
    return float(np.mean(recent)) if recent else 0.0


def usual_xi(hist, team, before, squad, lookback=LINEUP_LOOKBACK):
    """The 11 squad players with the most minutes in the team's last `lookback` matches this season."""
    games = hist[(hist.team == team) & (hist.kickoff < before)]
    recent = games[["fixture", "kickoff"]].drop_duplicates().sort_values("kickoff").fixture.tail(lookback)
    mins = games[games.fixture.isin(recent)].groupby("player").minutes.sum()
    mins = mins[mins.index.isin(squad) & (mins > 0)]
    return list(mins.sort_values(ascending=False, kind="stable").index[:11]), len(recent)


def star_absence(starters, usual, price, matches_before):
    """How far today's XI value falls more than STAR_THRESHOLD below the usual XI. Zero when unknown."""
    if matches_before < MIN_MATCHES_FOR_XI or len(usual) < 8 or len(starters) < 10:
        return 0.0
    drop = price.reindex(starters).mean() / price.reindex(usual).mean() - 1
    return float(max(0.0, -drop - STAR_THRESHOLD))


def add_star_absence(hist, lookback=LINEUP_LOOKBACK):
    """Training version: star absence for every team in every match, from who actually started."""
    out = {}
    for (team, season), tp in hist.groupby(["team", "season"]):
        order = tp[["fixture", "kickoff"]].drop_duplicates().sort_values("kickoff").fixture.tolist()
        mins = tp.pivot_table(index="fixture", columns="player", values="minutes", aggfunc="sum").reindex(order).fillna(0)
        recent = mins.shift(1).rolling(lookback, min_periods=1).sum()
        by_fixture = {f: s.set_index("player") for f, s in tp.groupby("fixture")}
        for i, fixture in enumerate(order):
            squad = by_fixture[fixture]
            prior = recent.loc[fixture]
            prior = prior[prior.index.isin(squad.index) & (prior > 0)]
            usual = list(prior.sort_values(ascending=False, kind="stable").index[:11])
            starters = list(squad.index[squad.starts == 1])
            out[(fixture, team)] = star_absence(starters, usual, squad.price, matches_before=min(i, lookback))
    return pd.Series(out, name="star_absence")


def expected_xi(usual, squad_info):
    """Best guess at the XI before lineups are out: the usual XI, with unavailable players swapped
    for the available squad player in the same position who has played most recently.

    `squad_info` is indexed by player with columns position, status, chance, recent_minutes, price.
    """
    ok = squad_info[(squad_info.status == "a") | (squad_info.chance.fillna(100) >= FIT_CHANCE)]
    xi = [p for p in usual if p in ok.index]
    out = [p for p in usual if p not in ok.index]
    for p in out:
        pos = squad_info.loc[p, "position"] if p in squad_info.index else None
        bench = ok[(ok.position == pos) & ~ok.index.isin(xi)].sort_values(["recent_minutes", "price"], ascending=False)
        if len(bench):
            xi.append(bench.index[0])
    return xi, out
