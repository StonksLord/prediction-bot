"""Historical data from public GitHub datasets, shaped the same way as the live data."""
import ast
import re
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

from .config import DATA_DIR, FIRST_FPL_SEASON, FPL_HISTORY_URL, HISTORY_RESULTS_URL, LEAGUES, canon


def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    return path


def label(start_year):
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def all_leagues():
    """Every match in the 22 leagues since 2000-01, from an archive of football-data.co.uk files.

    Club names are made consistent ("MGladbach" and "M'gladbach" are one club), and seasons are read from
    the calendar: a break of 45+ days ending between July and October starts a new season.
    """
    path = fetch(HISTORY_RESULTS_URL, DATA_DIR / "Matches.csv")
    m = pd.read_csv(path, low_memory=False, usecols=["Division", "MatchDate", "MatchTime", "HomeTeam", "AwayTeam", "FTHome", "FTAway",
                                                      "HTHome", "HTAway", "HomeTarget", "AwayTarget", "HomeCorners", "AwayCorners", "HomeYellow",
                                                      "AwayYellow", "HomeRed", "AwayRed", "OddHome", "OddDraw", "OddAway",
                                                      "MaxHome", "MaxDraw", "MaxAway", "Over25", "Under25"])
    m = m[m.Division.isin(LEAGUES)].dropna(subset=["FTHome", "FTAway"]).copy()
    key = lambda s: re.sub(r"[^a-z0-9]", "", s.strip().lower())
    names = pd.concat([m.HomeTeam, m.AwayTeam]).str.strip()
    spelling = names.groupby(names.map(key)).agg(lambda s: s.value_counts().index[0])
    m["home"] = m.HomeTeam.map(key).map(spelling).map(canon)
    m["away"] = m.AwayTeam.map(key).map(spelling).map(canon)
    date = pd.to_datetime(m.MatchDate)
    m["kickoff"] = date.dt.tz_localize("UTC") + pd.Timedelta(hours=14)
    m = m.drop_duplicates(["Division", "kickoff", "home", "away"])
    m = m.sort_values(["kickoff", "MatchTime", "Division", "home"], na_position="first", kind="stable").reset_index(drop=True)
    gap = m.groupby("Division").kickoff.diff().dt.days
    new_season = gap.isna() | ((gap >= 45) & m.kickoff.dt.month.isin([7, 8, 9, 10]))
    first = m.groupby([m.Division, new_season.groupby(m.Division).cumsum()]).kickoff.transform("min")
    m["season"] = [label(y) for y in np.where(first.dt.month >= 6, first.dt.year, first.dt.year - 1)]
    m["league"], m["fixture"] = m.Division, None
    m["hg"], m["ag"] = m.FTHome.astype(int), m.FTAway.astype(int)
    m["hst"], m["ast"] = m.HomeTarget.astype(float), m.AwayTarget.astype(float)
    m["hc"], m["ac"] = m.HomeCorners.astype(float), m.AwayCorners.astype(float)
    m["hcd"], m["acd"] = (m.HomeYellow + m.HomeRed).astype(float), (m.AwayYellow + m.AwayRed).astype(float)
    m["hhg"], m["hag"] = m.HTHome.astype(float), m.HTAway.astype(float)
    return m[["league", "season", "kickoff", "fixture", "home", "away", "hg", "ag", "hst", "ast", "hc", "ac", "hcd", "acd", "hhg", "hag",
              "OddHome", "OddDraw", "OddAway", "MaxHome", "MaxDraw", "MaxAway", "Over25", "Under25"]]


def fpl_path(season, name):
    remote = "gws/merged_gw.csv" if name == "merged_gw.csv" else name
    return fetch(f"{FPL_HISTORY_URL}/{season}/{remote}", DATA_DIR / "fpl" / season / name)


def fpl_seasons():
    """Every season the Fantasy Premier League archive has, from 2019-20 on."""
    seasons, year = [], int(FIRST_FPL_SEASON[:4])
    while True:
        try:
            fpl_path(label(year), "fixtures.csv")
        except urllib.error.HTTPError:
            return seasons
        seasons.append(label(year))
        year += 1


def shots_on_target(stats, home_goals, away_goals):
    """Shots on target for each side from a Fantasy Premier League fixture's stats.

    The feed doesn't list shots, but every shot on target is either saved or scored: a side's shots on target
    are the other keeper's saves plus its own goals (minus own goals the other side gifted it).
    """
    if isinstance(stats, str):
        stats = ast.literal_eval(stats) if stats.strip() else []
    if not stats or home_goals is None or pd.isna(home_goals):
        return np.nan, np.nan
    by_id = {s["identifier"]: s for s in stats}
    total = lambda what, side: sum(x["value"] for x in by_id.get(what, {}).get(side, []))
    return (total("saves", "a") + home_goals - total("own_goals", "a"),
            total("saves", "h") + away_goals - total("own_goals", "h"))


def cards(stats):
    """Yellow plus red cards for each side from a Fantasy Premier League fixture's stats (matches the official count 99% of the time)."""
    if isinstance(stats, str):
        stats = ast.literal_eval(stats) if stats.strip() else []
    if not stats:
        return np.nan, np.nan
    by_id = {s["identifier"]: s for s in stats}
    total = lambda side: sum(x["value"] for what in ("yellow_cards", "red_cards") for x in by_id.get(what, {}).get(side, []))
    return float(total("h")), float(total("a"))


def fpl_fixtures(season):
    fx = pd.read_csv(fpl_path(season, "fixtures.csv"))
    teams = pd.read_csv(fpl_path(season, "teams.csv"))
    names = {int(i): canon(n) for i, n in zip(teams.id, teams.name)}
    finished = fx.finished.astype(str).str.lower().eq("true")
    stats = fx.get("stats", pd.Series([None] * len(fx)))
    shots = [shots_on_target(s, h, a) if f else (np.nan, np.nan) for s, h, a, f in zip(stats, fx.team_h_score, fx.team_a_score, finished)]
    booked = [cards(s) if f else (np.nan, np.nan) for s, f in zip(stats, finished)]
    out = pd.DataFrame({"season": season, "fixture_id": fx.id.astype(int),
                        "fixture": [f"{season}:{i}" for i in fx.id.astype(int)],
                        "kickoff": pd.to_datetime(fx.kickoff_time, utc=True),
                        "home": fx.team_h.astype(int).map(names), "away": fx.team_a.astype(int).map(names),
                        "home_id": fx.team_h.astype(int), "away_id": fx.team_a.astype(int),
                        "finished": finished, "hg": fx.team_h_score, "ag": fx.team_a_score,
                        "hst": [s[0] for s in shots], "ast": [s[1] for s in shots],
                        "hcd": [c[0] for c in booked], "acd": [c[1] for c in booked]})
    return out, names


def fpl_results(season):
    fx, _ = fpl_fixtures(season)
    done = fx[fx.finished].copy()
    done["hg"], done["ag"] = done.hg.astype(int), done.ag.astype(int)
    done["league"] = "E0"
    return done[["league", "season", "kickoff", "fixture", "home", "away", "hg", "ag", "hst", "ast", "hcd", "acd"]]


def all_matches(last):
    """Every league's results up to season `last`, with Premier League matches from 2019-20 on taken from
    Fantasy Premier League (the same source the live bot uses), plus Bet365's and the best Premier League odds."""
    arch = all_leagues()
    e0 = arch[arch.league == "E0"].drop_duplicates(["season", "home", "away"])
    odds = e0[["season", "home", "away", "OddHome", "OddDraw", "OddAway", "MaxHome", "MaxDraw", "MaxAway", "Over25", "Under25"]]
    arch = arch[(arch.season <= last) & ~((arch.league == "E0") & (arch.season >= FIRST_FPL_SEASON))]
    fpl = [fpl_results(s) for s in fpl_seasons() if s <= last]
    if fpl:  # corners and half-time scores aren't in Fantasy Premier League: take them from the archive
        fpl = [f.merge(e0[["season", "home", "away", "hc", "ac", "hhg", "hag"]], on=["season", "home", "away"], how="left") for f in fpl]
    results = pd.concat([arch.drop(columns=["OddHome", "OddDraw", "OddAway", "MaxHome", "MaxDraw", "MaxAway", "Over25", "Under25"])] + fpl, ignore_index=True)
    results = results.sort_values(["kickoff", "league", "home"], kind="stable").reset_index(drop=True)
    return results, odds


def fpl_player_history(season):
    g = pd.read_csv(fpl_path(season, "merged_gw.csv"), low_memory=False)
    for c in ["starts", "expected_goals", "expected_goals_conceded", "expected_assists"]:
        if c not in g.columns:
            g[c] = np.nan
    g = g[g.position.replace({"GKP": "GK"}).isin(["GK", "DEF", "MID", "FWD"])]
    fx, _ = fpl_fixtures(season)
    keyed = fx.set_index("fixture_id")
    hist = pd.DataFrame({
        "player": [f"{season}:{e}" for e in g.element.astype(int)],
        "element": g.element.astype(int).to_numpy(),
        "name": g.name.to_numpy(),
        "team": g.team.map(canon).to_numpy(),
        "position": g.position.replace({"GKP": "GK"}).to_numpy(),
        "season": season,
        "fixture": [f"{season}:{f}" for f in g.fixture.astype(int)],
        "kickoff": keyed.kickoff.reindex(g.fixture.astype(int)).to_numpy(),
        "minutes": g.minutes.to_numpy(),
        "starts": g.starts.to_numpy(),
        "xg": g.expected_goals.to_numpy(),
        "xgc": g.expected_goals_conceded.to_numpy(),
        "xa": g.expected_assists.to_numpy(),
        "goals": g.goals_scored.to_numpy(), "assists": g.assists.to_numpy(),
        "cards": (g.yellow_cards + g.red_cards).to_numpy(),
        "price": (g.value / 10).to_numpy(),
    })
    sides = keyed[["home", "away"]].reindex(g.fixture.astype(int))
    hist = hist[(hist.team.to_numpy() == sides.home.to_numpy()) | (hist.team.to_numpy() == sides.away.to_numpy())]
    return hist.drop_duplicates(["fixture", "player"]).reset_index(drop=True)
