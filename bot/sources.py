"""Where live data comes from.

- Fantasy Premier League API: fixtures, results (with saves, which give shots on target), squads, prices,
  injury flags and every player's match-by-match minutes and xG. Free and needs no key, but it is unofficial,
  so field names can change.
- API-Football (optional, free key): confirmed starting XIs, usually out 20 to 40 minutes before kickoff.
- The Odds API (optional, free key): every bookie's match odds, for the best price and the market's view.

Replay (at the bottom) serves the same tables from the public archive for a past date, which is how
the bot is tested without touching the live APIs.
"""
import difflib
import re
import time
import unicodedata

import pandas as pd
import requests

from . import history
from .config import (API_FOOTBALL, API_FOOTBALL_KEY, API_FOOTBALL_LEAGUE, FPL_API, ODDS_API, ODDS_API_KEY,
                     ODDS_REGIONS, POSITIONS, canon)
from .model import fair_probs


def norm(text):
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", " ", text.lower()).split()


def history_rows(season, player_key, info, rows, fixtures):
    """Shape element-summary history rows like the training data."""
    out = []
    keyed = fixtures.set_index("fixture_id")
    for h in rows:
        fid = int(h["fixture"])
        if fid not in keyed.index:
            continue
        fx = keyed.loc[fid]
        out.append({"player": player_key, "name": info["name"], "position": info["position"], "season": season,
                    "team": fx.home if h["was_home"] else fx.away, "fixture": f"{season}:{fid}",
                    "kickoff": pd.Timestamp(h["kickoff_time"]), "minutes": h["minutes"], "starts": h.get("starts"),
                    "xg": float(h.get("expected_goals", "nan")), "xgc": float(h.get("expected_goals_conceded", "nan")),
                    "xa": float(h.get("expected_assists", "nan")), "goals": h.get("goals_scored", 0),
                    "assists": h.get("assists", 0), "cards": h.get("yellow_cards", 0) + h.get("red_cards", 0),
                    "price": h["value"] / 10})
    return out


class FplLive:
    def __init__(self):
        self.http = requests.Session()
        self.http.headers["User-Agent"] = "premier-league-prediction-bot (personal project)"
        self._boot, self._fixtures, self._summaries = None, None, {}

    def _get(self, path):
        for attempt in range(4):
            r = self.http.get(f"{FPL_API}/{path}", timeout=30)
            if r.status_code == 200:
                return r.json()
            time.sleep(3 * (attempt + 1))
        r.raise_for_status()

    def boot(self):
        if self._boot is None:
            self._boot = self._get("bootstrap-static/")
        return self._boot

    def season(self):
        first = pd.Timestamp(self.boot()["events"][0]["deadline_time"])
        return history.label(first.year if first.month >= 7 else first.year - 1)

    def teams(self):
        return {t["id"]: canon(t["name"]) for t in self.boot()["teams"]}

    def players(self):
        teams, season = self.teams(), self.season()
        rows = [{"player": f"{season}:{e['id']}", "element": e["id"], "name": f"{e['first_name']} {e['second_name']}",
                 "web_name": e["web_name"], "team": teams[e["team"]], "position": POSITIONS.get(e["element_type"]),
                 "price": e["now_cost"] / 10, "status": e["status"], "chance": e["chance_of_playing_next_round"],
                 "news": e["news"], "minutes": e["minutes"]} for e in self.boot()["elements"]]
        return pd.DataFrame(rows).dropna(subset=["position"]).set_index("player")

    def fixtures(self):
        if self._fixtures is None:
            teams, season = self.teams(), self.season()
            raw = self._get("fixtures/")
            rows = []
            for f in raw:
                hst, ast = (history.shots_on_target(f.get("stats"), f["team_h_score"], f["team_a_score"])
                            if f["finished"] else (float("nan"), float("nan")))
                hcd, acd = history.cards(f.get("stats")) if f["finished"] else (float("nan"), float("nan"))
                rows.append({"season": season, "fixture_id": f["id"], "fixture": f"{season}:{f['id']}",
                             "kickoff": pd.Timestamp(f["kickoff_time"]) if f["kickoff_time"] else pd.NaT,
                             "home": teams[f["team_h"]], "away": teams[f["team_a"]], "finished": bool(f["finished"]),
                             "hg": f["team_h_score"], "ag": f["team_a_score"], "hst": hst, "ast": ast, "hcd": hcd, "acd": acd})
            self._fixtures = pd.DataFrame(rows)
        return self._fixtures

    def histories(self, player_keys, info):
        season, fixtures, rows = self.season(), self.fixtures(), []
        for key in player_keys:
            if key not in self._summaries:
                self._summaries[key] = self._get(f"element-summary/{int(key.split(':')[1])}/")["history"]
                time.sleep(0.2)
            rows += history_rows(season, key, info.loc[key], self._summaries[key], fixtures)
        return pd.DataFrame(rows)


TEAM_ALIASES = {"manchester city": "Man City", "manchester united": "Man United", "nottingham forest": "Nott'm Forest",
                "sheffield utd": "Sheffield United", "sheffield united": "Sheffield United", "hull city": "Hull",
                "wolverhampton wanderers": "Wolves", "tottenham hotspur": "Tottenham", "newcastle united": "Newcastle",
                "west ham united": "West Ham", "brighton and hove albion": "Brighton", "brighton hove albion": "Brighton",
                "leicester city": "Leicester", "leeds united": "Leeds", "ipswich town": "Ipswich",
                "coventry city": "Coventry", "afc bournemouth": "Bournemouth", "luton town": "Luton"}


def match_team(api_name, names):
    key = " ".join(norm(api_name))
    if key in TEAM_ALIASES:
        return TEAM_ALIASES[key]
    lookup = {" ".join(norm(n)): n for n in names}
    best = difflib.get_close_matches(key, lookup, n=1, cutoff=0.5)
    return lookup[best[0]] if best else None


def match_players(api_names, squad):
    """Match lineup names like "E. Haaland" to squad rows (index = player key, columns name and web_name)."""
    taken, out, unmatched = set(), [], []
    for api_name in api_names:
        tokens = norm(api_name)
        best, best_score = None, 0.0
        for key, row in squad.iterrows():
            if key in taken:
                continue
            full, web = norm(row["name"]), norm(row["web_name"])
            score = max(difflib.SequenceMatcher(None, " ".join(tokens), " ".join(full)).ratio(),
                        difflib.SequenceMatcher(None, " ".join(tokens), " ".join(web)).ratio())
            if tokens and full and tokens[-1] in (full[-1], web[-1] if web else ""):
                score += 0.3
            if len(tokens) > 1 and len(tokens[0]) == 1 and full and full[0].startswith(tokens[0]):
                score += 0.1
            if score > best_score:
                best, best_score = key, score
        if best is not None and best_score >= 0.6:
            taken.add(best)
            out.append(best)
        else:
            unmatched.append(api_name)
    return out, unmatched


class ApiFootballLineups:
    def __init__(self, key=API_FOOTBALL_KEY):
        self.key = key

    def enabled(self):
        return bool(self.key)

    def _get(self, path, **params):
        r = requests.get(f"{API_FOOTBALL}/{path}", headers={"x-apisports-key": self.key}, params=params, timeout=30)
        r.raise_for_status()
        return r.json().get("response", [])

    def starting_xis(self, fixture, team_names):
        """{our team name: [player names]} for a fixture, or None if lineups aren't out yet."""
        season_year = int(fixture.season[:4])
        for item in self._get("fixtures", league=API_FOOTBALL_LEAGUE, season=season_year, date=fixture.kickoff.date().isoformat()):
            home = match_team(item["teams"]["home"]["name"], team_names)
            away = match_team(item["teams"]["away"]["name"], team_names)
            if (home, away) == (fixture.home, fixture.away):
                lineups = self._get("fixtures/lineups", fixture=item["fixture"]["id"])
                xis = {match_team(t["team"]["name"], team_names): [p["player"]["name"] for p in t.get("startXI", [])]
                       for t in lineups}
                return xis if len(xis) == 2 and all(len(v) >= 10 for v in xis.values()) else None
        return None


class OddsApi:
    """Match odds from every bookmaker on The Odds API (free key, 500 credits a month).

    One call fetches every upcoming Premier League match and costs one credit per region.
    """

    def __init__(self, key=ODDS_API_KEY, regions=ODDS_REGIONS):
        self.key, self.regions, self._events, self.remaining = key, regions, None, None

    def enabled(self):
        return bool(self.key)

    def events(self):
        if self._events is None:
            r = requests.get(f"{ODDS_API}/sports/soccer_epl/odds", timeout=30,
                             params={"apiKey": self.key, "regions": self.regions, "markets": "h2h", "oddsFormat": "decimal"})
            r.raise_for_status()
            self._events, self.remaining = r.json(), r.headers.get("x-requests-remaining")
        return self._events

    def prices(self, fixture, team_names):
        """For one fixture: the best price on each result with its bookmaker, and the market's view (every
        bookmaker's fair chances, averaged). None if the fixture isn't listed."""
        for ev in self.events():
            home, away = match_team(ev["home_team"], team_names), match_team(ev["away_team"], team_names)
            if (home, away) != (fixture.home, fixture.away):
                continue
            best = {"home": (0.0, ""), "draw": (0.0, ""), "away": (0.0, "")}
            books = []
            for book in ev.get("bookmakers", []):
                for market in book.get("markets", []):
                    if market.get("key") != "h2h":
                        continue
                    row = {}
                    for o in market.get("outcomes", []):
                        side = "draw" if o["name"] == "Draw" else "home" if match_team(o["name"], team_names) == home else "away"
                        row[side] = float(o["price"])
                        if row[side] > best[side][0]:
                            best[side] = (row[side], book["title"])
                    if len(row) == 3 and all(v > 1 for v in row.values()):
                        books.append([row["away"], row["draw"], row["home"]])
            if not all(price > 1 for price, _ in best.values()):
                return None
            fair = fair_probs(books).mean(axis=0) if books else None
            market = {"away": fair[0], "draw": fair[1], "home": fair[2], "books": len(books)} if books else None
            return {"best": best, "market": market}
        return None


class Replay:
    """Serves a past moment from the archive with the same methods as FplLive (for tests)."""

    def __init__(self, season, now):
        self.season_label, self.now = season, now
        self.fx, self.names = history.fpl_fixtures(season)
        self.hist = history.fpl_player_history(season)

    def season(self):
        return self.season_label

    def teams(self):
        return self.names

    def fixtures(self):
        fx = self.fx.copy()
        fx["finished"] = fx.finished & (fx.kickoff < self.now)
        return fx

    def players(self):
        before = self.hist[self.hist.kickoff < self.now].sort_values("kickoff")
        latest = before.groupby("player").tail(1).set_index("player")
        return pd.DataFrame({"element": latest.element, "name": latest["name"],
                             "web_name": latest["name"].str.split().str[-1], "team": latest.team,
                             "position": latest.position, "price": latest.price, "status": "a", "chance": None, "news": ""})

    def histories(self, player_keys, info=None):
        h = self.hist[self.hist.player.isin(list(player_keys)) & (self.hist.kickoff < self.now)]
        return h.drop(columns=["element"])

    def lineup_names(self, fixture):
        """Who really started, written the way API-Football writes names ("E. Haaland")."""
        rows = self.hist[(self.hist.fixture == fixture.fixture) & (self.hist.starts == 1)]
        fmt = lambda n: f"{n.split()[0][0]}. {' '.join(n.split()[1:])}" if len(n.split()) > 1 else n
        return {team: [fmt(n) for n in g["name"]] for team, g in rows.groupby("team")}
