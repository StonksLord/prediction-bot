"""Team strength, replayed one match at a time, in date order.

Four ratings for who wins, all tuned in the Simulation Lab:
- Elo: one number per club.
- Pi-ratings: a home and an away rating per club, in goals (Constantinou and Fenton, 2013).
- Attack and defence from goals: expected goals for each side.
- Attack and defence from shots on target: the same idea on a less noisy stat.
And two more for the side markets (Markets Lab): attack and defence for corners and for cards.

Each league keeps its own ratings. Over the summer, clubs that stay drift back toward average, and clubs
that arrive start from the clubs they replaced, nudged by how dominant they were in the league they came from.
"""
import math

import numpy as np
import pandas as pd

from .config import CARDS, CONTEXT_SEASONS, CORNERS, ELO, GOALS, LEAGUES, PI, PROMOTED_BETA, SHOTS

BY_TIER = {(country, tier): code for code, (country, tier) in LEAGUES.items()}


class Elo:
    default, newcomer = (1500.0,), (1450.0,)

    def __init__(self, k, home_adv, carry, mov):
        self.k, self.home, self.c, self.mov = k, home_adv, carry, mov

    def new_context(self):
        return {}

    def carry(self, v):
        return (self.c * v[0] + (1 - self.c) * 1500.0,)

    def update(self, hv, av, hg, ag, ctx):
        expected = 1 / (1 + 10 ** ((av[0] - hv[0] - self.home) / 400))
        actual = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        margin = abs(hg - ag)
        weight = (math.log1p(margin) if margin else 1.0) if self.mov == "log" else 1.0
        change = self.k * weight * (actual - expected)
        return (hv[0] + change,), (av[0] - change,)


class Pi:
    default, newcomer = (0.0, 0.0), (-0.3, -0.3)

    def __init__(self, lam, gamma, c, carry):
        self.lam, self.gamma, self.cc, self.c = lam, gamma, c, carry

    def new_context(self):
        return {}

    def carry(self, v):
        return (self.c * v[0], self.c * v[1])

    def goals(self, r):
        return math.copysign(10 ** (abs(r) / self.cc) - 1, r)

    def update(self, hv, av, hg, ag, ctx):
        error = (hg - ag) - (self.goals(hv[0]) - self.goals(av[1]))
        step = self.cc * math.log10(1 + abs(error)) * (1 if error > 0 else -1 if error < 0 else 0) * self.lam
        return (hv[0] + step, hv[1] + step * self.gamma), (av[0] - step * self.gamma, av[1] - step)


class AttackDefence:
    default, newcomer = (0.0, 0.0), (-0.1, -0.1)

    def __init__(self, eta, carry, eta_league, level, home=0.15, scaled=False):
        self.eta, self.c, self.eta_l, self.level, self.home, self.scaled = eta, carry, eta_league, level, home, scaled

    def new_context(self):
        return {"level": self.level, "home": self.home}

    def carry(self, v):
        return (self.c * v[0], self.c * v[1])

    def expected(self, hv, av, ctx):
        return (math.exp(ctx["level"] + ctx["home"] + hv[0] - av[1]), math.exp(ctx["level"] + av[0] - hv[1]))

    def update(self, hv, av, yh, ya, ctx):
        if yh != yh or ya != ya:  # not recorded for this match
            return hv, av
        lh, la = self.expected(hv, av, ctx)
        eh, ea = yh - lh, ya - la
        if self.scaled:  # big counts (corners, cards): step by the relative miss, so one busy match can't swing a rating
            eh, ea = eh / lh * min(lh, 1.5), ea / la * min(la, 1.5)
        ctx["level"] += self.eta_l * (eh + ea)
        ctx["home"] += self.eta_l * (eh - ea) / 2
        return (hv[0] + self.eta * eh, hv[1] - self.eta * ea), (av[0] + self.eta * ea, av[1] - self.eta * eh)


SYSTEMS = {"elo": Elo(**ELO), "pi": Pi(**PI), "goals": AttackDefence(level=0.25, **GOALS),
           "shots": AttackDefence(level=1.4, **SHOTS),
           "corners": AttackDefence(level=1.6, home=0.1, scaled=True, **CORNERS),
           "cards": AttackDefence(level=0.6, home=0.1, scaled=True, **CARDS)}
STATS = {"shots": ("hst", "ast"), "corners": ("hc", "ac"), "cards": ("hcd", "acd")}  # per-team counts behind each rating
TYPICAL = {"shots": 9.0, "corners": 10.0, "cards": 3.8}  # a league's usual match total, until it has a season of its own


def _mean(vals):
    return tuple(float(np.mean(v)) for v in zip(*vals)) if vals else None


class LeagueState:
    """Every club's ratings in one league, plus the league's own numbers (goal levels, home and draw rates)."""

    def __init__(self, code, season=None, teams=(), ratings=None, ctx=None, seasons=None, current=None, levels=None, tallies=None):
        self.code, self.season, self.teams = code, season, set(teams)
        self.ratings = {name: {} for name in SYSTEMS} | (ratings or {})
        self.ctx = {name: s.new_context() for name, s in SYSTEMS.items()} | (ctx or {})
        self.seasons = seasons or []          # finished seasons: [season, matches, home wins, draws]
        self.current = current or [0, 0, 0]   # this season so far: matches, home wins, draws
        self.levels = levels or {}            # last finished season's average match total per stat
        self.tallies = tallies or {k: [0.0, 0] for k in STATS}  # this season so far: total and matches recorded

    def context(self):
        last = self.seasons[-CONTEXT_SEASONS:]
        n = sum(s[1] for s in last)
        if not n:
            return {"lg_home": 0.45, "lg_draw": 0.27}
        return {"lg_home": sum(s[2] for s in last) / n, "lg_draw": sum(s[3] for s in last) / n}

    def features(self, home, away):
        r = self.ratings
        pi, goals = SYSTEMS["pi"], SYSTEMS["goals"]
        gh, ga = goals.expected(r["goals"][home], r["goals"][away], self.ctx["goals"])
        sh, sa = r["shots"][home], r["shots"][away]
        out = {"elo_diff": r["elo"][home][0] - r["elo"][away][0],
               "pi_diff": pi.goals(r["pi"][home][0]) - pi.goals(r["pi"][away][1]),
               "goal_ratio": math.log(gh / ga), "goal_total": gh + ga,
               "shot_gap": (sh[0] - sa[1]) - (sa[0] - sh[1]),
               **self.context(), "exp_home_goals": gh, "exp_away_goals": ga}
        for name in STATS:
            d = SYSTEMS[name].default
            eh, ea = SYSTEMS[name].expected(r[name].get(home, d), r[name].get(away, d), self.ctx[name])
            out[f"exp_home_{name}"], out[f"exp_away_{name}"] = eh, ea
            out[f"lvl_{name}"] = math.log(self.levels.get(name) or TYPICAL[name])
        return out

    def update(self, home, away, hg, ag, hst=float("nan"), ast=float("nan"), hc=float("nan"), ac=float("nan"),
               hcd=float("nan"), acd=float("nan")):
        hg, ag = int(hg), int(ag)
        counts = {"shots": (hst, ast), "corners": (hc, ac), "cards": (hcd, acd)}
        for name, system in SYSTEMS.items():
            yh, ya = (float(counts[name][0]), float(counts[name][1])) if name in counts else (hg, ag)
            r = self.ratings[name]
            r[home], r[away] = system.update(r.get(home, system.default), r.get(away, system.default), yh, ya, self.ctx[name])
            if name in counts and yh == yh and ya == ya:
                self.tallies[name] = [self.tallies[name][0] + yh + ya, self.tallies[name][1] + 1]
        self.current = [self.current[0] + 1, self.current[1] + (hg > ag), self.current[2] + (hg == ag)]

    def snapshot(self):
        return {name: {t: r[t] for t in self.teams if t in r} for name, r in self.ratings.items()}

    def start_season(self, season, teams, neighbours=(), beta=PROMOTED_BETA):
        """Move to a new season.

        `neighbours` describes the leagues just below and above: (their teams last season, their ratings at the
        end of last season, their teams this season or None if unknown).
        """
        teams = set(teams)
        if self.season is None:
            for name, system in SYSTEMS.items():
                for t in teams:
                    self.ratings[name][t] = system.default
            self.teams, self.season = teams, season
            return
        if self.current[0]:
            self.seasons.append([self.season, *self.current])
        self.current = [0, 0, 0]
        for name, (total, n) in self.tallies.items():
            if n >= 50:  # a season with the stat mostly recorded sets next season's level
                self.levels[name] = total / n
            elif name in self.levels and n == 0:
                del self.levels[name]  # not recorded last season: the old level is too stale to trust
        self.tallies = {k: [0.0, 0] for k in STATS}
        left, arrived = self.teams - teams, teams - self.teams
        for name, system in SYSTEMS.items():
            r = self.ratings[name]
            fallback = _mean([r[t] for t in left]) or system.newcomer
            placed = set()
            for prev_teams, final, new_teams in neighbours:
                final = final.get(name, {})
                came = [t for t in arrived if t in prev_teams and t in final]
                if not came:
                    continue
                went = [t for t in left if new_teams is not None and t in new_teams]
                base, avg = _mean([r[t] for t in went]) or fallback, _mean([final[t] for t in came])
                for t in came:
                    r[t] = tuple(b + beta * (v - a) for b, v, a in zip(base, final[t], avg))
                    placed.add(t)
            for t in arrived - placed:
                r[t] = fallback
            for t in teams:
                r[t] = system.carry(r[t])
        self.teams, self.season = teams, season

    def to_dict(self):
        return {"code": self.code, "season": self.season, "teams": sorted(self.teams),
                "ratings": {name: {t: list(v) for t, v in r.items()} for name, r in self.ratings.items()},
                "ctx": self.ctx, "seasons": self.seasons, "current": self.current, "levels": self.levels,
                "tallies": self.tallies}

    @classmethod
    def from_dict(cls, d):
        ratings = {name: {t: tuple(v) for t, v in r.items()} for name, r in d["ratings"].items()}
        return cls(d["code"], d["season"], d["teams"], ratings, {k: dict(v) for k, v in d["ctx"].items()},
                   [list(s) for s in d["seasons"]], list(d["current"]), dict(d.get("levels", {})),
                   {k: list(v) for k, v in d.get("tallies", {}).items()} or None)


def neighbour_leagues(code):
    country, tier = LEAGUES[code]
    return [BY_TIER.get((country, tier + 1)), BY_TIER.get((country, tier - 1))]  # the league below, then above


def replay_all(results):
    """Walk through every league's results in date order.

    `results` needs columns league, season, kickoff, home, away, hg, ag, hst, ast, sorted by kickoff.
    Matches that kick off together are predicted together before any of their results count, so no match
    ever sees a result it couldn't have known. Returns the pre-match features for every row and each
    league's state after its last match.
    """
    season_teams = {k: set(g.home) | set(g.away) for k, g in results.groupby(["league", "season"])}
    states, ended, feats = {}, {}, [None] * len(results)
    lg, ss, ko = results.league.to_numpy(), results.season.to_numpy(), results.kickoff.to_numpy()
    hm, aw = results.home.to_numpy(), results.away.to_numpy()
    hg, ag = results.hg.to_numpy(), results.ag.to_numpy()
    hst, ast = results.hst.to_numpy(float), results.ast.to_numpy(float)
    extra = {c: (results[c].to_numpy(float) if c in results else np.full(len(results), np.nan)) for c in ["hc", "ac", "hcd", "acd"]}
    countries = results.league.map(lambda c: LEAGUES[c][0])

    def prepare(L, S):
        st = states.get(L)
        if st is None:
            st = states[L] = LeagueState(L)
        if st.season != S:
            prev, neighbours = st.season, []
            if prev is not None:
                ended[(L, prev)] = st.snapshot()
                for other in neighbour_leagues(L):
                    if other is None:
                        continue
                    o = states.get(other)
                    final = o.snapshot() if o is not None and o.season == prev else ended.get((other, prev), {})
                    neighbours.append((season_teams.get((other, prev), set()), final, season_teams.get((other, S))))
            st.start_season(S, season_teams[(L, S)], neighbours)
        return st

    for _, idx in results.groupby(countries, sort=False).indices.items():
        start = 0
        while start < len(idx):
            end = start
            while end < len(idx) and ko[idx[end]] == ko[idx[start]]:
                end += 1
            group = idx[start:end]
            for i in group:
                feats[i] = prepare(lg[i], ss[i]).features(hm[i], aw[i])
            for i in group:
                states[lg[i]].update(hm[i], aw[i], hg[i], ag[i], hst[i], ast[i], extra["hc"][i], extra["ac"][i],
                                     extra["hcd"][i], extra["acd"][i])
            start = end
    return pd.DataFrame(feats, index=results.index), states
