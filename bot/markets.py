"""Probabilities for the side markets, all tested in the Markets Lab.

- Goals: a score grid from two expected-goal numbers (Poisson with the Dixon-Coles low-score tweak) gives totals,
  both teams to score, correct score, double chance, draw no bet, win to nil, team totals and same-game combos.
- Half-time: each half gets its usual share of the goals, which gives the half-time result and half-time/full-time.
- Corners, cards and shots on target: expected counts for each side (negative binomial), with match totals
  spread the way real totals spread (the two sides' counts are linked).
- Players: each starter's scoring, assist and card rates per 90 minutes, shrunk toward his position's average,
  scaled to tonight's expected goals.

Fitting uses scikit-learn and scipy; predicting needs numpy and scipy only.
"""
import math

import numpy as np
from scipy.stats import binom, gamma, nbinom, poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler

MAXG = 10
KMAX = 40
GOAL_FEATURES = ["lgh", "lga", "elo_diff", "pi_diff", "shot_gap"]
COUNTS = {  # stat: (home column, away column, features)
    "shots": ("hst", "ast", ["lsh", "lsa", "elo_diff", "pi_diff", "lgh", "lga", "lvl_shots"]),
    "corners": ("hc", "ac", ["lch", "lca", "elo_diff", "shot_gap", "lgh", "lga", "lvl_corners"]),
    "cards": ("hcd", "acd", ["lkh", "lka", "elo_diff", "abs_elo", "lvl_cards"]),
}
LINES = {"goals": [1.5, 2.5, 3.5], "shots": [7.5, 8.5, 9.5], "corners": [8.5, 9.5, 10.5, 11.5], "cards": [2.5, 3.5, 4.5, 5.5]}
PLAYER_PRIOR_90S = 8.0   # a player's own record counts like this many full games against his position's average
XG_WEIGHT_90S = 10.0     # how fast xG takes over from goals as the evidence for scoring rate


def add_inputs(table):
    """Log expected counts from the ratings, the inputs every market model uses (works on a DataFrame)."""
    t = table.copy()
    t["lgh"], t["lga"] = np.log(t.exp_home_goals), np.log(t.exp_away_goals)
    for short, name in [("s", "shots"), ("c", "corners"), ("k", "cards")]:
        t[f"l{short}h"], t[f"l{short}a"] = np.log(t[f"exp_home_{name}"]), np.log(t[f"exp_away_{name}"])
    t["abs_elo"] = t.elo_diff.abs()
    return t


def fit_poisson(table, features, target):
    """Poisson regression stored as JSON, plus its overdispersion: variance = mean + alpha * mean^2."""
    X = table[features].to_numpy(float)
    sc = StandardScaler().fit(X)
    reg = PoissonRegressor(alpha=1e-4, max_iter=1000).fit(sc.transform(X), table[target].to_numpy(float))
    mu, y = reg.predict(sc.transform(X)), table[target].to_numpy(float)
    return {"features": list(features), "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
            "coef": reg.coef_.tolist(), "intercept": float(reg.intercept_),
            "alpha": float(max(0.0, ((y - mu) ** 2 - y).sum() / (mu ** 2).sum()))}


def predict_mean(model, rows):
    X = rows[model["features"]].to_numpy(float)
    z = (X - np.array(model["mean"])) / np.array(model["scale"])
    return np.exp(z @ np.array(model["coef"]) + model["intercept"])


def fit_counts(table, name):
    """Home and away models for one stat, how match totals spread, and how linked the two sides are."""
    th, ta, feats = COUNTS[name]
    t = table.dropna(subset=[th, ta])
    out = {"home": fit_poisson(t, feats, th), "away": fit_poisson(t, feats, ta)}
    mh, ma = predict_mean(out["home"], t), predict_mean(out["away"], t)
    yh, ya = t[th].to_numpy(float), t[ta].to_numpy(float)
    total, mt = yh + ya, mh + ma
    out["total_alpha"] = float(((total - mt) ** 2 - total).sum() / (mt ** 2).sum())  # below 0: tighter than Poisson
    out["link"] = float(((yh - mh) * (ya - ma)).sum() / (mh * ma).sum())
    return out


def count_dist(mu, alpha):
    k = np.arange(KMAX + 1)
    if alpha < 1e-6:
        return poisson.pmf(k[None, :], mu[:, None])
    r = 1 / alpha
    return nbinom.pmf(k[None, :], r, (r / (r + mu))[:, None])


def total_dist(mu, total_alpha):
    k = np.arange(KMAX + 1)
    if total_alpha > 0:
        return count_dist(mu, total_alpha)
    ratio = 1 + total_alpha * mu.mean()  # variance / mean, below 1: a binomial with the same mean
    n = np.maximum(np.round(mu / max(1e-3, 1 - ratio)), mu + 1)
    return binom.pmf(k[None, :], n[:, None], (mu / n)[:, None])


def count_markets(name, model, rows):
    """Over/under lines for the match and each side, plus which side has more (arrays, one per match)."""
    mh, ma = predict_mean(model["home"], rows), predict_mean(model["away"], rows)
    dh, da, dt = count_dist(mh, model["home"]["alpha"]), count_dist(ma, model["away"]["alpha"]), total_dist(mh + ma, model["total_alpha"])
    out = {f"{name}_exp_home": mh, f"{name}_exp_away": ma}
    for line in LINES[name]:
        out[f"{name}_over_{line}"] = 1 - dt[:, : int(line) + 1].sum(axis=1)
    if model["link"] > 0.005:  # both sides' counts rise and fall together (cards): average over a shared match factor
        qs = gamma.ppf((np.arange(20) + 0.5) / 20, a=1 / model["link"], scale=model["link"])
        more, level = np.zeros(len(mh)), np.zeros(len(mh))
        for q in qs:
            ph, pa = count_dist(mh * q, model["home"]["alpha"]), count_dist(ma * q, model["away"]["alpha"])
            ca = np.cumsum(pa, axis=1)
            more += (ph[:, 1:] * ca[:, :-1]).sum(axis=1) / 20
            level += (ph * pa).sum(axis=1) / 20
    else:
        ca = np.cumsum(da, axis=1)
        more, level = (dh[:, 1:] * ca[:, :-1]).sum(axis=1), (dh * da).sum(axis=1)
    out[f"{name}_home_more"], out[f"{name}_level"] = more, level
    out[f"{name}_away_more"] = 1 - more - level
    return out


def score_grid(mu_h, mu_a, rho):
    g = np.arange(MAXG + 1)
    M = poisson.pmf(g[None, :, None], mu_h[:, None, None]) * poisson.pmf(g[None, None, :], mu_a[:, None, None])
    M[:, 0, 0] *= 1 - mu_h * mu_a * rho
    M[:, 0, 1] *= 1 + mu_h * rho
    M[:, 1, 0] *= 1 + mu_a * rho
    M[:, 1, 1] *= 1 - rho
    return M / M.sum(axis=(1, 2), keepdims=True)


def fit_rho(mu_h, mu_a, hg, ag):
    from scipy.optimize import minimize_scalar
    hg, ag = np.minimum(hg, MAXG).astype(int), np.minimum(ag, MAXG).astype(int)
    nll = lambda rho: -np.log(score_grid(mu_h, mu_a, rho)[np.arange(len(hg)), hg, ag]).mean()
    return float(minimize_scalar(nll, bounds=(-0.3, 0.3), method="bounded").x)


def goal_markets(M):
    g = np.arange(MAXG + 1)
    H, A = np.meshgrid(g, g, indexing="ij")
    tot, diff = H + A, H - A
    P = lambda mask: (M * mask[None]).sum(axis=(1, 2))
    out = {"goals_home": P(diff > 0), "goals_draw": P(diff == 0), "goals_away": P(diff < 0)}
    for line in LINES["goals"]:
        out[f"goals_over_{line}"] = P(tot > line)
    out["btts"] = P((H > 0) & (A > 0))
    out["home_scores"], out["away_scores"] = P(H > 0), P(A > 0)
    out["home_win_to_nil"], out["away_win_to_nil"] = P((diff > 0) & (A == 0)), P((diff < 0) & (H == 0))
    out["home_win_and_over_2.5"], out["home_win_and_btts"] = P((diff > 0) & (tot > 2)), P((diff > 0) & (H > 0) & (A > 0))
    out["away_win_and_over_2.5"], out["away_win_and_btts"] = P((diff < 0) & (tot > 2)), P((diff < 0) & (H > 0) & (A > 0))
    return out


def top_scores(M, n=4):
    """The n likeliest scorelines for one match: [(home goals, away goals, chance), ...]."""
    flat = np.argsort(M.ravel())[::-1][:n]
    return [(int(i // (MAXG + 1)), int(i % (MAXG + 1)), float(M.ravel()[i])) for i in flat]


def half_markets(mu_h, mu_a, share_h, share_a):
    """Half-time result and half-time/full-time, with each half's goals independent."""
    g = np.arange(MAXG + 1)
    pm = lambda lam: poisson.pmf(g[None, :], lam[:, None])
    h1 = pm(share_h * mu_h)[:, :, None] * pm(share_a * mu_a)[:, None, :]
    h2 = pm((1 - share_h) * mu_h)[:, :, None] * pm((1 - share_a) * mu_a)[:, None, :]
    D = g[:, None] - g[None, :]
    ds = np.arange(-MAXG, MAXG + 1)
    d1 = np.stack([(h1 * (D == d)[None]).sum(axis=(1, 2)) for d in ds], axis=1)
    d2 = np.stack([(h2 * (D == d)[None]).sum(axis=(1, 2)) for d in ds], axis=1)
    out = {"ht_home": (h1 * (D > 0)[None]).sum(axis=(1, 2)), "ht_draw": (h1 * (D == 0)[None]).sum(axis=(1, 2)),
           "ht_away": (h1 * (D < 0)[None]).sum(axis=(1, 2))}
    names = {1: "home", 0: "draw", -1: "away"}
    for a in (1, 0, -1):
        for b in (1, 0, -1):
            acc = np.zeros(len(mu_h))
            for i, x in enumerate(ds):
                if np.sign(x) == a:
                    acc += d1[:, i] * (d2 * (np.sign(x + ds) == b)[None]).sum(axis=1)
            out[f"htft_{names[a]}_{names[b]}"] = acc
    return out


def player_goal_chances(rate, minutes_share, scale):
    """Chance of 1+ goals (or assists, or a card) from a per-90 rate, the share of 90 minutes and tonight's scale."""
    x = np.asarray(rate) * np.asarray(minutes_share) * np.asarray(scale)
    return 1 - np.exp(-x), 1 - np.exp(-x) * (1 + x)


def shrunk_rate(events, minutes, prior, xg_events=None, xg_minutes=None):
    """Per-90 rate shrunk toward the position average; xG takes over as its own sample grows."""
    n90 = np.asarray(minutes, float) / 90
    r = (np.asarray(events, float) + PLAYER_PRIOR_90S * prior) / (n90 + PLAYER_PRIOR_90S)
    if xg_events is not None:
        nx = np.asarray(xg_minutes, float) / 90
        rx = (np.asarray(xg_events, float) + PLAYER_PRIOR_90S * prior) / (nx + PLAYER_PRIOR_90S)
        w = nx / (nx + XG_WEIGHT_90S)
        r = (1 - w) * r + w * rx
    return r


def all_markets(model, rows):
    """Every market for a table of matches (a DataFrame with the rating features). Returns a dict of arrays and the score grids."""
    t = add_inputs(rows)
    mu_h, mu_a = predict_mean(model["goals"]["home"], t), predict_mean(model["goals"]["away"], t)
    M = score_grid(mu_h, mu_a, model["goals"]["rho"])
    out = {"goals_exp_home": mu_h, "goals_exp_away": mu_a, **goal_markets(M),
           **half_markets(mu_h, mu_a, model["goals"]["share_h"], model["goals"]["share_a"])}
    for name in COUNTS:
        out.update(count_markets(name, model[name], t))
    return out, M


# ---------------------------------------------------------------------------------------------
# Players. Evidence before a match = this season's earlier games, plus last season's totals counted in
# full at the start and fading out over the first 19 games. Training and the live bot use the same rule.
# ---------------------------------------------------------------------------------------------
PLAYER_STATS = ["minutes", "goals", "assists", "cards", "xg", "xa", "xg_minutes"]
FADE_GAMES = 19


def player_evidence(hist, last_season):
    """For every row of `hist` (one season of appearances, any order), the evidence before that match.

    `last_season` maps player name -> totals of PLAYER_STATS for the previous season."""
    h = hist.sort_values(["kickoff", "name"]).copy()
    h["xg_minutes"] = np.where(h.xg.notna(), h.minutes, 0)
    h["xg"], h["xa"] = h.xg.fillna(0), h.xa.fillna(0)
    g = h.groupby("name", sort=False)
    apps = g.minutes.transform(lambda m: (m > 0).astype(int).shift(1, fill_value=0).cumsum())
    weight = np.clip(1 - apps / FADE_GAMES, 0, 1)
    for c in PLAYER_STATS:
        before = g[c].transform(lambda v: v.shift(1, fill_value=0).cumsum())
        prev = h.name.map(lambda n: last_season.get(n, {}).get(c, 0.0))
        h[f"ev_{c}"] = before + weight * prev
    h["start_min"] = np.where(h.starts == 1, h.minutes, np.nan)
    sm = g.start_min.transform(lambda v: v.shift(1).expanding().mean())
    h["exp_minutes"] = sm.fillna(h.name.map(lambda n: last_season.get(n, {}).get("minutes_per_start", np.nan))).fillna(75.0)
    return h


def player_probs(ev, priors, scale):
    """Chances for each player from his evidence row(s): anytime scorer, 2+ goals, assist, score or assist, card."""
    pos = ev.position
    share = np.clip(ev.exp_minutes / 90, 0.3, 1.0)
    g = shrunk_rate(ev.ev_goals, ev.ev_minutes, pos.map(priors["goals"]), ev.ev_xg, ev.ev_xg_minutes)
    a = shrunk_rate(ev.ev_assists, ev.ev_minutes, pos.map(priors["assists"]), ev.ev_xa, ev.ev_xg_minutes)
    c = shrunk_rate(ev.ev_cards, ev.ev_minutes, pos.map(priors["cards"]))
    xg, xa = g * share * scale, a * share * scale
    return {"p_score": 1 - np.exp(-xg), "p_two": 1 - np.exp(-xg) * (1 + xg), "p_assist": 1 - np.exp(-xa),
            "p_involved": 1 - np.exp(-(xg + xa)), "p_card": 1 - np.exp(-c * share)}


def season_totals(hist):
    """Each player's totals for one season, the carry-in for the next."""
    h = hist.copy()
    h["xg_minutes"] = np.where(h.xg.notna(), h.minutes, 0)
    h["xg"], h["xa"] = h.xg.fillna(0), h.xa.fillna(0)
    tot = h.groupby("name")[PLAYER_STATS].sum()
    starts = h[h.starts == 1].groupby("name").minutes.mean()
    tot["minutes_per_start"] = starts.reindex(tot.index)
    return {n: {k: float(v) for k, v in row.items() if v == v} for n, row in tot.iterrows()}
