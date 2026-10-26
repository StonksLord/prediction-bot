"""The prediction model, stored as plain JSON. Two layers:

1. The base model learns from every match in 22 leagues: Elo, pi-ratings, attack and defence (goals and
   shots on target) and the league's usual home and draw rates. A multinomial logistic regression.
2. The Premier League layer nudges the base model's home-versus-away balance for xG form and missing stars,
   which only the Fantasy Premier League data has (from 2022-23). It has just one number per factor, because
   a few seasons of data can't support more: a fuller version over-fitted and scored worse in backtests.

Training uses scikit-learn and scipy. Predicting only needs numpy, so the live bot stays light.
"""
import json

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

CLASSES = ["A", "D", "H"]


def fit_model(table, features):
    X = table[features].to_numpy(dtype=float)
    scaler = StandardScaler().fit(X)
    lr = LogisticRegression(max_iter=5000).fit(scaler.transform(X), table.result.to_numpy())
    assert list(lr.classes_) == CLASSES
    return {"features": list(features), "classes": CLASSES, "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(), "coef": lr.coef_.tolist(), "intercept": lr.intercept_.tolist()}


def predict(model, rows):
    """rows: list of dicts (or a DataFrame) holding the model's features. Returns an (n, 3) array: away, draw, home."""
    feats = model["features"]
    X = rows[feats].to_numpy(dtype=float) if hasattr(rows, "columns") else np.array([[r[f] for f in feats] for r in rows], float)
    z = (X - np.array(model["mean"])) / np.array(model["scale"])
    logits = z @ np.array(model["coef"]).T + np.array(model["intercept"])
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def shift(probs, X, beta):
    """Move probability between home and away: the home-vs-away log-odds change by X @ beta, the draw's share holds."""
    z = np.log(np.clip(probs, 1e-12, 1))
    move = np.asarray(X, float) @ np.asarray(beta, float)
    z[:, 2] += move / 2
    z[:, 0] -= move / 2
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def layer_inputs(rows, layer):
    """The Premier League layer's inputs, each capped where the layer says (big values are rare and the data runs out)."""
    feats = layer["features"]
    X = rows[feats].to_numpy(float) if hasattr(rows, "columns") else np.array([[r[f] for f in feats] for r in rows], float)
    for j, f in enumerate(feats):
        cap = layer.get("caps", {}).get(f)
        if cap is not None:
            X[:, j] = np.clip(X[:, j], -cap, cap)
    return X


def fit_shift(base_probs, table, features, caps=None):
    """The Premier League layer: one number per factor, fitted by maximum likelihood."""
    layer = {"features": list(features), "caps": dict(caps or {})}
    X = layer_inputs(table, layer)
    y = np.array([CLASSES.index(r) for r in table.result])
    loss = lambda beta: -np.log(shift(base_probs, X, beta)[np.arange(len(y)), y]).mean()
    layer["beta"] = minimize(loss, np.zeros(len(features))).x.tolist()
    return layer


def predict_full(model, rows):
    """Both layers. rows: list of dicts or a DataFrame with the base features and the Premier League factors.
    Returns (final probabilities, base-layer probabilities), each an (n, 3) array: away, draw, home."""
    base = predict(model["stage1"], rows)
    return shift(base, layer_inputs(rows, model["stage2"]), model["stage2"]["beta"]), base


def fair_probs(odds):
    """Bookie odds (n, 3) to fair chances, taking the margin off long shots more than favourites (the power method)."""
    inv = 1 / np.asarray(odds, float)
    k = np.ones(len(inv))
    for _ in range(30):
        powered = inv ** k[:, None]
        k = k - (powered.sum(axis=1) - 1) / (powered * np.log(inv)).sum(axis=1)
    p = inv ** k[:, None]
    return p / p.sum(axis=1, keepdims=True)


def log_loss(probs, results):
    pos = np.array([CLASSES.index(r) for r in results])
    return float(-np.log(np.clip(probs[np.arange(len(pos)), pos], 1e-12, 1)).mean())


def accuracy(probs, results):
    return float((np.array(CLASSES)[probs.argmax(axis=1)] == np.asarray(results)).mean())


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float))


def load_json(path):
    return json.loads(path.read_text())
