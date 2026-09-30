"""Detector stage: one XGBoost model per dataset, trained on detector_train, tuned on development only.

The operating threshold is fixed on development benign flows (1% false-alarm rate) so the certificate stage
works with a known decision rule. Metrics are reported per distinct record and weighted by multiplicity.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

DETECTOR_CONFIG = {
    "version": "detector_v1",
    "model": "xgboost binary:logistic, hist trees",
    "params": {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8,
               "colsample_bytree": 0.8, "min_child_weight": 1.0, "reg_lambda": 1.0, "tree_method": "hist",
               "eval_metric": "logloss", "early_stopping_rounds": 30, "random_state": 20260930},
    "train_role": "detector_train (distinct records, unweighted)",
    "tuning_role": "development (early stopping and threshold only)",
    "threshold_rule": "score threshold giving a 1% false-alarm rate on development benign flows",
    "untouched_roles": ["calibration", "locked_test"],
}
TARGET_FPR = 0.01


def fit(X_tr, y_tr, X_dev, y_dev, params=None):
    """Train with early stopping on development and return the booster trimmed to its best round, so the saved,
    verified and deployed model is the same object that makes every decision."""
    import xgboost as xgb
    p = dict(DETECTOR_CONFIG["params"] if params is None else params)
    model = xgb.XGBClassifier(objective="binary:logistic", n_jobs=-1, **p)
    model.fit(X_tr, y_tr, eval_set=[(X_dev, y_dev)], verbose=False)
    booster = model.get_booster()
    best = getattr(model, "best_iteration", None)
    if best is not None:
        booster = booster[: best + 1]
    return booster


def predict(booster, X: pd.DataFrame) -> np.ndarray:
    import xgboost as xgb
    return booster.predict(xgb.DMatrix(X, feature_names=list(X.columns)))


def threshold_at_fpr(score_benign: np.ndarray, fpr: float = TARGET_FPR) -> float:
    """Smallest threshold t with share(score >= t) <= fpr among benign scores."""
    s = np.sort(score_benign)
    k = int(np.floor(fpr * len(s)))
    return float(np.nextafter(s[len(s) - k - 1], np.inf)) if k < len(s) else float(s[0])


def metrics(y, score, thr, weight=None, family=None):
    from sklearn.metrics import average_precision_score, roc_auc_score
    w = None if weight is None else np.asarray(weight, dtype=np.float64)
    pred = score >= thr
    out = {"rows": int(len(y)), "attack_share": float(np.average(y, weights=w)),
           "auroc": float(roc_auc_score(y, score, sample_weight=w)),
           "pr_auc": float(average_precision_score(y, score, sample_weight=w))}
    ww = np.ones(len(y)) if w is None else w
    b, a = y == 0, y == 1
    out["false_alarm_rate"] = float(ww[b & pred].sum() / ww[b].sum())
    out["recall"] = float(ww[a & pred].sum() / ww[a].sum())
    tp, fp = ww[a & pred].sum(), ww[b & pred].sum()
    out["precision"] = float(tp / (tp + fp)) if tp + fp else float("nan")
    per_family = None
    if family is not None:
        f = pd.Series(family)
        per_family = (pd.DataFrame({"family": f, "y": y, "pred": pred, "w": ww})
                      .assign(hit=lambda d: d.w * d.pred)
                      .groupby("family").agg(rows=("y", "size"), weight=("w", "sum"), flagged=("hit", "sum"))
                      .assign(flag_rate=lambda d: d.flagged / d.weight).reset_index())
    return out, per_family


def single_feature_auroc(X: pd.DataFrame, y: np.ndarray, n_max: int = 400_000, seed: int = 20260930):
    """How well each feature separates the classes on its own (max(AUROC, 1 - AUROC)); near 1 means shortcut risk."""
    from sklearn.metrics import roc_auc_score
    idx = np.random.default_rng(seed).permutation(len(y))[:n_max]
    Xs, ys = X.iloc[idx], y[idx]
    rows = []
    for c in X.columns:
        v = Xs[c].to_numpy(dtype=np.float64)
        auc = roc_auc_score(ys, v) if np.unique(v).size > 1 else 0.5
        rows.append({"feature": c, "auroc_alone": round(max(auc, 1 - auc), 4), "distinct_values": int(np.unique(v).size)})
    return pd.DataFrame(rows).sort_values("auroc_alone", ascending=False).reset_index(drop=True)


def importance(booster, features):
    gain = booster.get_score(importance_type="total_gain")
    s = pd.Series({f: gain.get(f, 0.0) for f in features})
    return (s / s.sum()).sort_values(ascending=False).rename("gain_share").reset_index().rename(columns={"index": "feature"})


def save(booster, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_stats(booster) -> dict:
    df = booster.trees_to_dataframe()
    leaves = df[df["Feature"] == "Leaf"]
    return {"trees": int(df["Tree"].nunique()), "leaves": int(len(leaves)),
            "splits": int((df["Feature"] != "Leaf").sum()),
            "distinct_thresholds": int(df.loc[df["Feature"] != "Leaf", ["Feature", "Split"]].drop_duplicates().shape[0])}


def write_config(repo: Path) -> str:
    out = repo / "configs"
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(DETECTOR_CONFIG, indent=2, sort_keys=True)
    (out / "detector_v1.json").write_text(text + "\n")
    return hashlib.sha256(text.encode()).hexdigest()
