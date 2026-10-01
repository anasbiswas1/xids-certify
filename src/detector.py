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
    "excluded_host_identity": {
        "features": ["MIN_TTL", "MAX_TTL"],
        "reason": "TTL is set by the sending host and reduced by one per router hop; in a testbed it identifies the "
                  "sending machine, and an attacker sets it with one socket option. Notebook 02 (first run) found "
                  "MIN_TTL and MAX_TTL each separate NF-UNSW-NB15-v3 classes with AUROC 0.9994 on their own and carry "
                  "99.4% of that detector's gain. Excluded on both datasets, like IP addresses and ports.",
    },
}
TARGET_FPR = 0.01
HOST_IDENTITY = ["MIN_TTL", "MAX_TTL"]


def detector_features(model_features):
    """Detector inputs: the data-stage model features minus fields that identify the sending host."""
    return [f for f in model_features if f not in HOST_IDENTITY]


VARIANTS_CONFIG = {
    "version": "detector_v2",
    "fixed_on": "2026-09-30, after notebook 03 failed its pass rules and before any v2 result",
    "variants": {
        "v2a_no_timing": "detector_v1 inputs minus the eight inter-arrival and two per-second byte fields, which have "
                         "no audited formula (ablation)",
        "v2b_certifiable": "v2a, with the five size bins replaced by five cumulative counts (packets at least as large "
                           "as each bin edge) and monotone training: the score may only rise with every field the "
                           "declared operations can only raise, and only fall with receiver-side throughput "
                           "(primary; the pass rules apply to it)",
        "v2c_shaping_monotone": "v2b without the three fields the operations can move either way (smallest packet, "
                                "minimum IP length, source throughput): every field the operations can change is "
                                "monotone in the attacker's direction, so undoing an in-budget operation can never "
                                "raise the score and every benign verdict is certified by construction (the verifier "
                                "confirms each one; the price is detection accuracy)",
    },
    "training": "same parameters, roles, early stopping and threshold rule as detector_v1",
}


def variant_features(model_features, variant):
    from . import operations as ops
    base = [f for f in detector_features(model_features) if f not in ops.UNMODELLED]
    if variant == "v1":
        return detector_features(model_features)
    if variant == "v2a_no_timing":
        return base
    if variant == "v2b_certifiable":
        return [f for f in base if f not in ops.HIST] + ops.CUM
    if variant == "v2c_shaping_monotone":
        return [f for f in base if f not in ops.HIST and f not in ops.BOTH_WAYS] + ops.CUM
    raise ValueError(variant)


def variant_frame(df, model_features, variant):
    """The detector input table for one variant (adds the cumulative counts when the variant uses them)."""
    from . import operations as ops
    feats = variant_features(model_features, variant)
    out = df.copy()
    if any(c in feats for c in ops.CUM):
        ops.add_cumulative(out)
    return out[feats].astype(np.float32)


def context_features(model_features, variant):
    """Fields the verifier is given for each flow: the detector's inputs plus any field the operation model needs."""
    from . import certify as cf
    feats = variant_features(model_features, variant)
    return feats + [f for f in cf.REQUIRED if f not in feats]


def context_frame(df, model_features, variant):
    from . import operations as ops
    cols = context_features(model_features, variant)
    out = df.copy()
    if any(c in cols for c in ops.CUM):
        ops.add_cumulative(out)
    return out[cols].astype(np.float32)


def monotone(features, variant):
    """XGBoost monotone constraints: +1 for fields the operations can only raise, -1 for the one they can only
    lower, 0 otherwise. Only v2b and v2c are constrained."""
    from . import operations as ops
    if variant not in ("v2b_certifiable", "v2c_shaping_monotone"):
        return None
    return tuple(1 if f in ops.INCREASE_ONLY else -1 if f in ops.DECREASE_ONLY else 0 for f in features)


def monotone_violations(ens, X32, features, constraints, n_rows=300, seed=0):
    """Share of (row, feature) probes where the trained ensemble breaks its monotone constraint."""
    rng = np.random.default_rng(seed)
    rows = X32[rng.permutation(len(X32))[:n_rows]]
    bad = total = 0
    for j, (f, c) in enumerate(zip(features, constraints)):
        if c == 0:
            continue
        grid = np.unique(np.quantile(X32[:, j], np.linspace(0, 1, 12)).astype(np.float32))
        for r in rows:
            Z = np.repeat(r[None, :], len(grid), axis=0)
            Z[:, j] = grid
            m = ens.margin(Z)
            d = np.diff(m) * c
            bad += int((d < -1e-6).sum())
            total += len(d)
    return bad / max(total, 1)


def fit(X_tr, y_tr, X_dev, y_dev, params=None, monotone_constraints=None):
    """Train with early stopping on development and return the booster trimmed to its best round, so the saved,
    verified and deployed model is the same object that makes every decision."""
    import xgboost as xgb
    p = dict(DETECTOR_CONFIG["params"] if params is None else params)
    if monotone_constraints is not None:
        p["monotone_constraints"] = "(" + ",".join(str(int(c)) for c in monotone_constraints) + ")"
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
