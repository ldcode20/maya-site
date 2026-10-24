"""Shared configuration for the maya Phase 3 risk stack.

Single source of truth for the model feature columns, the XGBoost
hyperparameters, the recall@FPR metric, and the plain-English reason strings
used by src/reasons.py. Imported by train_xgb, conformal, reasons, federated_sim.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "n_t", "e_t", "g_t", "z_t", "pi_t", "d_R",
    "n_t_snd", "n_t_rcv",
    "e_t_snd", "e_t_rcv",
    "g_t_snd", "g_t_rcv",
    "z_t_snd", "z_t_rcv",
    "pi_t_snd", "pi_t_rcv",
    "d_R_snd", "d_R_rcv",
    # anti-evasion features (amount splitting / mule pass-through)
    "v_t", "zc_t", "ptr_t",
    "v_t_snd", "v_t_rcv",
    "zc_t_snd", "zc_t_rcv",
    "ptr_t_snd", "ptr_t_rcv",
]

TARGET = "is_scam"

# --smoke artifacts (data from journey_generator/features --smoke, model from
# train_xgb --smoke); kept separate so they never overwrite the full-run files.
SMOKE_FEATURES = "data/smoke/features.csv"
SMOKE_MODEL = "models/xgb_model_smoke.json"
SMOKE_CONFORMAL = "models/conformal_smoke.json"
SPLITS = ("train", "val", "calib", "test")

# Per-payment latency budget from the spec (API reports, GNN verdict).
LATENCY_SLO_MS = 150.0


def carve_calib(df: pd.DataFrame) -> pd.DataFrame:
    """Relabel the later half of the val days as `calib`, in place.

    `val` (earlier days) is used for early stopping and the 1%-FPR threshold;
    `calib` (later days) is held out for conformal calibration only, so the
    conformal scores are not computed on data the model was tuned on. Splitting
    by contiguous days keeps the generator's day-block leakage control.
    """
    is_val = df["split"] == "val"
    days = np.sort(df.loc[is_val, "day"].unique())
    if days.size >= 2:
        cut = days[days.size // 2]
        df.loc[is_val & (df["day"] >= cut), "split"] = "calib"
    return df

# XGBoost hyperparameters shared by train_xgb (CLI defaults) and federated_sim
# (per-bank locals), so both always train the same model family.
XGB_DEFAULTS = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}
EARLY_STOPPING_ROUNDS = 40


def xgb_params(scale_pos_weight: float, seed: int, **overrides) -> dict:
    """Full XGBClassifier kwargs: shared defaults, then any overrides."""
    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        **XGB_DEFAULTS,
        "scale_pos_weight": scale_pos_weight,
        "random_state": seed,
        "n_jobs": -1,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
    }
    params.update(overrides)
    return params


def recall_at_fpr(y_true, score, target_fpr, threshold=None):
    """Recall and FPR at `threshold`; if None, pick the best threshold with FPR <= target."""
    if threshold is None:
        from sklearn.metrics import roc_curve

        fpr, tpr, thr = roc_curve(y_true, score)
        mask = fpr <= target_fpr
        i = int(np.argmax(np.where(mask, tpr, -1)))
        threshold = thr[i]
    pred = score >= threshold
    pos = y_true == 1
    recall = float(pred[pos].mean()) if pos.any() else 0.0
    neg = ~pos
    fpr = float(pred[neg].mean()) if neg.any() else 0.0
    return threshold, recall, fpr

_BASE_REASONS = {
    "n_t": "Many new payees in the last 7 days",
    "e_t": "Amount is unusually high for a new payee",
    "g_t": "Received a credit from an unknown sender recently",
    "z_t": "Amount is unusually large vs the account's history",
    "pi_t": "Repeated tiny probe payments to the same payee within an hour",
    "d_R": "Within two payment hops of a reported fraudulent account",
    "v_t": "Several payments to this payee in the last 24 hours",
    "zc_t": "Total sent to this payee in 24 hours is unusually large",
    "ptr_t": "Account forwards most of the money it receives within a day",
}

ENGLISH_REASONS = {}
for _base, _text in _BASE_REASONS.items():
    ENGLISH_REASONS[_base] = _text
    ENGLISH_REASONS[f"{_base}_snd"] = "Payer side: " + _text
    ENGLISH_REASONS[f"{_base}_rcv"] = "Payee side: " + _text


def english_reason(feature: str) -> str:
    return ENGLISH_REASONS.get(feature, feature)


def fresh_parquet(path) -> Path | None:
    """The .parquet twin of a features CSV, if it exists and is not older than the CSV."""
    p = Path(path)
    pq = p.with_suffix(".parquet")
    if p.suffix == ".csv" and pq.exists() and (not p.exists() or pq.stat().st_mtime >= p.stat().st_mtime):
        return pq
    return None


def load_features(path: str, columns=None) -> pd.DataFrame:
    """Load a features file (+ calib split). Reads the fresh .parquet twin when
    available (several times faster than the ~1 GB CSV), else the CSV."""
    cols = list(dict.fromkeys(["txn_id", "day", "split", TARGET] + list(columns or FEATURE_COLUMNS)))
    pq = fresh_parquet(path)
    df = pd.read_parquet(pq, columns=cols) if pq else pd.read_csv(path, usecols=cols)
    for c in FEATURE_COLUMNS:
        if c in df.columns:
            df[c] = df[c].astype("float32")
    return carve_calib(df)
