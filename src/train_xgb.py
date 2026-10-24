"""Phase 3: train the XGBoost risk model on past-only features.

Trains on the train split, early-stops on val, evaluates on test, and reports
recall at a 1% false-positive rate (threshold chosen on val, reported on test).
Saves models/xgb_model.json and models/model_meta.json.

Adversarial training (default; --no-augment to disable): evasive copies of the
*train* scams (evasion.TRAIN_ATTACKS: split_2, split_4, extra_hop) are added as
extra positives, each attack family weighted 1 per original scam, and
scale_pos_weight is computed on the total positive weight. val / calib / test
are never augmented, so all reported metrics are on real transactions.

Run:  python3 src/train_xgb.py
      python3 src/train_xgb.py --smoke
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

from evasion import HELPER_COLUMNS, TRAIN_ATTACKS, augment
from risk_common import (
    FEATURE_COLUMNS, SMOKE_FEATURES, SMOKE_MODEL, TARGET, XGB_DEFAULTS, load_features,
    recall_at_fpr, xgb_params,
)


def run(args) -> None:
    t0 = time.time()
    df = load_features(args.features, columns=FEATURE_COLUMNS + (HELPER_COLUMNS if args.augment else []))
    tr = df[df["split"] == "train"]
    va = df[df["split"] == "val"]
    te = df[df["split"] == "test"]
    if args.smoke:
        tr = tr.sample(n=min(100_000, len(tr)), random_state=1)
        va = va.sample(n=min(20_000, len(va)), random_state=1)
        te = te.sample(n=min(20_000, len(te)), random_state=1)

    Xtr, ytr = tr[FEATURE_COLUMNS], tr[TARGET].to_numpy(np.int8)
    Xva, yva = va[FEATURE_COLUMNS], va[TARGET].to_numpy(np.int8)
    Xte, yte = te[FEATURE_COLUMNS], te[TARGET].to_numpy(np.int8)

    pos = int(ytr.sum())
    neg = int(len(ytr) - pos)
    weight = np.ones(len(ytr))
    n_aug = 0
    if args.augment:
        aug, w_aug = augment(tr[tr[TARGET] == 1][FEATURE_COLUMNS + HELPER_COLUMNS])
        n_aug = len(aug)
        Xtr = pd.concat([Xtr, aug[FEATURE_COLUMNS].astype("float32")], ignore_index=True)
        ytr = np.concatenate([ytr, np.ones(n_aug, np.int8)])
        weight = np.concatenate([weight, w_aug])
    spw = neg / max(float(weight[ytr == 1].sum()), 1.0)

    model = xgb.XGBClassifier(**xgb_params(
        spw, args.seed,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.lr,
        subsample=args.subsample,
        colsample_bytree=args.colsample,
    ))
    model.fit(Xtr, ytr, sample_weight=weight, eval_set=[(Xva, yva)], verbose=False)

    p_va = model.predict_proba(Xva)[:, 1]
    p_te = model.predict_proba(Xte)[:, 1]

    thr, rec, fpr = recall_at_fpr(yva, p_va, 0.01, None)
    _, rec_te, fpr_te = recall_at_fpr(yte, p_te, 0.01, thr)
    _, rec_te_only, fpr_te_only = recall_at_fpr(yte, p_te, 0.01, None)

    metrics = {
        "scale_pos_weight": float(spw),
        "train_rows": int(len(tr)), "train_pos": pos,
        "adversarial_attacks": list(TRAIN_ATTACKS) if args.augment else [],
        "adversarial_rows": n_aug,
        "val_rows": int(len(va)), "val_pos": int(yva.sum()),
        "calib_rows_held_out": int((df["split"] == "calib").sum()),
        "val_auroc": float(roc_auc_score(yva, p_va)),
        "val_auprc": float(average_precision_score(yva, p_va)),
        "test_auroc": float(roc_auc_score(yte, p_te)),
        "test_auprc": float(average_precision_score(yte, p_te)),
        "fpr_target": 0.01,
        "val_threshold_at_1pct_fpr": float(thr),
        "test_recall_at_1pct_fpr": rec_te,
        "test_fpr_at_val_threshold": fpr_te,
        "test_recall_at_1pct_fpr_test_only": rec_te_only,
        "best_iteration": int(getattr(model, "best_iteration", model.n_estimators)),
        "params": {
            "n_estimators": args.n_estimators, "max_depth": args.max_depth,
            "learning_rate": args.lr, "subsample": args.subsample,
            "colsample_bytree": args.colsample, "seed": args.seed,
        },
        "features": FEATURE_COLUMNS,
    }

    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)
    meta_path = Path(args.meta_out)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metrics, indent=2))

    print(f"[train_xgb] train={len(tr):,} (pos={pos}, +{n_aug:,} adversarial) val={len(va):,} "
          f"test={len(te):,} spw={spw:.1f} (calib={metrics['calib_rows_held_out']:,} held out for conformal)")
    print(f"[train_xgb] val  AUROC={metrics['val_auroc']:.4f} AUPRC={metrics['val_auprc']:.4f}")
    print(f"[train_xgb] test AUROC={metrics['test_auroc']:.4f} AUPRC={metrics['test_auprc']:.4f}")
    print(f"[train_xgb] recall@1%FPR (val thr={thr:.4f}) on test = {rec_te:.4f} "
          f"(test fpr={fpr_te:.4f}; test-only={rec_te_only:.4f})")
    print(f"[train_xgb] best_iteration={metrics['best_iteration']} "
          f"wrote {model_path} + {meta_path} ({time.time() - t0:.1f}s)")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Train the XGBoost risk model.")
    ap.add_argument("--features", default=None,
                    help="default: data/features.csv (data/smoke/features.csv with --smoke)")
    ap.add_argument("--model-out", default="models/xgb_model.json")
    ap.add_argument("--meta-out", default="models/model_meta.json")
    ap.add_argument("--n-estimators", type=int, default=XGB_DEFAULTS["n_estimators"])
    ap.add_argument("--max-depth", type=int, default=XGB_DEFAULTS["max_depth"])
    ap.add_argument("--lr", type=float, default=XGB_DEFAULTS["learning_rate"])
    ap.add_argument("--subsample", type=float, default=XGB_DEFAULTS["subsample"])
    ap.add_argument("--colsample", type=float, default=XGB_DEFAULTS["colsample_bytree"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-augment", dest="augment", action="store_false",
                    help="skip adversarial training on evasive copies of train scams")
    args = ap.parse_args(argv)
    if args.features is None:
        args.features = SMOKE_FEATURES if args.smoke else "data/features.csv"
    if args.smoke:
        args.model_out = SMOKE_MODEL
        args.meta_out = "models/model_meta_smoke.json"
    run(args)


if __name__ == "__main__":
    main()
