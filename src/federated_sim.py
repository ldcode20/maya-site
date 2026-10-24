"""Phase 8: federated learning (FedAvg-style) simulation across two banks.

Splits customers (sender_id) into two disjoint banks, trains a local XGBoost
model on each bank's data only, then averages the two models' prior-corrected
margins (log-odds) to build a global model. Evaluates the two locals vs the
global on the shared test split -- without either bank sharing raw customer
data -- and against a centralized model trained on the pooled data (what the
banks would get by sharing data), which is the cost-of-federation baseline.

Averaging note: tree ensembles have no aligned weights to average, so this is
output-level model averaging (an ensemble of the two locals), the standard
FedAvg stand-in for non-parametric models. Each local is trained with its own
scale_pos_weight, which shifts its log-odds by ~log(spw); that shift is
subtracted before averaging so neither bank's margins dominate.

Privacy: only model outputs and scalar thresholds leave a bank. The global
model's 1%-FPR threshold is the mean of the threshold each bank picks on its
*own* val rows -- val data is never pooled.

Run:  python3 src/federated_sim.py
      python3 src/federated_sim.py --smoke
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from risk_common import FEATURE_COLUMNS, TARGET, load_features, recall_at_fpr, xgb_params


def train_local(Xtr, ytr, Xva, yva, seed):
    pos = int(ytr.sum())
    neg = int(len(ytr) - pos)
    spw = neg / max(pos, 1)
    model = xgb.XGBClassifier(**xgb_params(spw, seed))
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return model, spw


def fedavg(model_a, spw_a, model_b, spw_b, X):
    """Global score = sigmoid(mean of the two prior-corrected margins)."""
    m = (
        model_a.predict(X, output_margin=True) - np.log(spw_a)
        + model_b.predict(X, output_margin=True) - np.log(spw_b)
    ) / 2.0
    return 1.0 / (1.0 + np.exp(-m))


def val_threshold(y_val, p_val):
    """Threshold at 1% FPR on one val set."""
    thr, _, _ = recall_at_fpr(y_val, p_val, 0.01, None)
    return thr


def evaluate(thr, y_test, p_test):
    """AUROC + recall/FPR on test at a threshold already chosen on val."""
    _, recall, fpr = recall_at_fpr(y_test, p_test, 0.01, thr)
    return roc_auc_score(y_test, p_test), recall, thr, fpr


def run(args) -> int:
    t0 = time.time()
    df = load_features(args.features, columns=FEATURE_COLUMNS + ["sender_id"])
    if args.smoke:
        parts = []
        for s, n in (("train", 100_000), ("val", 20_000), ("test", 20_000)):
            sub = df[df["split"] == s]
            parts.append(sub.sample(n=min(n, len(sub)), random_state=1))
        df = pd.concat(parts, ignore_index=True)

    # Horizontal split: Bank A holds the first half of customers, Bank B the rest.
    senders = np.sort(df["sender_id"].unique())
    mid = len(senders) // 2
    s_arr = df["sender_id"].to_numpy()
    mask_a = np.isin(s_arr, senders[:mid])
    df_a = df[mask_a]
    df_b = df[~mask_a]

    def split_bank(d):
        tr = d[d["split"] == "train"]
        va = d[d["split"] == "val"]
        te = d[d["split"] == "test"]
        return (
            tr[FEATURE_COLUMNS], tr[TARGET].to_numpy(np.int8),
            va[FEATURE_COLUMNS], va[TARGET].to_numpy(np.int8),
            te[FEATURE_COLUMNS], te[TARGET].to_numpy(np.int8),
        )

    Xa_tr, ya_tr, Xa_va, ya_va, Xa_te, ya_te = split_bank(df_a)
    Xb_tr, yb_tr, Xb_va, yb_va, Xb_te, yb_te = split_bank(df_b)

    model_a, spw_a = train_local(Xa_tr, ya_tr, Xa_va, ya_va, args.seed)
    model_b, spw_b = train_local(Xb_tr, yb_tr, Xb_va, yb_va, args.seed)

    # Shared test set = union of both banks' test rows (evaluation only).
    Xte = pd.concat([Xa_te, Xb_te], ignore_index=True)
    yte = np.concatenate([ya_te, yb_te])

    thr_a = val_threshold(ya_va, model_a.predict_proba(Xa_va)[:, 1])
    thr_b = val_threshold(yb_va, model_b.predict_proba(Xb_va)[:, 1])
    # Global threshold: each bank scores the global model on its own val only;
    # just the two scalar thresholds are shared and averaged.
    thr_g = float(np.mean([
        val_threshold(ya_va, fedavg(model_a, spw_a, model_b, spw_b, Xa_va)),
        val_threshold(yb_va, fedavg(model_a, spw_a, model_b, spw_b, Xb_va)),
    ]))

    auc_a, rec_a, _, fpr_a = evaluate(thr_a, yte, model_a.predict_proba(Xte)[:, 1])
    auc_b, rec_b, _, fpr_b = evaluate(thr_b, yte, model_b.predict_proba(Xte)[:, 1])
    auc_g, rec_g, _, fpr_g = evaluate(thr_g, yte, fedavg(model_a, spw_a, model_b, spw_b, Xte))

    # Centralized baseline: one model on the pooled data (requires data sharing).
    Xpool_va = pd.concat([Xa_va, Xb_va], ignore_index=True)
    ypool_va = np.concatenate([ya_va, yb_va])
    model_c, _ = train_local(
        pd.concat([Xa_tr, Xb_tr], ignore_index=True), np.concatenate([ya_tr, yb_tr]),
        Xpool_va, ypool_va, args.seed,
    )
    thr_c = val_threshold(ypool_va, model_c.predict_proba(Xpool_va)[:, 1])
    auc_c, rec_c, _, fpr_c = evaluate(thr_c, yte, model_c.predict_proba(Xte)[:, 1])

    print(f"[federated] customers: Bank A={mid} Bank B={len(senders) - mid} "
          f"(total {len(senders)})")
    print(f"[federated] txns: A={len(df_a):,} B={len(df_b):,} test={len(yte):,} "
          f"pos={int(yte.sum())} spw A={spw_a:.1f} B={spw_b:.1f}")

    hdr = f"{'model':<18}{'AUROC':>9}{'recall@1%FPR':>15}{'thr':>10}{'fpr':>9}"
    print(hdr)
    print("-" * len(hdr))
    print(f"{'Bank A (local)':<18}{auc_a:>9.4f}{rec_a:>15.4f}{thr_a:>10.2e}{fpr_a:>9.4f}")
    print(f"{'Bank B (local)':<18}{auc_b:>9.4f}{rec_b:>15.4f}{thr_b:>10.2e}{fpr_b:>9.4f}")
    print(f"{'Global (FedAvg)':<18}{auc_g:>9.4f}{rec_g:>15.4f}{thr_g:>10.2e}{fpr_g:>9.4f}")
    print(f"{'Pooled (no fed.)':<18}{auc_c:>9.4f}{rec_c:>15.4f}{thr_c:>10.2e}{fpr_c:>9.4f}")

    res_auc = outcome(auc_g, max(auc_a, auc_b))
    res_rec = outcome(rec_g, max(rec_a, rec_b))
    print(f"[federated] Global vs best local -> AUROC {res_auc} "
          f"({auc_g:.4f} vs {max(auc_a, auc_b):.4f}), "
          f"recall {res_rec} ({rec_g:.4f} vs {max(rec_a, rec_b):.4f})")
    if res_rec == "WIN":
        verdict = "the FedAvg global model catches more scams than either bank alone"
    elif res_rec == "TIE":
        verdict = "the FedAvg global model catches as many scams as the best bank alone"
    else:
        verdict = "the FedAvg global model catches fewer scams than the best bank alone"
    print(f"[federated] conclusion: {verdict}, without any bank sharing raw customer "
          "data (only model outputs and thresholds leave each bank).")
    print(f"[federated] cost of federation vs pooled data: AUROC {auc_g - auc_c:+.4f}, "
          f"recall {rec_g - rec_c:+.4f} (FedAvg {outcome(rec_g, rec_c)} vs pooled)")
    if outcome(rec_g, rec_c) == "WIN":
        print("[federated] note: the global model beats even the pooled-data model, so its gain "
              "over the locals comes from averaging two models (ensembling), not from federation.")
    print(f"[federated] done in {time.time() - t0:.1f}s")
    return 0


def outcome(global_score: float, best_local: float, tol: float = 1e-4) -> str:
    """WIN / TIE / LOSE of the global model vs the best local, at printed precision."""
    if global_score > best_local + tol:
        return "WIN"
    if global_score < best_local - tol:
        return "LOSE"
    return "TIE"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Federated learning (FedAvg) simulation across two banks.")
    ap.add_argument("--features", default="data/features.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
