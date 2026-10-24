"""Phase 3: class-conditional (Mondrian) split-conformal wrapper around the risk model.

LAC (least-ambiguous-set) nonconformity `1 - p(true class)` is calibrated
*separately per class* on the held-out `calib` split (the later half of the val
days, see risk_common.carve_calib), which gives a per-class coverage guarantee:

    P(scam in set | scam) >= 1 - alpha_scam     (default 0.90)
    P(fine in set | fine) >= 1 - alpha_fine     (default 0.99)

A single marginal qhat is useless at ~0.09% positives (it collapses to ~0 and
almost never emits {scam}); per-class qhats fix that. The prediction set is one of
    {fine}, {scam}, or {fine, scam} ("unsure")   -- never empty
(a txn that neither class admits is mapped to "unsure", not to an empty set).

Saves models/conformal.json with qhat_fine / qhat_scam. Run:
    python src/conformal.py
    python src/conformal.py --txn-id 12345
    python src/conformal.py --smoke   (smoke model + data/smoke/features.csv)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import xgboost as xgb

from risk_common import (
    FEATURE_COLUMNS, SMOKE_CONFORMAL, SMOKE_FEATURES, SMOKE_MODEL, TARGET, load_features,
)

SET_FINE, SET_SCAM, SET_UNSURE = 0, 1, 2
SET_NAMES = {SET_FINE: "{fine}", SET_SCAM: "{scam}", SET_UNSURE: "{fine, scam}"}


def fit_qhat(scores: np.ndarray, alpha: float) -> float:
    s = np.sort(scores)
    n = s.size
    if n == 0:
        return 1.0  # no calibration data for this class: admit it everywhere
    k = min(n, int(math.ceil((n + 1) * (1.0 - alpha))))
    return float(s[k - 1])


def fit_mondrian(p_scam: np.ndarray, y: np.ndarray, alpha_fine: float, alpha_scam: float):
    """Per-class LAC qhats: (qhat_fine, qhat_scam)."""
    qhat_fine = fit_qhat(p_scam[y == 0], alpha_fine)         # 1 - p(fine) = p_scam
    qhat_scam = fit_qhat(1.0 - p_scam[y == 1], alpha_scam)   # 1 - p(scam)
    return qhat_fine, qhat_scam


def predict_sets(p_scam: np.ndarray, qhat_fine: float, qhat_scam: float) -> np.ndarray:
    inc_scam = p_scam >= (1.0 - qhat_scam)
    inc_fine = p_scam <= qhat_fine
    out = np.full(p_scam.shape, SET_UNSURE, dtype=np.int8)
    out[inc_fine & ~inc_scam] = SET_FINE
    out[inc_scam & ~inc_fine] = SET_SCAM
    return out


def classify(p_scam: float, qhat_fine: float, qhat_scam: float) -> str:
    code = int(predict_sets(np.array([p_scam]), qhat_fine, qhat_scam)[0])
    return SET_NAMES[code]


def load_conformal(path) -> tuple[float, float]:
    """(qhat_fine, qhat_scam) from conformal.json; an old single-qhat file maps to both."""
    payload = json.loads(Path(path).read_text())
    if "qhat_fine" in payload:
        return float(payload["qhat_fine"]), float(payload["qhat_scam"])
    q = float(payload["qhat"])
    return q, q


def evaluate(sets: np.ndarray, y: np.ndarray) -> dict:
    scam_eff = (sets != SET_FINE)
    fine_eff = (sets != SET_SCAM)
    covered = np.where(y == 1, scam_eff, fine_eff)
    dist = {SET_NAMES[k]: int((sets == k).sum()) for k in (SET_FINE, SET_SCAM, SET_UNSURE)}
    pos = y == 1
    neg = ~pos
    return {
        "coverage": float(covered.mean()),
        "coverage_scam": float(scam_eff[pos].mean()) if pos.any() else 0.0,
        "coverage_fine": float(fine_eff[neg].mean()) if neg.any() else 0.0,
        "set_distribution": dist,
        "unsure_share": float((sets == SET_UNSURE).mean()),
        "scam_singleton_recall": float((sets[pos] == SET_SCAM).mean()) if pos.any() else 0.0,
        "scam_detection_rate": float(scam_eff[pos].mean()) if pos.any() else 0.0,
        "avg_set_size": float(np.where(sets == SET_UNSURE, 2, 1).mean()),
    }


def run(args) -> None:
    model = xgb.XGBClassifier()
    model.load_model(args.model)
    df = load_features(args.features)

    ca = df[df["split"] == "calib"]
    te = df[df["split"] == "test"]
    p_ca = model.predict_proba(ca[FEATURE_COLUMNS])[:, 1]
    p_te = model.predict_proba(te[FEATURE_COLUMNS])[:, 1]
    y_ca = ca[TARGET].to_numpy(np.int8)
    y_te = te[TARGET].to_numpy(np.int8)

    qhat_fine, qhat_scam = fit_mondrian(p_ca, y_ca, args.alpha_fine, args.alpha_scam)

    sets_te = predict_sets(p_te, qhat_fine, qhat_scam)
    rep = evaluate(sets_te, y_te)

    payload = {
        "method": "mondrian-lac",
        "score": "lac: 1 - p(true class), calibrated per class",
        "alpha_fine": args.alpha_fine,
        "alpha_scam": args.alpha_scam,
        "qhat_fine": qhat_fine,
        "qhat_scam": qhat_scam,
        "scam_threshold": 1.0 - qhat_scam,
        "calibration_split": "calib",
        "calibration_rows": int(len(ca)),
        "calibration_pos": int(y_ca.sum()),
        "test": rep,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    print(f"[conformal] mondrian alpha_fine={args.alpha_fine} alpha_scam={args.alpha_scam} "
          f"(calibrated on {len(ca):,} calib rows, {int(y_ca.sum())} scams)")
    print(f"[conformal] qhat_fine={qhat_fine:.6f} (fine if p<={qhat_fine:.6f})  "
          f"qhat_scam={qhat_scam:.6f} (scam if p>={1 - qhat_scam:.6f})")
    print(f"[conformal] test coverage scam={rep['coverage_scam']:.4f} (target {1 - args.alpha_scam:.2f}) "
          f"fine={rep['coverage_fine']:.4f} (target {1 - args.alpha_fine:.2f})")
    print(f"[conformal] test unsure_share={rep['unsure_share']:.4f} "
          f"scam_singleton_recall={rep['scam_singleton_recall']:.4f} avg_set_size={rep['avg_set_size']:.4f}")
    print(f"[conformal] set distribution: {rep['set_distribution']}")

    if args.txn_id is not None:
        row = df[df["txn_id"] == args.txn_id]
        if row.empty:
            print(f"[conformal] txn_id {args.txn_id} not found")
        else:
            p = float(model.predict_proba(row[FEATURE_COLUMNS])[:, 1][0])
            print(f"[conformal] txn_id={args.txn_id} p_scam={p:.4f} -> {classify(p, qhat_fine, qhat_scam)}")
    print(f"[conformal] wrote {out.resolve()}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Class-conditional conformal prediction sets for the risk model.")
    ap.add_argument("--model", default="models/xgb_model.json")
    ap.add_argument("--features", default="data/features.csv")
    ap.add_argument("--out", default="models/conformal.json")
    ap.add_argument("--alpha-fine", type=float, default=0.01,
                    help="max miss rate for legit txns (1 - fine coverage)")
    ap.add_argument("--alpha-scam", type=float, default=0.10,
                    help="max miss rate for scams (1 - scam coverage)")
    ap.add_argument("--txn-id", type=int, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="calibrate the train_xgb --smoke model on data/smoke/features.csv")
    args = ap.parse_args(argv)
    if args.smoke:
        args.model = SMOKE_MODEL
        args.features = SMOKE_FEATURES
        args.out = SMOKE_CONFORMAL
    run(args)


if __name__ == "__main__":
    main()
