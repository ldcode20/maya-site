"""Phase 3: plain-English SHAP reasons for a scored transaction.

Uses shap.TreeExplainer on the XGBoost booster and returns the top-3 features by
absolute SHAP value, mapped to English via risk_common.ENGLISH_REASONS.

Run:
    python3 src/reasons.py --txn-id 12345
    python3 src/reasons.py --demo 3
"""

from __future__ import annotations

import argparse

import numpy as np
import xgboost as xgb

from risk_common import FEATURE_COLUMNS, TARGET, english_reason, load_features


def make_explainer(model):
    import shap
    return shap.TreeExplainer(model.get_booster())


def shap_matrix(explainer, X) -> np.ndarray:
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[..., 1]
    return sv


def top_reasons(shap_row: np.ndarray, k: int = 3) -> list:
    order = np.argsort(-np.abs(shap_row))[:k]
    return [(FEATURE_COLUMNS[i], float(shap_row[i]), english_reason(FEATURE_COLUMNS[i]))
            for i in order]


def native_contrib(model, X) -> np.ndarray:
    """XGBoost's own SHAP values, over the same trees the model scores with
    (early stopping keeps best_iteration + 1 of them; TreeExplainer already does)."""
    booster = model.get_booster()
    dm = xgb.DMatrix(X, feature_names=FEATURE_COLUMNS)
    stop = int(getattr(model, "best_iteration", booster.num_boosted_rounds() - 1)) + 1
    contribs = booster.predict(dm, pred_contribs=True, iteration_range=(0, stop))
    return contribs[:, :-1]


def shap_rows(model, explainer, X) -> list:
    """All (feature, shap, text) rows for the first row of X, by |SHAP| (empty on failure)."""
    try:
        try:
            sv = shap_matrix(explainer, X)[0]
        except Exception:
            sv = native_contrib(model, X)[0]
        return top_reasons(sv, k=len(FEATURE_COLUMNS))
    except Exception:
        return []


def in_risky_range(base: str, value: float, thr=None) -> bool:
    """True when the feature value itself is anomalous (same bars as the policy's
    hard signals), so a reason's English text is actually true for this txn."""
    if thr is None:
        from policy import DEFAULT_THRESHOLDS as thr  # lazy: policy imports this module
    checks = {
        "n_t": value >= thr.n_t,
        "e_t": value >= thr.e_t,
        "z_t": value >= thr.z_t,
        "pi_t": value >= thr.pi_t,
        "d_R": value <= thr.d_r_max,
        "zc_t": value >= thr.zc_t,
        # Not policy hard signals; bars chosen so the English text is true:
        # "several payments" = 2+ earlier ones, "forwards most" = >= 80%.
        "v_t": value >= 2.0,
        "ptr_t": value >= 0.8,
        # g_t is ~0.997 on this synthetic graph (see AGENTS.md), so it never
        # distinguishes a txn and is not cited.
    }
    return bool(checks.get(base, False))


def risk_reasons(rows: list, values, k: int = 3, with_features: bool = False) -> list:
    """Up to k English reasons, ranked by SHAP: the feature must push risk *up*
    and its value (`values[feature]`, a full feature row) must be in the risky
    range. One per base feature, so `z_t` and `z_t_snd` don't both cite it.
    With `with_features`, returns (base_feature, text) pairs instead of texts."""
    out, seen = [], set()
    for feature, value, text in rows:
        base = feature.replace("_snd", "").replace("_rcv", "")
        if value > 0 and base not in seen and in_risky_range(base, float(values[feature])):
            seen.add(base)
            out.append((base, text) if with_features else text)
    return out[:k]


def explain_row(model, explainer, X, index: int) -> list:
    row = X.iloc[[index]]
    try:
        sv = shap_matrix(explainer, row)[0]
    except Exception:
        sv = native_contrib(model, row)[0]
    return top_reasons(sv)


def run(args) -> None:
    model = xgb.XGBClassifier()
    model.load_model(args.model)
    df = load_features(args.features)

    if args.txn_id is not None:
        mask = df.index[df["txn_id"] == args.txn_id]
        if len(mask) == 0:
            print(f"[reasons] txn_id {args.txn_id} not found")
            return
        sel = df.loc[mask]
    elif args.index is not None:
        sel = df.iloc[[args.index]]
    else:
        p = model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
        top = np.argsort(-p)[: args.demo]
        sel = df.iloc[top]

    X = sel[FEATURE_COLUMNS]
    p = model.predict_proba(X)[:, 1]
    try:
        sv = shap_matrix(make_explainer(model), X)
    except Exception:
        sv = native_contrib(model, X)

    for j, (_, r) in enumerate(sel.iterrows()):
        print(f"txn_id={int(r['txn_id'])} p_scam={p[j]:.4f} is_scam={int(r[TARGET])} "
              f"split={r['split']}")
        for feat, val, text in top_reasons(sv[j], args.top_k):
            direction = "increases risk" if val > 0 else "decreases risk"
            print(f"   - {feat} ({val:+.4f}, {direction}): {text}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Explain a scored transaction with SHAP.")
    ap.add_argument("--model", default="models/xgb_model.json")
    ap.add_argument("--features", default="data/features.csv")
    ap.add_argument("--txn-id", type=int, default=None)
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--demo", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
