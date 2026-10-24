"""Phase 4: the Maya Action Policy Engine (the Two-Signal Rule).

Turns a conformal prediction set plus raw features and mocked soft signals into
one of three actions -- release / nudge / hold -- under the hard rule that a hold
requires two independent signals to agree and at least one of them must be a
*hard* signal (Payment History or Receiving Account). Chat / voice / screen-share
are soft only and can never trigger a hold on their own.

Decision table (conformal set -> action):
    {fine}          -> release
    {scam}          -> same two-group safeguard as unsure
    {fine, scam}    -> hold if (payment-hard AND receiving-hard)
                              or (any hard AND any soft)
                       else nudge

Run:
    python3 src/policy.py --self-test
    python3 src/policy.py --eval         (release/nudge/hold rates on the test split)
    python3 src/policy.py --txn-id 2002885 --soft-signals '{"chat_stage": "escalated"}'
    python3 src/policy.py --demo 3 --soft-signals '{"is_screen_shared": true}'
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np
import xgboost as xgb

from conformal import (
    SET_FINE, SET_NAMES, SET_SCAM, SET_UNSURE, classify as conformal_classify, load_conformal,
    predict_sets,
)
from integrations import check_mulehunter
from reasons import make_explainer, native_contrib, risk_reasons, shap_matrix, top_reasons
from risk_common import FEATURE_COLUMNS, TARGET, load_features

# --- hard-rule constants ----------------------------------------------------

PAYMENT_HISTORY_FEATURES = ("n_t", "e_t", "z_t", "zc_t", "pi_t")
RECEIVING_ACCOUNT_FEATURES = ("d_R",)

HOLD_ANALYST_AMOUNT = 100_000.0
AUTO_RELEASE_MINUTES = 60

CHAT_SIGNAL_STAGES = {"escalated", "payment_demand"}


@dataclass(frozen=True)
class Thresholds:
    """Fixed defaults (val-percentile informed), CLI-overridable."""

    n_t: float = 59.0
    e_t: float = 1.0
    z_t: float = 2.6
    # ~val legit p99, the same percentile z_t's 2.6 sits at (repeat payments to
    # merchants make legit 24 h totals large, so zc_t needs its own bar)
    zc_t: float = 4.2
    pi_t: float = 1.0
    d_r_max: float = 1.0


DEFAULT_THRESHOLDS = Thresholds()

ACTION_RELEASE = "release"
ACTION_NUDGE = "nudge"
ACTION_HOLD = "hold"


def hard_signals(features, thr: Thresholds = DEFAULT_THRESHOLDS) -> dict:
    """One bool per hard trigger. `features` is a Series or a plain mapping.

    `zc_t` (24 h cumulative amount to this payee) keeps the signal alive when a
    payment is split into parts that each look normal; when it is absent (older
    feature rows) the z_t check stands in for it.
    """
    zc = features.get("zc_t")
    return {
        "n_t": float(features["n_t"]) >= thr.n_t,
        "e_t": float(features["e_t"]) >= thr.e_t,
        "z_t": float(features["z_t"]) >= thr.z_t,
        "zc_t": (float(zc) >= thr.zc_t) if zc is not None else (float(features["z_t"]) >= thr.z_t),
        "pi_t": float(features["pi_t"]) >= thr.pi_t,
        "d_R": float(features["d_R"]) <= thr.d_r_max,
    }


def soft_signals(soft=None) -> dict:
    """Normalize the mocked soft-signal inputs into booleans."""
    soft = soft or {}
    stage = str(soft.get("chat_stage", "none")).lower()
    return {
        "is_screen_shared": bool(soft.get("is_screen_shared", False)),
        "is_voice_active": bool(soft.get("is_voice_active", False)),
        "chat_stage": stage in CHAT_SIGNAL_STAGES,
    }


def decide(
    conf_set: str,
    features,
    soft=None,
    reasons=None,
    amount: float = 0.0,
    thr: Thresholds = DEFAULT_THRESHOLDS,
    mulehunter_status: str | None = None,
) -> dict:
    """Pure policy function: no I/O, no model. Returns the 4 contract fields.

    `mulehunter_status` is the mock MuleHunter.AI verdict for the receiving
    account ("clean"/"suspicious"/"mule"). A "mule" verdict while the conformal
    set is not fine is treated as a Receiving Account hard signal.
    """
    soft = soft_signals(soft)
    n_soft = sum(soft.values())

    hard = hard_signals(features, thr)
    payment = any(hard[f] for f in PAYMENT_HISTORY_FEATURES)
    receiving = any(hard[f] for f in RECEIVING_ACCOUNT_FEATURES)
    if conf_set not in SET_NAMES.values():
        raise ValueError(f"Unknown conformal set: {conf_set!r}")
    mule_signal = mulehunter_status == "mule" and conf_set != SET_NAMES[SET_FINE]
    receiving = receiving or mule_signal
    n_hard = int(payment) + int(receiving)

    if conf_set == SET_NAMES[SET_FINE]:
        action = ACTION_RELEASE
    else:  # A model prediction never bypasses the two-group safeguard.
        if (payment and receiving) or (n_hard >= 1 and n_soft >= 1):
            action = ACTION_HOLD
        else:
            action = ACTION_NUDGE

    is_hold = action == ACTION_HOLD
    return {
        "action": action,
        "reasons": list(reasons or [])[:3],
        "auto_release_minutes": AUTO_RELEASE_MINUTES if is_hold else 0,
        "requires_analyst": bool(is_hold and amount > HOLD_ANALYST_AMOUNT),
    }


# --- I/O plumbing -----------------------------------------------------------


def thresholds_from_args(args) -> Thresholds:
    return Thresholds(
        n_t=args.n_t_threshold,
        e_t=args.e_t_threshold,
        z_t=args.z_t_threshold,
        zc_t=args.zc_t_threshold,
        pi_t=args.pi_t_threshold,
        d_r_max=args.d_r_max,
    )


def active_signals(features, soft, thr: Thresholds, mulehunter_status: str | None = None,
                   conf_set: str | None = None) -> dict:
    """Diagnostic view of which signals fired (not part of the return contract)."""
    hard = hard_signals(features, thr)
    soft_norm = soft_signals(soft)
    if mulehunter_status == "mule" and conf_set != SET_NAMES[SET_FINE]:
        hard["mulehunter"] = True
    return {
        "hard": {k: v for k, v in hard.items() if v},
        "soft": {k: v for k, v in soft_norm.items() if v},
    }


def run(args) -> int:
    thr = thresholds_from_args(args)
    if args.self_test:
        return 0 if run_self_test() else 1

    model = xgb.XGBClassifier()
    model.load_model(args.model)
    df = load_features(args.features, columns=FEATURE_COLUMNS + ["amount", "receiver_id"])
    qhat_fine, qhat_scam = load_conformal(args.conformal)
    soft = json.loads(args.soft_signals) if args.soft_signals else {}

    if args.eval:
        return run_eval(model, df, qhat_fine, qhat_scam, thr, args.eval)

    if args.txn_id is not None:
        sel = df[df["txn_id"] == args.txn_id]
        if sel.empty:
            print(f"[policy] txn_id {args.txn_id} not found")
            return 1
        p = model.predict_proba(sel[FEATURE_COLUMNS])[:, 1]
    elif args.demo:
        p_all = model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
        top = np.argsort(-p_all)[: args.demo]
        sel = df.iloc[top]
        p = p_all[top]
    else:
        print("[policy] provide --txn-id N or --demo K (or --self-test)")
        return 1

    explainer = make_explainer(model)
    X = sel[FEATURE_COLUMNS]

    for j, (_, row) in enumerate(sel.iterrows()):
        conf_set = conformal_classify(float(p[j]), qhat_fine, qhat_scam)
        amount = float(args.amount) if args.amount is not None else float(row["amount"])
        mulehunter = check_mulehunter(row["receiver_id"])
        try:
            sv = shap_matrix(explainer, X.iloc[[j]])[0]
        except Exception:
            sv = native_contrib(model, X.iloc[[j]])[0]
        # same "only cite true reasons" filter as the API and UIs
        reasons = risk_reasons(top_reasons(sv, k=len(FEATURE_COLUMNS)), row)

        decision = decide(conf_set, row, soft, reasons, amount, thr, mulehunter)
        fired = active_signals(row, soft, thr, mulehunter, conf_set)

        print(f"txn_id={int(row['txn_id'])} p_scam={float(p[j]):.4f} set={conf_set} "
              f"amount={amount:,.2f} mulehunter={mulehunter!r}")
        print(f"   hard={fired['hard'] or '{}'} soft={fired['soft'] or '{}'}")
        for text in decision["reasons"]:
            print(f"   - {text}")
        print(f"   decision={json.dumps(decision)}")
    return 0


def run_eval(model, df, qhat_fine: float, qhat_scam: float, thr: Thresholds, split: str) -> int:
    """Run decide() over every txn of `split` and report the policy's action rates.

    Baseline view: no soft signals and no MuleHunter verdict (both are mocked
    inputs with no ground truth in the data), so this is what the model +
    hard signals do on their own. Soft signals can only turn nudges into holds.
    """
    part = df[df["split"] == split]
    if part.empty:
        print(f"[policy] split {split!r} has no rows")
        return 1
    p = model.predict_proba(part[FEATURE_COLUMNS])[:, 1]
    sets = predict_sets(p, qhat_fine, qhat_scam)
    feats = part[list(PAYMENT_HISTORY_FEATURES + RECEIVING_ACCOUNT_FEATURES)].to_dict("records")
    amounts = part["amount"].to_numpy(float)
    out = [decide(SET_NAMES[int(s)], f, None, None, a, thr) for s, f, a in zip(sets, feats, amounts)]
    actions = np.array([d["action"] for d in out])
    analyst = np.array([d["requires_analyst"] for d in out])
    y = part[TARGET].to_numpy(np.int8) == 1

    n, n_pos = len(part), int(y.sum())
    print(f"[policy] eval split={split}: {n:,} txns, {n_pos} scams "
          f"(no soft signals, no MuleHunter; thresholds {thr})")
    print(f"{'action':<10}{'txns':>10}{'share':>9}{'scams':>8}{'% of scams':>12}{'% of legit':>12}")
    for action in (ACTION_RELEASE, ACTION_NUDGE, ACTION_HOLD):
        hit = actions == action
        print(f"{action:<10}{int(hit.sum()):>10,}{hit.mean() * 100:>8.2f}%{int(hit[y].sum()):>8}"
              f"{hit[y].mean() * 100 if n_pos else 0.0:>11.2f}%{hit[~y].mean() * 100:>11.3f}%")
    held = actions == ACTION_HOLD
    precision = held[y].sum() / max(int(held.sum()), 1)
    print(f"[policy] hold precision={precision:.4f}  scams intercepted (hold or nudge)="
          f"{(actions[y] != ACTION_RELEASE).mean() * 100 if n_pos else 0.0:.2f}%  "
          f"analyst reviews={int(analyst.sum()):,}")
    return 0


# --- self-test (the truth table) --------------------------------------------


def _self_test_cases() -> list:
    none_feats = {"n_t": 0.0, "e_t": 0.0, "z_t": 0.0, "pi_t": 0.0, "d_R": 3.0}
    pay_feats = {"n_t": 0.0, "e_t": 0.0, "z_t": 3.0, "pi_t": 0.0, "d_R": 3.0}
    rcv_feats = {"n_t": 0.0, "e_t": 0.0, "z_t": 0.0, "pi_t": 0.0, "d_R": 0.0}
    both_feats = {"n_t": 0.0, "e_t": 0.0, "z_t": 3.0, "pi_t": 0.0, "d_R": 0.0}
    # split payment: each part's z_t is below the bar, the 24 h total zc_t is not
    split_feats = {"n_t": 0.0, "e_t": 0.0, "z_t": 1.6, "zc_t": 5.0, "pi_t": 0.0, "d_R": 3.0}

    fine, scam, unsure = SET_NAMES[SET_FINE], SET_NAMES[SET_SCAM], SET_NAMES[SET_UNSURE]
    chat = {"chat_stage": "escalated"}
    screen = {"is_screen_shared": True}
    soft_off = {"chat_stage": "none"}

    return [
        # name, set, features, soft, amount, action, minutes, analyst, mulehunter
        ("fine releases", fine, both_feats, chat, 500_000.0, "release", 0, False, None),
        ("scam without signals nudges", scam, none_feats, soft_off, 500.0, "nudge", 0, False, None),
        ("scam two groups + analyst", scam, both_feats, soft_off, 150_000.0, "hold", 60, True, None),
        ("unsure no signals -> nudge", unsure, none_feats, soft_off, 900.0, "nudge", 0, False, None),
        ("unsure soft only -> nudge", unsure, none_feats, chat, 900.0, "nudge", 0, False, None),
        ("unsure screen only -> nudge", unsure, none_feats, screen, 900.0, "nudge", 0, False, None),
        ("unsure 1 hard no soft -> nudge", unsure, pay_feats, soft_off, 900.0, "nudge", 0, False, None),
        ("unsure 1 rcv no soft -> nudge", unsure, rcv_feats, soft_off, 900.0, "nudge", 0, False, None),
        ("unsure pay + soft -> hold", unsure, pay_feats, chat, 900.0, "hold", 60, False, None),
        ("unsure rcv + soft -> hold", unsure, rcv_feats, screen, 900.0, "hold", 60, False, None),
        ("unsure pay + rcv -> hold", unsure, both_feats, soft_off, 900.0, "hold", 60, False, None),
        ("unsure pay + rcv big -> analyst", unsure, both_feats, soft_off, 150_000.0, "hold", 60, True, None),
        ("unsure hold analyst only if >100k", unsure, both_feats, soft_off, 99_999.0, "hold", 60, False, None),
        # MuleHunter.AI as a Receiving Account hard signal (unsure set only)
        ("unsure mule alone -> nudge", unsure, none_feats, soft_off, 900.0, "nudge", 0, False, "mule"),
        ("unsure mule + pay -> hold", unsure, pay_feats, soft_off, 900.0, "hold", 60, False, "mule"),
        ("unsure mule + soft -> hold", unsure, none_feats, chat, 900.0, "hold", 60, False, "mule"),
        ("unsure suspicious -> nudge", unsure, none_feats, soft_off, 900.0, "nudge", 0, False, "suspicious"),
        ("unsure clean -> nudge", unsure, none_feats, soft_off, 900.0, "nudge", 0, False, "clean"),
        ("fine mule ignored -> release", fine, none_feats, soft_off, 900.0, "release", 0, False, "mule"),
        ("scam mule alone -> nudge", scam, none_feats, soft_off, 900.0, "nudge", 0, False, "mule"),
        # zc_t (24 h cumulative amount) is a Payment History hard signal
        ("unsure split + soft -> hold", unsure, split_feats, chat, 900.0, "hold", 60, False, None),
        ("unsure split alone -> nudge", unsure, split_feats, soft_off, 900.0, "nudge", 0, False, None),
    ]


SELF_TEST_CASES = _self_test_cases()


def run_self_test() -> bool:
    cases = SELF_TEST_CASES
    n_pass = 0
    for name, cset, feats, soft, amt, exp_action, exp_min, exp_analyst, mh in cases:
        got = decide(cset, feats, soft, ["r1", "r2", "r3"], amt, DEFAULT_THRESHOLDS, mh)
        ok = (
            got["action"] == exp_action
            and got["auto_release_minutes"] == exp_min
            and got["requires_analyst"] == exp_analyst
            and got["reasons"] == ["r1", "r2", "r3"]
        )
        n_pass += int(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {got['action']} "
              f"min={got['auto_release_minutes']} analyst={got['requires_analyst']}")
        if not ok:
            print(f"   expected action={exp_action} min={exp_min} analyst={exp_analyst}")

    print(f"[policy] self-test {n_pass}/{len(cases)} passed")
    return n_pass == len(cases)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Maya action policy engine (Two-Signal Rule).")
    ap.add_argument("--model", default="models/xgb_model.json")
    ap.add_argument("--features", default="data/features.csv")
    ap.add_argument("--conformal", default="models/conformal.json")
    ap.add_argument("--txn-id", type=int, default=None)
    ap.add_argument("--demo", type=int, default=0)
    ap.add_argument("--amount", type=float, default=None, help="override the txn amount")
    ap.add_argument("--soft-signals", default="", help='JSON, e.g. \'{"is_screen_shared": true}\'')
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--eval", nargs="?", const="test", default=None, metavar="SPLIT",
                    help="report release/nudge/hold rates over a split (default: test)")
    ap.add_argument("--n-t-threshold", type=float, default=DEFAULT_THRESHOLDS.n_t)
    ap.add_argument("--e-t-threshold", type=float, default=DEFAULT_THRESHOLDS.e_t)
    ap.add_argument("--z-t-threshold", type=float, default=DEFAULT_THRESHOLDS.z_t)
    ap.add_argument("--zc-t-threshold", type=float, default=DEFAULT_THRESHOLDS.zc_t)
    ap.add_argument("--pi-t-threshold", type=float, default=DEFAULT_THRESHOLDS.pi_t)
    ap.add_argument("--d-r-max", type=float, default=DEFAULT_THRESHOLDS.d_r_max)
    args = ap.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
