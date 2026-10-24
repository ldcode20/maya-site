"""Maya red-teaming: measure how the risk model holds up against evasive scammers.

Two kinds of attack against real scam transactions from the test split:

1. **Text rewording** (the Maya PDF's "AI rewording cut text detectors' catch
   rate by 88%"). Maya's 18 features are journey/money-flow only -- no chat text
   is a model input -- so rewording leaves its score unchanged *by construction*.
   This row is an architectural illustration, not an empirical test. The
   phrase-matching "text-only detector" contrast is likewise illustrative: its
   88% collapse is set by REWORD_EVASION (the PDF figure), not measured.

2. **Adaptive journey attacks** (the real test): the scammer changes the parts
   of the journey they control (split_2/4/10, extra_hop, no_probes, and "all"
   = split_4 + extra_hop + no_probes), with features recomputed by the
   features.py formulas -- see src/evasion.py for the exact math. A split scam
   counts as caught if ANY of its parts is caught (one hold stops it). Each
   attack rescores the perturbed rows with the real model; catch rate = recall
   at the val-chosen 1%-FPR threshold in models/model_meta.json.
   train_xgb.py adversarially trains on evasion.TRAIN_ATTACKS (split_2,
   split_4, extra_hop); split_10 and "all" are held out, so their rows measure
   robustness to evasions the model never saw.
   The model's own feature list is used, so an older model (e.g. the 18-feature
   one) can be attacked with --model/--meta for a before/after comparison.

Exit code: 0 iff Maya's rewording drop < --drop-target; with --strict, every
adaptive attack's drop must also be < --drop-target.

Run:  python src/attack_sim.py
      python src/attack_sim.py --strict
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xgboost as xgb

from evasion import ATTACKS, HELPER_COLUMNS, TRAIN_ATTACKS
from risk_common import FEATURE_COLUMNS, TARGET, load_features

# The PDF's headline figure: how much AI rewording cut *text* detectors.
REWORD_EVASION = 0.88

# Trigger phrases a text-only classifier keys on. A stand-in for the production
# text detector, documented as such (Maya never consumes this text).
TRIGGERS = [
    "guaranteed returns",
    "send OTP",
    "verify your KYC",
    "CBI verification",
    "digital arrest",
    "double your money",
    "blocked account",
    "remote access app",
    "transfer now",
    "pay the fine",
]

SYNONYMS = {
    "guaranteed returns": "assured growth",
    "send OTP": "share the code",
    "verify your KYC": "confirm your details",
    "CBI verification": "a compliance check",
    "digital arrest": "a legal hold",
    "double your money": "grow your savings",
    "blocked account": "a hold on the wallet",
    "remote access app": "a support tool",
    "transfer now": "complete the payment",
    "pay the fine": "settle the amount",
}

STYLE_TEMPLATES = [
    "Sir, {t1} and {t2} to unlock your {t3}.",
    "Your {t3}. Please {t1} and {t2} immediately.",
    "This is {t1} for your {t3}; {t2} to avoid a {t3}.",
]


def build_script(txn_id: int) -> str:
    """Deterministic synthetic chat script for a scam txn (text-only demo)."""
    t = int(txn_id)
    t1, t2, t3 = TRIGGERS[t % len(TRIGGERS)], TRIGGERS[(t // 3) % len(TRIGGERS)], TRIGGERS[(t // 7) % len(TRIGGERS)]
    return STYLE_TEMPLATES[t % len(STYLE_TEMPLATES)].format(t1=t1, t2=t2, t3=t3)


def reword(script: str, keep_trigger: bool) -> str:
    """AI-style paraphrase: replace trigger phrases with synonyms (or keep one)."""
    out = script
    for k, v in SYNONYMS.items():
        if keep_trigger and k in out:
            keep_trigger = False  # leave exactly one trigger phrase intact
            continue
        out = out.replace(k, v)
    return out


def text_detector(script: str) -> bool:
    return any(k in script for k in TRIGGERS)


def catch_rate(p: np.ndarray, threshold: float) -> float:
    return float((p >= threshold).mean())


def run(args) -> int:
    model = xgb.XGBClassifier()
    model.load_model(args.model)
    meta = json.loads(Path(args.meta).read_text())
    threshold = float(meta["val_threshold_at_1pct_fpr"])

    cols = model.get_booster().feature_names or FEATURE_COLUMNS
    df = load_features(args.features, columns=FEATURE_COLUMNS + HELPER_COLUMNS)
    te = df[df["split"] == "test"]
    scams = te[te[TARGET] == 1]
    if args.n_attacks and len(scams) > args.n_attacks:
        scams = scams.sample(n=args.n_attacks, random_state=args.seed).sort_values("txn_id")
    n = len(scams)
    ids = scams["txn_id"].to_numpy(np.int64)
    X = scams[FEATURE_COLUMNS + HELPER_COLUMNS]

    p_scam = model.predict_proba(X[cols])[:, 1]
    maya_before = catch_rate(p_scam, threshold)

    # Text-only detector (illustrative): the reworder strips triggers from
    # REWORD_EVASION of the scripts.
    scripts = [build_script(int(i)) for i in ids]
    keep = int(round(n * (1.0 - REWORD_EVASION)))
    reworded = [reword(s, idx >= n - keep) for idx, s in enumerate(scripts)]
    text_before = sum(text_detector(s) for s in scripts) / n
    text_after = sum(text_detector(s) for s in reworded) / n

    print("=" * 78)
    print(f"[attack_sim] {n} test-split scam txns, operating threshold (val @1%FPR) = {threshold:.4f}, "
          f"model features = {len(cols)}")
    print("=" * 78)
    print(f"{'attack':<28}{'catch before':>14}{'catch after':>13}{'drop (pts)':>13}")
    print("-" * 68)
    print("text rewording (illustrative, not a model input):")
    print(f"  {'text-only detector':<26}{text_before * 100:>13.1f}%{text_after * 100:>12.1f}%"
          f"{(text_before - text_after) * 100:>12.1f}")
    print(f"  {'Maya (no text features)':<26}{maya_before * 100:>13.1f}%{maya_before * 100:>12.1f}%{0.0:>12.1f}")
    print("adaptive journey attacks (measured):")
    drops, caught = {}, {}
    for name, attack in ATTACKS.items():
        # Caught if any of the scammer's payments is flagged.
        p_after = np.max([model.predict_proba(part[cols])[:, 1] for part in attack(X)], axis=0)
        after = catch_rate(p_after, threshold)
        caught[name] = after
        drops[name] = (maya_before - after) * 100.0
        tag = "" if name in TRAIN_ATTACKS else "  (held out of adversarial training)"
        print(f"  {'Maya / ' + name:<26}{maya_before * 100:>13.1f}%{after * 100:>12.1f}%"
              f"{drops[name]:>12.1f}{tag}")
    print()

    worst = max(drops, key=drops.get)
    print(f"[attack_sim] worst adaptive attack: {worst} (-{drops[worst]:.1f} pts)")
    failed = [k for k, v in drops.items() if v >= args.drop_target]
    verdict = not (args.strict and failed)
    detail = (f"adaptive attacks over the {args.drop_target:.1f}-pt target: {', '.join(failed)}"
              if failed else f"every adaptive drop < {args.drop_target:.1f} pts")
    print(f"[attack_sim] VERDICT: {'PASS' if verdict else 'FAIL'} "
          f"({'strict' if args.strict else 'rewording-only; pass --strict to gate on adaptive attacks'}; {detail})")
    print("=" * 78)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "n_scams": n, "threshold": threshold, "baseline_catch_rate": maya_before,
            "model_features": len(cols),
            "attacks": [{"name": k, "catch_rate": caught[k], "drop_pts": drops[k],
                         "held_out_of_training": k not in TRAIN_ATTACKS} for k in ATTACKS],
            "note": "Known, white-box attacks on synthetic test scams; caught = recall at the "
                    "val 1%-FPR threshold, a split scam counts as caught if any part is.",
        }, indent=2))
        print(f"[attack_sim] wrote {out}")
    return 0 if verdict else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Red-team the Maya model with rewording and adaptive journey attacks.")
    ap.add_argument("--features", default="data/features.csv")
    ap.add_argument("--model", default="models/xgb_model.json")
    ap.add_argument("--meta", default="models/model_meta.json")
    ap.add_argument("--n-attacks", type=int, default=0,
                    help="sample this many test scams (default 0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drop-target", type=float, default=10.0,
                    help="max acceptable Maya catch-rate drop in points")
    ap.add_argument("--strict", action="store_true",
                    help="also fail if any adaptive journey attack drops >= --drop-target")
    ap.add_argument("--json-out", default=None,
                    help="also write the results as JSON (e.g. models/attack_results.json, "
                         "which the API's /v1/model and the judges' demo show)")
    args = ap.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
