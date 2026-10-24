"""Evasion perturbations of scam feature rows, shared by red-teaming and training.

Each function takes a frame with the model's FEATURE_COLUMNS plus HELPER_COLUMNS
(the raw 24 h sums that src/features.py writes) and returns perturbed copies,
recomputed with the same formulas as features.py:

  split_k    send the amount A as k equal parts to the same payee within 24 h.
             For part i (1..k), per side, with the side's past-log median
             recovered as log(A) - z_t:
               z_t   -= log(k)                        (exact)
               e_t   /= k, and min(., 1) for i >= 2   (<= the true value: if the
                     payee was new, part 1 joins the e_t denominator)
               v_t   += i - 1                         (exact)
               zc_t  = log(pair24 + i*A/k) - median   (exact)
               ptr_t sender outflow / receiver inflow += (i-1)*A/k (exact)
             Approximation: earlier parts also join the past-log median, which
             is ignored (small when the account has a long history).
  extra_hop  route through one more layering mule: d_R (min hops to a reported
             account, <=2 else 3) goes up by one, capped at 3. The receiver's
             behaviour features (ptr_t, n_t_rcv, ...) are kept, i.e. the new
             entry mule is assumed to forward money like the old one.
  no_probes  skip the tiny probe payments: pi_t = 0.

ATTACKS maps a name to a function returning the list of payment rows the
scammer sends (a scam is caught if any of them is flagged).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SIDES = ("", "_snd", "_rcv")
D_R_CAP = 3.0

HELPER_COLUMNS = ["amount", "pair24", "in24_snd", "out24_snd", "in24_rcv", "out24_rcv"]


def larger_abs(a: pd.Series, b: pd.Series) -> np.ndarray:
    """features.py's combine rule for z-like features: the side with larger |z|."""
    return np.where(a.isna(), b, np.where(b.isna(), a, np.where(a.abs() >= b.abs(), a, b)))


def split_part(X: pd.DataFrame, k: int, i: int) -> pd.DataFrame:
    """Part i (1..k) of k equal sub-payments to the same payee."""
    X = X.copy()
    amount = X["amount"]
    part = amount / k
    for s in ("_snd", "_rcv"):
        median = np.log(amount) - X[f"z_t{s}"]
        X[f"z_t{s}"] = X[f"z_t{s}"] - np.log(k)
        X[f"e_t{s}"] = X[f"e_t{s}"] / k if i == 1 else np.minimum(X[f"e_t{s}"] / k, 1.0)
        if f"v_t{s}" in X:
            X[f"v_t{s}"] = X[f"v_t{s}"] + (i - 1)
            X[f"zc_t{s}"] = np.log(X["pair24"] + i * part) - median
    X["z_t"] = larger_abs(X["z_t_snd"], X["z_t_rcv"])
    X["e_t"] = X["e_t"] / k if i == 1 else np.minimum(X["e_t"] / k, 1.0)
    if "v_t" in X:
        X["v_t"] = np.maximum(X["v_t_snd"], X["v_t_rcv"])
        X["zc_t"] = larger_abs(X["zc_t_snd"], X["zc_t_rcv"])
        prior = (i - 1) * part  # earlier parts: sender outflow, receiver inflow
        out_s, in_s = X["out24_snd"] + prior, X["in24_snd"]
        out_r, in_r = X["out24_rcv"], X["in24_rcv"] + prior
        X["ptr_t_snd"] = np.where(in_s > 0, out_s / in_s, np.nan)
        X["ptr_t_rcv"] = np.where(in_r > 0, out_r / in_r, np.nan)
        X["ptr_t"] = np.fmax(X["ptr_t_snd"], X["ptr_t_rcv"])
    return X


def split_parts(X: pd.DataFrame, k: int) -> list:
    return [split_part(X, k, i) for i in range(1, k + 1)]


def extra_hop(X: pd.DataFrame) -> pd.DataFrame:
    """One more layering mule between victim and the reported account."""
    X = X.copy()
    for s in SIDES:
        X[f"d_R{s}"] = np.minimum(X[f"d_R{s}"] + 1.0, D_R_CAP)
    return X


def no_probes(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for s in SIDES:
        X[f"pi_t{s}"] = 0.0
    return X


ATTACKS = {
    "split_2": lambda X: split_parts(X, 2),
    "split_4": lambda X: split_parts(X, 4),
    "split_10": lambda X: split_parts(X, 10),
    "extra_hop": lambda X: [extra_hop(X)],
    "no_probes": lambda X: [no_probes(X)],
    "all": lambda X: [no_probes(extra_hop(p)) for p in split_parts(X, 4)],
}

# Attacks used for adversarial training in train_xgb.py. split_10 and "all"
# are deliberately held out so attack_sim also measures unseen evasions.
TRAIN_ATTACKS = ("split_2", "split_4", "extra_hop")


def augment(X: pd.DataFrame, attacks=TRAIN_ATTACKS) -> tuple[pd.DataFrame, np.ndarray]:
    """Evasive copies of scam rows `X` plus per-row weights.

    Each attack's rows for one original scam share a total weight of 1, so an
    attack that emits k parts does not count k times more than a 1-row attack.
    """
    frames, weights = [], []
    for name in attacks:
        parts = ATTACKS[name](X)
        for p in parts:
            frames.append(p)
            weights.append(np.full(len(p), 1.0 / len(parts)))
    return pd.concat(frames, ignore_index=True), np.concatenate(weights)
