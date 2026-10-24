"""Local simulated payment service. SQLite owns operations and immutable reviews.

No bank connection, user authentication, or real settlement. The session check
models binding; it is not production access control. Callers never supply a
decision to execute: execution retrieves the service's persisted review.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
import time
import uuid
from copy import deepcopy
from pathlib import Path

from claims import compare_claim, extract_claim
from policy import decide
from risk_common import FEATURE_COLUMNS

# v2: zc_t (24 h cumulative amount) joined the payment-history hard signals.
POLICY_VERSION = "payment-review-v2"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def available_evidence(evidence, at):
    return [deepcopy(e) for e in evidence if e["event_at"] < at and e["available_at"] < at
            and (e["kind"] != "payment" or e["event_at"] >= at - 14 * 86400)]


def feature_snapshot(payment, evidence):
    """Small synthetic journey adapter; not a replacement for features.py.

    Derives the supported combined features from available events. Endpoint
    columns are mirrored explicitly for the demo, as in the existing UI.
    """
    history = [e for e in evidence if e["kind"] == "payment" and e["direction"] == "outgoing"]
    prior = [e["amount_paise"] for e in history]
    payees = {e["recipient"] for e in history}
    recipient = payment["recipient"]
    reports = {e["account"] for e in evidence if e["kind"] == "report"}
    edges = [(e["source"], e["target"]) for e in evidence if e["kind"] == "edge"]
    distance = 3
    frontier = {recipient}
    for depth in range(3):
        if frontier & reports:
            distance = depth
            break
        frontier = {dst for src, dst in edges if src in frontier}
    amount = payment["amount_paise"]
    ratio = amount / max(prior) if prior else 0.0
    # Same-recipient payments in the evidence window feed the anti-splitting
    # features: v_t counts them, zc_t is z_t on the cumulative amount.
    same = [e["amount_paise"] for e in history if e["recipient"] == recipient]
    cum_ratio = (amount + sum(same)) / max(prior) if prior else 0.0
    combined = dict(n_t=float(len(payees)), e_t=ratio if recipient not in payees else 0.0,
                    g_t=float(any(e["kind"] == "payment" and e["direction"] == "incoming" for e in evidence)),
                    z_t=max(0.0, math.log(ratio)) if ratio else 0.0,
                    pi_t=float(sum(e["amount_paise"] <= 1000 for e in history)), d_R=float(distance),
                    v_t=float(len(same)),
                    zc_t=max(0.0, math.log(cum_ratio)) if cum_ratio else 0.0,
                    # Pass-through needs the recipient's own flows, which this
                    # evidence lacks: None is JSON-safe and the model sees NaN.
                    ptr_t=None)
    return {column: combined[column.replace("_snd", "").replace("_rcv", "")] for column in FEATURE_COLUMNS}


class ModelAdapter:
    def __init__(self, model_path, calibration_path):
        import xgboost as xgb
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        from conformal import load_conformal
        self.qhats = load_conformal(calibration_path)  # (qhat_fine, qhat_scam)
        self.version = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
        self.calibration_version = hashlib.sha256(Path(calibration_path).read_bytes()).hexdigest()

    def score(self, features):
        import pandas as pd
        from conformal import classify
        row = pd.DataFrame([features], columns=FEATURE_COLUMNS).astype("float32")
        score = float(self.model.predict_proba(row)[0, 1])
        return dict(score=score, uncertainty=classify(score, *self.qhats))


def evaluate(adapter, payment, evidence, text, consent, enhanced=True):
    features = feature_snapshot(payment, evidence)
    model_result = adapter.score(features)
    claim = extract_claim(text, consent)
    comparison = compare_claim(claim, payment["direction"])
    soft = {"chat_stage": "payment_demand"} if enhanced and comparison in (
        "direction_contradiction", "unverified_future_benefit") else {}
    reasons = []
    if features["e_t"] >= 1 or features["z_t"] >= 2.6:
        reasons.append("This amount is high relative to the available payment history.")
    elif features["zc_t"] >= 4.2:
        reasons.append("Payments to this recipient add up to a high amount relative to the history.")
    if features["d_R"] <= 1:
        reasons.append("The recipient is connected to an account reported before this review.")
    if soft:
        reasons.append("The message promises incoming money, but this payment sends money." if
                       comparison == "direction_contradiction" else
                       "This payment cannot verify the promised investment withdrawal.")
    result = decide(model_result["uncertainty"], features, soft, reasons,
                    payment["amount_paise"] / 100)
    if enhanced and comparison == "direction_contradiction" and result["action"] == "release":
        result["action"] = "nudge"
        result["reasons"] = ["The message promises incoming money, but this payment sends money."]
    # Independent verifier of the proposed action; model certainty is not a group.
    history = any(features[k] >= t for k, t in (("n_t", 59), ("e_t", 1), ("z_t", 2.6), ("zc_t", 4.2), ("pi_t", 1)))
    receiving = features["d_R"] <= 1
    if result["action"] == "hold" and not ((history and receiving) or ((history or receiving) and soft)):
        raise ValueError("Policy verifier rejected a hold without independent groups")
    return dict(**result, **model_result, features=features, claim=claim,
                comparison=comparison, evidence_ids=[e["id"] for e in evidence],
                reason_kind="evidence templates; not SHAP attribution")


class PaymentService:
    def __init__(self, adapter, database=":memory:", clock=time.time):
        self.adapter, self.clock = adapter, clock
        self.lock = threading.RLock()
        self.db = sqlite3.connect(database, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS payments (id TEXT PRIMARY KEY, body TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS reviews (id TEXT PRIMARY KEY, body TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS holds (payment_id TEXT, revision INTEGER, until REAL, analyst_required INTEGER, approved INTEGER DEFAULT 0, PRIMARY KEY(payment_id, revision))")
        self.db.execute("CREATE TRIGGER IF NOT EXISTS immutable_review BEFORE UPDATE ON reviews BEGIN SELECT RAISE(ABORT, 'immutable review'); END")
        self.db.execute("CREATE TRIGGER IF NOT EXISTS retain_review BEFORE DELETE ON reviews BEGIN SELECT RAISE(ABORT, 'immutable review'); END")
        self.db.commit()

    def _payment(self, payment_id, session):
        row = self.db.execute("SELECT body FROM payments WHERE id=?", (payment_id,)).fetchone()
        if not row:
            raise ValueError("Unknown payment")
        p = json.loads(row[0])
        if p["session"] != session:
            raise ValueError("Wrong payer session")
        return p

    def _save(self, p):
        self.db.execute("UPDATE payments SET body=? WHERE id=?", (canonical(p), p["id"]))

    @staticmethod
    def _validate(recipient, amount, direction):
        if not isinstance(recipient, str) or not recipient.strip() or len(recipient) > 200:
            raise ValueError("Recipient is required")
        if type(amount) is not int or not 0 < amount <= 100_000_000:
            raise ValueError("Amount must be positive integer paise within the simulator limit")
        if direction not in ("incoming", "outgoing"):
            raise ValueError("Invalid direction")

    def create(self, session, recipient, amount_paise, direction="outgoing"):
        self._validate(recipient, amount_paise, direction)
        if not session:
            raise ValueError("Session is required")
        p = dict(id=uuid.uuid4().hex, session=session, recipient=recipient,
                 amount_paise=amount_paise, direction=direction, revision=1, state="pending")
        with self.lock, self.db:
            self.db.execute("INSERT INTO payments VALUES (?, ?)", (p["id"], canonical(p)))
        return deepcopy(p)

    def get(self, payment_id, session):
        with self.lock:
            return self._payment(payment_id, session)

    def change(self, payment_id, session, recipient, amount_paise):
        self._validate(recipient, amount_paise, "outgoing")
        with self.lock, self.db:
            self.db.execute("BEGIN IMMEDIATE")
            p = self._payment(payment_id, session)
            if p["state"] != "pending":
                raise ValueError("Payment is terminal")
            p.update(recipient=recipient, amount_paise=amount_paise, revision=p["revision"] + 1)
            self._save(p)
            return p

    def review(self, payment_id, session, evidence, text="", consent=False, parent=None):
        with self.lock, self.db:
            self.db.execute("BEGIN IMMEDIATE")
            p = self._payment(payment_id, session)
            if p["state"] != "pending":
                raise ValueError("Payment is terminal")
            if parent:
                self.replay(parent, session)  # parent must belong to this payer session
            now = self.clock()
            evidence = available_evidence(evidence, now)
            outcome = evaluate(self.adapter, p, evidence, text, consent)
            if outcome["action"] == "hold":
                self.db.execute("INSERT OR IGNORE INTO holds (payment_id, revision, until, analyst_required) VALUES (?, ?, ?, ?)",
                                (p["id"], p["revision"], now + 3600, int(outcome["requires_analyst"])))
            record = dict(id=uuid.uuid4().hex, payment=p, evaluated_at=now, expires_at=now + 300,
                          model_version=self.adapter.version, calibration_version=self.adapter.calibration_version,
                          policy_version=POLICY_VERSION, evidence=evidence,
                          shared_text=text if consent else "", consent=consent, outcome=outcome, parent=parent)
            record["digest"] = digest(record)
            self.db.execute("INSERT INTO reviews VALUES (?, ?)", (record["id"], canonical(record)))
            return deepcopy(record)

    def replay(self, review_id, session, recompute=False):
        with self.lock:
            row = self.db.execute("SELECT body FROM reviews WHERE id=?", (review_id,)).fetchone()
            if not row:
                raise ValueError("Unknown review")
            record = json.loads(row[0])
            if record["payment"]["session"] != session:
                raise ValueError("Wrong payer session")
            expected = record.pop("digest")
            if digest(record) != expected:
                raise ValueError("Review integrity failed")
            record["digest"] = expected
            if recompute:
                self._versions(record)
                result = evaluate(self.adapter, record["payment"], record["evidence"],
                                  record["shared_text"], record["consent"])
                if result != record["outcome"]:
                    raise ValueError("Replay differs from original outcome")
            return record

    def _versions(self, r):
        if (r["model_version"], r["calibration_version"], r["policy_version"]) != (
            self.adapter.version, self.adapter.calibration_version, POLICY_VERSION):
            raise ValueError("Review versions changed; reevaluate")

    def execute(self, payment_id, session, review_id, confirmed=False):
        with self.lock, self.db:
            self.db.execute("BEGIN IMMEDIATE")
            p = self._payment(payment_id, session)
            r = self.replay(review_id, session)
            self._versions(r)
            if p != r["payment"] or p["state"] != "pending":
                raise ValueError("Payment changed or completed; reevaluate")
            if self.clock() >= r["expires_at"]:
                raise ValueError("Review expired; reevaluate")
            hold = self.db.execute("SELECT until, analyst_required, approved FROM holds WHERE payment_id=? AND revision=?",
                                   (p["id"], p["revision"])).fetchone()
            if hold and self.clock() < hold[0]:
                raise ValueError("Payment held for one hour; cancellation remains available")
            if hold and hold[1] and not hold[2]:
                raise ValueError("Analyst approval required in this simulator")
            if (hold or r["outcome"]["action"] == "nudge") and not confirmed:
                raise ValueError("Explicit confirmation required")
            p["state"] = "completed"
            self._save(p)
            return p

    def approve_by_analyst(self, payment_id, session, review_id):
        """Explicit simulated analyst action, bound to the reviewed revision."""
        with self.lock, self.db:
            self.db.execute("BEGIN IMMEDIATE")
            p = self._payment(payment_id, session)
            r = self.replay(review_id, session)
            self._versions(r)
            if p != r["payment"] or p["state"] != "pending" or self.clock() >= r["expires_at"]:
                raise ValueError("Analyst review is stale")
            self.db.execute("UPDATE holds SET approved=1 WHERE payment_id=? AND revision=?", (p["id"], p["revision"]))

    def cancel(self, payment_id, session):
        with self.lock, self.db:
            self.db.execute("BEGIN IMMEDIATE")
            p = self._payment(payment_id, session)
            if p["state"] != "pending":
                raise ValueError("Payment is terminal")
            p["state"] = "cancelled"
            self._save(p)
