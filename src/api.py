"""Maya risk-scoring API: the service a bank or UPI app calls before releasing a payment.

Two families of endpoints (FastAPI; interactive docs at /docs):

  Stateless scoring (the bank's hot path)
    POST /v1/score            the caller's feature pipeline sends the 27 Maya
                              features (feature contract maya-features-v1) plus
                              the amount and optional soft signals / MuleHunter
                              verdict; Maya returns release / nudge / hold with
                              reasons. The conformal set is always computed
                              here -- a caller can never pass a decision in.

  Stateful payment review (thin wrappers over payment_review.PaymentService)
    POST /v1/payments, GET|PATCH /v1/payments/{id},
    POST /v1/payments/{id}/reviews, GET /v1/reviews/{id},
    POST /v1/payments/{id}/execute | /cancel | /analyst-approval
    The caller is identified by the X-Payer-Session header -- a simulated
    binding, not real authentication.

  Meta
    GET /v1/health, GET /v1/model, GET /v1/demo/scenarios

  Frontends (static, no build step; served when frontend/ exists)
    /app    everyday-user payment app      (frontend/user)
    /demo   judges' demo                   (frontend/judge)
    /static shared tokens, fonts, helpers  (frontend/shared)

Everything is local and simulated: no bank connection, no real money, and the
MuleHunter / FRI / NCRP feeds stay mocks. See docs/API.md.

Run:  uvicorn api:app --app-dir src            (from the repo root)
      MAYA_MODEL / MAYA_CONFORMAL / MAYA_REVIEW_DB override the default paths.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import mimetypes
import os
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

import payment_review
from conformal import classify, load_conformal
from integrations import check_mulehunter
from policy import DEFAULT_THRESHOLDS, active_signals, decide
from risk_common import FEATURE_COLUMNS, LATENCY_SLO_MS
from rooms import RateLimiter, RoomError, Rooms
from scenario_features import SCENARIO_FEATURES, SCENARIO_STORY

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
mimetypes.add_type("font/woff2", ".woff2")  # Windows' registry often lacks it
API_VERSION = "v1"
FEATURE_CONTRACT = "maya-features-v1"
SLO_MS = LATENCY_SLO_MS  # per-payment latency budget from the spec
# Policy semantics version, shared with the stateful review path.
POLICY_VERSION = payment_review.POLICY_VERSION


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- model scorer ------------------------------------------------------------


class Scorer:
    """Loaded model + conformal cutoffs + SHAP explainer, built once at startup."""

    def __init__(self, model_path, conformal_path, meta_path=None):
        import xgboost as xgb

        from reasons import make_explainer

        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        self.qhats = load_conformal(conformal_path)
        self.explainer = make_explainer(self.model)
        self.model_version = sha256_file(model_path)
        self.calibration_version = sha256_file(conformal_path)
        meta = Path(meta_path) if meta_path else None
        self.meta = json.loads(meta.read_text()) if meta and meta.exists() else {}
        # attack_sim.py --json-out writes this next to the model
        attacks = Path(model_path).with_name("attack_results.json")
        self.attacks = json.loads(attacks.read_text()) if attacks.exists() else None
        self._lock = threading.Lock()  # SHAP explainer thread-safety is not guaranteed

    def score(self, values: dict, explain: bool = False) -> dict:
        """values: all FEATURE_COLUMNS -> float (NaN = missing)."""
        import pandas as pd

        from reasons import in_risky_range, risk_reasons, shap_rows

        X = pd.DataFrame([values], columns=FEATURE_COLUMNS).astype("float32")
        with self._lock:
            p = float(self.model.predict_proba(X)[0, 1])
            rows = shap_rows(self.model, self.explainer, X)
        cited = risk_reasons(rows, values, with_features=True)
        out = {
            "p_scam": p,
            "conformal_set": classify(p, *self.qhats),
            "reasons": [text for _, text in cited],
            "reason_features": [base for base, _ in cited],
        }
        if explain:
            out["explanation"] = {
                "contributions": [
                    {"feature": f, "shap": v, "text": t,
                     "value": None if math.isnan(values[f]) else values[f],
                     "risky": in_risky_range(f.replace("_snd", "").replace("_rcv", ""), values[f])}
                    for f, v, t in rows
                ],
                "note": "Tree SHAP in log-odds; positive pushes towards scam. Not proof of fraud.",
            }
        return out

    def info(self) -> dict:
        qhat_fine, qhat_scam = self.qhats
        keys = ("test_auroc", "test_auprc", "test_recall_at_1pct_fpr", "best_iteration",
                "adversarial_attacks")
        return {
            "model_version": self.model_version,
            "calibration_version": self.calibration_version,
            "conformal": {"fine_if_p_at_most": qhat_fine, "scam_if_p_at_least": 1.0 - qhat_scam},
            "metrics": {k: self.meta[k] for k in keys if k in self.meta},
            "attacks": self.attacks,
        }


# --- request / response models ---------------------------------------------


class SoftSignals(BaseModel):
    model_config = ConfigDict(extra="forbid")
    is_screen_shared: bool = False
    is_voice_active: bool = False
    # Only "escalated" / "payment_demand" count as a soft signal (policy.CHAT_SIGNAL_STAGES).
    chat_stage: Literal["none", "escalated", "payment_demand"] = "none"


class ScoreContext(BaseModel):
    """Display-only labels for the demo screen (escaped there; never scored)."""
    model_config = ConfigDict(extra="forbid")
    scenario: Optional[str] = Field(None, max_length=32)
    payee: Optional[str] = Field(None, max_length=80)
    note: Optional[str] = Field(None, max_length=80)


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, Optional[float]] = Field(
        description=f"All {len(FEATURE_COLUMNS)} {FEATURE_CONTRACT} columns; null = missing.")
    amount_inr: float = Field(gt=0, le=10_000_000)
    soft_signals: SoftSignals = SoftSignals()
    mulehunter_status: Optional[Literal["clean", "suspicious", "mule"]] = None
    request_id: Optional[str] = Field(None, max_length=100)
    explain: bool = Field(False, description="Also return every feature's SHAP contribution (slower).")
    room: Optional[str] = Field(None, max_length=64,
                                description="Publish token from POST /v1/rooms/join: the server "
                                            "publishes this result to that demo screen.")
    context: Optional[ScoreContext] = None

    @field_validator("features")
    @classmethod
    def exact_contract(cls, v: dict) -> dict:
        missing = [c for c in FEATURE_COLUMNS if c not in v]
        unknown = sorted(set(v) - set(FEATURE_COLUMNS))
        if missing or unknown:
            raise ValueError(f"feature contract {FEATURE_CONTRACT}: missing={missing} unknown={unknown}")
        bad = [k for k, x in v.items() if x is not None and not math.isfinite(x)]
        if bad:
            raise ValueError(f"non-finite feature values (use null for missing): {bad}")
        return v


class ScoreResponse(BaseModel):
    action: Literal["release", "nudge", "hold"]
    reasons: list[str]
    auto_release_minutes: int
    requires_analyst: bool
    p_scam: float
    conformal_set: str
    fired_signals: dict
    model_version: str
    calibration_version: str
    policy_version: str
    request_id: str
    latency_ms: float
    reason_features: list[str] = Field(
        default=[], description="Base feature behind each reason, in order (for localised copy).")
    explanation: Optional[dict] = None
    room_published: Optional[bool] = Field(
        None, description="With `room`: whether the result reached the demo screen.")


class JoinRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str = Field(min_length=4, max_length=16)


class Evidence(BaseModel):
    """One evidence event for the payment-review service (see demo_scenarios.py)."""
    model_config = ConfigDict(extra="forbid")
    id: str = Field(max_length=200)
    kind: Literal["payment", "report", "edge"]
    event_at: float
    available_at: float
    recipient: Optional[str] = None
    direction: Optional[Literal["incoming", "outgoing"]] = None
    amount_paise: Optional[int] = Field(None, gt=0)
    account: Optional[str] = None
    source: Optional[str] = None
    target: Optional[str] = None

    @model_validator(mode="after")
    def fields_for_kind(self):
        need = {"payment": ("recipient", "direction", "amount_paise"),
                "report": ("account",), "edge": ("source", "target")}[self.kind]
        missing = [f for f in need if getattr(self, f) is None]
        if missing:
            raise ValueError(f"{self.kind} evidence needs {missing}")
        return self


class CreatePayment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    recipient: str
    amount_paise: int
    direction: Literal["incoming", "outgoing"] = "outgoing"


class ChangePayment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    recipient: str
    amount_paise: int


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence: list[Evidence] = []
    text: str = Field("", max_length=2000)
    consent: bool = False
    parent: Optional[str] = None


class ExecuteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    review_id: str
    confirmed: bool = False


class ApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    review_id: str


# --- error mapping -----------------------------------------------------------


def http_error(exc: ValueError) -> HTTPException:
    """Map PaymentService's ValueError messages onto HTTP status codes."""
    msg = str(exc)
    if msg.startswith("Unknown"):
        code = 404
    elif msg.startswith("Wrong payer session"):
        code = 403
    elif msg.startswith(("Recipient is required", "Amount must", "Invalid direction", "Session is required")):
        code = 422
    elif msg.startswith("Policy verifier"):
        code = 500  # internal invariant broken: never expected
    else:
        code = 409  # stale / expired / held / terminal / confirmation needed
    return HTTPException(status_code=code, detail=msg)


# --- app ---------------------------------------------------------------------


def default_scorer() -> Optional[Scorer]:
    model = Path(os.environ.get("MAYA_MODEL", ROOT / "models" / "xgb_model.json"))
    conf = Path(os.environ.get("MAYA_CONFORMAL", ROOT / "models" / "conformal.json"))
    if not (model.exists() and conf.exists()):
        return None
    return Scorer(model, conf, model.with_name("model_meta.json"))


def default_service() -> Optional[payment_review.PaymentService]:
    model = Path(os.environ.get("MAYA_MODEL", ROOT / "models" / "xgb_model.json"))
    conf = Path(os.environ.get("MAYA_CONFORMAL", ROOT / "models" / "conformal.json"))
    if not (model.exists() and conf.exists()):
        return None
    adapter = payment_review.ModelAdapter(str(model), str(conf))
    return payment_review.PaymentService(adapter, os.environ.get("MAYA_REVIEW_DB", ":memory:"))


def create_app(scorer=None, service=None, load_defaults: bool = True, rooms=None,
               rate_limits: Optional[dict] = None) -> FastAPI:
    """Build the API. Tests inject a fake `scorer` / `service`; by default both are
    loaded from models/ (and stay None, i.e. not ready, when the models are absent).

    `rate_limits` maps "score" / "rooms" / "join" to requests per minute per client
    (MAYA_RATE_LIMIT=0 turns limiting off). Behind a reverse proxy set
    MAYA_TRUST_PROXY=1 so the client is taken from X-Forwarded-For.
    """
    if load_defaults:
        scorer = scorer if scorer is not None else default_scorer()
        service = service if service is not None else default_service()
    rooms = rooms if rooms is not None else Rooms()
    limits_on = os.environ.get("MAYA_RATE_LIMIT", "1") != "0"
    limits = {"score": 120, "rooms": 20, "join": 30, **(rate_limits or {})}
    limiters = {k: RateLimiter(v) for k, v in limits.items()}
    trust_proxy = os.environ.get("MAYA_TRUST_PROXY") == "1"
    streams = {"open": 0}
    MAX_STREAMS = 200

    def limit(kind: str, request: Request) -> None:
        if not limits_on:
            return
        client = request.client.host if request.client else "unknown"
        if trust_proxy:
            client = request.headers.get("x-forwarded-for", client).split(",")[0].strip() or client
        if not limiters[kind].allow(client):
            raise HTTPException(429, "Too many requests; slow down a little", headers={"Retry-After": "5"})

    def room_call(fn, *args):
        try:
            return fn(*args)
        except RoomError as exc:
            raise HTTPException(exc.status, str(exc)) from exc

    app = FastAPI(
        title="Maya risk-scoring API",
        version=API_VERSION,
        description="Pre-payment scam risk scoring for UPI (local simulator; synthetic data).",
    )

    def need_scorer():
        if scorer is None:
            raise HTTPException(503, "model not loaded (build models/ first)")
        return scorer

    def need_service():
        if service is None:
            raise HTTPException(503, "payment review service not loaded (build models/ first)")
        return service

    @app.get("/v1/health")
    def health():
        ready = scorer is not None and service is not None
        return {"status": "ok" if ready else "not_ready", "api_version": API_VERSION,
                "model_loaded": scorer is not None, "review_service": service is not None}

    @app.get("/v1/model")
    def model_info():
        s = need_scorer()
        return {"api_version": API_VERSION, "policy_version": POLICY_VERSION,
                "feature_contract": {"name": FEATURE_CONTRACT, "columns": FEATURE_COLUMNS},
                "thresholds": asdict(DEFAULT_THRESHOLDS), "latency_slo_ms": SLO_MS,
                **s.info()}

    @app.post("/v1/score", response_model=ScoreResponse)
    def score(req: ScoreRequest, request: Request):
        limit("score", request)
        s = need_scorer()
        t0 = time.perf_counter()
        values = {c: (math.nan if req.features[c] is None else float(req.features[c]))
                  for c in FEATURE_COLUMNS}
        soft = req.soft_signals.model_dump()
        model_out = s.score(values, explain=req.explain)
        decision = decide(model_out["conformal_set"], values, soft, model_out["reasons"],
                          req.amount_inr, DEFAULT_THRESHOLDS, req.mulehunter_status)
        fired = active_signals(values, soft, DEFAULT_THRESHOLDS, req.mulehunter_status,
                               model_out["conformal_set"])
        response = {
            **decision,
            "p_scam": model_out["p_scam"],
            "conformal_set": model_out["conformal_set"],
            "fired_signals": fired,
            "model_version": s.model_version,
            "calibration_version": s.calibration_version,
            "policy_version": POLICY_VERSION,
            "request_id": req.request_id or uuid.uuid4().hex,
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
            "reason_features": model_out.get("reason_features", []),
            "explanation": model_out.get("explanation"),
        }
        if req.room:
            # The server publishes its own result: a phone can't post a decision.
            event = {k: response[k] for k in (
                "action", "reasons", "auto_release_minutes", "requires_analyst", "p_scam",
                "conformal_set", "fired_signals", "reason_features", "explanation",
                "latency_ms", "model_version", "policy_version")}
            event.update(amount_inr=req.amount_inr, soft_signals=soft,
                         context=req.context.model_dump() if req.context else {})
            try:
                rooms.publish(req.room, event)
                response["room_published"] = True
            except RoomError:
                response["room_published"] = False  # never fail the payment check itself
        return response

    # -- demo rooms: a judge screen follows payments made on phones ------------

    @app.post("/v1/rooms", status_code=201)
    def create_room(request: Request):
        limit("rooms", request)
        return room_call(rooms.create)

    @app.post("/v1/rooms/join")
    def join_room(body: JoinRequest, request: Request):
        limit("join", request)
        return room_call(rooms.join, body.code)

    @app.get("/v1/rooms/{room_id}")
    def room_status(room_id: str, watch: str = Query(..., max_length=64)):
        return room_call(rooms.status, room_id, watch)

    @app.get("/v1/rooms/{room_id}/events")
    async def room_events(room_id: str, request: Request, watch: str = Query(..., max_length=64),
                          after: int = Query(0, ge=0), format: Literal["sse", "json"] = "sse"):
        """Live events for the judge view (read-only). `format=json` is a polling
        fallback for proxies that buffer Server-Sent Events."""
        try:  # a reconnecting EventSource resumes after the last event it saw
            cursor = max(after, int(request.headers.get("last-event-id", after)))
        except ValueError:
            cursor = after
        start = room_call(rooms.events, room_id, watch, cursor)  # also checks the token
        if format == "json":
            return {"events": start, "status": room_call(rooms.status, room_id, watch)}
        if streams["open"] >= MAX_STREAMS:
            raise HTTPException(503, "Too many live screens; use format=json polling",
                                headers={"Retry-After": "5"})

        async def stream():
            nonlocal cursor
            streams["open"] += 1
            try:
                pending, quiet = start, 0.0
                deadline = time.monotonic() + 30 * 60  # clients reconnect with Last-Event-ID
                while time.monotonic() < deadline:
                    if await request.is_disconnected():
                        return
                    for e in pending:
                        cursor = max(cursor, e["seq"])
                        yield f"id: {e['seq']}\nevent: payment\ndata: {json.dumps(e, allow_nan=False)}\n\n"
                    if not pending:
                        quiet += 0.5
                        if quiet >= 15:
                            quiet = 0.0
                            status = rooms.status(room_id, watch)
                            yield f"event: status\ndata: {json.dumps(status)}\n\n"
                    else:
                        quiet = 0.0
                    await asyncio.sleep(0.5)
                    try:
                        pending = rooms.events(room_id, watch, cursor)
                    except RoomError:
                        yield "event: end\ndata: {}\n\n"
                        return
            finally:
                streams["open"] -= 1

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/v1/demo/scenarios")
    def demo_scenarios():
        """The four demo scenarios: story (en/hi), the 27 features, the payer's
        soft signals and the mock MuleHunter verdict, ready for /v1/score."""
        return [
            {"key": key, **SCENARIO_STORY[key],
             "amount_inr": sc["amount_inr"], "source_txn": sc["source_txn"],
             "features": sc["features"], "soft_signals": sc["soft"],
             "mulehunter_status": check_mulehunter(sc["account_id"]),
             "expected": {"action": sc["expected"][0], "requires_analyst": sc["expected"][1]}}
            for key, sc in SCENARIO_FEATURES.items()
        ]

    # -- stateful payment review --------------------------------------------

    def call(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ValueError as exc:
            raise http_error(exc) from exc

    @app.post("/v1/payments", status_code=201)
    def create_payment(body: CreatePayment, x_payer_session: str = Header(...)):
        return call(need_service().create, x_payer_session, body.recipient,
                    body.amount_paise, body.direction)

    @app.get("/v1/payments/{payment_id}")
    def get_payment(payment_id: str, x_payer_session: str = Header(...)):
        return call(need_service().get, payment_id, x_payer_session)

    @app.patch("/v1/payments/{payment_id}")
    def change_payment(payment_id: str, body: ChangePayment, x_payer_session: str = Header(...)):
        return call(need_service().change, payment_id, x_payer_session, body.recipient,
                    body.amount_paise)

    @app.post("/v1/payments/{payment_id}/reviews", status_code=201)
    def review_payment(payment_id: str, body: ReviewRequest, x_payer_session: str = Header(...)):
        evidence = [e.model_dump(exclude_none=True) for e in body.evidence]
        return call(need_service().review, payment_id, x_payer_session, evidence,
                    body.text, body.consent, body.parent)

    @app.get("/v1/reviews/{review_id}")
    def get_review(review_id: str, recompute: bool = Query(False),
                   x_payer_session: str = Header(...)):
        return call(need_service().replay, review_id, x_payer_session, recompute)

    @app.post("/v1/payments/{payment_id}/execute")
    def execute_payment(payment_id: str, body: ExecuteRequest, x_payer_session: str = Header(...)):
        return call(need_service().execute, payment_id, x_payer_session, body.review_id,
                    body.confirmed)

    @app.post("/v1/payments/{payment_id}/cancel")
    def cancel_payment(payment_id: str, x_payer_session: str = Header(...)):
        call(need_service().cancel, payment_id, x_payer_session)
        return call(need_service().get, payment_id, x_payer_session)

    @app.post("/v1/payments/{payment_id}/analyst-approval")
    def analyst_approval(payment_id: str, body: ApprovalRequest, x_payer_session: str = Header(...)):
        call(need_service().approve_by_analyst, payment_id, x_payer_session, body.review_id)
        return {"payment_id": payment_id, "review_id": body.review_id, "approved": True}

    # -- frontends -------------------------------------------------------------
    mounts = {"/app": "user", "/demo": "judge", "/static": "shared"}
    if all((FRONTEND / d).is_dir() for d in mounts.values()):
        for route, folder in mounts.items():
            app.mount(route, StaticFiles(directory=FRONTEND / folder, html=True), name=folder)

        @app.get("/", include_in_schema=False)
        def root():
            return RedirectResponse("/demo/")

    return app


app = create_app()
