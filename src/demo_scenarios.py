"""Synthetic evidence, shared by the demo and behavioral evaluation."""
from copy import deepcopy

SCENARIOS = {
    "Rent": dict(recipient="landlord@demo", amount_paise=2_000_000, text="Pay your rent", scam=False),
    "Investment journey": dict(recipient="traders@demo", amount_paise=2_000_000,
                               text="Pay ₹20,000 to release your investment withdrawal", scam=True),
    "Refund contradiction": dict(recipient="refund@demo", amount_paise=2_000_000,
                                 text="Approve to receive your refund", scam=True),
}


def scenario(name, now):
    s = deepcopy(SCENARIOS[name])
    evidence = []
    if name == "Rent":
        evidence = [dict(id="rent-history", kind="payment", recipient=s["recipient"],
                         direction="outgoing", amount_paise=2_000_000,
                         event_at=now-86400*7, available_at=now-86400*7)]
    if name == "Investment journey":
        evidence = [
            dict(id="profit-credit", kind="payment", recipient="ramesh@demo", direction="incoming",
                 amount_paise=120_000, event_at=now-86400*4, available_at=now-86400*4),
            dict(id="earlier-payment", kind="payment", recipient="first-trader@demo", direction="outgoing",
                 amount_paise=500_000, event_at=now-86400*2, available_at=now-86400*2),
            dict(id="collector-edge", kind="edge", source="traders@demo", target="collector@demo",
                 event_at=now-86400, available_at=now-86400),
            dict(id="collector-report", kind="report", account="collector@demo",
                 event_at=now-86400, available_at=now-3600),
            dict(id="other-payer-edge", kind="edge", source="payer2@demo", target="traders@demo",
                 event_at=now-86400, available_at=now-86400),
        ]
    s["evidence"] = evidence
    return s
