"""Maya ecosystem integrations (deterministic mocks).

Mock connectors for the Indian fraud ecosystem referenced in the Maya PDF, so the
policy engine and UI can exercise the real deployment wiring without live APIs:

    check_fri(phone)        -> DoT Financial Fraud Risk Indicator (FRI) score
    check_mulehunter(acct)  -> MuleHunter.AI account status
    file_1930_report(...)   -> National Cyber Crime Reporting Portal (NCRP) filing

All functions are pure and deterministic (sha256 hashing, no network, no data
files), so they are safe to call from policy.py and ui/app.py.

Run the demo:  python3 src/integrations.py
"""

from __future__ import annotations

import hashlib
import time
import uuid


def _digest(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h, 16)


def check_fri(phone_number) -> dict:
    """Mock DoT Financial Fraud Risk Indicator.

    Returns a 0-100 score and a risk band. Deterministic for a given number.
    """
    phone = str(phone_number)
    score = _digest("fri", phone) % 101
    if score >= 70:
        band = "HIGH"
    elif score >= 35:
        band = "MEDIUM"
    else:
        band = "LOW"
    return {"phone": phone, "fri_score": score, "risk_band": band}


# Known accounts the UI scenarios rely on for deterministic outcomes, plus a
# hashed fallback for everything else.
KNOWN_ACCOUNTS = {
    # scenario 1 (normal): clean
    "1001": "clean",
    # scenario 2 (investment scam): mule
    "2002": "mule",
    # scenario 3 (fake police): mule
    "3003": "mule",
    # scenario 4 (unclear edge): clean (keeps it a nudge)
    "4004": "clean",
}
_STATUSES = ("clean", "suspicious", "mule")


def check_mulehunter(account_id) -> str:
    """Mock MuleHunter.AI status for a receiving account."""
    key = str(account_id)
    if key in KNOWN_ACCOUNTS:
        return KNOWN_ACCOUNTS[key]
    return _STATUSES[_digest("mulehunter", key) % len(_STATUSES)]


def file_1930_report(transaction_id, amount) -> dict:
    """Simulate filing a complaint with the NCRP (dial 1930)."""
    txn = str(transaction_id)
    ref = f"NCRP-{int(_digest('ncrp', txn, amount) % 10_000_000):07d}"
    return {
        "status": "filed",
        "ref_id": ref,
        "transaction_id": txn,
        "amount": float(amount),
        "portal": "National Cyber Crime Reporting Portal (1930)",
        "filed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "internal_id": uuid.uuid4().hex[:12],
    }


def _demo() -> None:
    print("[integrations] mock ecosystem self-demo")
    for phone in ("9876543210", "9000000001", "9812345678"):
        print(f"  check_fri({phone}) -> {check_fri(phone)}")
    for acct in ("1001", "2002", "3003", "4004", "9999"):
        print(f"  check_mulehunter({acct}) -> {check_mulehunter(acct)!r}")
    rep = file_1930_report(2002885, 200000.0)
    print(f"  file_1930_report(2002885, 200000) -> {rep}")


if __name__ == "__main__":
    _demo()
