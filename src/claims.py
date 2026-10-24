"""Narrow English demo grammar: deterministic, consented, and abstaining.

Exact whole-message matching deliberately avoids extracting confident claims
from negations, quotations, or mixed instructions outside the supported grammar.
"""
import re

PATTERNS = (
    (r"approve to receive (?:a |your )?refund[.!]?", "incoming", "refund", False),
    (r"pay (?:₹|rs\.?\s*)?[\d,]+ to release your investment withdrawal[.!]?",
     "outgoing", "investment_withdrawal", True),
    (r"pay (?:the agreed )?booking deposit[.!]?", "outgoing", "booking_deposit", False),
    (r"pay (?:your |the )?rent[.!]?", "outgoing", "rent", False),
    (r"send (?:the |an? )?refund to (?:the |your )?customer[.!]?", "outgoing", "merchant_refund", False),
)


def extract_claim(text, consent=False):
    result = dict(status="unclear", direction=None, purpose=None,
                  future_benefit_unverified=False, amount_paise=None, span=None)
    if not consent:
        return {**result, "status": "not_shared"}
    if not isinstance(text, str) or len(text) > 2000:
        return result
    stripped = text.strip()
    for pattern, direction, purpose, future in PATTERNS:
        if re.fullmatch(pattern, stripped, re.IGNORECASE):
            start = text.index(stripped)
            amount = re.search(r"[\d,]+", stripped)
            return dict(status="extracted", direction=direction, purpose=purpose,
                        amount_paise=int(amount.group().replace(",", "")) * 100 if amount else None,
                        future_benefit_unverified=future,
                        span=dict(start=start, end=start + len(stripped), text=stripped))
    return result


def compare_claim(claim, direction):
    if claim["status"] != "extracted":
        return "unclear"
    if claim["direction"] != direction:
        return "direction_contradiction"
    if claim["future_benefit_unverified"]:
        return "unverified_future_benefit"
    return "consistent"
