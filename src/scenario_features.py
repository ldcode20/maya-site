"""The four demo scenarios' feature rows, copied from real test-split transactions.

Shared by ui/app.py (the Streamlit demo), scripts/demo_bank_client.py and
tests/test_api.py so all three score exactly the same rows. Values are the
source transaction's 27 features rounded to 4 dp; `amount_inr` is that
transaction's amount. If the model or features change, re-pick the rows so the
model-implied conformal sets still match `conf_set` (tests/test_ui.py checks).
"""

from __future__ import annotations

BASES = ("n_t", "e_t", "g_t", "z_t", "pi_t", "d_R", "v_t", "zc_t", "ptr_t")


def real_features(comb: tuple, snd: tuple, rcv: tuple) -> dict:
    """27-col feature dict from (n_t, e_t, g_t, z_t, pi_t, d_R, v_t, zc_t, ptr_t)
    combined / payer / payee."""
    feats = dict(zip(BASES, comb))
    feats.update({f"{b}_snd": v for b, v in zip(BASES, snd)})
    feats.update({f"{b}_rcv": v for b, v in zip(BASES, rcv)})
    return feats


# key -> source txn, amount, features, the scenario's soft signals and mock
# MuleHunter account, the scripted conformal set, and the Two-Signal outcome.
#
# These rows were picked for one specific trained model. Tests assert the
# scripted outcomes only when models/xgb_model.json has this sha256; after a
# retrain they skip and ask for a re-pick (python scripts/pick_scenarios.py).
PICKED_FOR_MODEL_SHA256 = "de9181e37fa52d37a970019f92a3e066dfa5f547527d586257d0dab5e653365b"


def model_matches(model_path) -> bool:
    """True when `model_path` is the model these scenario rows were picked for."""
    import hashlib
    from pathlib import Path

    p = Path(model_path)
    return p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == PICKED_FOR_MODEL_SHA256


SCENARIO_FEATURES = {
    "normal": {
        "source_txn": 2391121,
        "amount_inr": 36_307.70,
        "features": real_features(
            (38.0, 0.2644, 1.0, 4.6623, 0.0, 3.0, 2.0, 5.7600, 0.2515),
            (38.0, 0.2644, 1.0, 4.6623, 0.0, 3.0, 2.0, 5.7600, 0.2515),
            (40.0, 0.4234, 1.0, 1.2731, 0.0, 3.0, 2.0, 2.3709, 0.0),
        ),
        "soft": {},
        "account_id": "1001",
        "conf_set": "{fine}",
        "expected": ("release", False),
    },
    "investment": {
        "source_txn": 2861134,
        "amount_inr": 196_538.18,
        "features": real_features(
            (65.0, 6.3706, 1.0, 6.4313, 0.0, 0.0, 3.0, 6.6293, 1.0892),
            (65.0, 6.3706, 1.0, 6.4313, 0.0, 1.0, 3.0, 6.6293, 1.0892),
            (41.0, 0.1964, 1.0, 6.0641, 0.0, 0.0, 3.0, 6.2621, 0.8784),
        ),
        "soft": {},
        "account_id": "2002",
        "conf_set": "{scam}",
        "expected": ("hold", True),
    },
    "police": {
        "source_txn": 2599999,
        "amount_inr": 3_592.49,
        "features": real_features(
            (57.0, 0.0117, 1.0, 2.5225, 0.0, 0.0, 0.0, 2.5225, 0.8883),
            (57.0, 0.0104, 1.0, 1.8773, 0.0, 0.0, 0.0, 1.8773, 0.8883),
            (55.0, 0.0117, 1.0, 2.5225, 0.0, 1.0, 0.0, 2.5225, 0.3235),
        ),
        "soft": {"is_screen_shared": True, "chat_stage": "payment_demand"},
        "account_id": "3003",
        "conf_set": "{fine, scam}",
        "expected": ("hold", False),
    },
    "unclear": {
        "source_txn": 2415533,
        "amount_inr": 4_729.22,
        "features": real_features(
            (40.0, 0.2209, 1.0, 2.5972, 0.0, 3.0, 4.0, 3.0886, 0.4985),
            (40.0, 0.0440, 1.0, 2.5765, 0.0, 3.0, 4.0, 3.0678, 0.4953),
            (36.0, 0.2209, 1.0, 2.5972, 0.0, 3.0, 4.0, 3.0886, 0.4985),
        ),
        "soft": {"is_voice_active": True},
        "account_id": "4004",
        "conf_set": "{fine, scam}",
        "expected": ("nudge", False),
    },
}


# Display story for each scenario, in English and Hindi (the Hindi was drafted
# for the demo and should be checked by a native speaker before public use).
# `journey` / `money_path` are illustrative story props, not model inputs.
SCENARIO_STORY = {
    "normal": {
        "label": {"en": "Rent help to Mom", "hi": "माँ को किराए के लिए"},
        "story": {"en": "A larger-than-usual transfer to a long-time contact.",
                  "hi": "एक पुराने परिचित को सामान्य से बड़ी रकम भेजी जा रही है।"},
        "payee": {"en": "Mom", "hi": "माँ"},
        "upi": "mom@okhdfcbank",
        "note": {"en": "Rent help", "hi": "किराए में मदद"},
        "phone": "9876543210",
        "txn_id": 100001,
    },
    "investment": {
        "label": {"en": "Investment scam", "hi": "निवेश धोखाधड़ी"},
        "story": {"en": "After a fake profit, the victim is asked for a large 'IPO allotment fee'.",
                  "hi": "नकली मुनाफ़े के बाद पीड़ित से बड़ी 'IPO आवंटन फ़ीस' माँगी जा रही है।"},
        "payee": {"en": "WealthGrow Investments", "hi": "WealthGrow Investments"},
        "upi": "wealthgrow@ybl",
        "note": {"en": "IPO allotment fee", "hi": "IPO आवंटन फ़ीस"},
        "phone": "9812345678",
        "txn_id": 100002,
        "journey": [
            {"day": 1, "stage": {"en": "Hook", "hi": "जाल"}, "amount_inr": 10_000.0, "direction": "sent"},
            {"day": 2, "stage": {"en": "Fake profit", "hi": "नकली मुनाफ़ा"}, "amount_inr": 5_000.0,
             "direction": "received"},
            {"day": 3, "stage": {"en": "Trust", "hi": "भरोसा"}, "amount_inr": 50_000.0, "direction": "sent"},
            {"day": 5, "stage": {"en": "Escalation", "hi": "बड़ी माँग"}, "amount_inr": 196_538.18,
             "direction": "sent"},
        ],
        "money_path": [
            {"name": {"en": "You", "hi": "आप"}, "role": {"en": "Victim", "hi": "पीड़ित"}},
            {"name": {"en": "WealthGrow", "hi": "WealthGrow"}, "role": {"en": "Entry mule", "hi": "पहला म्यूल"}},
            {"name": {"en": "Mule #7782", "hi": "म्यूल #7782"}, "role": {"en": "Layering", "hi": "परतबंदी"}},
            {"name": {"en": "Aggregator #12", "hi": "एग्रीगेटर #12"}, "role": {"en": "Pooling", "hi": "इकट्ठा करना"}},
            {"name": {"en": "Crypto exit", "hi": "क्रिप्टो निकासी"}, "role": {"en": "Cash-out", "hi": "नकद निकासी"}},
        ],
    },
    "police": {
        "label": {"en": "Fake police call", "hi": "नकली पुलिस कॉल"},
        "story": {"en": "A caller posing as CBI shares the victim's screen and demands a 'verification deposit'.",
                  "hi": "CBI बनकर कॉल करने वाला स्क्रीन शेयर करवाता है और 'सत्यापन जमा' माँगता है।"},
        "payee": {"en": "Officer R. Sharma (CBI)", "hi": "अधिकारी आर. शर्मा (CBI)"},
        "upi": "cbi.verify@upi",
        "note": {"en": "Verification deposit", "hi": "सत्यापन जमा"},
        "phone": "9000000001",
        "txn_id": 100003,
    },
    "unclear": {
        "label": {"en": "Unclear case", "hi": "अस्पष्ट मामला"},
        "story": {"en": "A buyer is on a voice call while paying a marketplace seller.",
                  "hi": "खरीदार मार्केटप्लेस विक्रेता को भुगतान करते समय वॉइस कॉल पर है।"},
        "payee": {"en": "Rohan (marketplace)", "hi": "रोहन (मार्केटप्लेस)"},
        "upi": "rohan.p2p@okicici",
        "note": {"en": "Second-hand phone", "hi": "पुराना फ़ोन"},
        "phone": "9123456789",
        "txn_id": 100004,
    },
}
