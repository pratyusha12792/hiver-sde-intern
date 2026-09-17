from __future__ import annotations

import re
from dataclasses import asdict, dataclass

INTENTS = {
    "account_access_security": "Login, account recovery, compromised accounts, or identity checks.",
    "payments_refunds": "Charges, payment failures, refunds, or billing disputes.",
    "plan_management_eligibility": "Premium, Family, Duo, Student, trials, or plan changes.",
    "playback_connectivity": "Playback, downloads, offline use, networking, or audio failures.",
    "app_device_features": "App behavior, supported devices, playlists, library, or feature use.",
    "content_availability_metadata": "Missing content, artist or track metadata, or regional availability.",
    "feedback_other": "Feedback or requests that do not fit another intent.",
}

REASONS = {
    "billing_or_refund",
    "account_or_security_action",
    "privacy_or_sensitive_data",
    "low_evidence",
    "ambiguous_or_multi_intent",
    "legal_abuse_or_safety",
    "unsupported_language",
    "none",
}

_RULES = (
    ("privacy_or_sensitive_data", re.compile(r"\b(password|passcode|credit card|card number|cvv|social security|otp)\b", re.I)),
    ("account_or_security_action", re.compile(r"\b(hack(?:ed)?|compromis(?:e|ed)|stolen account|can't log in|cannot log in|locked out)\b", re.I)),
    ("billing_or_refund", re.compile(r"\b(refund|charged|chargeback|billing|payment|money back)\b", re.I)),
    ("legal_abuse_or_safety", re.compile(r"\b(lawsuit|lawyer|legal action|suicide|kill myself|threat)\b", re.I)),
)


@dataclass(frozen=True)
class Decision:
    intent: str
    draft_reply: str
    decision: str
    reason: str
    evidence_tweet_ids: list[str]
    intent_support: float
    retrieval_support: float
    model_error: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def hard_escalation_reason(text: str) -> str | None:
    for reason, pattern in _RULES:
        if pattern.search(text):
            return reason
    return None


def apply_gate(
    text: str,
    intent: str,
    requested_reason: str,
    evidence_ids: list[str],
    valid_evidence_ids: set[str],
    intent_support: float,
    retrieval_support: float,
    intent_threshold: float = 0.2,
    retrieval_threshold: float = 0.15,
) -> tuple[str, str]:
    hard_reason = hard_escalation_reason(text)
    if hard_reason:
        return "ESCALATE", hard_reason
    if intent not in INTENTS or requested_reason not in REASONS:
        return "ESCALATE", "ambiguous_or_multi_intent"
    if requested_reason != "none":
        return "ESCALATE", requested_reason
    if not evidence_ids or not set(evidence_ids).issubset(valid_evidence_ids):
        return "ESCALATE", "low_evidence"
    if intent_support < intent_threshold or retrieval_support < retrieval_threshold:
        return "ESCALATE", "low_evidence"
    return "AUTO_HANDLE", "none"
