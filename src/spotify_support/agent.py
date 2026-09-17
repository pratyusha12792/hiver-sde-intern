from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from openai import APIStatusError, OpenAI

from .config import api_key, load_local_env
from .core import Decision, INTENTS, REASONS, apply_gate, hard_escalation_reason
from .retrieval import load_indexes

ROOT = Path(__file__).resolve().parents[2]


def _query_text(text: str, context: list[dict] | None) -> str:
    prior = " ".join(item.get("text", "") for item in (context or []))
    return f"{prior} {text}".strip()


def _log_api_error(error: APIStatusError) -> None:
    body = error.body if isinstance(error.body, dict) else {}
    provider = body.get("error", body)
    provider = provider if isinstance(provider, dict) else {}
    allowed = {"model_not_found", "invalid_request_error", "not_found", "resource_not_found", "unknown_model", "invalid_model", "route_not_found"}
    safe = lambda value: value if isinstance(value, str) and value in allowed else None
    message = provider.get("message", "")
    message = message.lower() if isinstance(message, str) else ""
    category = "model_not_found" if "model" in message and ("not found" in message or "does not exist" in message) else "endpoint_not_found" if "endpoint" in message and "not found" in message else "unspecified"
    print(json.dumps({
        "event": "generation_api_error",
        "status": error.status_code,
        "provider_code": safe(provider.get("code")),
        "provider_type": safe(provider.get("type")),
        "message_category": category,
        "response_chars": len(error.response.content),
    }), file=sys.stderr)


class SupportAgent:
    def __init__(self, index_path: str | Path, client: OpenAI | None = None, model: str | None = None):
        load_local_env()
        self.indexes = load_indexes(index_path)
        self.client = client
        self.model = model or os.getenv("GENERATOR_MODEL", "qwen/qwen3.8-27b")

    def handle(self, text: str, context: list[dict] | None = None, system: str = "main") -> Decision:
        query = _query_text(text, context)
        intent_hits = self.indexes["intent"].query(query)
        reply_hits = self.indexes["reply"].query(query)
        intent_support = intent_hits[0]["score"]
        retrieval_support = reply_hits[0]["score"]

        if system == "trivial":
            return Decision(self.indexes["majority_intent"], "Thanks for reaching out. Please tell us more so we can help.", "ESCALATE", "low_evidence", [], intent_support, retrieval_support)
        if system == "simple":
            reason = hard_escalation_reason(text) or "none"
            evidence = [reply_hits[0]["tweet_id"]]
            decision, reason = apply_gate(text, intent_hits[0]["intent"], reason, evidence, set(evidence), intent_support, retrieval_support)
            return Decision(intent_hits[0]["intent"], reply_hits[0]["reply"], decision, reason, evidence, intent_support, retrieval_support)
        if system != "main":
            raise ValueError(f"Unknown system: {system}")
        return self._generate(text, context or [], intent_hits, reply_hits, intent_support, retrieval_support)

    def _generate(self, text: str, context: list[dict], intent_hits: list[dict], reply_hits: list[dict], intent_support: float, retrieval_support: float) -> Decision:
        valid_ids = {row["tweet_id"] for row in reply_hits}
        evidence = [
            {"tweet_id": row["tweet_id"], "customer": row["text"], "brand_reply": row["reply"]}
            for row in reply_hits
        ]
        payload = {
            "message": text,
            "context": context,
            "intent_examples": [{"message": row["text"], "intent": row["intent"]} for row in intent_hits],
            "historical_evidence": evidence,
            "allowed_intents": list(INTENTS),
            "allowed_escalation_reasons": sorted(REASONS),
        }
        client = self.client or OpenAI(base_url=os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1"), api_key=api_key())
        try:
            response = client.chat.completions.create(
                model=self.model,
                temperature=0,
                seed=42,
                reasoning_effort="none",
                max_completion_tokens=256,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": (ROOT / "prompts/agent.txt").read_text(encoding="utf-8")},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
            )
            raw = json.loads(response.choices[0].message.content)
            intent = raw.get("intent", "feedback_other")
            reason = raw.get("reason", "ambiguous_or_multi_intent")
            if intent not in INTENTS or reason not in REASONS or not isinstance(raw.get("evidence_tweet_ids"), list):
                raise ValueError("Model returned an invalid decision schema")
            evidence_ids = [str(value) for value in raw["evidence_tweet_ids"]]
            draft = str(raw.get("draft_reply", "Thanks for reaching out. A support specialist will review this."))
            model_error = None
        except APIStatusError as exc:
            _log_api_error(exc)
            if exc.status_code == 404:
                raise
            intent, reason, evidence_ids = intent_hits[0]["intent"], "low_evidence", []
            draft = "Thanks for reaching out. A support specialist will review this."
            model_error = type(exc).__name__
        except Exception as exc:
            intent, reason, evidence_ids = intent_hits[0]["intent"], "low_evidence", []
            draft = "Thanks for reaching out. A support specialist will review this."
            model_error = type(exc).__name__
        decision, reason = apply_gate(text, intent, reason, evidence_ids, valid_ids, intent_support, retrieval_support)
        return Decision(intent, draft, decision, reason, evidence_ids, intent_support, retrieval_support, model_error)
