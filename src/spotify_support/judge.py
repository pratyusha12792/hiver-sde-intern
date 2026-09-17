from __future__ import annotations

import json
import hashlib
import os
import time
from collections import Counter
from pathlib import Path

from openai import OpenAI

from .agent import ROOT
from .annotate import RATING_SAMPLE_COUNTS
from .config import api_key, load_local_env
from .retrieval import read_csv

AXES = ("groundedness", "helpfulness", "tone", "safety")
JUDGE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "support_judgment",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                **{axis: {"type": "integer", "enum": [1, 2, 3, 4, 5]} for axis in AXES},
                "reason": {"type": "string"},
            },
            "required": [*AXES, "reason"],
            "additionalProperties": False,
        },
    },
}


def _parse_score(content: str) -> dict:
    score = json.loads(content)
    if set(score) != {*AXES, "reason"} or any(not isinstance(score[axis], int) or score[axis] not in range(1, 6) for axis in AXES):
        raise ValueError("Judge returned an invalid score schema")
    if not isinstance(score["reason"], str) or not score["reason"].strip() or len(score["reason"].split()) > 8:
        raise ValueError("Judge returned an invalid reason")
    return score


def _sample_predictions(predictions: list[dict], sample_path: str | Path) -> list[dict]:
    sample_ids = {(row["system"], row["tweet_id"]) for row in read_csv(sample_path)}
    if Counter(system for system, _ in sample_ids) != Counter(RATING_SAMPLE_COUNTS):
        raise ValueError("Human-rating sample must contain 15 trivial, 15 simple, and 20 main predictions")
    systems = {row["system"] for row in predictions}
    if len(systems) != 1:
        raise ValueError("Judge input must contain exactly one system")
    system = systems.pop()
    selected = [row for row in predictions if (system, row["tweet_id"]) in sample_ids]
    if len(selected) != RATING_SAMPLE_COUNTS[system]:
        raise ValueError(f"Predictions do not cover the complete {system} human-rating sample")
    return selected


def judge_predictions(prediction_path: str | Path, index_path: str | Path, output_path: str | Path, sample_path: str | Path | None = None) -> int:
    from .evaluate import read_jsonl
    from .retrieval import load_indexes

    predictions = read_jsonl(prediction_path)
    if sample_path:
        predictions = _sample_predictions(predictions, sample_path)
    evidence_rows = load_indexes(index_path)["reply"].rows
    evidence = {row["tweet_id"]: row for row in evidence_rows}
    load_local_env()
    client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1"), api_key=api_key(), max_retries=0)
    model = os.getenv("JUDGE_MODEL", "openai/gpt-oss-20b")
    delay = float(os.getenv("JUDGE_DELAY_SECONDS", "2.1"))
    token_budget = int(os.getenv("JUDGE_TPM_BUDGET", "7000"))
    prompt = (ROOT / "prompts/judge.txt").read_text(encoding="utf-8")
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    existing = {(row["system"], row["tweet_id"]): row for row in read_jsonl(output_path)}
    if existing and any(row.get("prompt_sha256") != prompt_hash or row.get("model") != model for row in existing.values()):
        raise ValueError(f"Cached judge scores use a different model or prompt; remove {output_path} before regenerating")
    judged = 0
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(output_path).open("a", encoding="utf-8") as target:
        for prediction in predictions:
            if (prediction["system"], prediction["tweet_id"]) in existing:
                continue
            payload = {
                "message": prediction["text"],
                "reply": prediction["draft_reply"],
                "decision": prediction["decision"],
                "escalation_reason": prediction["reason"],
                "evidence": [[evidence[value]["text"], evidence[value]["reply"]] for value in prediction["evidence_tweet_ids"] if value in evidence],
            }
            payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                seed=42,
                reasoning_effort="low",
                max_completion_tokens=256,
                response_format=JUDGE_RESPONSE_FORMAT,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": payload_json},
                ],
            )
            score = _parse_score(response.choices[0].message.content)
            score.update({"tweet_id": prediction["tweet_id"], "system": prediction["system"], "model": model, "prompt_sha256": prompt_hash})
            if response.usage:
                score.update({"prompt_tokens": response.usage.prompt_tokens, "completion_tokens": response.usage.completion_tokens, "total_tokens": response.usage.total_tokens})
            score["pass"] = score["groundedness"] >= 4 and score["safety"] >= 4 and sum(score[name] for name in AXES) / 4 >= 4
            target.write(json.dumps(score, ensure_ascii=False) + "\n")
            target.flush()
            judged += 1
            used_tokens = response.usage.total_tokens if response.usage else len((prompt + payload_json).encode()) + 128
            time.sleep(max(delay, used_tokens * 60 / token_budget))
    return judged
