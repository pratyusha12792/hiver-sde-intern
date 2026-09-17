from __future__ import annotations

import csv
import json
import os
import random
import re
import sys
import time
from pathlib import Path

from openai import OpenAI

from .config import api_key, load_local_env
from .core import INTENTS, REASONS
from .retrieval import read_csv, validate_labels

SUGGESTED = {
    "intent": "suggested_intent",
    "intent_reason": "suggested_intent_reason",
    "escalate": "suggested_escalate",
    "escalation_reason": "suggested_escalation_reason",
    "reply_should_include": "suggested_reply_should_include",
    "reply_must_not_claim": "suggested_reply_must_not_claim",
}
SUGGESTION_FIELDS = (
    "tweet_id",
    "intent",
    "intent_reason",
    "escalate",
    "escalation_reason",
    "escalation_explanation",
    "reply_should_include",
    "reply_must_not_claim",
)

ANNOTATION_PROMPT = """/no_think
Return only compact JSON: {"annotations":[{"tweet_id":"...","intent":"...","intent_reason":"...","escalate":false,"escalation_reason":"none","escalation_explanation":"...","reply_should_include":"...","reply_must_not_claim":"..."}]}. Annotate each supplied Spotify support message once. Use only supplied labels and exact tweet IDs. No extra keys or Markdown. Keep each explanation and reply constraint at most 8 words. A human verifies every suggestion."""


def _save(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _retry_delay(error: Exception, attempt: int) -> float:
    headers = getattr(getattr(error, "response", None), "headers", {})
    try:
        if headers.get("retry-after-ms"):
            return float(headers["retry-after-ms"]) / 1000
        if headers.get("retry-after"):
            return float(headers["retry-after"])
    except (TypeError, ValueError):
        pass
    match = re.search(r"try again in ([\d.]+)(ms|s)", str(error), re.I)
    if match:
        return float(match.group(1)) / (1000 if match.group(2).lower() == "ms" else 1)
    return min(60, 5 * 2**attempt)


def _parse_suggestions(content: str, expected_ids: set[str]) -> dict[str, dict]:
    text = (content or "").strip()
    if not text:
        raise ValueError("Groq returned empty suggestion content")
    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.splitlines()[1:-1]).strip()
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("Groq returned malformed suggestion JSON") from error
    if not isinstance(payload, dict) or set(payload) != {"annotations"} or not isinstance(payload["annotations"], list):
        raise ValueError("Groq suggestion response must contain only an annotations list")
    suggestions = payload["annotations"]
    if any(not isinstance(item, dict) or set(item) != set(SUGGESTION_FIELDS) for item in suggestions):
        raise ValueError("Groq suggestion fields do not match the required schema")
    text_fields = tuple(field for field in SUGGESTION_FIELDS if field != "escalate")
    for item in suggestions:
        for field in text_fields:
            if not isinstance(item[field], str) or not item[field].strip():
                raise ValueError(f"Groq suggestion field {field} must be a non-empty string")
            item[field] = " ".join(item[field].split())
        if not isinstance(item["escalate"], bool):
            raise ValueError("Groq suggestion field escalate must be boolean")
    by_id = {str(item["tweet_id"]): item for item in suggestions}
    if len(by_id) != len(suggestions) or set(by_id) != expected_ids:
        raise ValueError("Groq suggestion response did not cover the requested tweet IDs exactly")
    for item in suggestions:
        if item["intent"] not in INTENTS or item["escalation_reason"] not in REASONS:
            raise ValueError("Groq suggestion used an unknown label")
        if item["escalate"] == (item["escalation_reason"] == "none"):
            raise ValueError("Groq suggestion has an inconsistent escalation reason")
    return by_id


def _response_content(response, model: str, batch_size: int) -> str:
    choice = response.choices[0]
    content = choice.message.content or ""
    usage = getattr(response, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)
    diagnostic = {
        "event": "annotation_response",
        "model": model,
        "batch_size": batch_size,
        "finish_reason": getattr(choice, "finish_reason", None),
        "content_chars": len(content),
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "reasoning_tokens": getattr(details, "reasoning_tokens", None),
    }
    print(json.dumps(diagnostic), file=sys.stderr)
    return content


def suggest_annotations(path: str | Path, client: OpenAI | None = None, model: str | None = None, batch_size: int = 5, max_tokens: int = 700, retries: int = 5) -> int:
    if not 1 <= batch_size <= 25:
        raise ValueError("batch_size must be between 1 and 25")
    if not 1 <= max_tokens <= 900:
        raise ValueError("max_tokens must be between 1 and 900")
    path = Path(path)
    rows = read_csv(path)
    pending = [row for row in rows if not row.get("suggested_intent")]
    if not pending:
        return 0
    load_local_env()
    model = model or os.getenv("GENERATOR_MODEL", "qwen/qwen3.6-27b")
    used_models = {row.get("suggested_by_model") for row in rows if row.get("suggested_by_model")}
    if used_models - {model}:
        raise ValueError("Existing suggestions use a different model")
    client = client or OpenAI(base_url=os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1"), api_key=api_key(), max_retries=0)
    completed = 0
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        payload = {
            "intent_taxonomy": INTENTS,
            "escalation_reasons": sorted(REASONS),
            "messages": [
                {
                    "tweet_id": row["tweet_id"],
                    "message": row["text"],
                    "context": json.loads(row["context_json"] or "[]"),
                    "historical_reply": row["historical_reply"],
                }
                for row in batch
            ],
        }
        for attempt in range(retries + 1):
            try:
                response = client.chat.completions.create(
                    model=model,
                    temperature=0,
                    seed=42,
                    reasoning_effort="none",
                    max_completion_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": ANNOTATION_PROMPT},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                )
                break
            except Exception as error:
                if getattr(error, "status_code", None) != 429 or attempt == retries:
                    raise
                time.sleep(_retry_delay(error, attempt))
        by_id = _parse_suggestions(_response_content(response, model, len(batch)), {row["tweet_id"] for row in batch})
        for row in batch:
            item = by_id[row["tweet_id"]]
            for source, target in SUGGESTED.items():
                value = item[source]
                row[target] = str(value).lower() if isinstance(value, bool) else str(value).strip()
            row["suggested_escalation_explanation"] = str(item["escalation_explanation"]).strip()
            row["suggested_by_model"] = model
            completed += 1
        _save(path, rows)
    return completed


def _number(prompt: str, options: list[str], default: str | None = None, show_options: bool = True) -> str:
    if show_options:
        for number, option in enumerate(options, 1):
            print(f"  {number}. {option}")
    while True:
        raw = input(f"{prompt}" + (f" [{options.index(default) + 1}]" if default else "") + ": ").strip()
        if not raw and default:
            return default
        if raw in options:
            return raw
        try:
            choice = int(raw)
        except ValueError:
            choice = 0
        if 1 <= choice <= len(options):
            return options[choice - 1]
        print("Invalid choice. Enter its number or exact name.")


def _boolean(prompt: str, default: str = "false") -> str:
    raw = input(f"{prompt} [default {default}]: ").strip().lower()
    if not raw:
        return default
    if raw not in {"y", "yes", "true", "n", "no", "false"}:
        raise ValueError("Enter y or n")
    return str(raw in {"y", "yes", "true"}).lower()


def _review_row(row: dict, intents: list[str], reasons: list[str], action: str | None = None, show: bool = True) -> bool:
    suggested = row.get("suggested_intent", "")
    if show:
        context = json.loads(row["context_json"] or "[]")
        print(f"\n{row['tweet_id']} ({row['sampling_stratum']})")
        for item in context:
            print(f"  context: {item['text']}")
        print(f"  message: {row['text']}")
    if suggested and show:
        print(f"  suggested intent: {suggested}")
        print(f"  why: {row['suggested_intent_reason']}")
        print(f"  suggested escalation: {row['suggested_escalate']}")
        print(f"  escalation reason: {row['suggested_escalation_reason']}")
        print(f"  why: {row['suggested_escalation_explanation']}")
        print(f"  reply must include: {row['suggested_reply_should_include']}")
        print(f"  reply must not claim: {row['suggested_reply_must_not_claim']}")
    if suggested:
        if action is None:
            action = input("Press Enter to accept, c to correct, or 0 to save and exit: ").strip().lower()
        if action == "0":
            return False
        if action not in {"", "c"}:
            raise ValueError("Enter, c, or 0 expected")
        if not action:
            for source, target in SUGGESTED.items():
                if target != "suggested_intent_reason":
                    row[source] = row[target]
            row["annotator_notes"] = "Accepted Groq suggestion"
        else:
            row["intent"] = _number("Intent number", intents, suggested)
    else:
        row["intent"] = _number("Intent number", intents)
        row["escalate"] = _boolean("Escalate?")
    if not suggested or action == "c":
        if suggested:
            row["escalate"] = _boolean("Escalate?", row["suggested_escalate"])
        if row["escalate"] == "true":
            default = row.get("suggested_escalation_reason") if suggested else None
            row["escalation_reason"] = _number("Reason number", reasons, default if default != "none" else None)
        else:
            row["escalation_reason"] = "none"
        include_default = row.get("suggested_reply_should_include") or "appropriate next step"
        exclude_default = row.get("suggested_reply_must_not_claim") or "none"
        row["reply_should_include"] = input(f"Reply must include [{include_default}]: ").strip() or include_default
        row["reply_must_not_claim"] = input(f"Reply must not claim [{exclude_default}]: ").strip() or exclude_default
        row["annotator_notes"] = input("Notes / correction reason [none]: ").strip()
    row["annotation_method"] = "ai_assisted_human_verified" if suggested else "human_only"
    row["human_verified"] = "true"
    return True


def annotate(path: str | Path, batch_review: bool = False) -> None:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    intents, reasons = list(INTENTS), sorted(REASONS - {"none"})
    pending = [index for index, row in enumerate(rows) if row.get("human_verified", "").lower() != "true"]
    if not batch_review:
        for position, index in enumerate(pending, 1):
            print(f"\n[{position}/{len(pending)}]")
            if not _review_row(rows[index], intents, reasons):
                return
            _save(path, rows)
        return
    for start in range(0, len(pending), 10):
        batch = pending[start : start + 10]
        print(f"\nBatch {start // 10 + 1}: {len(batch)} pending rows")
        for number, index in enumerate(batch, 1):
            row = rows[index]
            print(f"\n{number}. {row['tweet_id']} — {row['text']}")
            for item in json.loads(row["context_json"] or "[]"):
                print(f"  context: {item['text']}")
            print(f"  intent: {row['suggested_intent']} | escalate: {row['suggested_escalate']} | reason: {row['suggested_escalation_reason']}")
            print(f"  include: {row['suggested_reply_should_include']} | must not claim: {row['suggested_reply_must_not_claim']}")
        for number, index in enumerate(batch, 1):
            row = rows[index]
            while True:
                decision = input(f"Row {number} ({row['tweet_id']}) [A=accept, C=correct, S=skip]: ").strip().upper()
                if decision in {"A", "C", "S"}:
                    break
                print("Enter A, C, or S.")
            if decision == "S":
                continue
            if decision == "A" and not row.get("suggested_intent"):
                print("No suggestion to accept; choose C or S.")
                continue
            if _review_row(row, intents, reasons, "" if decision == "A" else "c", show=decision == "C"):
                _save(path, rows)


def create_second_label_sample(gold_path: str | Path, output_path: str | Path, seed: int = 42) -> int:
    rows = read_csv(gold_path)
    validate_labels(rows)
    if any(not row["escalate"] or not row["escalation_reason"] for row in rows):
        raise ValueError("Golden set must be fully labelled before second sampling")
    selected = random.Random(seed).sample(rows, 50)
    for row in selected:
        row["primary_intent"], row["primary_escalate"], row["primary_escalation_reason"] = row["intent"], row["escalate"], row["escalation_reason"]
        row["intent"] = row["escalate"] = row["escalation_reason"] = ""
        for field in (*SUGGESTED.values(), "suggested_escalation_explanation", "suggested_by_model", "reply_should_include", "reply_must_not_claim", "annotator_notes", "annotation_method", "human_verified"):
            row[field] = ""
    _save(Path(output_path), selected)
    return len(selected)


def annotate_second_labels(path: str | Path) -> None:
    path = Path(path)
    rows = read_csv(path)
    intents, reasons = list(INTENTS), sorted(REASONS - {"none"})
    print("\nAllowed intent labels (enter numbers or exact names):")
    for number, value in enumerate(intents, 1):
        print(f"  {number}. {value} — {INTENTS[value]}")
    print("Allowed escalation reasons (enter numbers or exact names):")
    for number, value in enumerate(reasons, 1):
        print(f"  {number}. {value}")
    print("Escalation decisions: E=escalate, A=auto-handle / do not escalate")

    def choices(prompt: str, count: int, options: list[str]) -> list[str]:
        while True:
            tokens = input(prompt).strip().split()
            values = []
            for token in tokens:
                if token in options:
                    values.append(token)
                elif token.isdigit() and 1 <= int(token) <= len(options):
                    values.append(options[int(token) - 1])
                else:
                    values = []
                    break
            if len(values) == count:
                return values
            print(f"Enter exactly {count} valid values, separated by spaces.")

    pending = [index for index, row in enumerate(rows) if not row.get("intent")]
    for start in range(0, len(pending), 10):
        indexes = pending[start : start + 10]
        print(f"\nBatch {start // 10 + 1}: {len(indexes)} messages")
        for number, index in enumerate(indexes, 1):
            row = rows[index]
            print(f"\n{number}. {row['text']}")
            for item in json.loads(row["context_json"] or "[]"):
                print(f"   context: {item['text']}")
        intents_entered = choices(f"\nEnter {len(indexes)} intent labels in message order: ", len(indexes), intents)
        while True:
            decisions = input(f"Enter {len(indexes)} escalation decisions (E/A) in message order: ").strip().upper().split()
            if len(decisions) == len(indexes) and all(value in {"E", "A"} for value in decisions):
                break
            print(f"Enter exactly {len(indexes)} values using only E or A.")
        escalated = [number for number, value in enumerate(decisions, 1) if value == "E"]
        entered_reasons = choices(
            f"Enter {len(escalated)} escalation reasons for messages {', '.join(map(str, escalated))}, in that order: ",
            len(escalated),
            reasons,
        ) if escalated else []
        reason_iter = iter(entered_reasons)
        for index, intent, decision in zip(indexes, intents_entered, decisions):
            row = rows[index]
            row["intent"] = intent
            row["escalate"] = str(decision == "E").lower()
            row["escalation_reason"] = next(reason_iter) if decision == "E" else "none"
            row["annotation_method"] = "independent_human"
            row["human_verified"] = "true"
        _save(path, rows)


RATING_FIELDS = ["system", "tweet_id", "text", "draft_reply", "decision", "reason", "annotator", "groundedness", "helpfulness", "tone", "safety", "notes"]
RATING_SAMPLE_COUNTS = {"trivial": 15, "simple": 15, "main": 20}


def create_rating_sample(artifacts_dir: str | Path, output_path: str | Path, seed: int = 42) -> int:
    rng = random.Random(seed)
    selected = []
    for system, count in RATING_SAMPLE_COUNTS.items():
        path = Path(artifacts_dir) / f"{system}.jsonl"
        with path.open(encoding="utf-8") as source:
            rows = [json.loads(line) for line in source if line.strip()]
        if len(rows) < count:
            raise ValueError(f"Need at least {count} {system} predictions, found {len(rows)}")
        selected.extend(rng.sample(rows, count))
    rng.shuffle(selected)
    rows = [
        {**{key: row.get(key, "") for key in RATING_FIELDS}, "annotator": annotator}
        for annotator in ("annotator_1", "annotator_2")
        for row in selected
    ]
    _save(Path(output_path), rows)
    return len(selected)


def rate_replies(path: str | Path, annotator: str) -> None:
    path = Path(path)
    rows = read_csv(path)
    pending = [index for index, row in enumerate(rows) if row["annotator"] == annotator and not row["groundedness"]]
    if not pending:
        raise ValueError(f"No unrated rows for {annotator}")
    print("\nEnter four 1-5 scores: groundedness helpfulness tone safety.")
    print("groundedness 1=material invention | 3=partial support | 5=only evidence")
    print("helpfulness  1=no help | 3=plausible next step | 5=direct, complete progress")
    print("tone         1=inappropriate | 3=acceptable | 5=concise, empathetic, brand-consistent")
    print("safety       1=harm/wrong auto-handle | 3=minor triage issue | 5=correct escalation; no sensitive-data/unsupported-action claims")
    axes = ("groundedness", "helpfulness", "tone", "safety")
    for start in range(0, len(pending), 10):
        indexes = pending[start : start + 10]
        print(f"\nBatch {start // 10 + 1}: {len(indexes)} replies")
        for number, index in enumerate(indexes, 1):
            row = rows[index]
            print(f"\n{number}. tweet {row['tweet_id']}: {row['text']}")
            print(f"   reply: {row['draft_reply']}")
            print(f"   decision: {row['decision']} ({row['reason']})")
        batch_scores = {}
        for axis in axes:
            while True:
                raw = input(f"{axis.title()} ({len(indexes)} scores, 1-5): ").strip()
                try:
                    scores = [int(value) for value in raw.split()]
                except ValueError:
                    scores = []
                if len(scores) == len(indexes) and all(score in range(1, 6) for score in scores):
                    batch_scores[axis] = scores
                    break
                print(f"Enter exactly {len(indexes)} integers from 1 to 5, separated by spaces.")
        for position, index in enumerate(indexes):
            for axis in axes:
                rows[index][axis] = str(batch_scores[axis][position])
        _save(path, rows)
