from __future__ import annotations

import csv
import json
import random
import re
import sqlite3
from collections import defaultdict
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

REQUIRED_COLUMNS = {
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
}

LABEL_COLUMNS = [
    "tweet_id",
    "thread_id",
    "author_id",
    "created_at",
    "text",
    "context_json",
    "historical_reply",
    "sampling_stratum",
    "suggested_intent",
    "suggested_intent_reason",
    "suggested_escalate",
    "suggested_escalation_reason",
    "suggested_escalation_explanation",
    "suggested_reply_should_include",
    "suggested_reply_must_not_claim",
    "suggested_by_model",
    "intent",
    "escalate",
    "escalation_reason",
    "reply_should_include",
    "reply_must_not_claim",
    "annotator_notes",
    "annotation_method",
    "human_verified",
]


def timestamp(value: str) -> float:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return parsedate_to_datetime(value).timestamp()


def _chunks(values: set[str], size: int = 800):
    values = list(values)
    for start in range(0, len(values), size):
        yield values[start : start + size]


def prepare_twitter_csv(raw_csv: str | Path, output_jsonl: str | Path, brand: str = "SpotifyCares") -> int:
    raw_csv, output_jsonl = Path(raw_csv), Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    db_path = output_jsonl.with_suffix(".sqlite")
    if db_path.exists():
        db_path.unlink()

    with sqlite3.connect(db_path) as db, raw_csv.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")
        db.execute("CREATE TABLE tweets (tweet_id TEXT PRIMARY KEY, author_id TEXT, inbound INTEGER, created_at TEXT, text TEXT, response_ids TEXT, parent_id TEXT)")
        rows = []
        for row in reader:
            rows.append((row["tweet_id"], row["author_id"], int(row["inbound"].lower() == "true"), row["created_at"], row["text"], row["response_tweet_id"], row["in_response_to_tweet_id"]))
            if len(rows) == 10_000:
                db.executemany("INSERT OR IGNORE INTO tweets VALUES (?,?,?,?,?,?,?)", rows)
                rows.clear()
        if rows:
            db.executemany("INSERT OR IGNORE INTO tweets VALUES (?,?,?,?,?,?,?)", rows)
        db.execute("CREATE INDEX tweets_author ON tweets(author_id)")
        db.execute("CREATE INDEX tweets_parent ON tweets(parent_id)")

        relevant = {row[0] for row in db.execute("SELECT tweet_id FROM tweets WHERE author_id=?", (brand,))}
        frontier = relevant.copy()
        while frontier:
            found: set[str] = set()
            for chunk in _chunks(frontier):
                marks = ",".join("?" for _ in chunk)
                found.update(row[0] for row in db.execute(f"SELECT parent_id FROM tweets WHERE tweet_id IN ({marks}) AND parent_id != ''", chunk))
                found.update(row[0] for row in db.execute(f"SELECT tweet_id FROM tweets WHERE parent_id IN ({marks})", chunk))
            frontier = found - relevant
            relevant.update(frontier)

        count = 0
        with output_jsonl.open("w", encoding="utf-8") as target:
            for chunk in _chunks(relevant):
                marks = ",".join("?" for _ in chunk)
                rows = db.execute(f"SELECT tweet_id,author_id,created_at,text,parent_id FROM tweets WHERE inbound=1 AND tweet_id IN ({marks}) ORDER BY created_at", chunk)
                for tweet_id, author_id, created_at, text, parent_id in rows:
                    context, cursor, seen = [], parent_id, {tweet_id}
                    while cursor and cursor not in seen and len(context) < 6:
                        seen.add(cursor)
                        parent = db.execute("SELECT tweet_id,author_id,inbound,text,parent_id FROM tweets WHERE tweet_id=?", (cursor,)).fetchone()
                        if not parent:
                            break
                        context.append({"tweet_id": parent[0], "author_id": parent[1], "inbound": bool(parent[2]), "text": parent[3]})
                        cursor = parent[4]
                    context.reverse()
                    thread_id = context[0]["tweet_id"] if context else tweet_id
                    reply = db.execute("SELECT tweet_id,text FROM tweets WHERE parent_id=? AND author_id=? ORDER BY created_at LIMIT 1", (tweet_id, brand)).fetchone()
                    record = {
                        "tweet_id": tweet_id,
                        "thread_id": thread_id,
                        "author_id": author_id,
                        "created_at": created_at,
                        "text": text.strip(),
                        "context": context,
                        "brand_reply_id": reply[0] if reply else "",
                        "brand_reply": reply[1].strip() if reply else "",
                    }
                    target.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
    return count


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _stratum(record: dict) -> str:
    text = record["text"]
    if re.search(r"\b(refund|charged|hack|password|payment)\b", text, re.I):
        return "challenge_sensitive"
    if record["context"]:
        return "challenge_followup"
    if len(text) < 45:
        return "challenge_short"
    if sum(ord(char) > 127 for char in text) > max(3, len(text) // 4):
        return "challenge_language"
    return "challenge_rare"


def _write_label_csv(path: str | Path, records: list[dict], strata: dict[str, str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow({
                "tweet_id": record["tweet_id"],
                "thread_id": record["thread_id"],
                "author_id": record["author_id"],
                "created_at": record["created_at"],
                "text": record["text"],
                "context_json": json.dumps(record["context"], ensure_ascii=False),
                "historical_reply": record["brand_reply"],
                "sampling_stratum": strata[record["tweet_id"]],
                "suggested_intent": "",
                "suggested_intent_reason": "",
                "suggested_escalate": "",
                "suggested_escalation_reason": "",
                "suggested_escalation_explanation": "",
                "suggested_reply_should_include": "",
                "suggested_reply_must_not_claim": "",
                "suggested_by_model": "",
                "intent": "",
                "escalate": "",
                "escalation_reason": "",
                "reply_should_include": "",
                "reply_must_not_claim": "",
                "annotator_notes": "",
                "annotation_method": "",
                "human_verified": "",
            })


def sample_for_annotation(prepared_jsonl: str | Path, pilot_csv: str | Path, gold_csv: str | Path, seed: int = 42) -> tuple[int, int]:
    for output in (pilot_csv, gold_csv):
        path = Path(output)
        if path.exists():
            with path.open(newline="", encoding="utf-8") as source:
                if any(row.get("intent") or row.get("suggested_intent") or row.get("human_verified") for row in csv.DictReader(source)):
                    raise ValueError(f"Refusing to overwrite annotation progress in {path}")
    records = sorted(read_jsonl(prepared_jsonl), key=lambda row: timestamp(row["created_at"]))
    if len(records) < 400:
        raise ValueError("Need at least 400 SpotifyCares inbound messages")
    rng = random.Random(seed)
    cutoff = int(len(records) * 0.8)
    train, evaluation = records[:cutoff], records[cutoff:]

    shuffled = evaluation.copy()
    rng.shuffle(shuffled)
    evaluation = []
    seen_authors, seen_threads = set(), set()
    for record in shuffled:
        if record["author_id"] in seen_authors or record["thread_id"] in seen_threads:
            continue
        evaluation.append(record)
        seen_authors.add(record["author_id"])
        seen_threads.add(record["thread_id"])

    buckets: dict[str, list[dict]] = defaultdict(list)
    for record in evaluation:
        buckets[_stratum(record)].append(record)
    challenge = []
    for name in sorted(buckets):
        rng.shuffle(buckets[name])
        challenge.extend(buckets[name][:10])
    used = {row["tweet_id"] for row in challenge}
    remaining = [row for row in evaluation if row["tweet_id"] not in used]
    if len(challenge) < 50:
        challenge.extend(remaining[: 50 - len(challenge)])
        used = {row["tweet_id"] for row in challenge}
        remaining = [row for row in evaluation if row["tweet_id"] not in used]
    representative = rng.sample(remaining, 100)
    gold = representative + challenge[:50]
    rng.shuffle(gold)

    gold_authors = {row["author_id"] for row in gold}
    gold_threads = {row["thread_id"] for row in gold}
    pilot_pool = [row for row in train if row["author_id"] not in gold_authors and row["thread_id"] not in gold_threads]
    pilot = rng.sample(pilot_pool, 20)
    _write_label_csv(pilot_csv, pilot, {row["tweet_id"]: "pilot" for row in pilot})
    _write_label_csv(gold_csv, gold, {row["tweet_id"]: ("representative" if row in representative else _stratum(row)) for row in gold})
    return len(pilot), len(gold)
