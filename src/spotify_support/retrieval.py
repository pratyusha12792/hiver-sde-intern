from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.metrics.pairwise import cosine_similarity


class TextIndex:
    def __init__(self, rows: list[dict], text_key: str = "text"):
        self.rows = rows
        texts = [row[text_key] for row in rows]
        self.vectorizer = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)),
        ])
        self.matrix = self.vectorizer.fit_transform(texts)

    def query(self, text: str, limit: int = 3) -> list[dict]:
        scores = cosine_similarity(self.vectorizer.transform([text]), self.matrix)[0]
        indices = scores.argsort()[::-1][:limit]
        return [{**self.rows[index], "score": float(scores[index])} for index in indices]


def read_csv(path: str | Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def validate_labels(rows: list[dict], *, require_intent: bool = True) -> None:
    missing = [row["tweet_id"] for row in rows if require_intent and not row.get("intent")]
    if missing:
        raise ValueError(f"Unlabelled examples: {len(missing)}; first={missing[0]}")


def train_indexes(prepared_jsonl: str | Path, gold_csv: str | Path, output: str | Path) -> dict:
    from .core import INTENTS
    from .data import read_jsonl, timestamp

    gold = read_csv(gold_csv)
    gold_authors = {row["author_id"] for row in gold}
    gold_ids = {row["tweet_id"] for row in gold}
    gold_threads = {row["thread_id"] for row in gold}
    oldest_gold = min(timestamp(row["created_at"]) for row in gold)
    knowledge = [
        row for row in read_jsonl(prepared_jsonl)
        if row["brand_reply"] and timestamp(row["created_at"]) < oldest_gold
        and row["author_id"] not in gold_authors and row["tweet_id"] not in gold_ids
        and row["thread_id"] not in gold_threads
    ]
    if not knowledge:
        raise ValueError("No leakage-safe historical replies found")
    intent_examples = [
        {"tweet_id": f"taxonomy:{intent}", "text": f"{intent.replace('_', ' ')}. {description}", "intent": intent}
        for intent, description in INTENTS.items()
    ]
    bundle = {
        "intent": TextIndex(intent_examples),
        "reply": TextIndex([{"tweet_id": row["tweet_id"], "thread_id": row["thread_id"], "author_id": row["author_id"], "text": row["text"], "reply_id": row["brand_reply_id"], "reply": row["brand_reply"]} for row in knowledge]),
        "majority_intent": "feedback_other",
        "gold_tweet_ids": sorted(gold_ids),
        "gold_author_ids": sorted(gold_authors),
        "gold_thread_ids": sorted(gold_threads),
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
    return {"intent_examples": len(intent_examples), "knowledge": len(knowledge)}


def load_indexes(path: str | Path) -> dict:
    return joblib.load(path)
