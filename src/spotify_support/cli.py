from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from .agent import SupportAgent
from .annotate import annotate, annotate_second_labels, create_rating_sample, create_second_label_sample, rate_replies, suggest_annotations
from .data import prepare_twitter_csv, sample_for_annotation
from .evaluate import evaluate_all, read_jsonl
from .judge import judge_predictions
from .retrieval import read_csv, train_indexes


def _write_predictions(system: str, gold_path: str, index_path: str, output_path: str) -> int:
    rows = read_csv(gold_path)
    agent = SupportAgent(index_path)
    output = Path(output_path)
    prompt_hash = hashlib.sha256((Path("prompts/agent.txt").read_bytes() if system == "main" else system.encode())).hexdigest()
    existing = {row["tweet_id"]: row for row in read_jsonl(output)}
    if existing and any(
        row.get("prompt_sha256") != prompt_hash
        or (system == "main" and row.get("model_error") != "NotFoundError" and row.get("model") != agent.model)
        for row in existing.values()
    ):
        raise ValueError(f"Cached {system} predictions use a different model or prompt; remove {output} before regenerating")
    generated = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        cached = existing.get(row["tweet_id"])
        if cached and not (system == "main" and cached.get("model_error") == "NotFoundError"):
            continue
        temporary = output.with_name(output.name + ".tmp")
        if temporary.exists():
            raise ValueError(f"Refusing to overwrite temporary prediction file {temporary}")
        context = json.loads(row["context_json"] or "[]")
        decision = agent.handle(row["text"], context, system)
        existing[row["tweet_id"]] = {"tweet_id": row["tweet_id"], "system": system, "text": row["text"], "model": agent.model if system == "main" else None, "prompt_sha256": prompt_hash, **decision.as_dict()}
        with temporary.open("w", encoding="utf-8") as target:
            for prediction in existing.values():
                target.write(json.dumps(prediction, ensure_ascii=False) + "\n")
        os.replace(temporary, output)
        generated += 1
    return generated


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="spotify-support")
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("raw_csv")
    prepare.add_argument("output_jsonl")
    prepare.add_argument("--brand", default="SpotifyCares")
    sample = commands.add_parser("sample")
    sample.add_argument("prepared_jsonl")
    sample.add_argument("pilot_csv")
    sample.add_argument("gold_csv")
    sample.add_argument("--seed", type=int, default=42)
    train = commands.add_parser("train")
    train.add_argument("prepared_jsonl")
    train.add_argument("gold_csv")
    train.add_argument("output_index")
    suggest = commands.add_parser("suggest")
    suggest.add_argument("csv")
    suggest.add_argument("--batch-size", type=int, default=5)
    suggest.add_argument("--max-tokens", type=int, default=700)
    annotation = commands.add_parser("annotate")
    annotation.add_argument("csv")
    annotation.add_argument("--batch-review", action="store_true")
    annotation.add_argument("--second-label", action="store_true")
    second = commands.add_parser("second-sample")
    second.add_argument("gold_csv")
    second.add_argument("output_csv")
    second.add_argument("--seed", type=int, default=42)
    ratings = commands.add_parser("rating-sample")
    ratings.add_argument("artifacts_dir")
    ratings.add_argument("output_csv")
    ratings.add_argument("--seed", type=int, default=42)
    rate = commands.add_parser("rate")
    rate.add_argument("csv")
    rate.add_argument("annotator", choices=("annotator_1", "annotator_2"))
    reply = commands.add_parser("reply")
    reply.add_argument("message")
    reply.add_argument("--context", action="append", default=[])
    reply.add_argument("--index", default="artifacts/index.joblib")
    reply.add_argument("--system", choices=("trivial", "simple", "main"), default="main")
    generate = commands.add_parser("generate")
    generate.add_argument("system", choices=("trivial", "simple", "main"))
    generate.add_argument("gold_csv")
    generate.add_argument("index")
    generate.add_argument("output")
    judge = commands.add_parser("judge")
    judge.add_argument("predictions")
    judge.add_argument("index")
    judge.add_argument("output")
    judge.add_argument("--sample")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("gold_csv")
    evaluate.add_argument("artifacts_dir")
    evaluate.add_argument("output")
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "prepare":
        result = {"prepared": prepare_twitter_csv(args.raw_csv, args.output_jsonl, args.brand)}
    elif args.command == "sample":
        pilot, gold = sample_for_annotation(args.prepared_jsonl, args.pilot_csv, args.gold_csv, args.seed)
        result = {"pilot": pilot, "gold": gold}
    elif args.command == "train":
        result = train_indexes(args.prepared_jsonl, args.gold_csv, args.output_index)
    elif args.command == "suggest":
        result = {"suggested": suggest_annotations(args.csv, batch_size=args.batch_size, max_tokens=args.max_tokens)}
    elif args.command == "annotate":
        if args.second_label:
            annotate_second_labels(args.csv)
        else:
            annotate(args.csv, batch_review=args.batch_review)
        result = {"saved": args.csv}
    elif args.command == "second-sample":
        result = {"sampled": create_second_label_sample(args.gold_csv, args.output_csv, args.seed)}
    elif args.command == "rating-sample":
        result = {"sampled": create_rating_sample(args.artifacts_dir, args.output_csv, args.seed)}
    elif args.command == "rate":
        rate_replies(args.csv, args.annotator)
        result = {"saved": args.csv}
    elif args.command == "reply":
        context = [{"text": text} for text in args.context]
        result = SupportAgent(args.index).handle(args.message, context, args.system).as_dict()
    elif args.command == "generate":
        result = {"generated": _write_predictions(args.system, args.gold_csv, args.index, args.output)}
    elif args.command == "judge":
        result = {"judged": judge_predictions(args.predictions, args.index, args.output, args.sample)}
    else:
        result = evaluate_all(args.gold_csv, args.artifacts_dir, args.output)
    print(json.dumps(result, indent=2))
