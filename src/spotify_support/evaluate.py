from __future__ import annotations

import json
import math
import random
from pathlib import Path

from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_recall_fscore_support

from .retrieval import read_csv


def _finite(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _ci(values: list[int], seed: int = 42, rounds: int = 10_000) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(rounds))
    return [means[int(rounds * 0.025)], means[int(rounds * 0.975)]]


def evaluate_system(gold: list[dict], predictions: list[dict], judge: list[dict]) -> dict:
    by_id = {row["tweet_id"]: row for row in predictions}
    pairs = [(row, by_id[row["tweet_id"]]) for row in gold if row["tweet_id"] in by_id]
    if len(pairs) != len(gold):
        raise ValueError(f"Predictions cover {len(pairs)}/{len(gold)} gold examples")
    true_intent = [row[0]["intent"] for row in pairs]
    pred_intent = [row[1]["intent"] for row in pairs]
    true_escalate = [row[0]["escalate"].strip().lower() in {"1", "true", "yes"} for row in pairs]
    pred_escalate = [row[1]["decision"] == "ESCALATE" for row in pairs]
    precision, recall, escalation_f1, _ = precision_recall_fscore_support(true_escalate, pred_escalate, average="binary", zero_division=0)
    policy_safe_auto = [int(not predicted and not expected) for expected, predicted in zip(true_escalate, pred_escalate)]
    false_auto = [int(expected and not predicted) for expected, predicted in zip(true_escalate, pred_escalate)]
    judged = {row["tweet_id"]: row for row in judge}
    judge_pass = [bool(judged[row[0]["tweet_id"]]["pass"]) for row in pairs if row[0]["tweet_id"] in judged]
    safe_automation = [
        int(not predicted and not expected and bool(judged[gold_row["tweet_id"]]["pass"]))
        for (gold_row, _), expected, predicted in zip(pairs, true_escalate, pred_escalate)
        if gold_row["tweet_id"] in judged
    ]
    return {
        "count": len(pairs),
        "intent_accuracy": accuracy_score(true_intent, pred_intent),
        "intent_macro_f1": f1_score(true_intent, pred_intent, average="macro", zero_division=0),
        "escalation_precision": precision,
        "escalation_recall": recall,
        "escalation_f1": escalation_f1,
        "policy_safe_auto_rate": sum(policy_safe_auto) / len(policy_safe_auto),
        "safe_automation_rate": sum(safe_automation) / len(safe_automation) if safe_automation else None,
        "safe_automation_95ci": _ci(safe_automation) if safe_automation else None,
        "unsafe_auto_handle_rate": sum(false_auto) / max(1, sum(not value for value in pred_escalate)),
        "judge_reply_pass_rate": sum(judge_pass) / len(judge_pass) if judge_pass else None,
        "model_error_rate": sum(bool(row[1].get("model_error")) for row in pairs) / len(pairs),
    }


def agreement(human_csv: str | Path, judge_rows: list[dict]) -> dict:
    path = Path(human_csv)
    if not path.exists():
        return {}
    human: dict[tuple[str, str], list[dict]] = {}
    for row in read_csv(path):
        if row.get("groundedness"):
            human.setdefault((row["system"], row["tweet_id"]), []).append(row)
    paired = [(human[(row["system"], row["tweet_id"])], row) for row in judge_rows if (row["system"], row["tweet_id"]) in human]
    if not paired:
        return {}
    axes = ["groundedness", "helpfulness", "tone", "safety"]
    consensus = lambda rows, axis: round(sum(int(row[axis]) for row in rows) / len(rows))
    result = {f"weighted_kappa_{axis}": _finite(cohen_kappa_score([consensus(h, axis) for h, _ in paired], [int(j[axis]) for _, j in paired], weights="quadratic")) for axis in axes}
    human_total = [sum(consensus(h, axis) for axis in axes) for h, _ in paired]
    judge_total = [sum(int(j[axis]) for axis in axes) for _, j in paired]
    result["spearman_total"] = _finite(spearmanr(human_total, judge_total).statistic)
    human_pass = [consensus(h, "groundedness") >= 4 and consensus(h, "safety") >= 4 and total / 4 >= 4 for (h, _), total in zip(paired, human_total)]
    judge_pass = [bool(row["pass"]) for _, row in paired]
    precision, recall, binary_f1, _ = precision_recall_fscore_support(human_pass, judge_pass, average="binary", zero_division=0)
    result.update({"pass_accuracy": accuracy_score(human_pass, judge_pass), "pass_precision": precision, "pass_recall": recall, "pass_f1": binary_f1})
    double = [(rows[0], rows[1]) for rows in human.values() if len(rows) >= 2]
    result["human_human_kappa"] = {
        axis: _finite(cohen_kappa_score([int(left[axis]) for left, _ in double], [int(right[axis]) for _, right in double], weights="quadratic"))
        for axis in axes
    } if double else {}
    result["count"] = len(paired)
    return result


def label_agreement(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    rows = [row for row in read_csv(path) if row.get("intent") and row.get("primary_intent")]
    if not rows:
        return {}
    primary_escalate = [row["primary_escalate"].lower() in {"1", "true", "yes"} for row in rows]
    second_escalate = [row["escalate"].lower() in {"1", "true", "yes"} for row in rows]
    return {
        "count": len(rows),
        "intent_accuracy": accuracy_score([row["primary_intent"] for row in rows], [row["intent"] for row in rows]),
        "intent_kappa": _finite(cohen_kappa_score([row["primary_intent"] for row in rows], [row["intent"] for row in rows])),
        "escalation_accuracy": accuracy_score(primary_escalate, second_escalate),
        "escalation_kappa": _finite(cohen_kappa_score(primary_escalate, second_escalate)),
    }


def evaluate_all(gold_path: str | Path, artifacts_dir: str | Path, output_path: str | Path) -> dict:
    gold = read_csv(gold_path)
    if len(gold) != 150 or any(not row["intent"] or not row["escalate"] or row.get("human_verified", "").lower() != "true" for row in gold):
        raise ValueError("Golden set must contain exactly 150 human-verified labels before evaluation")
    artifacts = Path(artifacts_dir)
    result = {"representative": {}, "challenge": {}}
    all_judges = []
    for system in ("trivial", "simple", "main"):
        judge = read_jsonl(artifacts / f"judge-{system}.jsonl")
        all_judges.extend(judge)
        predictions = read_jsonl(artifacts / f"{system}.jsonl")
        representative = [row for row in gold if row["sampling_stratum"] == "representative"]
        challenge = [row for row in gold if row["sampling_stratum"] != "representative"]
        result["representative"][system] = evaluate_system(representative, predictions, judge)
        result["challenge"][system] = evaluate_system(challenge, predictions, judge)
    result["judge_human_agreement"] = agreement(artifacts / "human_reply_scores.csv", all_judges)
    result["label_agreement"] = label_agreement(artifacts.parent / "data/annotations/second_labels.csv")
    Path(output_path).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
