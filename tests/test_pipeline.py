import csv
import hashlib
import json
from types import SimpleNamespace

import joblib
import httpx
import pytest
from openai import NotFoundError

from spotify_support.agent import SupportAgent
from spotify_support.cli import _write_predictions
from spotify_support.core import apply_gate, hard_escalation_reason
from spotify_support.config import api_key
from spotify_support.data import LABEL_COLUMNS, prepare_twitter_csv, sample_for_annotation, timestamp
from spotify_support.annotate import RATING_FIELDS, _number, _parse_suggestions, annotate, annotate_second_labels, create_rating_sample, create_second_label_sample, rate_replies, suggest_annotations
from spotify_support.evaluate import _finite, evaluate_system, label_agreement
from spotify_support.judge import _parse_score, _sample_predictions, judge_predictions
from spotify_support.retrieval import TextIndex, train_indexes


def test_prepare_reconstructs_only_brand_threads(tmp_path):
    output = tmp_path / "spotify.jsonl"
    count = prepare_twitter_csv("tests/fixtures/twcs.csv", output)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert count == 3
    assert {row["tweet_id"] for row in rows} == {"1", "3", "5"}
    followup = next(row for row in rows if row["tweet_id"] == "3")
    assert [item["tweet_id"] for item in followup["context"]] == ["1", "2"]
    assert next(row for row in rows if row["tweet_id"] == "1")["brand_reply_id"] == "2"
    assert timestamp("Fri Apr 07 10:36:18 +0000 2017") < timestamp("Wed Sep 27 21:58:44 +0000 2017")


def test_sampling_is_seeded_and_leakage_safe(tmp_path):
    prepared = tmp_path / "prepared.jsonl"
    with prepared.open("w") as target:
        for index in range(1_200):
            target.write(json.dumps({
                "tweet_id": str(index), "thread_id": str(index), "author_id": f"user-{index}",
                "created_at": f"2017-01-{index // 50 + 1:02d}T00:00:00Z", "text": f"Playback issue number {index}",
                "context": [] if index % 2 else [{"text": "Earlier message"}], "brand_reply": "Restart the app", "brand_reply_id": f"r-{index}",
            }) + "\n")
    pilot, gold = tmp_path / "pilot.csv", tmp_path / "gold.csv"
    assert sample_for_annotation(prepared, pilot, gold) == (20, 150)
    with pilot.open() as source:
        pilot_rows = list(csv.DictReader(source))
    with gold.open() as source:
        gold_rows = list(csv.DictReader(source))
    assert not ({row["author_id"] for row in pilot_rows} & {row["author_id"] for row in gold_rows})
    assert len({row["thread_id"] for row in gold_rows}) == 150
    assert sum(row["sampling_stratum"] == "representative" for row in gold_rows) == 100
    with prepared.open("a") as target:
        for tweet_id, thread_id, author_id in (
            (gold_rows[0]["tweet_id"], "other-thread", "other-author"),
            ("gold-author-sibling", "other-thread-2", gold_rows[0]["author_id"]),
            ("gold-thread-sibling", gold_rows[0]["thread_id"], "other-author-2"),
        ):
            target.write(json.dumps({
                "tweet_id": tweet_id, "thread_id": thread_id, "author_id": author_id,
                "created_at": "2017-01-01T00:00:00Z", "text": "leak candidate", "context": [],
                "brand_reply": "reply", "brand_reply_id": f"reply-{tweet_id}",
            }) + "\n")
    index = tmp_path / "index.joblib"
    result = train_indexes(prepared, gold, index)
    assert result["intent_examples"] == 7
    retrieval_rows = joblib.load(index)["reply"].rows
    assert not ({row["tweet_id"] for row in gold_rows} & {row["tweet_id"] for row in retrieval_rows})
    assert not ({row["author_id"] for row in gold_rows} & {row["author_id"] for row in retrieval_rows})
    assert not ({row["thread_id"] for row in gold_rows} & {row["thread_id"] for row in retrieval_rows})
    gold_rows[0]["suggested_intent"] = "feedback_other"
    with gold.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=gold_rows[0].keys())
        writer.writeheader()
        writer.writerows(gold_rows)
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        sample_for_annotation(prepared, pilot, gold)


def test_groq_suggestions_are_not_truth_until_human_accepts(tmp_path, monkeypatch, capsys):
    path = tmp_path / "gold.csv"
    row = {field: "" for field in LABEL_COLUMNS}
    row.update({"tweet_id": "1", "text": "I was charged twice", "context_json": "[]", "sampling_stratum": "representative"})
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    suggestion = {
        "annotations": [{
            "tweet_id": "1", "intent": "payments_refunds", "intent_reason": "The customer reports a duplicate charge.",
            "escalate": True, "escalation_reason": "billing_or_refund", "escalation_explanation": "A human must review charges.", "reply_should_include": "a billing support handoff",
            "reply_must_not_claim": "that a refund was issued",
        }]
    }
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(suggestion)))])
    calls, waits = [], []

    class TooManyRequests(Exception):
        status_code = 429

    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise TooManyRequests("Please try again in 0.01s")
        return response

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr("spotify_support.annotate.time.sleep", waits.append)
    assert suggest_annotations(path, client=client, model="test-model") == 1
    assert len(calls) == 2 and waits == [0.01]
    assert calls[-1]["max_completion_tokens"] == 700
    assert calls[-1]["reasoning_effort"] == "none"
    assert "response_format" not in calls[-1]
    assert '"content_chars":' in capsys.readouterr().err
    suggested = list(csv.DictReader(path.open()))[0]
    assert suggested["suggested_intent"] == "payments_refunds"
    assert suggested["intent"] == suggested["human_verified"] == ""
    assert suggest_annotations(path, client=client, model="test-model") == 0
    assert len(calls) == 2
    prompts = []
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt) or "")
    annotate(path)
    screen = capsys.readouterr().out
    assert "I was charged twice" in screen
    assert "The customer reports a duplicate charge." in screen
    assert "A human must review charges." in screen
    assert "a billing support handoff" in screen
    assert "that a refund was issued" in screen
    verified = list(csv.DictReader(path.open()))[0]
    assert prompts == ["Press Enter to accept, c to correct, or 0 to save and exit: "]
    assert verified["intent"] == "payments_refunds"
    assert verified["escalation_reason"] == "billing_or_refund"
    assert verified["human_verified"] == "true"
    assert verified["annotation_method"] == "ai_assisted_human_verified"


def test_malformed_or_wrong_schema_suggestions_are_rejected():
    with pytest.raises(ValueError, match="empty"):
        _parse_suggestions("", {"1"})
    with pytest.raises(ValueError, match="malformed"):
        _parse_suggestions("not json", {"1"})
    with pytest.raises(ValueError, match="required schema"):
        _parse_suggestions('{"annotations":[{"tweet_id":"1"}]}', {"1"})


def test_suggestion_whitespace_and_reasonable_verbosity_are_accepted():
    item = {
        "tweet_id": " 1 ",
        "intent": " playback_connectivity ",
        "intent_reason": "  Playback fails repeatedly across several different tracks and artists. ",
        "escalate": False,
        "escalation_reason": " none ",
        "escalation_explanation": " Standard public troubleshooting can safely address this playback problem. ",
        "reply_should_include": " Ask for device details and provide an appropriate troubleshooting step. ",
        "reply_must_not_claim": " Do not claim that playback has already been permanently fixed. ",
    }
    parsed = _parse_suggestions(json.dumps({"annotations": [item]}), {"1"})["1"]
    assert parsed["intent"] == "playback_connectivity"
    assert parsed["tweet_id"] == "1"
    assert "  " not in parsed["intent_reason"]
    item["escalate"] = "false"
    with pytest.raises(ValueError, match="escalate must be boolean"):
        _parse_suggestions(json.dumps({"annotations": [item]}), {"1"})
    item["escalate"], item["reply_should_include"] = False, "   "
    with pytest.raises(ValueError, match="reply_should_include"):
        _parse_suggestions(json.dumps({"annotations": [item]}), {"1"})


def test_empty_response_logs_safe_metadata_fails_without_saving(tmp_path, capsys):
    path = tmp_path / "gold.csv"
    row = {field: "" for field in LABEL_COLUMNS}
    row.update({"tweet_id": "1", "text": "Help", "context_json": "[]", "sampling_stratum": "representative"})
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    usage = SimpleNamespace(prompt_tokens=300, completion_tokens=700, completion_tokens_details=SimpleNamespace(reasoning_tokens=700))
    response = SimpleNamespace(choices=[SimpleNamespace(finish_reason="length", message=SimpleNamespace(content=""))], usage=usage)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: response)))
    with pytest.raises(ValueError, match="empty"):
        suggest_annotations(path, client=client, model="test-model")
    diagnostic = capsys.readouterr().err
    assert '"finish_reason": "length"' in diagnostic
    assert '"content_chars": 0' in diagnostic
    assert '"reasoning_tokens": 700' in diagnostic
    saved = list(csv.DictReader(path.open()))[0]
    assert saved["suggested_intent"] == saved["suggested_by_model"] == ""


def test_human_can_correct_a_suggestion(tmp_path, monkeypatch):
    path = tmp_path / "gold.csv"
    row = {field: "" for field in LABEL_COLUMNS}
    row.update({
        "tweet_id": "1", "text": "Just sharing feedback", "context_json": "[]", "sampling_stratum": "representative",
        "suggested_intent": "playback_connectivity", "suggested_intent_reason": "Possibly playback related.",
        "suggested_escalate": "false", "suggested_escalation_reason": "none",
        "suggested_escalation_explanation": "No private action is required.",
        "suggested_reply_should_include": "acknowledgement", "suggested_reply_must_not_claim": "none",
        "suggested_by_model": "test-model",
    })
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    answers = iter(["c", "7", "", "", "", "Corrected intent after review"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    annotate(path)
    corrected = list(csv.DictReader(path.open()))[0]
    assert corrected["intent"] == "feedback_other"
    assert corrected["escalate"] == "false"
    assert corrected["annotator_notes"] == "Corrected intent after review"
    assert corrected["human_verified"] == "true"


def test_numbered_choice_accepts_number_or_exact_name_and_reprompts(monkeypatch, capsys):
    options = ["account_access_security", "payments_refunds"]
    answers = iter(["not_an_intent", "payments_refunds"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert _number("Intent number", options, "payments_refunds") == "payments_refunds"
    assert "Invalid choice" in capsys.readouterr().out
    monkeypatch.setattr("builtins.input", lambda _: "2")
    assert _number("Intent number", options) == "payments_refunds"


def test_batch_review_accepts_and_skips_without_losing_progress(tmp_path, monkeypatch, capsys):
    path = tmp_path / "gold.csv"
    rows = []
    for number in range(11):
        row = {field: "" for field in LABEL_COLUMNS}
        row.update({
            "tweet_id": str(number), "text": f"Message {number}", "context_json": "[]",
            "sampling_stratum": "representative", "suggested_intent": "playback_connectivity",
            "suggested_intent_reason": "Playback issue", "suggested_escalate": "false",
            "suggested_escalation_reason": "none", "suggested_escalation_explanation": "Public help is safe",
            "suggested_reply_should_include": "next step", "suggested_reply_must_not_claim": "already fixed",
        })
        rows.append(row)
    rows[0]["human_verified"] = "true"
    rows[0]["intent"] = "feedback_other"
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    answers = iter(["A", "S"] + ["A"] * 8)
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    annotate(path, batch_review=True)
    screen = capsys.readouterr().out
    assert "Batch 1: 10 pending rows" in screen
    assert "Message 10" in screen
    saved = list(csv.DictReader(path.open()))
    assert saved[0]["intent"] == "feedback_other"
    assert saved[1]["human_verified"] == "true"
    assert saved[1]["annotation_method"] == "ai_assisted_human_verified"
    assert saved[1]["reply_should_include"] == "next step"
    assert saved[2]["human_verified"] == ""
    assert sum(row["human_verified"] == "true" for row in saved) == 10
    monkeypatch.setattr("builtins.input", lambda _: "A")
    annotate(path, batch_review=True)
    assert all(row["human_verified"] == "true" for row in csv.DictReader(path.open()))


def test_batch_review_corrects_only_selected_row(tmp_path, monkeypatch):
    path = tmp_path / "gold.csv"
    rows = []
    for number in range(2):
        row = {field: "" for field in LABEL_COLUMNS}
        row.update({
            "tweet_id": str(number), "text": f"Message {number}", "context_json": "[]",
            "sampling_stratum": "representative", "suggested_intent": "playback_connectivity",
            "suggested_intent_reason": "Playback issue", "suggested_escalate": "false",
            "suggested_escalation_reason": "none", "suggested_escalation_explanation": "Public help is safe",
            "suggested_reply_should_include": "next step", "suggested_reply_must_not_claim": "already fixed",
        })
        rows.append(row)
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    answers = iter(["C", "feedback_other", "", "", "", "Reviewed correction", "S"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    annotate(path, batch_review=True)
    saved = list(csv.DictReader(path.open()))
    assert saved[0]["intent"] == "feedback_other"
    assert saved[0]["annotator_notes"] == "Reviewed correction"
    assert saved[0]["annotation_method"] == "ai_assisted_human_verified"
    assert saved[1]["human_verified"] == ""


def test_gate_escalates_sensitive_and_unsupported_outputs():
    assert hard_escalation_reason("I was charged twice") == "billing_or_refund"
    assert apply_gate("Help", "playback_connectivity", "none", ["1"], {"1"}, 0.8, 0.8) == ("AUTO_HANDLE", "none")
    assert apply_gate("Refund me", "payments_refunds", "none", ["1"], {"1"}, 0.8, 0.8) == ("ESCALATE", "billing_or_refund")
    assert apply_gate("Help", "playback_connectivity", "none", ["fake"], {"1"}, 0.8, 0.8) == ("ESCALATE", "low_evidence")


def test_agent_prompt_forbids_future_action_becoming_completed_action():
    prompt = open("prompts/agent.txt", encoding="utf-8").read()
    assert '"we\'ll pass this on"' in prompt
    assert '"we\'ve passed"' in prompt
    assert "claim an action is complete only when historical_evidence explicitly establishes completion" in prompt


def test_simple_agent_returns_evidence_and_trivial_escalates(tmp_path):
    intent = TextIndex([{"tweet_id": "p1", "text": "music will not play", "intent": "playback_connectivity"}])
    reply = TextIndex([{"tweet_id": "k1", "text": "music will not play", "reply_id": "k2", "reply": "Please restart the app."}])
    path = tmp_path / "index.joblib"
    joblib.dump({"intent": intent, "reply": reply, "majority_intent": "playback_connectivity"}, path)
    agent = SupportAgent(path)
    simple = agent.handle("music will not play", system="simple")
    assert simple.decision == "AUTO_HANDLE"
    assert simple.evidence_tweet_ids == ["k1"]
    assert agent.handle("anything", system="trivial").decision == "ESCALATE"


def test_invalid_hosted_model_output_fails_closed(tmp_path):
    intent = TextIndex([{"tweet_id": "p1", "text": "music will not play", "intent": "playback_connectivity"}])
    reply = TextIndex([{"tweet_id": "k1", "text": "music will not play", "reply_id": "k2", "reply": "Restart the app."}])
    path = tmp_path / "index.joblib"
    joblib.dump({"intent": intent, "reply": reply, "majority_intent": "playback_connectivity"}, path)
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"intent":"invented","reason":"none","evidence_tweet_ids":[]}'))])
    calls = []
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: calls.append(kwargs) or response)))
    result = SupportAgent(path, client=client).handle("music will not play")
    assert calls[0]["max_completion_tokens"] == 256
    assert calls[0]["reasoning_effort"] == "none"
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert result.decision == "ESCALATE"
    assert result.intent == "playback_connectivity"
    assert result.model_error == "ValueError"


def test_generation_404_logs_safe_details_and_stops(tmp_path, capsys):
    intent = TextIndex([{"tweet_id": "p1", "text": "music will not play", "intent": "playback_connectivity"}])
    reply = TextIndex([{"tweet_id": "k1", "text": "music will not play", "reply_id": "k2", "reply": "Restart the app."}])
    path = tmp_path / "index.joblib"
    joblib.dump({"intent": intent, "reply": reply, "majority_intent": "playback_connectivity"}, path)
    body = {"error": {"code": "model_not_found", "type": "invalid_request_error", "message": "Model was not found; Bearer fake-secret; private customer text"}}
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(404, request=request, json=body)
    error = NotFoundError("Not Found", response=response, body=body)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: (_ for _ in ()).throw(error))))
    with pytest.raises(NotFoundError):
        SupportAgent(path, client=client).handle("music will not play")
    diagnostic = capsys.readouterr().err
    assert '"status": 404' in diagnostic
    assert '"provider_code": "model_not_found"' in diagnostic
    assert '"provider_type": "invalid_request_error"' in diagnostic
    assert "fake-secret" not in diagnostic
    assert "private customer text" not in diagnostic


def test_judge_bounds_reasoning_retries_and_token_rate(tmp_path, monkeypatch):
    predictions = tmp_path / "main.jsonl"
    predictions.write_text(json.dumps({
        "system": "main", "tweet_id": "1", "text": "Help", "draft_reply": "Restart the app.",
        "decision": "AUTO_HANDLE", "reason": "none", "evidence_tweet_ids": ["e1"],
    }) + "\n")
    index = tmp_path / "index.joblib"
    joblib.dump({"reply": SimpleNamespace(rows=[{"tweet_id": "e1", "text": "Playback fails", "reply": "Restart the app."}])}, index)
    output = tmp_path / "judge.jsonl"
    calls, sleeps, clients = [], [], []
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
            "groundedness": 5, "helpfulness": 4, "tone": 5, "safety": 5, "reason": "Grounded and safe",
        })))],
        usage=SimpleNamespace(prompt_tokens=300, completion_tokens=50, total_tokens=350),
    )

    def fake_openai(**kwargs):
        clients.append(kwargs)
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **call: calls.append(call) or response)))

    monkeypatch.setattr("spotify_support.judge.OpenAI", fake_openai)
    monkeypatch.setattr("spotify_support.judge.time.sleep", sleeps.append)
    assert judge_predictions(predictions, index, output) == 1
    assert clients[0]["max_retries"] == 0
    assert calls[0]["model"] == "openai/gpt-oss-20b"
    assert calls[0]["reasoning_effort"] == "low"
    assert calls[0]["max_completion_tokens"] == 256
    response_format = calls[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert schema["required"] == ["groundedness", "helpfulness", "tone", "safety", "reason"]
    assert schema["additionalProperties"] is False
    assert sleeps == [3.0]
    saved = json.loads(output.read_text())
    assert (saved["prompt_tokens"], saved["completion_tokens"], saved["total_tokens"]) == (300, 50, 350)


def test_judge_rejects_response_outside_strict_schema():
    with pytest.raises(ValueError, match="invalid score schema"):
        _parse_score('{"groundedness":5,"helpfulness":5,"tone":5,"safety":5}')
    with pytest.raises(ValueError, match="invalid reason"):
        _parse_score('{"groundedness":5,"helpfulness":5,"tone":5,"safety":5,"reason":"one two three four five six seven eight nine"}')


def test_stale_notfound_predictions_retry_without_duplicate_or_data_loss(tmp_path, monkeypatch):
    gold = tmp_path / "gold.csv"
    with gold.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=["tweet_id", "text", "context_json"])
        writer.writeheader()
        writer.writerows([{"tweet_id": "1", "text": "help one", "context_json": "[]"}, {"tweet_id": "2", "text": "help two", "context_json": "[]"}])
    prompt_hash = hashlib.sha256(open("prompts/agent.txt", "rb").read()).hexdigest()
    output = tmp_path / "main.jsonl"
    stale = {"tweet_id": "1", "system": "main", "model": "retired-model", "prompt_sha256": prompt_hash, "model_error": "NotFoundError"}
    valid = {"tweet_id": "2", "system": "main", "model": "test-model", "prompt_sha256": prompt_hash, "model_error": None, "intent": "feedback_other"}
    output.write_text(json.dumps(stale) + "\n" + json.dumps(valid) + "\n")
    original = output.read_bytes()

    class FakeAgent:
        model = "test-model"
        calls = []
        fail = True

        def __init__(self, _):
            pass

        def handle(self, text, context, system):
            self.calls.append(text)
            if self.fail:
                raise NotFoundError("Not Found", response=httpx.Response(404, request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")), body={})
            return SimpleNamespace(as_dict=lambda: {"intent": "playback_connectivity", "model_error": None})

    monkeypatch.setattr("spotify_support.cli.SupportAgent", FakeAgent)
    with pytest.raises(NotFoundError):
        _write_predictions("main", str(gold), "unused-index", str(output))
    assert output.read_bytes() == original
    FakeAgent.fail = False
    assert _write_predictions("main", str(gold), "unused-index", str(output)) == 1
    saved = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(saved) == 2
    assert saved[0]["tweet_id"] == "1" and saved[0]["model_error"] is None
    assert saved[1] == valid
    assert FakeAgent.calls == ["help one", "help one"]


def test_metrics_penalize_unsafe_automation():
    gold = [{"tweet_id": "1", "intent": "payments_refunds", "escalate": "true"}]
    prediction = [{"tweet_id": "1", "intent": "payments_refunds", "decision": "AUTO_HANDLE"}]
    judge = [{"tweet_id": "1", "pass": False}]
    metrics = evaluate_system(gold, prediction, judge)
    assert metrics["intent_macro_f1"] == 1
    assert metrics["unsafe_auto_handle_rate"] == 1
    assert metrics["safe_automation_rate"] == 0


def test_gitignored_env_loads_api_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    (tmp_path / ".env").write_text("GROQ_API_KEY=test-key\n")
    assert api_key() == "test-key"


def test_human_review_samples_are_fixed_and_blinded(tmp_path):
    gold = tmp_path / "gold.csv"
    fields = ["tweet_id", "intent", "escalate", "escalation_reason"]
    with gold.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows({"tweet_id": str(i), "intent": "feedback_other", "escalate": "false", "escalation_reason": "none"} for i in range(60))
    second = tmp_path / "second.csv"
    assert create_second_label_sample(gold, second) == 50
    second_rows = list(csv.DictReader(second.open()))
    assert all(not row["intent"] and row["primary_intent"] == "feedback_other" for row in second_rows)

    for system in ("trivial", "simple", "main"):
        with (tmp_path / f"{system}.jsonl").open("w") as target:
            for i in range(20):
                target.write(json.dumps({"system": system, "tweet_id": str(i), "text": "help", "draft_reply": "reply", "decision": "ESCALATE", "reason": "low_evidence"}) + "\n")
    ratings = tmp_path / "ratings.csv"
    assert create_rating_sample(tmp_path, ratings) == 50
    rating_rows = list(csv.DictReader(ratings.open()))
    assert len(rating_rows) == 100
    assert {row["annotator"] for row in rating_rows} == {"annotator_1", "annotator_2"}
    expected = {"trivial": 15, "simple": 15, "main": 20}
    for system, count in expected.items():
        predictions = [json.loads(line) for line in (tmp_path / f"{system}.jsonl").read_text().splitlines()]
        assert len(_sample_predictions(predictions, ratings)) == count


def test_second_labels_and_reply_ratings_save_compact_batch_input(tmp_path, monkeypatch, capsys):
    gold = tmp_path / "gold.csv"
    fields = ["tweet_id", "text", "context_json", "intent", "escalate", "escalation_reason", "primary_intent", "primary_escalate", "primary_escalation_reason", "suggested_intent_reason", "annotation_method", "human_verified"]
    with gold.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows({"tweet_id": str(i), "text": f"Message {i}", "context_json": "[]", "intent": "", "escalate": "", "escalation_reason": "", "primary_intent": "feedback_other", "primary_escalate": "true", "primary_escalation_reason": "HIDDEN_PRIMARY", "suggested_intent_reason": "HIDDEN_AI", "annotation_method": "", "human_verified": ""} for i in range(11))
    answers = iter(["4 4", "4 4 4 4 4 4 4 4 4 4", "A A", "A A A A A A A A A A", KeyboardInterrupt()])

    def answer(_):
        value = next(answers)
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr("builtins.input", answer)
    with pytest.raises(KeyboardInterrupt):
        annotate_second_labels(gold)
    labelled = list(csv.DictReader(gold.open()))
    assert all((row["intent"], row["escalate"], row["escalation_reason"]) == ("playback_connectivity", "false", "none") for row in labelled[:10])
    assert labelled[10]["intent"] == ""
    screen = capsys.readouterr().out
    assert "HIDDEN_PRIMARY" not in screen and "HIDDEN_AI" not in screen
    assert "Enter exactly 10 valid values" in screen and "using only E or A" in screen

    ratings = tmp_path / "ratings.csv"
    with ratings.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=RATING_FIELDS)
        writer.writeheader()
        writer.writerows({field: {"system": "main", "tweet_id": str(i), "text": f"Help {i}", "draft_reply": "Try restarting.", "decision": "AUTO_HANDLE", "reason": "none", "annotator": "annotator_1"}.get(field, "") for field in RATING_FIELDS} for i in range(11))
        writer.writerow({field: {"system": "main", "tweet_id": "other", "text": "HIDDEN_OTHER", "draft_reply": "HIDDEN_OTHER", "decision": "ESCALATE", "reason": "low_evidence", "annotator": "annotator_2", "groundedness": "1", "helpfulness": "1", "tone": "1", "safety": "1", "notes": "HIDDEN_OTHER"}.get(field, "") for field in RATING_FIELDS})
    answers = iter(["5 4", "5 5 5 5 5 5 5 5 5 5", "4 4 4 4 4 4 4 4 4 4", "5 5 5 5 5 5 5 5 5 5", "5 5 5 5 5 5 5 5 5 5", KeyboardInterrupt()])

    def rating_answer(_):
        value = next(answers)
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr("builtins.input", rating_answer)
    with pytest.raises(KeyboardInterrupt):
        rate_replies(ratings, "annotator_1")
    rated = list(csv.DictReader(ratings.open()))
    assert all([row[axis] for axis in ("groundedness", "helpfulness", "tone", "safety")] == ["5", "4", "5", "5"] for row in rated[:10])
    assert rated[10]["groundedness"] == ""
    assert rated[11]["groundedness"] == "1"
    screen = capsys.readouterr().out
    assert "HIDDEN_OTHER" not in screen
    assert "Enter exactly 10 integers" in screen


def test_label_agreement_reports_independent_annotations(tmp_path):
    path = tmp_path / "second.csv"
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=["intent", "primary_intent", "escalate", "primary_escalate"])
        writer.writeheader()
        writer.writerows([
            {"intent": "feedback_other", "primary_intent": "feedback_other", "escalate": "false", "primary_escalate": "false"},
            {"intent": "payments_refunds", "primary_intent": "feedback_other", "escalate": "true", "primary_escalate": "false"},
        ])
    result = label_agreement(path)
    assert result["count"] == 2
    assert result["intent_accuracy"] == result["escalation_accuracy"] == 0.5
    assert _finite(float("nan")) is None
