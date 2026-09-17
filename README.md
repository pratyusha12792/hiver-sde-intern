# SpotifyCares Support Agent

Evidence-grounded support triage built from historical `SpotifyCares` Twitter conversations. The system predicts an intent, drafts a reply, and either auto-handles or escalates with a reason. Evaluation quality is the product: every headline result comes from a frozen human-labelled set and reproducible cached outputs.

> Status: pipeline, downloaded dataset preparation, sampling, and evaluation harness are implemented. Human labels, hosted-model runs, and measured results remain. No result below is fabricated.

## Reproduce Headline Results

Requirements: Python 3.9+ and `make`. Cached reproduction needs no dataset download or API key.

```bash
make setup
make reproduce
```

`make reproduce` runs tests and regenerates `artifacts/metrics.json` from committed gold labels, predictions, judge scores, and human scores. It is designed to finish within 15 minutes from a clean clone. Record the measured clean-clone duration here after artifacts exist.

## Build Results From Source Data

Dataset download is public and needs no Kaggle token. Create a free [Groq API key](https://console.groq.com/keys), then:

```bash
make data
make sample
make suggest-gold       # about 30 batched Groq calls; resumable
make annotate-batch     # review ten pending rows at once: A, C, or S
make second-sample
make annotate-second  # run by a second person
make train
make generate
make judge
make rating-sample
make rate-1            # first human
make rate-2            # second human
make evaluate
```

The raw dataset, SQLite staging file, and trained index are intentionally gitignored. Setup installs only the Python environment (about 190 MB); source-data work additionally uses about 530 MB. Repeated `make data` runs do not redownload existing data. Put `GROQ_API_KEY=...` in a gitignored `.env` file before generation. Another OpenAI-compatible provider can use `OPENAI_BASE_URL`, `LLM_API_KEY`, `GENERATOR_MODEL`, and `JUDGE_MODEL`.

Free-tier limits may interrupt a generation or judging run. Every completed row is flushed to disk, so rerun the same command after the provider limit resets; compatible caches resume rather than repeat calls.

Try one message after training:

```bash
make demo MESSAGE="I was charged twice for Premium"
```

## Problem Framing

“Good” means correctly routing the customer's current need, grounding advice in similar historical SpotifyCares exchanges, and auto-handling only when the reply needs no private account or payment action. The primary metric is safe automation rate: representative messages both auto-handled and safe according to human policy labels.

This project does not execute account actions, send tweets, train a model, deploy a service, or claim production readiness. Banking77 is excluded because banking intents do not represent Spotify support traffic.

## Data and Labels

The Kaggle CSV contains one tweet per row with anonymized author IDs and reply links. `make data` loads it into temporary SQLite storage, finds connected SpotifyCares threads, and emits each inbound message with up to six preceding turns and its immediate historical brand reply.

`make sample` creates an optional 20-row pilot fixture from the older 80% of messages and exactly 150 gold rows from the newer 20%. Gold contains 100 representative rows and 50 challenge rows covering sensitive requests, follow-ups, short messages, language variation, and rare cases. Gold rows have unique authors and threads; every gold author, tweet, and complete thread is excluded from pilot and retrieval data.

Labels follow [the annotation guide](data/annotations/annotation_guide.md). Groq returns concise plain JSON in configurable batches of five with a 700-token completion cap; an exact local schema check runs before suggestions reach separate CSV columns. Rate-limit responses use bounded backoff, and saved batches resume without regeneration. The primary annotator must view every suggestion and either press Enter to accept it or correct it. Accepted rows record `annotation_method=ai_assisted_human_verified` and `human_verified=true`. A second human independently labels a fixed 50-row copy without seeing primary labels or Groq suggestions. Review disagreements and update the frozen gold label only when both annotators agree it is wrong. Dataset use and attribution are documented in [LICENSE-DATA.md](LICENSE-DATA.md).

`make annotate-batch` displays ten pending messages and suggestions together. Enter `A` to accept a row, `C` to open its detailed correction prompts, or `S` to leave it pending. Each accepted or corrected row saves atomically; rerunning starts with unverified rows. `make annotate-gold` remains the one-at-a-time option.

## Systems

- **Trivial:** fixed `feedback_other` intent, fixed acknowledgement, always escalate.
- **Simple:** TF-IDF nearest-taxonomy intent and nearest historical reply, then deterministic safety rules.
- **Main:** Groq-hosted Qwen 3.6 27B receives three matching taxonomy descriptions and three retrieved SpotifyCares exchanges. A deterministic gate overrides unsafe auto-handling. Groq-hosted GPT-OSS 20B judges replies independently.

Each output includes intent, draft, decision, reason, evidence tweet IDs, and retrieval-support scores. Billing, refunds, compromised accounts, sensitive data, and legal or safety language always escalate.

## Evaluation

Automated metrics include intent macro-F1, accuracy, escalation precision/recall/F1, unsafe auto-handle rate, safe automation rate, reply pass rate, and a seeded bootstrap 95% confidence interval. Representative and challenge results must be reported separately.

The judge scores groundedness, helpfulness, tone, and safety from 1–5. A reply passes only when groundedness and safety are at least 4 and the four-score mean is at least 4. Two people blindly rate the same stratified 50-draft sample through the CLI. Judge agreement is measured using quadratic weighted kappa, Spearman correlation, and pass/fail F1; human-human kappa provides context.

### Results vs. Baselines

Pending real labels and model runs. Replace this paragraph with the generated table from `artifacts/metrics.json`; do not hand-edit numbers.

### Failure Analysis

Pending evaluation. Select the five observed failure categories with highest frequency multiplied by severity. For each, include a redacted real example, expected output, actual output, retrieved evidence, hypothesis, and smallest credible fix.

## What Is Misleading About My Headline Number?

Even a strong safe automation rate would cover one brand, historical Twitter traffic, and only 100 representative examples plus 50 deliberately difficult cases. “Resolved” is inferred from public replies rather than verified customer outcomes. Labels are human-verified but initially suggested by a model, which can anchor reviewers. Human escalation policy and reply ratings are subjective. Free-tier hosted models, cached outputs, sampling choices, and conservative escalation thresholds limit generalization. Offline drafting does not measure production latency, policy drift, private-account context, or customer satisfaction.

## With One More Week

Collect more double-labelled examples around confused intent pairs, calibrate thresholds on a separate development set, test temporal drift, add retrieval ablations, and run a small blinded support-agent review. Add production integration only after those results justify it.

## Decision Log

1. Chose SpotifyCares because product issues have reusable public troubleshooting patterns.
2. Restricted evaluation to one brand instead of optimizing across incompatible support policies.
3. Preserved thread context rather than treating every tweet as independent.
4. Used a temporal split and excluded held-out tweet IDs, authors, and entire threads from retrieval.
5. Used the minimum allowed 150 gold examples: 100 representative and 50 challenge cases.
6. Kept human-verified labels as ground truth; Groq suggestions remain separate and are disclosed to avoid presenting them as independent human labels.
7. Used TF-IDF before semantic-vector infrastructure because tweets are short and lexical evidence is auditable.
8. Used Groq-hosted Qwen 3.6 27B to avoid local memory and model downloads while retaining free inference.
9. Used GPT-OSS 20B for judging to reduce same-model self-preference.
10. Put deterministic safety rules after generation so model confidence cannot bypass escalation.
11. Treated historical replies as evidence, not verified resolutions.
12. Cached outputs so reviewers can reproduce metrics without model downloads or API cost.
13. Chose a CLI and repository skill instead of a web application because proof, not UI, is graded.
14. Excluded Banking77 because cross-domain labels would distort the Spotify taxonomy.
15. Removed mandatory pilot labelling by matching intents against the fixed taxonomy, avoiding 100 extra manual labels and evaluation leakage.
