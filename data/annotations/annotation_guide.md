# Annotation Guide

## Unit

Label the current inbound customer message using all preceding `context_json`. Groq suggestions are time-saving proposals, not ground truth. Historical replies are weak context, not ground truth.

## Assisted Review

Run `make suggest-gold` once, then `make annotate-gold`. For every row, inspect the message, suggested intent and reason, escalation decision and reason, and reply constraints. Press Enter only when all fields are correct; otherwise enter `c` and correct them. Final fields count only when `human_verified=true`; suggestion fields remain separate for auditability.

For faster review, use `make annotate-batch`. It shows ten pending rows together with thread context and constraints. Enter `A` to accept, `C` to correct that row in the detailed prompts, or `S` to leave it pending for a later run. Never accept without checking all displayed fields.

## Intent

Choose one primary intent from `taxonomy.json`. Use the action the customer most needs now. For multiple requests, select the highest-risk actionable request and note the secondary request. Use `feedback_other` only when no specific category fits.

## Escalation

Set `escalate=true` when a safe answer requires private account access, a refund or billing action, identity verification, sensitive data, legal or safety review, language support unavailable to the agent, or facts absent from historical evidence. Otherwise set `false`.

Use exactly one reason: `billing_or_refund`, `account_or_security_action`, `privacy_or_sensitive_data`, `low_evidence`, `ambiguous_or_multi_intent`, `legal_abuse_or_safety`, `unsupported_language`, or `none`.

## Reply Constraints

In `reply_should_include`, record the minimum useful next step. In `reply_must_not_claim`, record actions or facts that would be unsafe to invent, such as “your refund was issued.” Leave neither field blank. Use `none` when no special prohibition applies.

## Quality Check

Re-read every ambiguous row after the first pass. A second annotator independently labels the fixed-seed 50-row agreement sample without primary labels or Groq suggestions. Resolve disagreements before freezing the file; preserve original annotations separately.
