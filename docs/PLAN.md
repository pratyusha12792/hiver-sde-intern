# Implementation Plan and Completion Audit

## Current Workflow

1. Prepare SpotifyCares threads from the Kaggle CSV with `make data`.
2. Create a fixed-seed 150-row gold sample with unique authors and threads: 100 representative and 50 challenge rows. Keep 20 older pilot rows only as an optional fixture.
3. Generate batched Groq suggestions with `make suggest-gold`; a human reviews every row with `make annotate-gold`. Double-label 50 gold rows independently and adjudicate disagreements.
4. Train TF-IDF taxonomy-intent and leakage-safe historical-reply indexes with `make train`; pilot labels are not required.
5. Generate trivial, simple, and Groq Qwen 3.6 27B predictions with `make generate`.
6. Grade replies with Groq GPT-OSS 20B, then collect blinded ratings from two humans.
7. Run `make evaluate`, replace pending README sections with measured results and five real failure modes, then time `make reproduce` from a clean clone.

Resource rule: use hosted free-tier models only. Do not install local model runtimes or download weights. Python setup is cache-free; dataset and generated artifacts stay inside this repository.

## Completion Checklist

- [x] Runnable data preparation, sampling, retrieval, triage, judging, and evaluation code
- [x] Public dataset downloaded; 48,543 inbound messages prepared, including 41,585 with direct SpotifyCares replies
- [x] Fixed-seed sampling produces exactly 150 unique-thread gold rows; gold tweet, author, and thread overlap with retrieval is zero
- [x] Two baselines and deterministic safety gate implemented
- [x] Test gate passes (11 tests); annotation progress and model/prompt caches are protected, and undefined agreement metrics serialize as `null`
- [x] Repository-scoped Codex skill implemented and validated
- [x] Local model runtime and downloaded model fully removed
- [ ] Groq suggestions and human verification for all 150 gold examples
- [ ] Independent second-human labels for 50 gold examples
- [x] Free Groq API key supplied and both hosted models validated
- [ ] Cached predictions and judge scores generated
- [ ] Two-human reply ratings collected and judge agreement calculated
- [ ] Headline results, confidence intervals, and five observed failures written into README
- [ ] Clean-clone reproduction verified in under 15 minutes
- [x] Local Git repository initialized on `main`
- [ ] Public remote created and commits pushed manually by the repository owner

No headline metric may be published before all unchecked evaluation items are complete.

## Next Execution Gate

After reviewing this implementation, run `make suggest-gold` and then `make annotate-gold`. Suggestions populate only `suggested_*` columns; each final label requires explicit human acceptance or correction and records its provenance. After that checkpoint, the order is `make second-sample`, `make annotate-second`, `make train`, `make generate`, `make judge`, `make rating-sample`, and `make evaluate`. The second annotation and two blinded reply ratings remain human tasks.
