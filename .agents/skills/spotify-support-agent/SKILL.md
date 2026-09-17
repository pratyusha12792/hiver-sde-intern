---
name: spotify-support-agent
description: Triage a Spotify customer-support message using this repository's trained historical-reply index. Use for classifying Spotify support intent, drafting an evidence-grounded reply, and deciding auto-handle versus human escalation. Do not use before artifacts/index.joblib exists.
---

# Spotify Support Agent

Run the repository CLI instead of answering from memory:

```bash
.venv/bin/python -m spotify_support reply "<customer message>" --index artifacts/index.joblib
```

Add each earlier thread message with `--context "<message>"` in chronological order.

Return the CLI's intent, draft reply, decision, reason, evidence tweet IDs, and support scores. Never change `ESCALATE` to `AUTO_HANDLE`. If the index is missing, direct the user to complete dataset preparation, human annotation, and `make train`; do not invent a support response. Model generation also requires `GROQ_API_KEY`.
