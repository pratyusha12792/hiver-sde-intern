PYTHON ?= python3
VENV := .venv
RUN := $(VENV)/bin/python
ANNOTATION_BATCH_SIZE ?= 5
ANNOTATION_MAX_TOKENS ?= 700

.PHONY: setup data sample suggest-gold annotate-gold annotate-batch second-sample annotate-second train generate judge rating-sample rate-1 rate-2 evaluate reproduce test demo

setup:
	$(PYTHON) -m venv $(VENV)
	$(RUN) -m pip install --no-cache-dir -e '.[dev]'

data:
	mkdir -p data/raw data/processed
	test -f data/raw/twcs/twcs.csv || (curl -L --fail --output data/raw/customer-support-on-twitter.zip https://www.kaggle.com/api/v1/datasets/download/thoughtvector/customer-support-on-twitter && unzip -o data/raw/customer-support-on-twitter.zip -d data/raw && rm data/raw/customer-support-on-twitter.zip)
	test -f data/processed/spotify.jsonl || ($(RUN) -m spotify_support prepare data/raw/twcs/twcs.csv data/processed/spotify.jsonl && rm data/processed/spotify.sqlite)

sample:
	$(RUN) -m spotify_support sample data/processed/spotify.jsonl data/annotations/pilot.csv data/gold/golden_set.csv

suggest-gold:
	$(RUN) -m spotify_support suggest data/gold/golden_set.csv --batch-size $(ANNOTATION_BATCH_SIZE) --max-tokens $(ANNOTATION_MAX_TOKENS)

annotate-gold:
	$(RUN) -m spotify_support annotate data/gold/golden_set.csv

annotate-batch:
	$(RUN) -m spotify_support annotate data/gold/golden_set.csv --batch-review

second-sample: data/annotations/second_labels.csv

data/annotations/second_labels.csv:
	$(RUN) -m spotify_support second-sample data/gold/golden_set.csv data/annotations/second_labels.csv

annotate-second: data/annotations/second_labels.csv
	$(RUN) -m spotify_support annotate data/annotations/second_labels.csv --second-label

train:
	$(RUN) -m spotify_support train data/processed/spotify.jsonl data/gold/golden_set.csv artifacts/index.joblib

generate:
	for system in trivial simple main; do $(RUN) -m spotify_support generate $$system data/gold/golden_set.csv artifacts/index.joblib artifacts/$$system.jsonl; done

judge: artifacts/human_reply_scores.csv
	set -e; for system in trivial simple main; do $(RUN) -m spotify_support judge artifacts/$$system.jsonl artifacts/index.joblib artifacts/judge-$$system.jsonl --sample artifacts/human_reply_scores.csv; done

rating-sample: artifacts/human_reply_scores.csv

artifacts/human_reply_scores.csv:
	$(RUN) -m spotify_support rating-sample artifacts artifacts/human_reply_scores.csv

rate-1: artifacts/human_reply_scores.csv
	$(RUN) -m spotify_support rate artifacts/human_reply_scores.csv annotator_1

rate-2: artifacts/human_reply_scores.csv
	$(RUN) -m spotify_support rate artifacts/human_reply_scores.csv annotator_2

evaluate:
	$(RUN) -m spotify_support evaluate data/gold/golden_set.csv artifacts artifacts/metrics.json

reproduce: test evaluate

test:
	$(RUN) -m pytest -q

demo:
	$(RUN) -m spotify_support reply "$(MESSAGE)" --index artifacts/index.joblib
