# Intelligent-IOT — common project commands.
# Override the interpreter:  make PY=python test
# Auto-uses ./.venv if present, else `python`.

PY := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python)
START ?= 2025-05-01
END ?= 2025-12-31
TRAIN_CSV ?= data/external/multistation/train.csv
DEMO_MODE ?= eval
DEMO_INTERVAL ?= 1.0

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) | sort | \
		awk -F':.*## ' '{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ---- Setup ----------------------------------------------------------------
.PHONY: setup
setup: ## Create the py3.11 conda env (canonical; trains all model families)
	conda env create -f environment.yml || conda env update -f environment.yml
	@echo "Then: conda activate Intelligent-IOT && pip install -r requirements.txt"

.PHONY: venv
venv: ## Quick py3.13 venv for ingestion + tests (NOT for full model training)
	python3 -m venv .venv
	.venv/bin/pip install -U pip
	.venv/bin/pip install -r requirements.txt

# ---- Quality --------------------------------------------------------------
.PHONY: test
test: ## Run unit tests (excludes integration)
	$(PY) -m pytest -q -m "not integration"

.PHONY: test-int
test-int: ## Run integration tests (requires the Kafka stack up)
	$(PY) -m pytest -q -m integration

.PHONY: lint
lint: ## ruff + black --check (if installed)
	-$(PY) -m ruff check .
	-$(PY) -m black --check .

# ---- Kafka stack ----------------------------------------------------------
.PHONY: up
up: ## Start Kafka + Schema Registry + Flink (docker compose)
	docker compose up -d

.PHONY: down
down: ## Stop the stack
	docker compose down

.PHONY: ps
ps: ## Show stack containers
	docker compose ps

.PHONY: topics
topics: ## Create Kafka topics
	$(PY) -m infrastructure.kafka.create_topics

.PHONY: schemas
schemas: ## Register Avro schemas
	$(PY) -m infrastructure.kafka.register_schemas

.PHONY: bootstrap
bootstrap: up topics schemas ## Bring up stack, create topics, register schemas

# ---- Pipeline stages (run each in its own terminal) -----------------------
.PHONY: producer
producer: ## Run the ingestion producer (needs OPENAQ_API_KEY in .env)
	$(PY) -m infrastructure.kafka.producers.run_ingestion

.PHONY: normalizer
normalizer: ## raw -> aq.measurements
	$(PY) -m infrastructure.kafka.consumers.normalizer

.PHONY: features
features: ## aq.measurements -> aq.features (diffusion + recovery; FEATURE_TICK_SECONDS overridable)
	$(PY) -m infrastructure.kafka.consumers.features

.PHONY: inference
inference: ## aq.features -> aq.predictions
	$(PY) -m infrastructure.kafka.consumers.inference

.PHONY: alerts
alerts: ## aq.predictions -> aq.alerts
	$(PY) -m infrastructure.kafka.consumers.alerts

.PHONY: live-state
live-state: ## aq.predictions/aq.alerts -> live_state.json
	$(PY) -m infrastructure.kafka.consumers.live_state

.PHONY: sink
sink: ## aq.measurements -> Parquet audit
	$(PY) -m infrastructure.kafka.consumers.sink

# ---- Serving --------------------------------------------------------------
.PHONY: api
api: ## Run the FastAPI service (uvicorn)
	$(PY) -m uvicorn infrastructure.deployment.app:app --reload

.PHONY: dashboard
dashboard: ## Run the Streamlit dashboard
	$(PY) -m streamlit run infrastructure/deployment/dashboard/streamlit_app.py

.PHONY: demo
demo: ## Replay recorded features for a live demo (DEMO_MODE=eval|wildfire, DEMO_INTERVAL=secs)
	$(PY) -m infrastructure.kafka.scripts.demo_replay --mode $(DEMO_MODE) --interval $(DEMO_INTERVAL)

# ---- Training / evaluation ------------------------------------------------
.PHONY: backfill
backfill: ## Build multi-station training CSV (START/END overridable)
	$(PY) -m infrastructure.kafka.scripts.backfill_multistation --start $(START) --end $(END) --out $(TRAIN_CSV)

.PHONY: train
train: ## Train all families + save bundles (use the py3.11 env)
	$(PY) scripts/save_deployment_models.py

.PHONY: eval
eval: ## Recovery degradation eval (5/10/20% missing)
	$(PY) -m analytics.recovery.degradation_eval --path $(TRAIN_CSV)

# ---- Housekeeping ---------------------------------------------------------
.PHONY: clean
clean: ## Remove caches and __pycache__
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
