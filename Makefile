# ExoVet pipeline targets.
#
# Notes:
# - All targets run from the repository root.
# - On Windows without GNU Make, run the recipe commands directly, e.g.
#     python scripts/fetch_data.py
#     python -m pytest tests/ -q
#
PYTHON ?= python

.PHONY: data train ablate cross test api lint ruff fmt clean

data:		## Pull the raw KOI table, save CSV + SHA256 + retrieval date
	$(PYTHON) scripts/fetch_data.py

train:		## Train models, calibrate, fit conformal sets, evaluate (Phase 2)
	$(PYTHON) scripts/train.py

ablate:		## Leakage ablation: feature set A vs B, grouped vs sibling leak (Phase 2)
	$(PYTHON) scripts/ablate_leakage.py

cross:		## Kepler -> TESS zero-shot evaluation (Phase 5)
	$(PYTHON) scripts/cross_mission.py

test:		## Run the test suite
	$(PYTHON) -m pytest tests/ -q

api:		## Start the FastAPI service (Phase 6)
	uvicorn api.main:app --reload --port 8000

ruff:		## Lint with ruff
	ruff check .
	ruff format --check src scripts tests configs

fmt:		## Format with ruff
	ruff format src scripts tests configs

lint: ruff

clean:		## Remove generated outputs
	rm -rf data/processed results paper/figures