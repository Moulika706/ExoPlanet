# ExoVet pipeline targets.
#
# Notes:
# - All targets run from the repository root.
# - On Windows without GNU Make, run the recipe commands directly, e.g.
#     python scripts/fetch_data.py
#     python -m pytest tests/ -q
#
PYTHON ?= python

.PHONY: all data toi train ablate stability phase2 cross web web-install serve figures test api lint ruff fmt clean

all: data toi phase2 cross figures	## Everything, from the raw data to the figures
	@echo "done: results/ and paper/figures/ are up to date"

data:		## Pull the raw KOI table, save CSV + SHA256 + retrieval date
	$(PYTHON) scripts/fetch_data.py

toi:		## Pull the raw TESS TOI table (Phase 5 test set)
	$(PYTHON) scripts/fetch_toi.py

train:		## Train models, calibrate, fit conformal sets, evaluate (Phase 2)
	$(PYTHON) scripts/train.py

ablate:		## Leakage ablation: feature set A vs B, grouped vs random split (Phase 2)
	$(PYTHON) scripts/ablate_leakage.py

stability:	## SHAP top-feature stability across folds (Phase 2)
	$(PYTHON) scripts/shap_stability.py

phase2:		## Everything in Phase 2: train, ablate, stability
	$(PYTHON) scripts/train.py
	$(PYTHON) scripts/ablate_leakage.py
	$(PYTHON) scripts/shap_stability.py

cross:		## Kepler -> TESS zero-shot evaluation (Phase 5)
	$(PYTHON) scripts/cross_mission.py

figures:	## Publication figures, built from the results bundle (Phase 7)
	$(PYTHON) scripts/make_figures.py

test:		## Run the test suite
	$(PYTHON) -m pytest tests/ -q

api:		## Start the FastAPI service (Phase 6)
	uvicorn api.main:app --reload --port 8000

serve:		## Build the front end and serve it from the API (Phase 6)
	cd web && npm install --no-audit --no-fund && npm run build
	uvicorn api.main:app --port 8000

web-install:	## Install the front end dependencies
	cd web && npm install --no-audit --no-fund

web:		## Build the front end (Phase 6)
	cd web && npm run build

ruff:		## Lint with ruff
	ruff check .
	ruff format --check src scripts tests configs

fmt:		## Format with ruff
	ruff format src scripts tests configs

lint: ruff

clean:		## Remove generated outputs
	rm -rf data/processed results paper/figures