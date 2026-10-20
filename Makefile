# ExoVet pipeline targets.
#
# Notes:
# - All targets run from the repository root.
# - On Windows without GNU Make, run the recipe commands directly, e.g.
#     python scripts/fetch_data.py
#     python -m pytest tests/ -q
#
PYTHON ?= python

.PHONY: data train ablate stability phase2 cross test api lint ruff fmt clean

data:		## Pull the raw KOI table, save CSV + SHA256 + retrieval date
	$(PYTHON) scripts/fetch_data.py

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