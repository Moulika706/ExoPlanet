# ExoVet

Explainable, uncertainty-aware exoplanet candidate vetting on NASA Kepler KOI
data. Produces a class (CONFIRMED / CANDIDATE / FALSE POSITIVE), calibrated
probabilities, split-conformal prediction sets (90% target coverage), and
SHAP-based plain-English reports.

## Setup

```bash
pip install -r requirements.txt
pip install -e .   # optional; scripts add the repo root to sys.path themselves
```

## Commands

```bash
make data       # Phase 1: fetch KOI cumulative table (CSV + SHA256 + retrieval date)
make test       # run pytest suite
make ruff       # lint
```

`make train`, `make ablate`, `make cross`, and `make api` correspond to
Phases 2, 2, 5, and 6 and are stubbed until those phases are implemented.

## Results summary

To be filled in after Phase 2 (see `results/`).

## Limitations

- KOI labels come from the NASA Exoplanet Archive cumulative table; live
  values may differ from DR25 paper numbers (see docs/DATA_NOTES.md).
- Grouped splits use `kepid` (host-star id); multi-planet systems share a
  host, so sibling KOIs never straddle train/test.

## Paper support

`scripts/make_figures.py` and `docs/METHODS.md` (Phase 7).
