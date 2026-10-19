# Data Notes — ExoVet KOI pipeline

## Source

NASA Exoplanet Archive, Kepler KOI cumulative table, fetched with
`astroquery.ipac.nexsci.nasa_exoplanet_archive.NasaExoplanetArchive`
(`table="cumulative"`) with a TAP CSV fallback
(`https://exoplanetarchive.ipac.caltech.edu/tap/sync?query=SELECT * FROM cumulative&format=csv`).

Raw outputs live in `data/raw/`: `koi_cumulative.csv`,
`koi_cumulative.csv.sha256`, and `retrieval.json` (retrieval date, source,
row/column counts, checksum).

## Observed schema (live archive)

9,564 rows × 108 columns. Host-star id column `kepid` exists (int32) and is
used as the grouping key for train/test splits, so sibling KOIs of the same
star never straddle train and test.

Target column `koi_disposition` contains exactly three values in this pull:

- CONFIRMED: 2,748
- CANDIDATE: 1,977
- FALSE POSITIVE: 4,839

No rows required dropping for label reasons; cleaning keeps these three
classes and drops anything else defensively.

## Leakage-risk columns (excluded from the main model)

These are outputs of the Kepler vetting process and encode the label
directly or near-directly. They are kept out of feature set A and added only
for the set-B ablation (see `scripts/ablate_leakage.py`):

| Column | Meaning |
|---|---|
| `koi_score` | Kepler vetting confidence, 0-1 |
| `koi_pdisposition` | Kepler's own CANDIDATE / FALSE POSITIVE call |
| `koi_fpflag_nt` | false-positive test: neutral transit |
| `koi_fpflag_ss` | false-positive test: stellar secondary |
| `koi_fpflag_co` | false-positive test: centroid offset |
| `koi_fpflag_ec` | false-positive test: eclipsing binary |

`koi_pdisposition` is particularly dangerous: it is the archive's own
disposition prediction and is essentially label leakage. `koi_score` is a
vetting confidence derived from the same tests.

## Core physical features (set A)

`koi_period`, `koi_duration`, `koi_depth`, `koi_prad`, `koi_teq`,
`koi_impact`, `koi_insol`, `koi_steff`, `koi_slogg`, `koi_srad`, `koi_kepmag`.

## Missing values

Several physical columns contain missing values (see the missing-value-rate
printout of `scripts/fetch_data.py`); imputation is performed inside the CV
pipeline only, never on the full data set.

## Discrepancies vs PROJECT_CONTEXT.md

None observed: row count (~9,564), target values, core features, leakage
columns and the `kepid` grouping key all match the document. If a future
pull differs, record it here before proceeding.
