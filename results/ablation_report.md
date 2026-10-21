# ExoVet - leakage ablation (Phase 2)

The `grouped` and `sibling_leak` rows are scored on the same
untouched test fold: the fold of the grouped split whose host stars
never appear in training. The `random` rows use an ordinary
stratified split, so their test rows differ as well.

- **grouped** - training data respects the host-star grouping.
- **random** - ordinary stratified split; sibling KOIs of a test
  object land in training, and the test rows differ from the
  grouped ones. This is the grouped-vs-random comparison.
- **sibling_leak** - same test rows as `grouped`, but the training
  pool is drawn at random, so a host star in the test fold usually
  has a sibling KOI in training. The difference between the two
  is therefore attributable to the sibling leak alone.

| model | features | split | macro-F1 | PR-AUC | log-loss | ECE | recall@99% |
|---|---|---|---|---|---|---|---|
| logistic_regression | A (clean) | grouped | 0.5272 | 0.6088 | 0.7797 | 0.0234 | 0.0548 |
| logistic_regression | A (clean) | random | 0.4986 | 0.5784 | 0.7881 | 0.0132 | 0.0000 |
| logistic_regression | A (clean) | sibling_leak | 0.5283 | 0.6067 | 0.7751 | 0.0252 | 0.0563 |
| logistic_regression | A+B (leaky) | grouped | 0.8352 | 0.8570 | 0.2579 | 0.0216 | 0.3467 |
| logistic_regression | A+B (leaky) | random | 0.8334 | 0.8538 | 0.2788 | 0.0195 | 0.3629 |
| logistic_regression | A+B (leaky) | sibling_leak | 0.8350 | 0.8583 | 0.2644 | 0.0178 | 0.3450 |
| random_forest | A (clean) | grouped | 0.6981 | 0.7666 | 0.5796 | 0.0189 | 0.1485 |
| random_forest | A (clean) | random | 0.6504 | 0.7489 | 0.6133 | 0.0213 | 0.0074 |
| random_forest | A (clean) | sibling_leak | 0.6960 | 0.7605 | 0.5844 | 0.0210 | 0.1273 |
| random_forest | A+B (leaky) | grouped | 0.8848 | 0.9285 | 0.2173 | 0.0080 | 0.4226 |
| random_forest | A+B (leaky) | random | 0.8787 | 0.9240 | 0.2237 | 0.0160 | 0.3333 |
| random_forest | A+B (leaky) | sibling_leak | 0.8903 | 0.9286 | 0.2221 | 0.0138 | 0.4352 |
| xgboost | A (clean) | grouped | 0.7290 | 0.7794 | 0.5598 | 0.0239 | 0.1320 |
| xgboost | A (clean) | random | 0.6857 | 0.7604 | 0.5714 | 0.0216 | 0.0869 |
| xgboost | A (clean) | sibling_leak | 0.7195 | 0.7692 | 0.5691 | 0.0220 | 0.1150 |
| xgboost | A+B (leaky) | grouped | 0.8921 | 0.9364 | 0.1997 | 0.0097 | 0.3746 |
| xgboost | A+B (leaky) | random | 0.8913 | 0.9303 | 0.1982 | 0.0156 | 0.4388 |
| xgboost | A+B (leaky) | sibling_leak | 0.8927 | 0.9335 | 0.2168 | 0.0134 | 0.3333 |

## Headline deltas (macro-F1)

| model | clean + grouped | leaky features | feature-leak delta | clean + random split | random-split delta | clean + sibling leak | sibling-leak delta |
|---|---|---|---|---|---|---|---|
| logistic_regression | 0.5272 | 0.8352 | +0.3080 | 0.4986 | -0.0286 | 0.5283 | +0.0012 |
| random_forest | 0.6981 | 0.8848 | +0.1867 | 0.6504 | -0.0477 | 0.6960 | -0.0021 |
| xgboost | 0.7290 | 0.8921 | +0.1631 | 0.6857 | -0.0433 | 0.7195 | -0.0095 |

## `koi_pdisposition` versus the target label

| koi_pdisposition | CANDIDATE | CONFIRMED | FALSE POSITIVE |
|---|---|---|---|
| CANDIDATE | 1977 | 2739 | 1 |
| FALSE POSITIVE | 0 | 9 | 4838 |

- P(disposition = FALSE POSITIVE | Kepler called FALSE POSITIVE) = 0.9981
- P(disposition = CONFIRMED | Kepler called CANDIDATE) = 0.5807

A single column that already decides the class for the large
majority of rows is why feature set B scores so much better
and why it cannot be used for the product.

## How much of the table is exposed to the sibling leak

- 8214 host stars, of which 938 (11.4%) carry more than one KOI.
- Those stars account for 2288 rows (23.9% of the table).

This is why the sibling-leak delta is small: most host stars have a
single KOI, and the physical features of one planet do not reveal
the disposition of a sibling. The leak is real but narrow, and it
is reported here as measured rather than as assumed.


## Reading the table

- Feature set A uses only physical measurements, so it generalises
  to a new mission. Set B includes vetting outputs and does not.
- The feature leak is large and one-directional: adding the
  vetting-output columns lifts macro-F1 by 0.16 to 0.31.
- The random split is the comparison the brief asks for, and it is
  the one other papers report. It is not a free choice: it puts
  sibling KOIs of a test object into training.
- The sibling leak is small on this table, but the grouped split is
  kept as the project default because it costs nothing and removes
  the possibility of a sibling KOI of a test object appearing in
  training.
- The headline model of this project is set A with the grouped
  split, whatever the other columns say.

