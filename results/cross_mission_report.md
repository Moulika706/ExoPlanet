# ExoVet - cross-mission transfer, Kepler to TESS (Phase 5)

Model `xgboost` trained on 9564 Kepler KOIs with 9 shared physical features, grouped by host star.
It is applied to 8146 TESS objects without retraining the model.

## Features carried across the mission boundary

| Kepler feature | TOI column |
|---|---|
| `koi_period` | `pl_orbper` |
| `koi_duration` | `pl_trandurh` |
| `koi_depth` | `pl_trandep` |
| `koi_prad` | `pl_rade` |
| `koi_teq` | `pl_eqt` |
| `koi_insol` | `pl_insol` |
| `koi_steff` | `st_teff` |
| `koi_slogg` | `st_logg` |
| `koi_srad` | `st_rad` |

Two core features are left out, and the reason is not convenience:

- `koi_impact` - the TOI table has no impact parameter.
- `koi_kepmag` - the TOI table reports TESS magnitudes, a different photometric system from the Kepler magnitude.

## Labels

The TOI table carries the TFOP working group disposition, which is
mapped onto the three canonical labels:

| TFOPWG disposition | canonical label | rows |
|---|---|---|
| `PC` | CANDIDATE | 4820 |
| `APC` | CANDIDATE | 490 |
| `CP` | CONFIRMED | 815 |
| `KP` | CONFIRMED | 606 |
| `FP` | FALSE POSITIVE | 1315 |
| `FA` | FALSE POSITIVE | 100 |

The two vetting pipelines are not the same, so the classes are close
but not identical. A KOI CONFIRMED is a planet confirmed by
follow-up; a TOI `CP` is a planet confirmed by the TFOP program.
The FALSE POSITIVE classes differ more: the Kepler class is the
automatic pipeline's rejection of a large population (eclipsing
binaries among them), while the TESS class is a rejection by human
vetting of objects that looked plausible enough to be worth vetting.

## Results

| setting | set | macro-F1 | PR-AUC | log-loss | ECE | recall CONFIRMED | recall CANDIDATE | recall FALSE POSITIVE | conformal cov. | mean set |
|---|---|---|---|---|---|---|---|---|---|
| in domain | Kepler test fold | 0.6749 | 0.7462 | 0.6213 | 0.0205 | 0.8309 | 0.3384 | 0.8647 | 0.8966 | 1.48 |
| zero shot | TESS, Kepler calibrator | 0.2930 | 0.4407 | 1.6864 | 0.4040 | 0.7071 | 0.0626 | 0.7022 | 0.4740 | 1.48 |
| recalibrated | TESS, isotonic | 0.2939 | 0.4439 | 0.8301 | 0.0079 | 0.0054 | 0.9917 | 0.0446 | 0.9020 | 1.99 |
| recalibrated | TESS, label shift | 0.4144 | 0.4317 | 1.0816 | 0.2046 | 0.5777 | 0.4410 | 0.3626 | 0.8853 | 2.12 |
| in domain | TESS test fold | 0.4341 | 0.5757 | 0.7324 | 0.0274 | 0.0739 | 0.9623 | 0.2544 | 0.9085 | 1.75 |

Macro-F1 drop from Kepler to TESS, zero shot: -0.3819. After recalibration on TESS, isotonic: -0.3810, label shift: -0.2605.

## Where the failure is

One-vs-rest PR-AUC per class, with the class prior for reference.
A PR-AUC below the base rate means the model ranks that class
worse than chance on the target population.

| setting | PR-AUC CONFIRMED | PR-AUC CANDIDATE | PR-AUC FALSE POSITIVE | prior CONFIRMED | prior CANDIDATE | prior FALSE POSITIVE |
|---|---|---|---|---|---|---|
| Kepler test fold | 0.8257 | 0.4906 | 0.9222 | 0.2874 | 0.2069 | 0.5057 |
| TESS, zero shot | 0.3192 | 0.6773 | 0.3255 | 0.1799 | 0.6498 | 0.1703 |
| TESS, in domain | 0.4307 | 0.7894 | 0.5071 | 0.1743 | 0.6519 | 0.1737 |

Two separate things go wrong, and they are worth telling apart.

The ranking collapses on two of the three classes. On Kepler the
model separates FALSE POSITIVES very well (PR-AUC 0.92 against a
prior of 0.51), because the Kepler automatic pipeline rejected a
large population of obviously non-planet signals. That population
does not exist in the TESS table, where the rejected objects are the
subtle ones a human looked at and threw out. A model trained on
TESS from scratch only reaches PR-AUC 0.51 on that class, so most
of that collapse is the task being different, not the transfer
failing.

The prior moves a long way: CANDIDATE is 21% of the Kepler table
and 65% of the TOI table. A model fitted under the Kepler prior
barely ever predicts CANDIDATE on TESS (recall 0.06). The
label-shift correction is what repairs that part, and it is worth
most of the difference between the zero-shot and in-domain
macro-F1.

The CANDIDATE class is the one that survives the move: PR-AUC 0.68
zero-shot, against a prior of 0.65.

## Feature shift

A median that moves by a large fraction of the Kepler interquartile
range is a warning sign for the transfer.

| feature | Kepler median | TESS median | shift | shift in Kepler IQR | Kepler missing | TESS missing |
|---|---|---|---|---|---|---|
| koi_period | 9.753 | 4.147 | -5.606 | -0.148 | 0.00% | 1.36% |
| koi_duration | 3.793 | 2.752 | -1.041 | -0.271 | 0.00% | 0.05% |
| koi_depth | 421.1 | 4743 | +4322 | +3.290 | 3.80% | 0.00% |
| koi_prad | 2.39 | 10.49 | +8.097 | +0.598 | 3.80% | 6.38% |
| koi_teq | 878 | 1194 | +316.5 | +0.377 | 3.80% | 3.85% |
| koi_insol | 141.6 | 378.5 | +236.9 | +0.279 | 3.36% | 2.26% |
| koi_steff | 5767 | 5799 | +32.3 | +0.040 | 3.80% | 2.15% |
| koi_slogg | 4.438 | 4.33 | -0.108 | -0.332 | 3.80% | 11.07% |
| koi_srad | 1 | 1.236 | +0.236 | +0.457 | 3.80% | 7.71% |

## Conformal coverage

Target coverage 0.90.

| setting | coverage | cov. CONFIRMED | cov. CANDIDATE | cov. FALSE POSITIVE | mean set | singleton |
|---|---|---|---|---|---|---|
| in domain (Kepler test fold) | 0.8966 | 0.9109 | 0.8207 | 0.9194 | 1.48 | 0.5627 |
| zero shot (TESS, Kepler calibrator) | 0.4740 | 0.8501 | 0.2840 | 0.8014 | 1.48 | 0.5635 |
| recalibrated (TESS, isotonic) | 0.9020 | 0.7888 | 1.0000 | 0.6475 | 1.99 | 0.0632 |
| recalibrated (TESS, label shift) | 0.8853 | 0.8638 | 0.9291 | 0.7410 | 2.12 | 0.1108 |
| in domain (TESS test fold) | 0.9085 | 0.8028 | 0.9962 | 0.6855 | 1.75 | 0.3051 |

The split-conformal guarantee is an exchangeability statement: it
holds for a fresh sample from the same distribution as the
calibration set. TESS objects are not a fresh sample of Kepler
objects, so the coverage measured on TESS is an observation, not a
guarantee. The zero-shot row is the honest one: it is what the
product would do on a TESS object it had never seen.

## Reading the table

- The zero-shot row is the headline result. Nothing about the
  model changes between the two missions.
- The label-shift row says how much of the drop was a prior
  problem rather than a ranking problem: it recovers most of the
  gap to the in-domain ceiling, which is what one would expect
  from a class prior that moved from 21% to 65%.
- The isotonic row does not recover it, and the reason is visible
  in the recall column: a per-class isotonic map fitted on a
  population that is 65% CANDIDATE pushes the arg-max onto that
  class and buries the other two. On a shifted population the
  flexible calibrator is the wrong tool.
- The in-domain TESS row is the reference. It shows the ceiling of
  the same features on the same objects, and it is far below the
  Kepler number.
- The honest conclusion is that a Kepler-trained model does not
  transfer to TESS, that a prior correction recovers a good part
  of the damage, and that the ceiling itself is far lower because
  the two tables answer different questions. A product that claims
  to work on TESS needs to be trained on TESS.

