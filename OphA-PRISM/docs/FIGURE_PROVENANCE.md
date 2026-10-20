# Figure specification and provenance

All planned displays are main-text figures. Each numerical panel requires a verified source table, analysis version, patient denominator, event count, and uncertainty definition. Reported scalar metrics alone do not specify empirical curves.

| Figure | Content | Required source |
| --- | --- | --- |
| 1 | Cohort flow and study chronology | Mutually exclusive screening ledger; documented data, model-lock, access, and assessment dates |
| 2 | VISION/CURE ROC, precision–recall, and calibration | Unique-patient held-out probabilities and outcomes |
| 3 | ACCESS/CONTROL/RECON ROC, precision–recall, and calibration | Task-specific unique-patient held-out predictions |
| 4 | CURE components, diameter comparator, and decision curves | Component flags; diameter; matched baseline and model predictions |
| 5 | Optimization trajectories, optimism, and permutation diagnostics | Iteration histories; bootstrap and permutation replicate records |
| 6 | Anatomical attribution and feature stability | Held-out SHAP/ALE outputs; training-only reference information; resample ranks |
| 7 | LOEO and rolling-origin evaluation | Patient-level temporal predictions; test-window counts and dates |
| 8 | Class-specific prediction-set performance | One documented held-out set per patient, with true class |
| 9 | Dependence among the five targets | Final paired binary labels and evaluable denominators |
| 10 | Ordered anatomical-domain analysis | Matched-split predictions for each prespecified domain stage |

## Statistical conventions

ROC panels include the diagonal reference and precision–recall panels the relevant outcome prevalence. Calibration panels include the identity line, probability distribution, and event support. Patient-bootstrap bands must state whether model fitting is repeated or predictions are held fixed.

Training curves retain inner fitting/validation provenance and the selected iteration. Log-loss and AUC histories are shown only when retained in the actual logs. Feature-domain additions are displayed separately from optimization trajectories.

Decision curves include the model, simple baselines, act-on-all, and act-on-none. Prediction-set panels report true-class coverage, positive/negative singleton accuracy, and empty/singleton/two-class proportions, with counts and Wilson intervals. Marginal-method class-specific results are empirical diagnostics, not class-conditional guarantees.

## Planning-action definitions

| Task | Proposed preoperative action | Risk-threshold range |
| --- | --- | ---: |
| VISION | Extended visual-risk review or additional simulation | 0.05–0.30 |
| ACCESS | Additional exposure and adjunct planning | 0.20–0.50 |
| CONTROL | Preparation for temporary vascular control | 0.15–0.40 |
| RECON | Preparation of reconstructive clip options | 0.30–0.60 |
| CURE | Multidisciplinary review of treatment options | 0.10–0.40 |

The threshold step is 0.01. These are developmental planning policies; the selected surgical cohort does not estimate the effect of changing treatment modality.

## Current rendering scope

`ophaprism/figures.py` produces ROC, precision–recall, and binned calibration diagnostics, plus aggregate decision-curve coordinates. The full composed layouts, smoothed calibration bands, training-history displays, and explanation uncertainty require additional verified source material and figure assembly. No empirical clinical figure is distributed in this release.
