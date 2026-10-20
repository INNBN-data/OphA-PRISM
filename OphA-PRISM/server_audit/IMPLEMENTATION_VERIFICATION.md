# Implementation verification requirements

Exact reproduction requires reconciliation of the manuscript, controlled source records, and historical executable analysis. Only approved metadata, aggregate summaries, and relevant code excerpts are needed outside the server.

| Area | Required evidence |
| --- | --- |
| Cohort accounting | Screening-ledger totals and exclusions; index-patient rules; large-aneurysm component intersections |
| Schema and timing | Actual headers; 408-variable dictionary; ordered 142-feature list; source hashes and temporal eligibility |
| Visual adjudication | Exact acuity/perimetry rules, persistence requirement, documentation source, and evaluable denominators |
| Imaging measurement | Operational definitions, modality visibility, reliability pairs, and final dataset version |
| Imputation | Historical `mice()` call, methods, `predictorMatrix`, `ignore`, `where`, outcome masking, and held-out application |
| Data roles | Patient and event counts for fitting, tuning/stopping, probability calibration, conformal calibration, and outer evaluation |
| Repeated evaluation | Actual saved outer assignments, probability pooling, set-generation calls, and the repeat or rule defining one reported set per patient |
| Final estimators | Model-family selection rule, original search spaces, final parameters, preprocessing/calibration objects, and iteration histories |
| Bootstrap | Apparent-original statistic, refitting/evaluation loop, replicate records, failures, and any claimed BCa statistic/jackknife implementation |
| Permutations | Shuffle unit, event-count preservation, changed-label counts, repeated fitting steps, and observed comparison statistic |
| Clinician agreement | Rating provenance, operating-surgeon exclusions, estimator, missing-rater handling, and actual agreement denominator |
| Sample size | Candidate design matrix before selection, pruning record, anticipated fit justification, package version, and full console output |
| Integer scores | Coefficient transformation, bins, scale, intercept/slope mapping, missing-input rule, and separately evaluated integer-score performance |
| Temporal evaluation | LOEO and rolling-origin predictions, test-window event counts, final reserved-cohort metrics, and documented evaluation dates |
| Provenance | Historical lock evidence, original model identifier, executable code version, environment capture, and holdout access history |

## Specific numerical checks

The supplied CURE sample-size inputs are `type="b"`, `parameters=15`, `rsquared=0.598`, `prevalence=50/261`, and `shrinkage=0.90`. The reported requirement is 306. A source-equivalent check of pmsampsize 1.1.0 gives 309 after intermediate rounding; this is not an executed package result. Retain the actual version-specific output from `audit_r_environment.R` for the final report. The candidate sample-size assessment does not establish adequacy of the broader 142-variable search.

Report the VISION calculation separately with an admissible anticipated Cox–Snell R². A post-selection six-variable representation does not determine the original candidate parameter count.

Missingness proportions require explicit eligible denominators. Structural inapplicability is distinguished from an unrecorded measurement. The R MICE completed-data extraction call alone is insufficient to document prediction for a new patient without an outcome.

The clinician agreement record must identify whether subjects with an excluded operating-surgeon rating entered the estimator. The supplied R diagnostic checks the installed function's missing-rater handling; it does not estimate clinical agreement.

For each metric, retain the exact patient set, event count, source version, probability scale, and interval method. Average precision and trapezoidal precision–recall area are separate measures. Optimism-replicate quantiles and BCa corrected-performance intervals are separate calculations.

## Source documentation

- [MICE completed-data extraction](https://amices.org/mice/reference/complete.mids.html)
- [MICE methods and row masks](https://amices.org/mice/reference/mice.html)
- [MAPIE 0.8.0 classifier](https://mapie.readthedocs.io/en/v0.8.0/generated/mapie.classification.MapieClassifier.html)
- [irr 0.84.1 Fleiss κ implementation](https://github.com/cran/irr/blob/0.84.1/R/kappam.fleiss.R)
- [pmsampsize 1.1.0 implementation](https://github.com/cran/pmsampsize/blob/1.1.0/R/pmsampsize.R)

These references specify software behavior. The study implementation is established by its own code and execution records.
