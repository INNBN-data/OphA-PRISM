# Statistical and computational methods

## Targets and information eligibility

The analysis unit is the patient and index aneurysm. The dictionary contains 408 unique variables; exactly 142 variables from T0 (presentation), T1 (preoperative visual assessment), and T2 (preoperative anatomy) are eligible predictors. Identifiers, treatment year, era, surgeon, clinician ratings, operative variables, and postoperative outcomes are excluded from the anatomical predictor set. The encoded design-matrix dimension is determined from the actual data.

VISION uses the adjudicated persistent visual outcome. ACCESS and CONTROL use supplied, fixed cohort-derived phenotype labels. RECON and CURE are checked against their available source components. Missing required outcomes cause validation failure; unknown observations are not assigned to the negative class.

## Nested evaluation and calibration

The default design is five-fold outer cross-validation repeated ten times, with four inner folds. Splits are stratified by the task outcome and grouped by patient. The executable splitter does not jointly stratify by era. Era is used separately for temporal evaluation.

Within each outer training partition, 20% of patients are reserved for conformal calibration. Of the remainder, 20% are reserved for probability calibration. The approximate fractions of the complete development cohort are therefore:

| Role | Fraction before rounding |
| --- | ---: |
| Model fitting and inner tuning | 0.512 |
| Probability calibration | 0.128 |
| Conformal calibration | 0.160 |
| Outer evaluation | 0.200 |

Duplicate bootstrap instances of a patient remain in the same role. Preprocessing and feature selection use the relevant fitting partition. Probability calibration uses an isotonic regression fitted after model tuning. Conformal calibration follows probability calibration on separate observations. Sparse event support may make this procedure non-estimable; isotonic calibration also remains unstable with few events.

These allocations define the reference implementation. They require reconciliation with the historical study workflow before the release can be used to reproduce the manuscript.

## Preprocessing, selection, and fitting

The default preprocessing ensemble comprises 20 seeded numeric `IterativeImputer` fits, each with ten update cycles and posterior sampling. Binary and categorical variables use fitting-partition mode imputation and one-hot encoding. Numeric scaling is estimated from fitting observations. Outcomes are excluded from the imputation matrix. During prediction, each row is transformed using a separate copy of the frozen posterior-imputer state, preserving prediction invariance to batch order.

This procedure differs from R MICE using PMM/logreg/polyreg. The deterministic median/mode backend is used for software tests.

Sparse selection uses 1,000 grouped, class-stratified bootstrap Elastic Net fits on the first imputation. An original feature is eligible for retention when at least one encoded column has a nonzero coefficient in at least 75% of fits. At most six original features are retained and applied across the imputation ensemble. Selection failure is reported when no feature qualifies.

Model families and fixed parameters are specified in `config/model_settings.yaml`. Optuna searches the ranges in `ophaprism/training.py`, minimizing mean inner-validation log-loss over 100 serial trials. Boosting uses up to 1,000 rounds and early-stopping patience of 50. Outer evaluation patients do not enter the stopping sets. The selected configuration is refitted using the median inner selected round. The final probability is the mean across the 20 imputation-specific estimators, followed by the fitted probability calibrator. Held-out probabilities are averaged across the ten outer repeats for patient-level evaluation.

## Performance measures

For outcomes $y_i\in\{0,1\}$, probabilities $p_i$, and evaluation prevalence $\pi=n^{-1}\sum_i y_i$, the Brier score and prevalence-reference skill score are

$$
\mathrm{BS}=\frac{1}{n}\sum_{i=1}^{n}(p_i-y_i)^2,
\qquad
\mathrm{BSS}=1-\frac{\mathrm{BS}}{\pi(1-\pi)}.
$$

The denominator is the evaluation-cohort prevalence benchmark. It is not a separately fitted cross-validated null model.

ROC-AUC is calculated from positive–negative pairs:

$$
\mathrm{AUC}=\frac{N_{\mathrm{concordant}}+\tfrac12N_{\mathrm{tied}}}{n_1n_0}.
$$

Average precision and trapezoidal precision–recall area are reported separately. Calibration output distinguishes the intercept and slope of joint logistic recalibration from the calibration-in-the-large intercept obtained with the predicted logit as an offset. Probabilities are clipped to $[10^{-6},1-10^{-6}]$ for these calibration fits. Nonconvergent or non-estimable fits retain their status.

Metric and paired-difference intervals use patient-level stratified percentile resampling of fixed prediction vectors, with 2,000 replicates by default. These intervals condition on the saved predictions; they do not include full training or model-selection uncertainty.

## Prediction sets

The direct split-conformal implementation uses $\alpha=0.10$. For $m$ calibration observations, the nonconformity score is $s_i=1-\hat p_{y_i}(x_i)$, with order-statistic rank

$$
k=\left\lceil(m+1)(1-\alpha)\right\rceil.
$$

The threshold is the $k$-th ordered score when $k\le m$, and $+\infty$ otherwise. A candidate class $c$ is included when $1-\hat p_c(x)\le q$. Ties use deterministic inclusion; empty sets remain empty.

The prespecified reporting repeat is 0, giving one held-out set per patient. Probability metrics instead average all ten repeats. Set unions across repeats are not used. Coverage is reported overall and by true class, together with singleton PPV/NPV and the complete true-class-by-set table. For binary targets,

$$
\overline{|\Gamma|}=P(|\Gamma|=1)+2P(|\Gamma|=2).
$$

Coverage and singleton-accuracy intervals use Wilson intervals. The implementation is marginal, not Mondrian; validity depends on exchangeability and can be affected by temporal selection.

## Baselines and temporal evaluation

CURE comparators comprise a hard diameter $\ge10$ mm rule, a fitted binary-indicator logistic model, a three-knot cubic B-spline diameter model, and a diameter/neck/compression logistic model. Outer evaluation folds are matched. The logistic baselines use the full outer training partition; the principal procedure reserves separate calibration subsets. The reference B-spline specification differs from a restricted-cubic-spline design.

For a hard binary rule, $\mathrm{AUC}=(\mathrm{sensitivity}+\mathrm{specificity})/2$ and the Brier score equals the misclassification proportion.

Development eras are 1998–2006, 2007–2016, and 2017–2025. LOEO trains in the other two eras. Rolling-origin evaluation trains in Era 1 and tests in Era 2, then trains in Eras 1–2 and tests in Era 3. Pooled AUC uses concatenated held-out patient predictions. Non-estimable windows retain their status. The 28-patient 2026 cohort is separate. Evaluation of this new implementation cannot retroactively establish that it was frozen before historical holdout access.

## Optimism and permutation diagnostics

Each bootstrap replicate refits the reference pipeline on a class-stratified patient sample and evaluates that fitted bundle in both the bootstrap sample and the original cohort:

$$
\widehat O=\frac{1}{B}\sum_{b=1}^{B}
\left(\mathrm{AUC}^{\mathrm{bootstrap}}_b-
\mathrm{AUC}^{\mathrm{original}}_b\right),
\qquad
\mathrm{AUC}_{\mathrm{corrected}}=
\mathrm{AUC}_{\mathrm{apparent,original}}-\widehat O.
$$

Apparent-original performance must come from the same configured procedure. Replicate quantiles describe the optimism distribution; the reference code does not calculate BCa intervals for corrected performance.

Permutation diagnostics globally shuffle the outcome vector, preserve its marginal count, and repeat nested evaluation. The recorded number of changed labels checks the shuffle. With $P$ complete replicates, the one-sided empirical value is

$$
p_{\mathrm{perm}}=\frac{1+\sum_{b=1}^{P}
I(\mathrm{AUC}_b\ge\mathrm{AUC}_{\mathrm{observed}})}{P+1}.
$$

Failed replicates are retained. Corrected-performance and permutation inference require all requested replicates and matching provenance. The intended totals are 2,000 bootstrap and 1,000 permutation replicates; these full runs have not been executed for this release.

## Explanation and decision analysis

Permutation importance uses held-out observations. ALE bin boundaries are derived from fitting data; empty evaluation bins are identified explicitly. The optional KernelSHAP adapter explains calibrated probabilities using sampled fitting-row backgrounds and currently requires complete numeric inputs. Bootstrap bands and the historical explanation pipeline require the corresponding verified server outputs. Explanations characterize fitted predictions rather than causal effects.

At planning threshold $t$, net benefit is

$$
\mathrm{NB}(t)=\frac{\mathrm{TP}(t)}{n}
-\frac{\mathrm{FP}(t)}{n}\frac{t}{1-t},
$$

with $\mathrm{NB}_{\mathrm{all}}(t)=\pi-(1-\pi)t/(1-t)$ and $\mathrm{NB}_{\mathrm{none}}(t)=0$. Frozen probabilities determine the actions directly. The proposed planning actions and threshold ranges are specified in [FIGURE_PROVENANCE.md](FIGURE_PROVENANCE.md); they do not establish a treatment effect.
