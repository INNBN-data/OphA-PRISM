# Software verification

**Release:** 1.0.1. **Verification date:** 20 October 2026.

| Check | Result |
| --- | --- |
| Unit tests | 48 discovered; 45 passed; 3 skipped; no failures or errors |
| Optional model checks | CatBoost, LightGBM, and XGBoost skipped because the libraries were unavailable locally |
| Python sources | Syntax checked |
| YAML, JSON, TOML, and citation metadata | Parsed |
| Slurm script | Bash syntax checked |
| Documentation links and release versions | Checked |
| Public archive | Directory structure, file checksums, and ZIP integrity checked |

Tests cover endpoint consistency, accepted remnants, missing labels, score bounds, conformal order statistics, empty sets, patient-role isolation, frozen imputation, feature-selection failure, bootstrap counts, paired metrics, serialization, plotting, and release exclusions. Fixtures are artificial software inputs, not clinical observations.

## Runtime

Python 3.12.14; NumPy 2.3.5; pandas 2.2.3; SciPy 1.17.0; scikit-learn 1.8.0; PyYAML 6.0.3. The declared analysis environment targets Python 3.10 and separately specified dependencies.

## Verification boundary

The controlled clinical inputs, complete dictionary, original predictions, and historical fitted objects were unavailable. Full tuning/imputation runs, diagnostic refits, optional model branches, R diagnostics, the conda environment, and the Docker build were not executed locally. Manuscript numerical results and clinical performance remain unverified.
