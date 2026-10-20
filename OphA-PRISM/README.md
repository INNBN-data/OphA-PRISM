# OphA-PRISM

**Reference analysis software for preoperative prediction of visual outcomes and operative demands in ophthalmic-segment aneurysm microsurgery.**

Version **1.0.1** · MIT License

This repository provides endpoint validation, nested model evaluation, temporal assessment, calibration, conformal prediction, diagnostic resampling, and empirical plotting. It accompanies *OphA-PRISM for Preoperative Prediction of Visual Outcomes and Operative Demands in Ophthalmic Segment Aneurysm Microsurgery*.

Clinical records, individual predictions, and fitted study models are excluded from this archive.

## Scope

The reported study comprises 261 development patients treated during 1998–2025 and 28 patients in a reserved 2026 cohort. Of 408 registry variables, 142 T0–T2 variables are eligible preoperative predictors. Five binary targets are evaluated:

| Target | Positive class |
| --- | --- |
| VISION | Persistent new visual morbidity |
| ACCESS | High observed exposure burden |
| CONTROL | High observed vascular-control burden |
| RECON | Reconstructive clipping |
| CURE | Failure of favorable angiography, ophthalmic artery patency, or visual preservation |

Clinical outcomes and observed operative choices have different interpretations. Accepted RROC II remnants contribute to favorable angiography; complete occlusion refers to RROC I alone. Definitions are specified in [ENDPOINTS.md](docs/ENDPOINTS.md).

## Installation

Use Python 3.10 in the approved analysis environment. Dependency constraints and the reported R environment are documented in [ENVIRONMENT.md](docs/ENVIRONMENT.md).

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python repository_audit.py
```

For offline execution, use the institutional package mirror or approved local wheels.

## Required inputs

Paths are relative to `--data-root`; filenames and dictionary-column mappings are configured in `config/model_settings.yaml`.

| Input | Requirement |
| --- | --- |
| `cohort_flow_export_v2.csv` | 261 development patients; one row per index patient |
| `holdout_cohort_v1.csv` | 28 reserved patients; separate identifiers |
| `data_dictionary_v2.csv` | 408 unique variable definitions with temporal layers and eligibility flags |
| `eligible_t0_t2_features.txt` | 142 unique feature names, one per line, in prediction order |

The public `data/` directory contains a header-only schema and 26 documented fields. Training requires the complete server dictionary and feature list.

## Execution

Run commands from the repository root.

### Audit and validate

```bash
python run_pipeline.py audit --data-root /server/study/project --output aggregate_audit_01.json
Rscript server_audit/audit_r_environment.R
python run_pipeline.py validate --data-root /server/study/project
```

The audit reads existing source files and produces aggregate checks. Validation excludes the reserved cohort unless `--include-holdout` is supplied.

### Nested evaluation

```bash
python run_pipeline.py train --data-root /server/study/project --task VISION --run reference_01 --repeat 0 --fold 0
```

The default design uses five outer folds, ten repeats, four inner folds, 100 tuning trials, and 20 imputations. An array job executes all 50 outer jobs for one task:

```bash
sbatch scripts/reference_fold.slurm /server/study/project VISION reference_01
python run_pipeline.py collect --task VISION --run reference_01
python run_pipeline.py fit-final --data-root /server/study/project --task VISION --run reference_01
```

Benchmark one fold before scheduling a complete analysis. The array allows 16 concurrent one-core jobs.

### Temporal assessment and diagnostics

```bash
python run_pipeline.py temporal --data-root /server/study/project --task CURE --run reference_01 --test-era 3
python run_pipeline.py temporal --data-root /server/study/project --task CURE --run reference_01 --test-era 3 --rolling
python run_pipeline.py baselines --data-root /server/study/project --task CURE --run reference_01 --repeat 0 --fold 0
python run_pipeline.py diagnostic --data-root /server/study/project --task CURE --run reference_01 --kind bootstrap --index 0
```

Each diagnostic command executes one replicate. `python run_pipeline.py --help` lists the available commands; append `--help` to a command for its arguments.

### Empirical plots

```bash
python run_pipeline.py figures --source runs/private/reference_01/VISION/pooled_oof_predictions.csv --task VISION --run reference_01
```

Plotting requires unique-patient outcomes and probabilities. The module exports diagnostic panels and aggregate coordinates. The complete main-text display specification is in [FIGURE_PROVENANCE.md](docs/FIGURE_PROVENANCE.md).

## Methods and provenance

[METHODS.md](docs/METHODS.md) specifies the executable procedure and metric equations. Material differences from the reported study workflow include Python numeric chained imputation with categorical mode imputation, separate probability/conformal calibration partitions, and prediction sets from one prespecified repeat. The reported R MICE workflow and historical fitted models require separate verification.

Generated predictions, preprocessing objects, models, and explanations remain in `runs/private/`. Author-reported summaries in `results/author_reported_metrics.yaml` are stored separately from computed outputs and are not read by the training code. [RELEASE_STATUS.md](docs/RELEASE_STATUS.md) distinguishes software verification from empirical reproduction.

## Structure

| Directory | Content |
| --- | --- |
| `ophaprism/` | Shared analytical implementation |
| `config/` | Model settings, paths, and seeds |
| `data/` | Public metadata and input specification |
| `preprocessing/`, `models/`, `validation/` | Validation, fitting, and temporal entry points |
| `calibration/`, `uncertainty/` | Probability and prediction-set evaluation entry points |
| `score_distillation/`, `explainability/`, `figures/` | Developmental scores, explanations, and plotting |
| `server_audit/` | Aggregate verification of existing institutional files |
| `results/` | Reported summaries and approved aggregate-output policy |
| `docs/`, `scripts/`, `tests/` | Documentation, release tools, and software tests |

## Release

```bash
python scripts/submission_preflight.py
python scripts/build_release.py
```

The builder preserves directory paths and includes only `RELEASE_FILES.txt`, with file checksums in `MANIFEST.sha256`. `--require-verified-results` returns a nonzero status until empirical verification is available. Upload instructions are in [PUBLIC_UPLOAD.md](docs/PUBLIC_UPLOAD.md).

## Citation and governance

Citation metadata are supplied in `CITATION.cff`. The code is licensed under MIT; third-party dependencies retain their own licenses. Clinical-data access is subject to institutional approval. Ethics approval, funding, affiliations, and correspondence are recorded in [ETHICS_AND_CONTACTS.md](docs/ETHICS_AND_CONTACTS.md).
