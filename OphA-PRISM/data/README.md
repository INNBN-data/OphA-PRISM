# Input specification

`registry_schema.csv` contains column headings only. `confirmed_variable_subset.csv` documents 26 named fields; it is a partial schema, not the complete 408-variable dictionary.

The private server inputs are configured in `config/model_settings.yaml`:

| File | Content |
| --- | --- |
| `cohort_flow_export_v2.csv` | Development records |
| `holdout_cohort_v1.csv` | Reserved-cohort records |
| `data_dictionary_v2.csv` | Full variable definitions and temporal eligibility |
| `eligible_t0_t2_features.txt` | Ordered list of 142 predictor names |

Each dictionary entry should specify name, definition, unit, source, temporal layer, derivation, missingness/applicability rules, and allowed use. Eligible names must match the flagged T0–T2 dictionary entries exactly. Cohort fields must use the canonical schema or be transformed by a documented server-side mapping before validation.

Use `--data-root` for the controlled input location. Patient records and images remain outside the public repository.
