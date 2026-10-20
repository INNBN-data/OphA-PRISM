# Computational environment

## Reported study environment

| Component | Reported version |
| --- | --- |
| Operating system | Ubuntu 22.04.4 LTS |
| Python | 3.10.12 |
| R | 4.3.1 |
| NumPy / pandas / scikit-learn | 1.24.3 / 2.0.3 / 1.3.0 |
| XGBoost / LightGBM / CatBoost | 1.7.6 / 4.0.0 / 1.2 |
| SHAP / MAPIE | 0.42.1 / 0.8.0 |
| mice / pmsampsize | 3.16.0 / 1.1.0 |
| mclust / logistf / irr | 5.4.10 / 1.24.1 / 0.84.1 |

These author-supplied versions have not been verified against an original environment capture.

## Reference environment

`requirements-core.txt` and `requirements-models.txt` specify the Python installation constraints. The reference dependencies include bounded versions of SciPy, PyYAML, joblib, matplotlib, Optuna, and statsmodels in addition to the reported model-library versions. These constraints are not a complete historical lock file. The conformal rule is implemented directly; MAPIE is not required by the reference implementation.

`environment.yml` specifies Python 3.10.12. `Dockerfile` defines an optional Python-only image and excludes R and clinical data. The conda environment has not been solved and the image has not been built in the local verification environment.

For a completed server run, retain `pip freeze`, `sessionInfo()`, hardware and BLAS settings, thread limits, job seeds, and the container digest when applicable. Offline installation uses institutional packages or approved wheels. Local test versions are recorded in [SOFTWARE_TEST_REPORT.md](SOFTWARE_TEST_REPORT.md).
