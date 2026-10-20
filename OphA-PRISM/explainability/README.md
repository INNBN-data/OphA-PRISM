# Model explanation

`ophaprism/explain.py` provides held-out permutation importance, ALE, and an optional KernelSHAP adapter. Evaluation patients and background observations must retain their fold provenance; background data come from the fitting partition.

The KernelSHAP adapter explains calibrated probabilities using sampled fitting rows and currently requires complete numeric inputs. Missing or categorical inputs require a compatible, verified explanation adapter. Empirical SHAP values, ALE curves, and uncertainty are generated from the institutional data, not supplied in this archive.
