from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from .common import configuration, identifiers, private_output, ValidationError
from .data import load_cohort
from .endpoints import labels
from .workflow import split_outer


def comparisons(data_root, output_root, repeat, fold, cfg=None):
    """CURE baselines on exactly the configured outer evaluation split."""
    cfg = cfg or configuration()
    data, _ = load_cohort(data_root, cfg=cfg)
    y, groups = labels(data, "CURE"), identifiers(data).to_numpy()
    x = data[["max_diameter_mm", "neck_width_mm", "optic_compression"]]
    a, b = split_outer(x, y, groups, cfg, repeat)[fold]
    out = []
    for name in ("diameter_ge10_hard", "diameter_ge10_probability", "diameter_continuous_spline", "size_neck_compression"):
        if name == "diameter_ge10_hard":
            if x.iloc[b].max_diameter_mm.isna().any():
                raise ValidationError("hard_diameter_rule_requires_observed_size")
            probability = (x.iloc[b].max_diameter_mm >= 10).astype(float).to_numpy()
        else:
            if name == "diameter_ge10_probability":
                z = pd.DataFrame({"size_ge10": (x.max_diameter_mm >= 10).astype(float).where(x.max_diameter_mm.notna())})
            elif name == "diameter_continuous_spline":
                z = x[["max_diameter_mm"]]
            else:
                z = x
            steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
            if name == "diameter_continuous_spline":
                steps.append(("spline", SplineTransformer(n_knots=3, degree=3, include_bias=False)))
            steps.extend([("scale", StandardScaler()), ("model", LogisticRegression(penalty=None, max_iter=5000))])
            fitted = Pipeline(steps).fit(z.iloc[a], y[a])
            probability = fitted.predict_proba(z.iloc[b])[:, 1]
        out.append(pd.DataFrame({"patient_id": groups[b], "task": "CURE", "model": name,
                                 "repeat": repeat, "outer_fold": fold, "y_true": y[b], "probability": probability}))
    destination = private_output(Path(output_root) / "CURE/baselines")
    pd.concat(out, ignore_index=True).to_csv(destination / f"repeat_{repeat}_fold_{fold}.csv", index=False)
    return {"baseline_models": len(out), "test_n": len(b), "repeat": repeat, "fold": fold,
            "note": "Logistic baselines fit on full outer training set; primary reference reserves calibration data."}
