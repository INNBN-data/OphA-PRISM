from __future__ import annotations
import numpy as np
import pandas as pd
from .common import ValidationError, binary, require_columns


def integers(data, task):
    common = ["max_diameter_mm", "neck_width_mm", "optic_compression", "opha_incorporation", "intraluminal_thrombus"]
    extra = ["neck_opha_involvement_pct"] if task == "VISION" else ["preoperative_visual_symptoms"]
    if task not in ("VISION", "CURE"):
        raise ValidationError("integer_score_defined_only_for_vision_and_cure")
    require_columns(data, common + extra)
    d = data[common + extra].apply(pd.to_numeric, errors="raise")
    if d.isna().any().any() or not np.isfinite(d.to_numpy(float)).all():
        raise ValidationError("missing_score_inputs_require_locked_training_derived_imputation")
    if (d.max_diameter_mm < 0).any() or (d.neck_width_mm < 0).any():
        raise ValidationError("negative_anatomical_measurement")
    for c in ["optic_compression", "opha_incorporation", "intraluminal_thrombus"] + ([] if task == "VISION" else extra):
        binary(d[c])
    if task == "VISION":
        if not d.neck_opha_involvement_pct.between(0, 100).all():
            raise ValidationError("neck_opha_percentage_outside_0_100")
        score = 2 * np.digitize(d.max_diameter_mm, [5, 10, 15])
        score += 2 * np.digitize(d.neck_width_mm, [3, 5, 7])
        score += 8 * d.optic_compression.astype(int) + 5 * d.opha_incorporation.astype(int)
        score += 4 * (d.neck_opha_involvement_pct >= 20).astype(int) + 2 * d.intraluminal_thrombus.astype(int)
    else:
        score = 8 * (d.max_diameter_mm >= 10).astype(int) + 6 * (d.neck_width_mm >= 5).astype(int)
        score += 7 * d.optic_compression.astype(int) + 5 * d.opha_incorporation.astype(int)
        score += 4 * d.preoperative_visual_symptoms.astype(int) + 3 * d.intraluminal_thrombus.astype(int)
    return np.asarray(score, int)


def probabilities(score, intercept=None, slope=None):
    if intercept is None or slope is None or not np.isfinite([intercept, slope]).all():
        raise ValidationError("actual_validated_score_probability_mapping_required")
    z = np.clip(intercept + slope * np.asarray(score, float), -700, 700)
    return 1 / (1 + np.exp(-z))
