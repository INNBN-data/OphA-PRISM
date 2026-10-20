from __future__ import annotations
import numpy as np
import pandas as pd
from .common import ValidationError, binary, require_columns

ACCESS_COMPONENTS = ["clinoidectomy_complete", "optic_canal_unroofing", "optic_strut_drilling",
                     "distal_ring_opening", "falciform_opening", "optic_mobilization",
                     "cervical_ica_exposure", "suction_decompression"]
LABELS = {"VISION": "vision_event", "ACCESS": "access_high_burden", "CONTROL": "control_high_burden",
          "RECON": "recon_label", "CURE": "cure_failure"}


def derive(data):
    """Derive transparent definitions; retain uncertainty as missing, never zero."""
    require_columns(data, ["rroc_grade", "remnant_intentional_stable", "opha_patency_loss", "vision_event"])
    grade = pd.to_numeric(data.rroc_grade, errors="raise").astype("Float64")
    if (grade.notna() & ~grade.isin([1, 2, 3])).any():
        raise ValidationError("invalid_rroc_grade")
    accepted = binary(data.remnant_intentional_stable)
    favorable = pd.Series(pd.NA, index=data.index, dtype="boolean")
    favorable.loc[grade == 1] = True
    favorable.loc[grade == 3] = False
    favorable.loc[grade == 2] = (accepted.loc[grade == 2] == 1).astype("boolean")
    loss, vision = binary(data.opha_patency_loss), binary(data.vision_event)
    technical = favorable & (loss == 0).astype("boolean")
    cure = ~(technical & (vision == 0).astype("boolean"))
    out = pd.DataFrame({"favorable_angiography": favorable.astype("Float64"),
                        "technical_success": technical.astype("Float64"),
                        "cure_failure": cure.astype("Float64")}, index=data.index)
    if all(c in data for c in ACCESS_COMPONENTS):
        values = pd.concat([binary(data[c]) for c in ACCESS_COMPONENTS], axis=1)
        count = values.sum(axis=1, min_count=len(ACCESS_COMPONENTS)).astype("Float64")
        out["access_count"] = count
        out["access_count_ge5"] = (count >= 5).astype("Float64")
    flags = ["fenestrated_clip", "tandem_clip", "booster_clip"]
    if "clip_count" in data and all(c in data for c in flags):
        count = pd.to_numeric(data.clip_count, errors="raise").astype("Float64")
        if (count.notna() & ((count < 1) | (count % 1 != 0))).any():
            raise ValidationError("invalid_clip_count")
        label = (count > 1).astype("boolean")
        for c in flags:
            label |= (binary(data[c]) == 1).astype("boolean")
        out["recon_label"] = label.astype("Float64")
    return out


def validate_and_add(data):
    out = data.copy()
    derived = derive(data)
    for c in derived:
        if c in out:
            saved = pd.to_numeric(out[c], errors="raise").astype("Float64")
            compare = saved.notna() & derived[c].notna()
            if (saved[compare] != derived.loc[compare, c]).any():
                raise ValidationError("saved_endpoint_disagrees_with_source_rule: " + c)
            # Do not accept a supplied label when the required components remain unknown.
            if c in ("cure_failure", "technical_success") and (saved.notna() & derived[c].isna()).any():
                raise ValidationError("saved_endpoint_has_unresolved_components: " + c)
        out[c] = derived[c]
    for c in LABELS.values():
        require_columns(out, [c])
        out[c] = binary(out[c])
    if "vision_6mo_confirmed" in out:
        confirmed = binary(out.vision_6mo_confirmed)
        if ((out.vision_event == 1) & (confirmed.isna() | (confirmed != 1))).any():
            raise ValidationError("vision_event_lacks_six_month_confirmation")
    return out


def labels(data, task):
    if task not in LABELS:
        raise ValidationError("unknown_task")
    value = binary(data[LABELS[task]])
    if value.isna().any():
        raise ValidationError("missing_outcome_requires_explicit_evaluable_cohort")
    return value.astype(int).to_numpy()
