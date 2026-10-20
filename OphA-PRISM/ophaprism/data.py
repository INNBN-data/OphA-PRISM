from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
from .common import ValidationError, configuration, read_csv, identifiers, require_columns
from .endpoints import validate_and_add, LABELS

FORBIDDEN = set(LABELS.values()) | {
    "patient_id", "lesion_id", "op_id", "cohort_flag", "surgery_year", "era", "surgeon_id",
    "technical_success", "rroc_grade", "opha_patency_loss", "remnant_intentional_stable",
    "followup_days", "vision_6mo_confirmed", "outcome_missing_flag", "angiography_days",
    "angiography_modality", "clinician_rating", "mean_clinician_rating"}


def file_under(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValidationError("input_path_escapes_data_root")
    if not path.is_file():
        raise ValidationError("required_server_input_missing: " + relative)
    return path


def feature_spec(root, cfg=None):
    cfg = cfg or configuration()
    names = [x.strip() for x in file_under(root, cfg["paths"]["eligible_features"]).read_text(encoding="utf-8-sig").splitlines()
             if x.strip() and not x.lstrip().startswith("#")]
    if len(names) != 142 or len(set(names)) != 142:
        raise ValidationError("actual_142_unique_eligible_feature_names_required")
    if set(names) & FORBIDDEN:
        raise ValidationError("outcome_identifier_era_or_operator_in_predictor_list")
    dictionary = read_csv(file_under(root, cfg["paths"]["dictionary"]))
    m = cfg["dictionary_columns"]
    require_columns(dictionary, list(m.values()))
    if len(dictionary) != 408 or dictionary[m["name"]].nunique() != 408:
        raise ValidationError("actual_complete_408_variable_dictionary_required")
    flags = dictionary[m["eligible"]].astype(str).str.lower().map(
        {"true": True, "false": False, "1": True, "0": False, "1.0": True, "0.0": False})
    if flags.isna().any():
        raise ValidationError("dictionary_eligibility_must_be_boolean")
    layers = dictionary[m["layer"]].astype(str).str.extract(r"^(T[0-7])(?:\b|$)")[0]
    if layers.isna().any() or (flags & ~layers.isin(["T0", "T1", "T2"])).any():
        raise ValidationError("invalid_temporal_eligibility")
    eligible = dictionary.loc[flags, m["name"]].astype(str)
    if set(eligible) != set(names):
        raise ValidationError("dictionary_and_feature_list_disagree")
    return names


def load_cohort(root, cohort="development", cfg=None, include_features=True):
    cfg = cfg or configuration()
    source = file_under(root, cfg["paths"][cohort])
    data = read_csv(source)
    identifiers(data)
    if len(data) != (261 if cohort == "development" else 28):
        raise ValidationError("cohort_size_disagrees_with_fixed_study_definition")
    require_columns(data, ["cohort_flag", "surgery_year", "era"])
    expected_flag = 0 if cohort == "development" else 1
    if data.cohort_flag.isna().any() or not data.cohort_flag.eq(expected_flag).all():
        raise ValidationError("wrong_or_missing_cohort_flag")
    year = pd.to_numeric(data.surgery_year, errors="raise")
    if cohort == "development":
        assigned = pd.Series(np.nan, index=data.index)
        for era, low, high in cfg["eras"]:
            assigned.loc[year.between(low, high)] = era
        if assigned.isna().any() or not assigned.eq(pd.to_numeric(data.era, errors="raise")).all():
            raise ValidationError("era_and_year_disagree")
    elif not year.eq(2026).all():
        raise ValidationError("holdout_year_does_not_match_reported_cohort")
    data = validate_and_add(data)
    if include_features:
        features = feature_spec(root, cfg)
        require_columns(data, features)
    return data.reset_index(drop=True), source


def validate_inputs(root, include_holdout=False):
    cfg = configuration()
    features = feature_spec(root, cfg)
    development, source = load_cohort(root, cfg=cfg)
    if len(development) != 261:
        raise ValidationError("development_count_disagrees_with_reported_261")
    result = {"development_n": len(development), "eligible_features": len(features), "dictionary_rows": 408,
              "source_hash_required_at_run": True}
    if include_holdout:
        holdout, _ = load_cohort(root, "holdout", cfg)
        if len(holdout) != 28 or set(holdout.patient_id) & set(development.patient_id):
            raise ValidationError("holdout_count_or_patient_independence_error")
        result["holdout_n"] = len(holdout)
    return result
