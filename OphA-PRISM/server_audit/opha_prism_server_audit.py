#!/usr/bin/env python3
"""Offline, read-only audit of existing OphA-PRISM records and saved predictions.

This is audit software, version 0.1.0, written in October 2026. It is not the
historical v1.2.0 training pipeline. No model fitting, label repair, internet,
pickle loading, clinical simulation, or patient-level export is performed.
"""
import argparse
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
import numpy as np
import pandas as pd

VERSION = "0.1.0"
TASKS = ("VISION", "ACCESS", "CONTROL", "RECON", "CURE")


class AuditError(Exception):
    """Only fixed, non-patient error codes may be returned to the report."""


def guarded(fn, *args, **kwargs):
    try:
        return {"status": "calculated", "result": fn(*args, **kwargs)}
    except AuditError as exc:
        return {"status": "needs_attention", "code": str(exc)}
    except Exception as exc:
        # Deliberately do not export exception messages, values, paths or rows.
        return {"status": "error", "exception_type": type(exc).__name__}


def resolve_column(df, spec, aliases=(), required=True):
    if isinstance(spec, str):
        if spec not in df.columns:
            raise AuditError("configured_column_not_found")
        return spec
    choices = [x for x in aliases if x in df.columns]
    if len(choices) > 1:
        raise AuditError("ambiguous_columns_configure_one_explicitly")
    if not choices:
        if required:
            raise AuditError("required_column_not_found")
        return None
    return choices[0]


def binary(series):
    value = pd.to_numeric(series, errors="coerce")
    if ((series.notna() & value.isna()) | (value.notna() & ~value.isin([0, 1]))).any():
        raise AuditError("invalid_binary_value")
    return value.astype("Float64")


def numeric(series):
    value = pd.to_numeric(series, errors="coerce")
    if (series.notna() & value.isna()).any():
        raise AuditError("invalid_numeric_value")
    if not np.isfinite(value.dropna().astype(float)).all():
        raise AuditError("nonfinite_numeric_value")
    return value


def wilson(k, n):
    if not n:
        return {"numerator": int(k), "denominator": 0, "proportion": None, "ci95": None}
    z = 1.959963984540054
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"numerator": int(k), "denominator": int(n), "proportion": p,
            "ci95": [max(0., c - h), min(1., c + h)]}


def auc_details(y, p):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return {"auc": None, "reason": "one_outcome_class", "positive_negative_pairs": 0}
    concordant = int((pos[:, None] > neg[None, :]).sum())
    tied = int((pos[:, None] == neg[None, :]).sum())
    pairs = len(pos) * len(neg)
    return {"auc": (concordant + 0.5 * tied) / pairs,
            "concordant_pairs": concordant, "tied_pairs": tied,
            "positive_negative_pairs": int(pairs),
            "half_pair_auc_step": 0.5 / pairs}


def pr_metrics(y, p):
    """Average precision and trapezoidal PR area, with ties grouped together."""
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    if y.sum() == 0:
        return {"average_precision": None, "pr_auc_trapezoidal": None}
    order = np.argsort(-p, kind="mergesort")
    ys, ps = y[order], p[order]
    ends = np.r_[np.flatnonzero(ps[:-1] != ps[1:]), len(ps) - 1]
    tp = np.cumsum(ys)[ends]
    precision = tp / (ends + 1)
    recall = tp / y.sum()
    r = np.r_[0., recall]
    q = np.r_[1., precision]
    return {"average_precision": float(np.sum(np.diff(r) * precision)),
            "pr_auc_trapezoidal": float(np.sum(np.diff(r) * (q[1:] + q[:-1]) / 2))}


def decision_rows(y, p, thresholds):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    n = len(y)
    out = []
    for threshold in thresholds:
        if not 0 < threshold < 1:
            raise AuditError("risk_threshold_outside_open_unit_interval")
        act = p >= threshold
        tp = int(((y == 1) & act).sum())
        fp = int(((y == 0) & act).sum())
        odds = threshold / (1 - threshold)
        out.append({"threshold": float(threshold), "tp": tp, "fp": fp,
                    "net_benefit": (tp - fp * odds) / n,
                    "act_on_all": (int(y.sum()) - (n - int(y.sum())) * odds) / n,
                    "act_on_none": 0.})
    return out


def calibration_points(y, p):
    """Optional unpenalized calibration fits; errors are not replaced by estimates."""
    try:
        import statsmodels.api as sm
    except ImportError:
        return {"status": "not_calculated", "reason": "optional_statsmodels_not_installed"}
    try:
        z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        if len(np.unique(y)) < 2 or np.ptp(z) == 0:
            return {"status": "not_estimable"}
        import warnings
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            joint = sm.GLM(y, sm.add_constant(z), family=sm.families.Binomial()).fit()
            offset = sm.GLM(y, np.ones((len(y), 1)), offset=z,
                            family=sm.families.Binomial()).fit()
        if captured or not joint.converged or not offset.converged:
            return {"status": "not_reported", "reason": "fit_warning_or_nonconvergence"}
        if not np.isfinite(np.r_[joint.params, offset.params]).all():
            return {"status": "not_estimable"}
        return {"status": "calculated", "joint_intercept": float(joint.params[0]),
                "slope": float(joint.params[1]),
                "calibration_in_the_large_offset_intercept": float(offset.params[0]),
                "logit_probability_clip": 1e-6,
                "uncertainty": "not_calculated; point estimates only"}
    except Exception as exc:
        return {"status": "not_estimable", "exception_type": type(exc).__name__}


def metric_summary(y, p, thresholds, bootstrap=0, seed=42, curves=False):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    if not len(y) or len(y) != len(p):
        raise AuditError("empty_or_mismatched_prediction_vectors")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise AuditError("invalid_outcome_or_probability_vector")
    prevalence = float(y.mean())
    brier = float(np.mean((p - y) ** 2))
    reference = prevalence * (1 - prevalence)
    out = {"n": len(y), "events": int(y.sum()), "prevalence": prevalence,
           "roc": auc_details(y, p), **pr_metrics(y, p), "brier": brier,
           "cohort_prevalence_reference_brier": reference,
           "brier_skill": 1 - brier / reference if reference else None,
           "probabilities_exactly_zero_or_one": int(((p == 0) | (p == 1)).sum()),
           "calibration": calibration_points(y, p),
           "decision_curve": decision_rows(y, p, thresholds)}
    if bootstrap and len(np.unique(y)) == 2:
        rng = np.random.default_rng(seed)
        positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        vals = []
        for _ in range(bootstrap):
            ix = np.r_[rng.choice(positive, len(positive), replace=True),
                       rng.choice(negative, len(negative), replace=True)]
            vals.append([auc_details(y[ix], p[ix])["auc"], np.mean((p[ix] - y[ix]) ** 2)])
        ci = np.quantile(vals, [0.025, 0.975], axis=0)
        out["fixed_prediction_bootstrap"] = {
            "replicates": bootstrap, "seed": seed, "method": "stratified percentile",
            "auc_ci95": ci[:, 0].tolist(), "brier_ci95": ci[:, 1].tolist(),
            "scope": "Patient resampling of saved predictions; no model refitting or optimism correction."}
    if curves:
        # A fixed public grid avoids exporting individual probability thresholds.
        rows = []
        for threshold in np.linspace(0, 1, 21):
            a = p >= threshold
            tp, fp = int(((y == 1) & a).sum()), int(((y == 0) & a).sum())
            rows.append({"threshold": float(threshold), "tp": tp, "fp": fp,
                         "recall": tp / y.sum() if y.sum() else None,
                         "fpr": fp / (len(y) - y.sum()) if len(y) - y.sum() else None,
                         "precision": tp / (tp + fp) if tp + fp else None})
        out["fixed_grid_roc_pr"] = rows
    return out


def clean_ids(series):
    if series.isna().any() or (series.astype(str).str.strip() == "").any():
        raise AuditError("missing_patient_identifiers")
    return series.astype(str)


def derived_endpoints(df, columns):
    grade = numeric(df[columns["rroc"]])
    if (grade.notna() & ~grade.isin([1, 2, 3])).any():
        raise AuditError("invalid_rroc_grade")
    loss, vision = binary(df[columns["opha_loss"]]), binary(df[columns["vision"]])
    accepted = binary(df[columns["accepted_remnant"]])
    favorable = pd.Series(pd.NA, index=df.index, dtype="boolean")
    favorable.loc[grade == 1] = True
    favorable.loc[grade == 3] = False
    favorable.loc[grade == 2] = (accepted.loc[grade == 2] == 1).astype("boolean")
    technical = favorable & (loss == 0).astype("boolean")
    cure = ~(favorable & (loss == 0).astype("boolean") & (vision == 0).astype("boolean"))
    return grade, loss, vision, favorable, technical.astype("Float64"), cure.astype("Float64")


def joint_counts(grade, loss, vision):
    keep = grade.notna() & loss.notna() & vision.notna()
    a, b, c = (grade.loc[keep] == 3).astype(int), loss.loc[keep], vision.loc[keep]
    return [{"residual_sac": x, "opha_loss": z, "vision_event": v,
             "n": int(((a == x) & (b == z) & (c == v)).sum())}
            for x, z, v in itertools.product([0, 1], repeat=3)]


def mismatch(saved, derived):
    saved = binary(saved)
    keep = saved.notna() & derived.notna()
    return {"compared": int(keep.sum()), "disagreements": int((saved[keep] != derived[keep]).sum()),
            "not_comparable": int((~keep).sum())}


def label_series(df, task, columns):
    col = columns.get("labels", {}).get(task)
    if col and col in df:
        return binary(df[col])
    if task == "CURE":
        return derived_endpoints(df, columns)[5]
    raise AuditError("task_label_column_requires_configuration")


def audit_cohort(df, cfg, flag):
    c = cfg["columns"]
    ids = clean_ids(df[c["id"]])
    if ids.duplicated().any():
        raise AuditError("duplicate_patient_identifiers_in_master_data")
    n = len(df)
    out = {"n": n, "columns": len(df.columns), "column_names": list(df.columns),
           "expected_n": cfg["expected"]["development_n" if flag == 0 else "holdout_n"],
           "unique_patient_count": int(ids.nunique())}
    out["n_matches_reported"] = n == out["expected_n"]
    if c["cohort"] in df:
        cohort = numeric(df[c["cohort"]])
        out["wrong_or_missing_cohort_flag"] = int((cohort.isna() | (cohort != flag)).sum())
    grade, loss, vision, favorable, technical, cure = derived_endpoints(df, c)
    out["angiography"] = {"rroc_i": int((grade == 1).sum()), "rroc_ii": int((grade == 2).sum()),
                          "rroc_iii": int((grade == 3).sum()),
                          "unaccepted_or_unknown_class_ii": int(((grade == 2) & (favorable.isna() | ~favorable.fillna(False))).sum()),
                          "favorable": int(favorable.sum()), "technical_success": int(technical.sum()),
                          "technical_unknown": int(technical.isna().sum()),
                          "cure_failure": int(cure.sum()), "cure_unknown": int(cure.isna().sum())}
    out["cure_joint_distribution"] = joint_counts(grade, loss, vision)
    if "vision_6mo_confirmed" in df:
        confirmed = binary(df.vision_6mo_confirmed)
        out["vision_events_without_documented_six_month_confirmation"] = int(
            ((vision == 1) & (confirmed.isna() | (confirmed != 1))).sum())
    if "outcome_missing_flag" in df:
        missing_flag = binary(df.outcome_missing_flag)
        out["nonspecific_missing_outcome_flag"] = {
            "flagged": int((missing_flag == 1).sum()),
            "flag_itself_missing": int(missing_flag.isna().sum()),
            "scope": "Task-specific missing-label rules require code review; this flag is not silently recoded."}
    for key, derived in (("technical_success", technical), ("cure_failure", cure)):
        if c.get(key) in df:
            out[key + "_saved_vs_derived"] = mismatch(df[c[key]], derived)
    out["outcomes"] = {}
    for task in TASKS:
        out["outcomes"][task] = guarded(lambda t=task: {
            "events": int(label_series(df, t, c).sum()),
            "missing_labels": int(label_series(df, t, c).isna().sum())})
    if c["era"] in df and flag == 0:
        era = numeric(df[c["era"]])
        year = numeric(df[c["year"]])
        expected_era = pd.Series(np.nan, index=df.index)
        for e, lo, hi in cfg["eras"]:
            expected_era.loc[year.between(lo, hi)] = e
        out["era_year_disagreements_or_unknown"] = int((era.isna() | expected_era.isna() | (era != expected_era)).sum())
        out["eras"] = []
        for e, lo, hi in cfg["eras"]:
            d = df.loc[era == e]
            entry = {"era": e, "years": [lo, hi], "n": len(d), "events": {}}
            for task in TASKS:
                entry["events"][task] = guarded(lambda t=task, part=d: int(label_series(part, t, c).sum()))
            out["eras"].append(entry)
    diameter = numeric(df[c["diameter"]])
    keep = diameter.notna() & cure.notna()
    if keep.any():
        out["diameter_10mm_rule"] = metric_summary(
            cure[keep].astype(int), (diameter[keep] >= 10).astype(float), [0.1, 0.2, 0.3, 0.4])
        large = diameter >= 10
        k = large & grade.notna() & loss.notna() & vision.notna()
        out["large_aneurysms"] = {"n": int(large.sum()), "cure_failures": int(cure[large].sum()),
                                  "cure_unknown": int(cure[large].isna().sum()),
                                  "joint_distribution": joint_counts(grade[k], loss[k], vision[k])}
    out["phi"] = []
    for a, b in itertools.combinations(TASKS, 2):
        def correlation():
            x, z = label_series(df, a, c), label_series(df, b, c)
            keep = x.notna() & z.notna()
            x, z = x[keep].astype(float), z[keep].astype(float)
            return {"n": len(x), "phi": float(np.corrcoef(x, z)[0, 1])
                    if len(x) and x.std() > 0 and z.std() > 0 else None}
        out["phi"].append({"tasks": [a, b], **guarded(correlation)})
    out["operative_coding"] = guarded(operative_coding, df, cfg)
    return out


def operative_coding(df, cfg):
    components = cfg.get("access_components", [])
    out = {}
    if components and all(c in df for c in components):
        values = pd.concat([binary(df[c]).rename(c) for c in components], axis=1)
        counts = values.sum(axis=1, min_count=len(components))
        out["access_count"] = {"complete_records": int(counts.notna().sum()),
                               "unknown": int(counts.isna().sum()),
                               "count_ge5": int((counts >= 5).sum())}
        if "access_count" in df:
            saved = numeric(df.access_count)
            keep = saved.notna() & counts.notna()
            out["access_count"]["saved_count_disagreements"] = int((saved[keep] != counts[keep]).sum())
        target = cfg["columns"]["labels"].get("ACCESS")
        if target in df:
            latent = binary(df[target])
            keep = counts.notna() & latent.notna()
            count_label = (counts[keep] >= 5).astype(int)
            latent = latent[keep].astype(int)
            cells = {str(a) + str(b): int(((latent == a) & (count_label == b)).sum())
                     for a, b in itertools.product([0, 1], repeat=2)}
            n = int(keep.sum())
            observed = (cells["00"] + cells["11"]) / n if n else None
            p1, q1 = float(latent.mean()) if n else 0, float(count_label.mean()) if n else 0
            expected = p1 * q1 + (1 - p1) * (1 - q1)
            out["access_latent_vs_count_rule"] = {"n": n, "cells_latent_then_count": cells,
                                                 "kappa": (observed - expected) / (1 - expected)
                                                 if n and expected < 1 else None}
    else:
        out["access_count"] = {"status": "component_columns_require_configuration"}
    flags = ["fenestrated_clip", "tandem_clip", "booster_clip"]
    if "clip_count" in df and all(c in df for c in flags):
        count = numeric(df.clip_count).astype("Float64")
        if (count.notna() & ((count < 1) | (count % 1 != 0))).any():
            raise AuditError("invalid_clip_count")
        recon = (count > 1).astype("boolean")
        for c in flags:
            recon = recon | (binary(df[c]) == 1).astype("boolean")
        label = cfg["columns"]["labels"]["RECON"]
        if label in df:
            out["recon_saved_vs_component_rule"] = mismatch(df[label], recon.astype("Float64"))
    return out


def pool_predictions(d, id_col, y, p, repeat=None, imputation=None, fold=None):
    """Average imputations within repeat, then equally average repeats per patient."""
    work = pd.DataFrame({"id": clean_ids(d[id_col]), "y": binary(y), "p": numeric(p)})
    if work[["y", "p"]].isna().any().any():
        raise AuditError("missing_prediction_or_evaluation_label_not_dropped")
    if ((work.p < 0) | (work.p > 1)).any():
        raise AuditError("probability_outside_unit_interval")
    work["repeat"] = clean_ids(d[repeat]) if repeat else "preaggregated"
    if imputation:
        work["imputation"] = clean_ids(d[imputation])
    if work.groupby("id").y.nunique().max() > 1:
        raise AuditError("inconsistent_outcome_across_repeated_predictions")
    keys = ["id", "repeat"] + (["imputation"] if imputation else [])
    if work.duplicated(keys).any():
        raise AuditError("duplicate_predictions_require_explicit_repeat_imputation_or_variant_mapping")
    if fold:
        work["fold"] = clean_ids(d[fold])
        if work.groupby(["id", "repeat"]).fold.nunique().max() > 1:
            raise AuditError("patient_has_multiple_test_folds_in_one_repeat")
    stage = work.groupby(["id", "repeat"], as_index=False).agg(y=("y", "first"), p=("p", "mean"))
    pooled = stage.groupby("id", as_index=False).agg(y=("y", "first"), p=("p", "mean"))
    def distribution(s):
        return [{"count_per_group": int(k), "groups": int(v)} for k, v in s.value_counts().sort_index().items()]
    info = {"input_records": len(d), "unique_patients": len(pooled),
            "repeats_per_patient": distribution(stage.groupby("id").size()),
            "imputations_per_patient_repeat": distribution(work.groupby(["id", "repeat"]).size()),
            "pooling": "mean within patient/repeat, then mean across repeats",
            "historical_pooling_provenance": "not verified by numerical aggregation alone"}
    return pooled, info


def prediction_tasks(df, spec):
    long_col = resolve_column(df, spec.get("task"), ("task", "engine", "endpoint"), False)
    if long_col:
        normalized = df[long_col].astype(str).str.upper().str.replace("PRISM-", "", regex=False)
        prob = resolve_column(df, spec.get("probability"),
                              ("probability", "y_prob", "p_calibrated", "predicted_probability", "y_pred_proba", "prob"))
        for task in TASKS:
            if (normalized == task).any():
                yield task, df.loc[normalized == task].copy(), prob
    else:
        found = False
        for task in TASKS:
            names = spec.get("wide_probabilities", {}).get(task)
            if isinstance(names, str):
                names = [names]
            names = names or [task.lower() + "_prob", task.lower() + "_probability", "p_" + task.lower()]
            col = resolve_column(df, None, names, False)
            if col:
                found = True
                yield task, df.copy(), col
        if not found:
            raise AuditError("prediction_layout_requires_column_mapping")


def audit_predictions(df, spec, master, cfg, bootstrap, curves):
    cols = cfg["columns"]
    id_col = resolve_column(df, spec.get("id"), (cols["id"],))
    # Group only configured model metadata. Neither IDs nor folds are variants.
    group_cols = [c for c in spec.get("groups", []) if c in df]
    if id_col in group_cols:
        raise AuditError("patient_identifier_cannot_be_export_group")
    results = []
    role = resolve_column(df, spec.get("role"), ("role", "partition", "prediction_partition"), False)
    excluded_role_rows = 0
    if role:
        test_roles = spec.get("test_roles", ["test", "outer_test", "heldout", "oof", "out_of_fold"])
        keep = df[role].astype(str).str.lower().isin(test_roles)
        if not keep.any():
            raise AuditError("no_recognized_test_rows_configure_prediction_role")
        excluded_role_rows = int((~keep).sum())
        df = df.loc[keep].copy()
    master_ids = clean_ids(master[cols["id"]])
    if master_ids.duplicated().any():
        raise AuditError("duplicate_master_ids_prevent_prediction_join")
    indexed = master.set_index(master_ids)
    for task, part, prob in prediction_tasks(df, spec):
        grouping = group_cols[0] if len(group_cols) == 1 else group_cols
        groups = part.groupby(grouping, dropna=False, sort=False) if group_cols else [((), part)]
        for value, d in groups:
            if len(results) >= 100:
                raise AuditError("too_many_prediction_groups_check_mapping")
            values = value if isinstance(value, tuple) else (value,)
            metadata = {name: safe_metadata(v) for name, v in zip(group_cols, values)}
            def evaluate():
                ids = clean_ids(d[id_col])
                if not ids.isin(indexed.index).all():
                    raise AuditError("prediction_contains_ids_outside_designated_cohort")
                expected = label_series(master, task, cols)
                truth = pd.Series(expected.to_numpy(), index=master_ids).reindex(ids.to_numpy())
                truth.index = d.index
                y_col = resolve_column(d, spec.get("outcome"), ("y_true", "observed", "outcome", "label"), False)
                if y_col:
                    recorded = binary(d[y_col])
                    if ((recorded.notna() & truth.notna()) & (recorded != truth)).any():
                        raise AuditError("saved_prediction_labels_disagree_with_master")
                    if truth.isna().any():
                        raise AuditError("master_endpoint_missing_saved_labels_not_substituted")
                repeat = resolve_column(d, spec.get("repeat"), ("repeat", "repeat_id"), False)
                imp = resolve_column(d, spec.get("imputation"), ("imputation", "imputation_id", ".imp"), False)
                fold = resolve_column(d, spec.get("fold"), ("outer_fold", "fold", "fold_id"), False)
                pooled, info = pool_predictions(d, id_col, truth, d[prob], repeat, imp, fold)
                info["cohort_patients_without_prediction"] = int((~master_ids.isin(pooled.id)).sum())
                info["expected_outer_repeats"] = 10 if spec["kind"] == "oof" else None
                thresholds = cfg["decision_thresholds"][task]
                metrics = metric_summary(pooled.y, pooled.p, thresholds, bootstrap, cfg["seed"], curves)
                by_era = []
                if spec["kind"] in ("loeo", "rolling_origin"):
                    eras = pooled.id.map(indexed[cols["era"]])
                    for e, _, _ in cfg["eras"]:
                        k = eras == e
                        if k.any():
                            by_era.append({"era": e, "metrics": metric_summary(
                                pooled.loc[k, "y"], pooled.loc[k, "p"], thresholds,
                                bootstrap, cfg["seed"], curves)})
                return {"pooling": info, "metrics": metrics, "by_test_era": by_era,
                        "test_partition_provenance": "requires source-code and role-membership review"}
            results.append({"task": task, "model_metadata": metadata, **guarded(evaluate)})
    if not results:
        raise AuditError("no_recognized_prediction_tasks_configure_endpoint_mapping")
    return {"headers": list(df.columns), "excluded_non_test_rows": excluded_role_rows, "groups": results}


def safe_metadata(value):
    # Model metadata only. Free-text clinical fields must not be configured here.
    if pd.isna(value):
        return None
    text = str(value)
    if len(text) > 80 or not re.fullmatch(r"[A-Za-z0-9 _./:+()=-]+", text):
        return "metadata_value_withheld"
    return text


def conformal_summary(y, zero, one):
    y, zero, one = binary(y), binary(zero), binary(one)
    if y.isna().any() or zero.isna().any() or one.isna().any():
        raise AuditError("missing_conformal_label_or_membership")
    y, zero, one = y.astype(int).to_numpy(), zero.astype(bool).to_numpy(), one.astype(bool).to_numpy()
    cover = np.where(y == 0, zero, one)
    size = zero.astype(int) + one.astype(int)
    out = {"n": len(y), "class_counts": [], "overall_coverage": wilson(int(cover.sum()), len(y)),
           "empty": int((size == 0).sum()), "singleton": int((size == 1).sum()),
           "two_class": int((size == 2).sum()), "mean_set_size": float(size.mean())}
    for label in (0, 1):
        k = y == label
        out["class_counts"].append({"true_class": label, "n": int(k.sum()),
                                    "empty": int((k & ~zero & ~one).sum()),
                                    "class_0_only": int((k & zero & ~one).sum()),
                                    "class_1_only": int((k & ~zero & one).sum()),
                                    "both": int((k & zero & one).sum()),
                                    "coverage": wilson(int(cover[k].sum()), int(k.sum()))})
    for label, mask in ((0, zero & ~one), (1, one & ~zero)):
        out["negative_singleton_accuracy" if label == 0 else "positive_singleton_ppv"] = wilson(
            int((mask & (y == label)).sum()), int(mask.sum()))
    out["scope"] = "Observed set counts only; no certification of conformal calibration independence or validity."
    return out


def audit_conformal(df, spec, master, cfg):
    c = cfg["columns"]
    id_col = resolve_column(df, spec.get("id"), (c["id"],))
    task_col = resolve_column(df, spec.get("task"), ("task", "engine", "endpoint"))
    zero = resolve_column(df, spec.get("include_0"), ("include_0", "contains_0", "set_0", "class_0_in_set"))
    one = resolve_column(df, spec.get("include_1"), ("include_1", "contains_1", "set_1", "class_1_in_set"))
    mids = clean_ids(master[c["id"]])
    if mids.duplicated().any():
        raise AuditError("duplicate_master_ids_prevent_set_join")
    out = []
    normalized = df[task_col].astype(str).str.upper().str.replace("PRISM-", "", regex=False)
    for task in TASKS:
        d = df.loc[normalized == task]
        if not len(d):
            continue
        def evaluate():
            ids = clean_ids(d[id_col])
            if ids.duplicated().any():
                raise AuditError("multiple_sets_per_patient_select_documented_repeat_no_union_or_averaging")
            if not ids.isin(mids).all():
                raise AuditError("conformal_ids_outside_designated_cohort")
            truth = pd.Series(label_series(master, task, c).to_numpy(), index=mids).reindex(ids.to_numpy())
            truth.index = d.index
            ycol = resolve_column(d, spec.get("outcome"), ("y_true", "observed", "outcome", "label"), False)
            if ycol and mismatch(d[ycol], binary(truth))["disagreements"]:
                raise AuditError("conformal_labels_disagree_with_master")
            result = conformal_summary(truth, d[zero], d[one])
            result["cohort_patients_without_set"] = int((~mids.isin(ids)).sum())
            return result
        out.append({"task": task, **guarded(evaluate)})
    return {"headers": list(df.columns), "tasks": out}


def audit_folds(df, cfg):
    c = cfg["columns"]
    id_col = resolve_column(df, None, (c["id"],))
    repeat = resolve_column(df, None, ("repeat", "repeat_id"), False)
    outer = resolve_column(df, None, ("outer_fold", "fold", "fold_id"), False)
    inner = resolve_column(df, None, ("inner_fold",), False)
    role = resolve_column(df, None, ("role", "partition"), False)
    out = {"headers": list(df.columns), "rows": len(df), "unique_patients": int(df[id_col].nunique()),
           "role_membership_available": bool(role), "groups": []}
    if not outer:
        return out
    keys = [x for x in (repeat, outer, inner) if x]
    role_map = {"inner_train": "fit", "fit": "fit", "model_fit": "fit",
                "conformal_calibration": "conformal", "conformal": "conformal",
                "probability_calibration": "probability_calibration", "early_stopping": "early_stopping",
                "inner_validation": "early_stopping", "outer_test": "outer_test", "test": "outer_test"}
    for key, d in df.groupby(keys, dropna=False, sort=False):
        if len(out["groups"]) >= 300:
            out["additional_groups_not_expanded"] = True
            break
        key = key if isinstance(key, tuple) else (key,)
        entry = {"split": {a: safe_metadata(b) for a, b in zip(keys, key)}, "unique_patients": int(d[id_col].nunique())}
        if role:
            memberships = {}
            canonical = d[role].astype(str).str.lower().map(role_map)
            entry["unrecognized_role_rows"] = int(canonical.isna().sum())
            for r in role_map.values():
                memberships[r] = set(d.loc[canonical == r, id_col].dropna().astype(str))
            entry["role_sizes"] = {a: len(b) for a, b in memberships.items()}
            entry["overlap_counts"] = {a + "__" + b: len(memberships[a] & memberships[b])
                                       for a, b in itertools.combinations(sorted(memberships), 2)}
        out["groups"].append(entry)
    out["scope"] = "Membership counts do not prove the recorded roles were respected by fitting code."
    return out


class Audit:
    def __init__(self, root, cfg, bootstrap=0, curves=False):
        self.root, self.cfg = Path(root).resolve(), cfg
        self.bootstrap, self.curves = bootstrap, curves

    def path(self, relative):
        p = (self.root / relative).resolve()
        if not p.is_relative_to(self.root):
            raise AuditError("input_path_escapes_project_root")
        return p

    def read_csv(self, relative):
        p = self.path(relative)
        if not p.is_file():
            raise AuditError("input_file_missing")
        with p.open(encoding="utf-8-sig", newline="") as handle:
            import csv
            header = next(csv.reader(handle), [])
        if len(set(header)) != len(header):
            raise AuditError("duplicate_csv_column_names")
        ids = {x: "string" for x in (self.cfg["columns"]["id"], "lesion_id", "op_id") if x in header}
        df = pd.read_csv(p, dtype=ids, low_memory=False)
        return df.replace([-999, "-999"], np.nan)

    def inventory(self):
        entries = []
        paths = {"development": self.cfg["files"]["development"], "holdout": self.cfg["files"]["holdout"],
                 **self.cfg["files"]["supporting"]}
        paths.update({k: s["path"] for k, s in self.cfg["prediction_files"].items()})
        for label, relative in paths.items():
            def describe():
                p = self.path(relative)
                if not p.exists():
                    return {"exists": False}
                if p.is_dir():
                    return {"exists": True, "type": "directory", "historical_contents": "not_verified"}
                h = hashlib.sha256()
                with p.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        h.update(chunk)
                entry = {"exists": True, "bytes": p.stat().st_size, "sha256": h.hexdigest()}
                if p.suffix.lower() == ".csv":
                    import csv
                    with p.open(encoding="utf-8-sig", newline="") as handle:
                        entry["column_names"] = next(csv.reader(handle), [])
                return entry
            entries.append({"material": label, "configured_relative_path": relative, **guarded(describe)})
        return entries

    def feature_audit(self, dev):
        s = self.cfg["files"]["supporting"]
        p = self.path(s["eligible_features"])
        if not p.is_file():
            raise AuditError("eligible_feature_file_missing")
        features = [line.strip() for line in p.read_text(encoding="utf-8-sig").splitlines()
                    if line.strip() and not line.lstrip().startswith("#")]
        present = [f for f in features if f in dev]
        missing = [f for f in features if f not in dev]
        forbidden = set(self.cfg["columns"].get("forbidden_predictors", []))
        forbidden.update(x for x in self.cfg["columns"]["labels"].values() if x)
        out = {"listed_count": len(features), "unique_count": len(set(features)), "feature_names": features,
               "expected_count": 142, "listed_features_absent_from_raw_master": missing,
               "forbidden_listed_features": sorted(set(features) & forbidden),
               "note": "Absent fields may be generated features; their derivations require source-code review.",
               "missingness": []}
        for f in present:
            groups = [("all", dev)]
            for era, _, _ in self.cfg["eras"]:
                groups.append(("era_" + str(era), dev.loc[dev[self.cfg["columns"]["era"]] == era]))
            for label, d in groups:
                out["missingness"].append({"feature": f, "group": label, "n": len(d),
                                           "missing": int(d[f].isna().sum())})
        out["dictionary"] = guarded(self.dictionary_audit)
        return out

    def dictionary_audit(self):
        df = self.read_csv(self.cfg["files"]["supporting"]["dictionary"])
        mapping = self.cfg["dictionary_columns"]
        name = resolve_column(df, mapping.get("name"), ("Variable_Name", "variable_name", "name"))
        layer = resolve_column(df, mapping.get("layer"), ("Temporal_Layer", "temporal_layer", "layer"))
        eligible = resolve_column(df, mapping.get("eligible"),
                                  ("Eligible_for_Preop_Prediction", "eligible_for_preop_prediction", "eligible"))
        flags = df[eligible].astype(str).str.lower().map({"true": True, "false": False, "1": True,
                                                        "0": False, "1.0": True, "0.0": False})
        out = {"rows": len(df), "unique_variables": int(df[name].nunique()),
               "eligible": int(flags.sum()), "invalid_eligibility_values": int(flags.isna().sum()),
               "eligible_outside_t0_t2": int((flags.fillna(False) & ~df[layer].isin(["T0", "T1", "T2"])).sum())}
        # Names and layer labels are dictionary metadata, not patient records.
        out["eligible_variable_names"] = df.loc[flags.fillna(False), name].dropna().astype(str).tolist()
        return out

    def bootstrap_audit(self):
        df = self.read_csv(self.cfg["files"]["supporting"]["bootstrap"])
        task_col = resolve_column(df, None, ("task", "engine", "endpoint"))
        a = resolve_column(df, None, ("apparent_boot_auc",))
        b = resolve_column(df, None, ("test_orig_auc",))
        out = []
        for task in TASKS:
            d = df.loc[df[task_col].astype(str).str.upper() == task]
            if not len(d):
                continue
            for column in ("model", "model_family", "variant", "representation", "calibration_status"):
                if column in d and d[column].nunique(dropna=False) > 1:
                    raise AuditError("multiple_bootstrap_variants_require_explicit_adapter")
            x, z = numeric(d[a]), numeric(d[b])
            if x.isna().any() or z.isna().any() or (~x.between(0, 1)).any() or (~z.between(0, 1)).any():
                raise AuditError("invalid_bootstrap_auc_values")
            delta = x - z
            entry = {"task": task, "replicates": len(d), "matches_reported_2000_replicates": len(d) == 2000,
                     "mean_optimism": float(delta.mean()),
                     "optimism_replicate_quantiles": np.quantile(delta, [.025, .975]).tolist(),
                     "scope": "Quantiles describe replicate optimism, not a BCa interval for corrected performance."}
            if "optimism" in d:
                saved = numeric(d.optimism)
                entry["saved_optimism_missing"] = int(saved.isna().sum())
                entry["saved_optimism_disagreements"] = int((abs(saved - delta) > 1e-10).sum())
            out.append(entry)
        if not out:
            raise AuditError("no_recognized_bootstrap_tasks_configure_mapping")
        return {"tasks": out, "full_pipeline_refitting": "not verified from aggregate replicates"}

    def permutation_audit(self):
        df = self.read_csv(self.cfg["files"]["supporting"]["permutations"])
        task_col = resolve_column(df, None, ("task", "engine", "endpoint"))
        auc = resolve_column(df, None, ("auc", "oof_auc", "test_auc"))
        out = []
        for task in TASKS:
            d = df.loc[df[task_col].astype(str).str.upper() == task]
            if not len(d):
                continue
            for column in ("model", "model_family", "variant", "representation", "calibration_status"):
                if column in d and d[column].nunique(dropna=False) > 1:
                    raise AuditError("multiple_permutation_variants_require_explicit_adapter")
            values = numeric(d[auc])
            if values.isna().any() or (~values.between(0, 1)).any():
                raise AuditError("invalid_permutation_auc_values")
            entry = {"task": task, "replicates": len(d), "matches_reported_1000_replicates": len(d) == 1000,
                     "mean_auc": float(values.mean()),
                     "sd_auc": float(values.std(ddof=1)) if len(values) > 1 else None,
                     "null_quantiles": np.quantile(values, [.025, .975]).tolist()}
            if "n_labels_changed" in d:
                changed = numeric(d.n_labels_changed)
                entry["replicates_with_no_changed_labels"] = int((changed == 0).sum())
            out.append(entry)
        if not out:
            raise AuditError("no_recognized_permutation_tasks_configure_mapping")
        return {"tasks": out, "shuffle_and_refitting_provenance": "requires source code; labels must actually change"}

    def git_provenance(self):
        ref = self.cfg.get("claimed_lock_git_ref")
        if not ref or not re.fullmatch(r"[0-9a-fA-F]{7,40}", ref):
            return {"status": "reference_not_configured_as_commit_hash"}
        try:
            process = subprocess.run(["git", "rev-parse", "--verify", "--end-of-options", ref + "^{commit}"],
                                     cwd=self.root, capture_output=True, text=True, timeout=10, check=False)
            text = process.stdout.strip()
            if process.returncode == 0 and re.fullmatch(r"[0-9a-f]{40,64}", text):
                return {"status": "resolved_in_current_checkout", "full_commit": text,
                        "historical_lock": "commit existence alone does not prove timing or holdout isolation"}
            return {"status": "not_resolved_in_current_checkout"}
        except (OSError, subprocess.TimeoutExpired):
            return {"status": "git_unavailable_or_timeout"}

    def run(self):
        report = {"audit_software_version": VERSION,
                  "generated_utc": datetime.now(timezone.utc).isoformat(),
                  "purpose": "Metadata and aggregate verification of current server files, not historical pipeline certification.",
                  "export_policy": "No patient IDs, rows, individual predictions, fitted objects or data-derived probability thresholds.",
                  "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__},
                  "inventory": self.inventory(), "git_provenance": self.git_provenance()}
        masters = {}
        for name, flag in (("development", 0), ("holdout", 1)):
            try:
                masters[name] = self.read_csv(self.cfg["files"][name])
                report[name] = guarded(audit_cohort, masters[name], self.cfg, flag)
            except AuditError as exc:
                report[name] = {"status": "needs_attention", "code": str(exc)}
            except Exception as exc:
                report[name] = {"status": "error", "exception_type": type(exc).__name__}
        if len(masters) == 2:
            c = self.cfg["columns"]["id"]
            report["development_holdout_id_overlap"] = guarded(lambda: len(
                set(clean_ids(masters["development"][c])) & set(clean_ids(masters["holdout"][c]))))
        if "development" in masters:
            report["features_and_missingness"] = guarded(self.feature_audit, masters["development"])
        report["saved_predictions"] = {}
        for name, spec in self.cfg["prediction_files"].items():
            master = masters.get("holdout" if spec["kind"] == "holdout" else "development")
            if master is None:
                report["saved_predictions"][name] = {"status": "not_run", "code": "master_cohort_unavailable"}
                continue
            def evaluate_prediction():
                df = self.read_csv(spec["path"])
                return audit_predictions(df, spec, master, self.cfg, self.bootstrap, self.curves)
            report["saved_predictions"][name] = guarded(evaluate_prediction)
        if "development" in masters:
            def sets():
                spec = self.cfg["conformal"]
                return audit_conformal(self.read_csv(spec["path"]), spec, masters["development"], self.cfg)
            report["conformal"] = guarded(sets)
        report["fold_membership"] = guarded(lambda: audit_folds(
            self.read_csv(self.cfg["files"]["supporting"]["folds"]), self.cfg))
        report["bootstrap_replicates"] = guarded(self.bootstrap_audit)
        report["permutation_replicates"] = guarded(self.permutation_audit)
        report["not_verified_by_this_audit"] = [
            "Historical data/model lock and analyst blinding; clinical record provenance and ethics scope.",
            "Exact training, MICE held-out application, tuning, calibration, selection and conformal separation.",
            "Refitting in each bootstrap/permutation; correctness of saved model objects or training histories.",
            "BCa inference, treatment effects, external validity, and clinical usefulness."]
        report["completion"] = "audit_executed; inspect every section and mapping error before interpreting results"
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Existing institutional project root; inputs remain read-only.")
    parser.add_argument("--config", default=str(Path(__file__).with_name("audit_config.json")))
    parser.add_argument("--output", help="New aggregate JSON file. Existing paths are never overwritten.")
    parser.add_argument("--bootstrap", type=int, default=0, help="Optional fixed-prediction percentile CIs; 0 disables.")
    parser.add_argument("--curves", action="store_true", help="Include aggregate ROC/PR coordinates on a fixed public grid.")
    args = parser.parse_args()
    if not 0 <= args.bootstrap <= 10000:
        parser.error("--bootstrap must be between 0 and 10000")
    if not Path(args.root).is_dir():
        parser.error("--root must be an existing project directory")
    try:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        report = Audit(args.root, config, args.bootstrap, args.curves).run()
        payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        if args.output:
            # Exclusive create: cannot replace data, code, or earlier audit results.
            with Path(args.output).open("x", encoding="utf-8") as handle:
                handle.write(payload + "\n")
            print("Aggregate audit written. Review it on the server before sharing.")
        else:
            print(payload)
        return 0
    except FileExistsError:
        print("Output already exists; choose a new output filename.", file=sys.stderr)
    except Exception as exc:
        print("Audit could not complete: " + type(exc).__name__ + ". No patient values were printed.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
