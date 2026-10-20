from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from .common import (ROOT, ValidationError, configuration, fingerprint, identifiers, private_output,
                     save_json, software_record)
from .data import load_cohort, feature_spec
from .endpoints import labels
from .training import fit_pipeline, assert_disjoint, two_classes
from .metrics import metric_summary, conformal_summary, auc_details


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def code_hash():
    h = hashlib.sha256()
    for path in sorted((ROOT / "ophaprism").glob("*.py")):
        h.update(path.name.encode()); h.update(path.read_bytes())
    h.update((ROOT / "run_pipeline.py").read_bytes())
    return h.hexdigest()


def split_outer(x, y, groups, cfg, repeat):
    cv = StratifiedGroupKFold(n_splits=int(cfg["outer_folds"]), shuffle=True,
                              random_state=int(cfg["seed"]) + repeat)
    return list(cv.split(x, y, groups))


def evaluate_frame(bundle, x, frame, task, repeat, fold):
    probability = bundle.predict(x)
    membership = bundle.prediction_sets(x)
    return pd.DataFrame({"patient_id": frame.patient_id.to_numpy(), "task": task,
                         "repeat": repeat, "outer_fold": fold, "y_true": labels(frame, task),
                         "probability": probability, "include_0": membership[:, 0].astype(int),
                         "include_1": membership[:, 1].astype(int)})


def train_fold(data_root, output_root, task, repeat, fold, cfg=None, resume=False):
    cfg = copy.deepcopy(cfg or configuration())
    if not 0 <= repeat < cfg["outer_repeats"] or not 0 <= fold < cfg["outer_folds"]:
        raise ValidationError("repeat_or_fold_out_of_configured_range")
    data, source = load_cohort(data_root, cfg=cfg)
    features = feature_spec(data_root, cfg)
    x, y, groups = data[features], labels(data, task), identifiers(data).to_numpy()
    a, b = split_outer(x, y, groups, cfg, repeat)[fold]
    assert_disjoint(groups, [a, b]); two_classes(y[a]); two_classes(y[b])
    destination = private_output(Path(output_root) / task / f"repeat_{repeat}_fold_{fold}")
    stamp = {"dataset_sha256": fingerprint(source), "config_sha256": config_hash(cfg),
             "reference_code_sha256": code_hash(),
             "feature_order_sha256": hashlib.sha256("\n".join(features).encode()).hexdigest(),
             "task": task, "repeat": repeat, "outer_fold": fold}
    checkpoint = destination / "complete.json"
    if checkpoint.exists():
        previous = json.loads(checkpoint.read_text())
        if resume and previous["stamp"] == stamp and all((destination / p).is_file() for p in ["bundle.joblib", "predictions.csv"]):
            if fingerprint(destination / "predictions.csv") != previous["predictions_sha256"] or fingerprint(destination / "bundle.joblib") != previous["bundle_sha256"]:
                raise ValidationError("checkpoint_output_hash_mismatch")
            return {"status": "resumed", "task": task, "repeat": repeat, "fold": fold}
        raise ValidationError("existing_fold_requires_matching_resume_or_new_run_directory")
    seed = cfg["seed"] + 1000 * repeat + 10 * fold
    bundle, fitting = fit_pipeline(x.iloc[a], y[a], groups[a], cfg, cfg["models"][task], seed)
    predictions = evaluate_frame(bundle, x.iloc[b], data.iloc[b], task, repeat, fold)
    predictions.to_csv(destination / "predictions.csv", index=False)
    joblib.dump(bundle, destination / "bundle.joblib")
    save_json(destination / "fitting.json", fitting)
    save_json(checkpoint, {"stamp": stamp, "train_n": len(a), "test_n": len(b),
                          "predictions_sha256": fingerprint(destination / "predictions.csv"),
                          "bundle_sha256": fingerprint(destination / "bundle.joblib"),
                          "software": software_record(), "status": "reference_fold_complete"})
    return {"status": "complete", "task": task, "repeat": repeat, "fold": fold,
            "training_n": len(a), "test_n": len(b)}


def collect(output_root, task, bootstrap=2000, cfg=None):
    cfg = cfg or configuration()
    output_root = private_output(output_root)
    records, stamps = [], []
    for repeat in range(cfg["outer_repeats"]):
        for fold in range(cfg["outer_folds"]):
            folder = output_root / task / f"repeat_{repeat}_fold_{fold}"
            if not (folder / "complete.json").is_file():
                raise ValidationError("all_configured_outer_folds_required_before_collecting")
            meta = json.loads((folder / "complete.json").read_text())
            if fingerprint(folder / "predictions.csv") != meta["predictions_sha256"]:
                raise ValidationError("saved_prediction_hash_mismatch")
            records.append(pd.read_csv(folder / "predictions.csv", dtype={"patient_id": str}))
            stamps.append(meta["stamp"])
    for key in ("dataset_sha256", "config_sha256", "feature_order_sha256", "reference_code_sha256"):
        if len({s[key] for s in stamps}) != 1:
            raise ValidationError("cannot_pool_incompatible_fold_versions")
    data = pd.concat(records, ignore_index=True)
    if not data.task.eq(task).all():
        raise ValidationError("prediction_task_does_not_match_requested_engine")
    if data.duplicated(["patient_id", "repeat"]).any() or data.groupby("patient_id").y_true.nunique().max() != 1:
        raise ValidationError("invalid_repeated_oof_structure")
    if not data.groupby("patient_id")["repeat"].nunique().eq(cfg["outer_repeats"]).all():
        raise ValidationError("patient_does_not_have_every_configured_repeat")
    pooled = data.groupby("patient_id", as_index=False).agg(y_true=("y_true", "first"), probability=("probability", "mean"))
    if len(pooled) != 261:
        raise ValidationError("pooled_development_patient_count_is_not_261")
    result = metric_summary(pooled.y_true, pooled.probability, cfg["thresholds"][task], bootstrap, cfg["seed"], True)
    # A prespecified repeat defines sets. Averaging probabilities does not average conformal validity.
    chosen = data.loc[data["repeat"] == cfg["conformal_reporting_repeat"]].copy()
    result["conformal"] = conformal_summary(chosen.y_true, chosen.include_0, chosen.include_1)
    result["conformal_reporting_repeat"] = cfg["conformal_reporting_repeat"]
    result["result_origin"] = "new reference run; compare with author-reported metrics, never overwrite them"
    result["reference_run_stamp"] = stamps[0]
    pooled["task"] = task
    pooled.to_csv(output_root / task / "pooled_oof_predictions.csv", index=False)
    chosen.to_csv(output_root / task / "single_repeat_prediction_sets.csv", index=False)
    save_json(output_root / task / "reference_metrics.json", result)
    return result


def fit_final(data_root, output_root, task, cfg=None):
    cfg = cfg or configuration()
    data, source = load_cohort(data_root, cfg=cfg)
    features = feature_spec(data_root, cfg)
    output = private_output(Path(output_root) / task / "final")
    if (output / "bundle.joblib").exists():
        raise ValidationError("final_model_already_exists_use_new_run_directory")
    bundle, metadata = fit_pipeline(data[features], labels(data, task), identifiers(data).to_numpy(),
                                     cfg, cfg["models"][task], cfg["seed"] + 90000)
    joblib.dump(bundle, output / "bundle.joblib")
    save_json(output / "metadata.json", {"fit": metadata, "dataset_sha256": fingerprint(source),
                                        "config_sha256": config_hash(cfg), "reference_code_sha256": code_hash(),
                                        "software": software_record(),
                                        "apparent_original_auc": auc_details(labels(data, task), bundle.predict(data[features]))["auc"]})
    return {"task": task, "status": "new_reference_final_model_fitted", "roles": metadata["roles"]}


def evaluate_holdout(data_root, output_root, task, cfg=None):
    cfg = cfg or configuration()
    output = private_output(Path(output_root) / task / "final")
    development, _ = load_cohort(data_root, cfg=cfg)
    holdout, source = load_cohort(data_root, "holdout", cfg)
    if set(development.patient_id) & set(holdout.patient_id):
        raise ValidationError("holdout_contains_development_patient")
    if (output / "holdout_predictions.csv").exists():
        raise ValidationError("holdout_evaluation_already_recorded_use_existing_predictions")
    metadata = json.loads((output / "metadata.json").read_text())
    if metadata["config_sha256"] != config_hash(cfg):
        raise ValidationError("configuration_changed_after_final_fit")
    if metadata["reference_code_sha256"] != code_hash():
        raise ValidationError("reference_code_changed_after_final_fit")
    # Only load objects created/trusted within the controlled institutional run.
    bundle = joblib.load(output / "bundle.joblib")
    predictions = evaluate_frame(bundle, holdout[bundle.feature_order], holdout, task, -1, -1)
    predictions.to_csv(output / "holdout_predictions.csv", index=False)
    metrics = metric_summary(predictions.y_true, predictions.probability, cfg["thresholds"][task],
                              2000, cfg["seed"], True)
    metrics["holdout_source_sha256"] = fingerprint(source)
    metrics["interpretation"] = "same-center assessment of newly written reference model; historical holdout already previously evaluated"
    save_json(output / "holdout_metrics.json", metrics)
    return metrics


def temporal_fit(data_root, output_root, task, test_era, rolling=False, cfg=None):
    cfg = cfg or configuration()
    data, source = load_cohort(data_root, cfg=cfg)
    features = feature_spec(data_root, cfg)
    train = data.era < test_era if rolling else data.era != test_era
    test = data.era == test_era
    if not train.any() or not test.any():
        raise ValidationError("empty_temporal_partition")
    output = private_output(Path(output_root) / task / ("rolling" if rolling else "loeo") / str(test_era))
    support = {"train_n": int(train.sum()), "test_n": int(test.sum()),
               "dataset_sha256": fingerprint(source), "config_sha256": config_hash(cfg),
               "reference_code_sha256": code_hash(),
               "train_events": int(labels(data.loc[train], task).sum()),
               "test_events": int(labels(data.loc[test], task).sum())}
    try:
        bundle, fit = fit_pipeline(data.loc[train, features], labels(data.loc[train], task),
            identifiers(data.loc[train]).to_numpy(), cfg, cfg["models"][task], cfg["seed"] + test_era)
        predictions = evaluate_frame(bundle, data.loc[test, features], data.loc[test], task, -1, test_era)
        predictions.to_csv(output / "predictions.csv", index=False)
        support.update(status="complete", metrics=metric_summary(predictions.y_true, predictions.probability,
                                                                 cfg["thresholds"][task], 2000, cfg["seed"]))
    except (ValidationError, ValueError) as exc:
        support.update(status="not_estimable_under_prespecified_reference_procedure", error_type=type(exc).__name__)
    save_json(output / "temporal_result.json", support)
    return support


def collect_temporal(output_root, task, rolling=False, cfg=None):
    cfg = cfg or configuration()
    folder = private_output(Path(output_root) / task / ("rolling" if rolling else "loeo"))
    eras = [2, 3] if rolling else [1, 2, 3]
    reports, predictions = [], []
    for era in eras:
        path = folder / str(era) / "temporal_result.json"
        if not path.is_file():
            raise ValidationError("every_requested_temporal_test_result_required")
        report = json.loads(path.read_text())
        reports.append(report)
        if report["status"] == "complete":
            predictions.append(pd.read_csv(folder / str(era) / "predictions.csv", dtype={"patient_id": str}))
    for key in ("dataset_sha256", "config_sha256", "reference_code_sha256"):
        if len({r[key] for r in reports}) != 1:
            raise ValidationError("temporal_versions_disagree")
    result = {"requested_eras": eras, "era_results": reports}
    if len(predictions) != len(eras):
        result.update(status="not_estimable_for_all_prespecified_temporal_windows", pooled_metrics=None)
    else:
        pooled = pd.concat(predictions, ignore_index=True)
        if pooled.patient_id.duplicated().any():
            raise ValidationError("patient_occurs_in_multiple_temporal_test_windows")
        result.update(status="complete", pooled_metrics=metric_summary(pooled.y_true, pooled.probability,
                      cfg["thresholds"][task], 2000, cfg["seed"]))
        pooled.to_csv(folder / "pooled_predictions.csv", index=False)
    save_json(folder / "pooled_temporal_result.json", result)
    return result
