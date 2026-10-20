from __future__ import annotations
from pathlib import Path
import numpy as np
from .common import configuration, private_output, identifiers, save_json, ValidationError, fingerprint
from .data import load_cohort, feature_spec
from .endpoints import labels
from .training import fit_pipeline
from .workflow import split_outer, config_hash, code_hash
from .metrics import auc_details


def stratified_bootstrap_indices(y, rng):
    y = np.asarray(y)
    return np.concatenate([rng.choice(np.flatnonzero(y == value), int((y == value).sum()), replace=True)
                           for value in (0, 1)])


def diagnostic_replicate(data_root, output_root, task, kind, index, cfg=None):
    """One independently checkpointed full-pipeline reference replicate per job."""
    cfg = cfg or configuration()
    if kind not in ("bootstrap", "permutation") or index < 0:
        raise ValidationError("invalid_diagnostic_job")
    destination = private_output(Path(output_root) / task / kind)
    path = destination / f"replicate_{index}.json"
    if path.exists():
        raise ValidationError("replicate_already_exists")
    data, source = load_cohort(data_root, cfg=cfg)
    features = feature_spec(data_root, cfg)
    x, y, groups = data[features], labels(data, task), identifiers(data).to_numpy()
    seed = cfg["seed"] + 100000 + index
    rng = np.random.default_rng(seed)
    out = {"task": task, "kind": kind, "replicate": index, "seed": seed,
           "dataset_sha256": fingerprint(source), "config_sha256": config_hash(cfg), "reference_code_sha256": code_hash(),
           "procedure": "new reference procedure including preprocessing, selection, tuning and calibration"}
    try:
        if kind == "bootstrap":
            ix = stratified_bootstrap_indices(y, rng)
            bundle, _ = fit_pipeline(x.iloc[ix], y[ix], groups[ix], cfg, cfg["models"][task], seed)
            apparent = auc_details(y[ix], bundle.predict(x.iloc[ix]))["auc"]
            tested = auc_details(y, bundle.predict(x))["auc"]
            out.update(apparent_boot_auc=apparent, test_orig_auc=tested, optimism=apparent - tested)
        else:
            permuted = rng.permutation(y)
            out["n_labels_changed"] = int((permuted != y).sum())
            probabilities = np.empty((cfg["outer_repeats"], len(y)))
            for repeat in range(cfg["outer_repeats"]):
                for fold, (a, b) in enumerate(split_outer(x, permuted, groups, cfg, repeat)):
                    bundle, _ = fit_pipeline(x.iloc[a], permuted[a], groups[a], cfg, cfg["models"][task],
                                             seed + repeat * 100 + fold)
                    probabilities[repeat, b] = bundle.predict(x.iloc[b])
            out["oof_auc"] = auc_details(permuted, probabilities.mean(axis=0))["auc"]
        out["status"] = "complete"
    except Exception as exc:
        # Retain failed replicates for complete accounting.
        out.update(status="failed", error_type=type(exc).__name__)
    save_json(path, out)
    return out


def summarize_replicates(output_root, task, kind, expected, apparent_original=None, observed_oof=None):
    import json
    folder = private_output(Path(output_root) / task / kind)
    paths = [folder / f"replicate_{i}.json" for i in range(expected)]
    if not all(p.is_file() for p in paths):
        raise ValidationError("all_requested_replicate_results_including_failures_required")
    records = [json.loads(p.read_text()) for p in paths]
    for key in ("dataset_sha256", "config_sha256", "reference_code_sha256"):
        if len({r[key] for r in records}) != 1:
            raise ValidationError("diagnostic_replicates_have_incompatible_versions")
    valid = [r for r in records if r["status"] == "complete"]
    out = {"requested": expected, "completed": len(valid), "failed": expected - len(valid),
           "scope": "No BCa corrected-performance interval is implied by replicate quantiles."}
    if len(valid) != expected:
        out["incomplete_run_warning"] = "Distribution summaries condition on successful fits; failed fits must be resolved before full-procedure inference."
    if not valid:
        raise ValidationError("no_valid_diagnostic_replicates")
    if kind == "bootstrap":
        values = [r["optimism"] for r in valid]
        out["mean_optimism"] = float(np.mean(values))
        out["replicate_optimism_quantiles"] = np.quantile(values, [.025, .975]).tolist()
        final = folder.parent / "final/metadata.json"
        if apparent_original is None and final.is_file():
            metadata = json.loads(final.read_text())
            if any(metadata[key] != records[0][key] for key in ("dataset_sha256", "config_sha256", "reference_code_sha256")):
                raise ValidationError("apparent_original_metadata_does_not_match_bootstrap")
            apparent_original = metadata["apparent_original_auc"]
        if apparent_original is not None and len(valid) == expected:
            out["optimism_corrected_auc"] = float(apparent_original) - out["mean_optimism"]
    else:
        values = [r["oof_auc"] for r in valid]
        out["null_auc_mean"] = float(np.mean(values))
        out["null_auc_quantiles"] = np.quantile(values, [.025, .975]).tolist()
        metrics_file = folder.parent / "reference_metrics.json"
        if observed_oof is None and metrics_file.is_file():
            metrics = json.loads(metrics_file.read_text())
            stamp = metrics["reference_run_stamp"]
            if any(stamp[key] != records[0][key] for key in ("dataset_sha256", "config_sha256", "reference_code_sha256")):
                raise ValidationError("observed_oof_does_not_match_permutation_procedure")
            observed_oof = metrics["roc"]["auc"]
        if observed_oof is not None and len(valid) == expected:
            out["empirical_p_value"] = (1 + sum(v >= observed_oof for v in values)) / (expected + 1)
    save_json(folder / "summary.json", out)
    return out
