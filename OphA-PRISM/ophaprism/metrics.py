"""Shared numerical metrics from the accompanying read-only audit implementation."""
from server_audit.opha_prism_server_audit import (
    auc_details, pr_metrics, decision_rows, wilson, metric_summary, conformal_summary)


def paired_difference(y, first, second, repetitions=2000, seed=42):
    import numpy as np
    y = np.asarray(y, int)
    first, second = np.asarray(first, float), np.asarray(second, float)
    if len(y) != len(first) or len(y) != len(second) or set(np.unique(y)) != {0, 1}:
        raise ValueError("paired_complete_two_class_prediction_vectors_required")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("nonfinite_probabilities")
    if ((first < 0) | (first > 1)).any() or ((second < 0) | (second > 1)).any():
        raise ValueError("probabilities_outside_unit_interval")
    rng = np.random.default_rng(seed)
    cases, controls = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    def difference(ix):
        a = auc_details(y[ix], first[ix])["auc"] - auc_details(y[ix], second[ix])["auc"]
        b = np.mean((first[ix] - y[ix]) ** 2) - np.mean((second[ix] - y[ix]) ** 2)
        return [a, b]
    observed = difference(np.arange(len(y)))
    samples = [difference(np.r_[rng.choice(cases, len(cases), True), rng.choice(controls, len(controls), True)])
               for _ in range(repetitions)]
    interval = np.quantile(samples, [.025, .975], axis=0)
    return {"auc_difference": observed[0], "auc_difference_ci95": interval[:, 0].tolist(),
            "brier_difference": observed[1], "brier_difference_ci95": interval[:, 1].tolist(),
            "method": "paired patient-stratified percentile bootstrap of fixed predictions; no refitting",
            "replicates": repetitions}
