from __future__ import annotations
import math
import numpy as np


def quantile(y, probability, alpha=.10):
    """Exact split-conformal order statistic, including the small-sample infinity case."""
    y, p = np.asarray(y), np.asarray(probability, float)
    if not len(y) or len(y) != len(p) or not np.isin(y, [0, 1]).all():
        raise ValueError("valid_calibration_labels_required")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any() or not 0 < alpha < 1:
        raise ValueError("invalid_calibration_probability_or_alpha")
    scores = np.where(y == 1, 1 - p, p)
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    return float(np.sort(scores)[rank - 1]) if rank <= len(scores) else float("inf")


def sets(probability, qhat):
    p = np.asarray(probability, float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any() or np.isnan(qhat):
        raise ValueError("invalid_probability_or_quantile")
    # Deterministic <= ties are conservative. Empty sets remain empty.
    return np.column_stack([p <= qhat, (1 - p) <= qhat])
