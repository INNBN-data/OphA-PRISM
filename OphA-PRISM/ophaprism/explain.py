from __future__ import annotations
import numpy as np
import pandas as pd


def permutation_importance(bundle, x_test, y_test, repeats=20, seed=42):
    """Evaluate only patients excluded from this fitted bundle."""
    from .metrics import auc_details
    baseline = auc_details(y_test, bundle.predict(x_test))["auc"]
    if baseline is None:
        raise ValueError("two_classes_required_for_auc_importance")
    rng = np.random.default_rng(seed)
    records = []
    for c in x_test:
        drops = []
        for _ in range(repeats):
            changed = x_test.copy()
            changed[c] = rng.permutation(changed[c].to_numpy())
            drops.append(baseline - auc_details(y_test, bundle.predict(changed))["auc"])
        records.append({"feature": c, "auc_drop_mean": np.mean(drops), "auc_drop_sd": np.std(drops, ddof=1)})
    return pd.DataFrame(records)


def ale(bundle, x_test, x_training, feature, bins=10):
    """Training-defined bins; local differences evaluated on held-out observations."""
    train = pd.to_numeric(x_training[feature], errors="raise").dropna()
    edges = np.unique(np.quantile(train, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        raise ValueError("insufficient_training_support_for_ale")
    values = pd.to_numeric(x_test[feature], errors="raise")
    indices = np.searchsorted(edges[1:-1], values, side="left")
    records = []
    for i in range(len(edges) - 1):
        mask = values.notna() & values.between(edges[0], edges[-1]) & (indices == i)
        if not mask.any():
            records.append({"lower": edges[i], "upper": edges[i+1], "n": 0, "local_difference": np.nan})
            continue
        low, high = x_test.loc[mask].copy(), x_test.loc[mask].copy()
        low[feature], high[feature] = edges[i], edges[i+1]
        records.append({"lower": edges[i], "upper": edges[i+1], "n": int(mask.sum()),
                        "local_difference": float(np.mean(bundle.predict(high) - bundle.predict(low)))})
    result = pd.DataFrame(records)
    # Empty evaluation bins leave the accumulated effect undefined.
    if result.n.eq(0).any():
        result["ale"] = np.nan
        result["status"] = "empty_test_bins_require_pooled_fold_analysis"
    else:
        effect = result.local_difference.cumsum()
        result["ale"] = effect - np.average(effect, weights=result.n)
        result["status"] = "calculated"
    return result


def kernel_shap(bundle, x_test, x_training, seed=42, background_rows=10):
    """Optional final-probability SHAP; background is sampled from fitting rows only.

    Complete numeric inputs are required by this reference adapter. Missing or
    categorical inputs require the actual server-specific explanation adapter.
    """
    import shap
    if any(not pd.api.types.is_numeric_dtype(x_training[c]) for c in x_training):
        raise ValueError("numeric_shap_adapter_required_for_categorical_inputs")
    if x_test.isna().any().any() or x_training.isna().any().any():
        raise ValueError("server_specific_shap_adapter_required_for_missing_inputs")
    background = x_training.sample(min(background_rows, len(x_training)), random_state=seed)
    columns = list(x_training.columns)
    def predict(array):
        return bundle.predict(pd.DataFrame(array, columns=columns))
    explainer = shap.KernelExplainer(predict, background.to_numpy(), link="identity")
    values = explainer.shap_values(x_test.to_numpy())
    return np.asarray(values), {"scale": "final calibrated probability", "method": "KernelSHAP",
                                "background": "sampled fitting patients; not k-means",
                                "background_n": len(background), "seed": seed}
