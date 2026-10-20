from __future__ import annotations
import copy
import warnings
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from .common import ValidationError


def canonical(data):
    value = data.copy()
    # Remove pandas nullable scalar objects before sklearn's numeric checks.
    for c in value:
        if pd.api.types.is_numeric_dtype(value[c]):
            value[c] = pd.to_numeric(value[c], errors="raise").astype(float)
        else:
            value[c] = value[c].astype(object).where(value[c].notna(), np.nan)
    return value


def processor(train, seed, mode="iterative_numeric"):
    """Fit training-partition numeric and categorical preprocessing."""
    train = canonical(train)
    numeric, categorical = [], []
    for c in train:
        observed = train[c].dropna()
        if pd.api.types.is_numeric_dtype(train[c]) and not set(observed.unique()).issubset({0, 1}):
            numeric.append(c)
        else:
            categorical.append(c)
    blocks = []
    if numeric:
        if mode == "iterative_numeric":
            imputer = IterativeImputer(max_iter=10, sample_posterior=True, random_state=seed,
                                       keep_empty_features=True)
        elif mode == "simple":
            imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        else:
            raise ValidationError("unsupported_reference_imputation_mode")
        blocks.append(("numeric", Pipeline([("impute", imputer), ("scale", StandardScaler())]), numeric))
    if categorical:
        blocks.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical))
    return ColumnTransformer(blocks, remainder="drop", verbose_feature_names_out=True)


def transform_frozen(preprocessor, data):
    # IterativeImputer posterior sampling mutates RNG state during transform.
    # Use a frozen state separately for each row: a patient's probability must
    # not change merely because other patients are added to the prediction batch.
    numeric = preprocessor.named_transformers_.get("numeric")
    iterative = numeric is not None and isinstance(numeric.named_steps.get("impute"), IterativeImputer)
    value = canonical(data)
    if iterative:
        return np.vstack([copy.deepcopy(preprocessor).transform(value.iloc[[i]]) for i in range(len(value))])
    return copy.deepcopy(preprocessor).transform(value)


def imputed_partitions(train, validation, config, seed):
    count = int(config["imputations"])
    mode = config["imputation_backend"]
    if count < 1 or (mode == "simple" and count != 1):
        raise ValidationError("simple_imputation_requires_one_dataset; iterative_requires_positive_m")
    result = []
    for imp in range(count):
        pre = processor(train, seed + imp, mode)
        a = pre.fit_transform(canonical(train))
        b = transform_frozen(pre, validation)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValidationError("nonfinite_preprocessed_values")
        result.append((pre, a, b))
    return result


def feature_parents(names, columns):
    parents = []
    for name in names:
        matches = [c for c in columns if name == "numeric__" + c or name == "categorical__" + c
                   or name.startswith("categorical__" + c + "_")]
        if not matches:
            raise ValidationError("transformed_feature_has_no_raw_parent")
        parents.append(max(matches, key=len))
    return np.asarray(parents, object)


def stability_columns(matrix, y, groups, names, raw_columns, config, seed):
    if not config["enabled"]:
        return np.arange(matrix.shape[1]), {"method": "all eligible encoded columns"}
    parents = feature_parents(names, raw_columns)
    raw = list(dict.fromkeys(parents))
    counts = {c: 0 for c in raw}
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    group_classes = np.asarray([np.asarray(y)[np.flatnonzero(groups == g)[0]] for g in unique])
    reps = int(config["resamples"])
    for _ in range(reps):
        # Preserve all duplicate instances of an original patient together.
        sampled = np.concatenate([rng.choice(unique[group_classes == label],
            int((group_classes == label).sum()), replace=True) for label in (0, 1)])
        ix = np.concatenate([np.flatnonzero(groups == g) for g in sampled])
        if np.unique(np.asarray(y)[ix]).size < 2:
            raise ValidationError("stability_resample_has_one_outcome_class")
        fit = LogisticRegression(penalty="elasticnet", solver="saga", C=.1, l1_ratio=.5,
                                 max_iter=5000, random_state=seed)
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always", ConvergenceWarning)
            fit.fit(matrix[ix], np.asarray(y)[ix])
        if any(issubclass(w.category, ConvergenceWarning) for w in recorded):
            raise ValidationError("stability_selection_fit_nonconvergence")
        selected = set(parents[np.abs(fit.coef_[0]) > 1e-8])
        for c in selected:
            counts[c] += 1
    frequencies = {c: counts[c] / reps for c in raw}
    kept = [c for c in raw if frequencies[c] >= float(config["threshold"])]
    kept.sort(key=lambda c: (-frequencies[c], c))
    kept = kept[:int(config["max_raw_features"])]
    if not kept:
        raise ValidationError("no_features_meet_stability_threshold; target_count_not_forced")
    indices = np.flatnonzero(np.isin(parents, kept))
    return indices, {"method": "training-only grouped elastic-net bootstrap on first imputation",
                     "resamples": reps, "frequencies": frequencies, "selected_raw_features": kept,
                     "threshold": config["threshold"], "other_imputations": "same selected raw variables"}
