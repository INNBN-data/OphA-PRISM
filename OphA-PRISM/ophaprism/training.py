from __future__ import annotations
from dataclasses import dataclass
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.exceptions import ConvergenceWarning
from .common import ValidationError
from .preprocessing import imputed_partitions, transform_frozen, stability_columns
from .conformal import quantile, sets


def two_classes(y):
    if set(np.unique(y)) != {0, 1}:
        raise ValidationError("partition_requires_two_outcome_classes")


def split_roles(y, groups, seed, conformal_fraction=.20, probability_fraction=.20):
    """Split whole patients: fitting, probability calibration, conformal calibration."""
    y, groups = np.asarray(y, int), np.asarray(groups)
    unique, first = np.unique(groups, return_index=True)
    for g in unique:
        if np.unique(y[groups == g]).size != 1:
            raise ValidationError("patient_group_contains_conflicting_labels")
    classes = y[first]
    rest, conformal = train_test_split(np.arange(len(unique)), test_size=conformal_fraction,
                                      stratify=classes, random_state=seed)
    fitting, probability = train_test_split(rest, test_size=probability_fraction,
                                           stratify=classes[rest], random_state=seed + 1)
    result = []
    for rows in (fitting, probability, conformal):
        ix = np.flatnonzero(np.isin(groups, unique[rows]))
        result.append(ix)
    assert_disjoint(groups, result)
    two_classes(y[result[0]])
    two_classes(y[result[1]])
    return tuple(result)


def assert_disjoint(groups, partitions):
    values = [set(np.asarray(groups)[p]) for p in partitions]
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if values[i] & values[j]:
                raise ValidationError("patient_group_crosses_fit_or_calibration_partitions")


def model(family, params, seed):
    params = dict(params)
    if family == "elastic_net":
        return LogisticRegression(penalty="elasticnet", solver="saga", max_iter=5000,
                                  random_state=seed, **params)
    if family == "l1_logistic":
        return LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000,
                                  random_state=seed, **params)
    if family == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(loss_function="Logloss", eval_metric="Logloss", verbose=False,
                                  random_seed=seed, thread_count=1, allow_writing_files=False, **params)
    if family == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(objective="binary", verbosity=-1, n_jobs=1, random_state=seed,
                              deterministic=True, force_col_wise=True, **params)
    if family == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(objective="binary:logistic", eval_metric="logloss", n_jobs=1,
                             random_state=seed, **params)
    raise ValidationError("unknown_model_family")


def fit_model(family, params, seed, x, y, validation=None):
    estimator = model(family, params, seed)
    history = {}
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        if validation is not None and family == "lightgbm":
            from lightgbm import early_stopping, record_evaluation
            estimator.fit(x, y, eval_set=[(x, y), validation], eval_metric="binary_logloss",
                          callbacks=[early_stopping(50, verbose=False), record_evaluation(history)])
        elif validation is not None and family == "catboost":
            estimator.fit(x, y, eval_set=validation, early_stopping_rounds=50, use_best_model=True, verbose=False)
            history = estimator.get_evals_result()
        elif validation is not None and family == "xgboost":
            estimator.fit(x, y, eval_set=[(x, y), validation], early_stopping_rounds=50, verbose=False)
            history = estimator.evals_result()
        else:
            estimator.fit(x, y)
    if any(issubclass(w.category, ConvergenceWarning) for w in recorded):
        raise ValidationError("model_fit_nonconvergence")
    if family == "catboost":
        best = estimator.get_best_iteration()
        rounds = best + 1 if best >= 0 else params["iterations"]
    elif family == "lightgbm":
        rounds = getattr(estimator, "best_iteration_", 0) or params["n_estimators"]
    elif family == "xgboost" and validation is not None:
        rounds = estimator.best_iteration + 1
    else:
        rounds = None
    return estimator, history, rounds


def parameter_space(trial, family, fixed):
    params = dict(fixed)
    if family == "elastic_net":
        params.update(C=trial.suggest_float("C", .01, 10, log=True),
                      l1_ratio=trial.suggest_categorical("l1_ratio", [.1, .5, .9]))
    elif family == "l1_logistic":
        params["C"] = trial.suggest_float("C", .01, 10, log=True)
    else:
        params["learning_rate"] = trial.suggest_float("learning_rate", .01, .10, log=True)
        params["depth" if family == "catboost" else "max_depth"] = trial.suggest_int("depth", 2, 4)
        if family == "lightgbm":
            params["num_leaves"] = trial.suggest_categorical("num_leaves", [5, 9, 15])
    return params


def prepared_fold(train, valid, y, groups, cfg, spec, seed):
    parts = imputed_partitions(train, valid, cfg, seed)
    names = list(parts[0][0].get_feature_names_out())
    selected, selection = stability_columns(parts[0][1], y, groups, names, list(train.columns),
                                             spec["selection"], seed)
    raw_selected = selection.get("selected_raw_features")
    cooked = []
    from .preprocessing import feature_parents
    for pre, a, b in parts:
        current_names = list(pre.get_feature_names_out())
        if raw_selected:
            indices = np.flatnonzero(np.isin(feature_parents(current_names, list(train.columns)), raw_selected))
        else:
            indices = np.arange(a.shape[1])
        cooked.append((pre, a[:, indices], b[:, indices], indices))
    return cooked, selection


def tune(x, y, groups, cfg, spec, seed):
    split = StratifiedGroupKFold(n_splits=int(cfg["inner_folds"]), shuffle=True, random_state=seed)
    folds = []
    for fold, (a, b) in enumerate(split.split(x, y, groups)):
        assert_disjoint(groups, [a, b])
        two_classes(np.asarray(y)[a]); two_classes(np.asarray(y)[b])
        prepared, selection = prepared_fold(x.iloc[a], x.iloc[b], np.asarray(y)[a], np.asarray(groups)[a],
                                             cfg, spec, seed + fold * 100)
        folds.append((a, b, prepared, selection))
    trials = int(cfg["optuna_trials"])
    def evaluate(params):
        losses, histories, rounds = [], [], []
        for inner, (a, b, prepared, selection) in enumerate(folds):
            values = []
            for imp, (_, train, valid, _) in enumerate(prepared):
                fitted, history, best = fit_model(spec["family"], params, seed + imp,
                                                  train, np.asarray(y)[a], (valid, np.asarray(y)[b]))
                values.append(fitted.predict_proba(valid)[:, 1])
                histories.append({"inner_fold": inner, "imputation": imp, "metrics": history,
                                  "selected_iteration": best, "selection": selection})
                if best is not None:
                    rounds.append(best)
            losses.append(log_loss(np.asarray(y)[b], np.mean(values, axis=0), labels=[0, 1]))
        return float(np.mean(losses)), histories, rounds
    if trials == 1:
        best = dict(spec["parameters"])
        loss, histories, rounds = evaluate(best)
        trial_records = [{"trial": 0, "value": loss, "parameters": best}]
    else:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        def objective(trial):
            params = parameter_space(trial, spec["family"], spec["parameters"])
            value, _, _ = evaluate(params)
            return value
        study.optimize(objective, n_trials=trials, n_jobs=1)
        best = parameter_space(optuna.trial.FixedTrial(study.best_trial.params), spec["family"], spec["parameters"])
        _, histories, rounds = evaluate(best)
        trial_records = [{"trial": t.number, "value": t.value, "state": t.state.name, "parameters": t.params}
                         for t in study.trials]
    if rounds:
        key = "iterations" if spec["family"] == "catboost" else "n_estimators"
        best[key] = max(1, int(np.median(rounds)))
    return best, histories, trial_records


@dataclass
class Bundle:
    members: list
    calibrator: object
    calibration_method: str
    qhat: float
    feature_order: list

    def raw_predict(self, data):
        if list(data.columns) != self.feature_order:
            raise ValidationError("prediction_feature_order_differs_from_training")
        values = [m.predict_proba(transform_frozen(p, data)[:, ix])[:, 1] for p, m, ix in self.members]
        return np.mean(values, axis=0)

    def predict(self, data):
        raw = self.raw_predict(data)
        if self.calibration_method == "isotonic":
            return self.calibrator.predict(raw)
        if self.calibration_method == "logistic":
            logits = np.log(np.clip(raw, 1e-6, 1-1e-6) / (1-np.clip(raw, 1e-6, 1-1e-6))).reshape(-1, 1)
            return self.calibrator.predict_proba(logits)[:, 1]
        raise ValidationError("unknown_calibration_method")

    def prediction_sets(self, data):
        return sets(self.predict(data), self.qhat)


def fit_pipeline(x, y, groups, cfg, spec, seed):
    """Fit an entire reference procedure using only the supplied training patients."""
    y, groups = np.asarray(y, int), np.asarray(groups)
    fit, probability, conformity = split_roles(y, groups, seed,
        cfg["conformal_fraction"], cfg["probability_calibration_fraction"])
    best, histories, trials = tune(x.iloc[fit], y[fit], groups[fit], cfg, spec, seed + 10000)
    prepared, selection = prepared_fold(x.iloc[fit], x.iloc[probability], y[fit], groups[fit],
                                         cfg, spec, seed + 20000)
    members, cal_probabilities = [], []
    for imp, (pre, a, b, indices) in enumerate(prepared):
        fitted, _, _ = fit_model(spec["family"], best, seed + imp, a, y[fit])
        cal_probabilities.append(fitted.predict_proba(b)[:, 1])
        members.append((pre, fitted, indices))
    raw = np.mean(cal_probabilities, axis=0)
    method = cfg["calibration_method"]
    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw, y[probability])
    elif method == "logistic":
        z = np.log(np.clip(raw, 1e-6, 1-1e-6) / (1-np.clip(raw, 1e-6, 1-1e-6))).reshape(-1, 1)
        calibrator = LogisticRegression(C=1e6, solver="lbfgs").fit(z, y[probability])
    else:
        raise ValidationError("unknown_calibration_method")
    bundle = Bundle(members, calibrator, method, float("inf"), list(x.columns))
    conf_probability = bundle.predict(x.iloc[conformity])
    bundle.qhat = quantile(y[conformity], conf_probability, cfg["conformal_alpha"])
    partition = []
    for role, ix in (("fit", fit), ("probability_calibration", probability), ("conformal_calibration", conformity)):
        partition.append({"role": role, "n": len(ix), "events": int(y[ix].sum())})
    return bundle, {"parameters": best, "selection": selection, "roles": partition,
                    "inner_training_histories": histories, "optuna_trials": trials,
                    "qhat": bundle.qhat, "qhat_infinite": bool(np.isinf(bundle.qhat)),
                    "procedure_origin": "new reference implementation; not historical numerical reproduction"}
