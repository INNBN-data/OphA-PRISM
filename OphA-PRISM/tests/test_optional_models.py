"""Optional small software checks, skipped when model dependencies are absent."""
import importlib.util
import unittest
import numpy as np
from ophaprism.training import fit_model


class OptionalModelInterfaceTests(unittest.TestCase):
    def exercise(self, family, params):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(80, 3))
        y = np.tile([0, 1], 40)
        fitted, history, rounds = fit_model(family, params, 42, x[:60], y[:60], (x[60:], y[60:]))
        self.assertTrue(history)
        self.assertTrue(np.isfinite(fitted.predict_proba(x[60:])[:, 1]).all())
        self.assertTrue(1 <= rounds <= 20)

    @unittest.skipUnless(importlib.util.find_spec("catboost"), "CatBoost is not installed locally")
    def test_catboost_inner_history(self):
        self.exercise("catboost", {"iterations": 20, "depth": 2, "learning_rate": .05})

    @unittest.skipUnless(importlib.util.find_spec("lightgbm"), "LightGBM is not installed locally")
    def test_lightgbm_inner_history(self):
        self.exercise("lightgbm", {"n_estimators": 20, "max_depth": 2, "num_leaves": 4, "learning_rate": .05})

    @unittest.skipUnless(importlib.util.find_spec("xgboost"), "XGBoost is not installed locally")
    def test_xgboost_inner_history(self):
        self.exercise("xgboost", {"n_estimators": 20, "max_depth": 2, "learning_rate": .05})
