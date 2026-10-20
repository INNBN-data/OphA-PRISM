"""Nonclinical engineering fixtures only; none are study observations or outputs."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ophaprism.common import ROOT, ValidationError, configuration, identifiers, save_json
from ophaprism.endpoints import derive, validate_and_add
from ophaprism.conformal import quantile, sets
from ophaprism.scores import integers, probabilities
from ophaprism.training import split_roles, assert_disjoint, fit_pipeline
from ophaprism.preprocessing import processor, transform_frozen, stability_columns, imputed_partitions
from ophaprism.diagnostics import stratified_bootstrap_indices
from ophaprism.metrics import paired_difference
from sklearn.exceptions import ConvergenceWarning
from repository_audit import release_path_allowed


def endpoint_fixture():
    return pd.DataFrame({"rroc_grade": [1, 2, 3, 1], "remnant_intentional_stable": [0, 1, 0, 0],
                         "opha_patency_loss": [0, 0, 0, 1], "vision_event": [0, 1, 0, 0]})


class ClinicalRuleSoftwareTests(unittest.TestCase):
    def test_favorable_remnant_is_not_complete_occlusion(self):
        d = derive(endpoint_fixture())
        self.assertEqual(d.favorable_angiography.tolist(), [1, 1, 0, 1])
        self.assertEqual(d.technical_success.tolist(), [1, 1, 0, 0])
        self.assertEqual(d.cure_failure.tolist(), [0, 1, 1, 1])

    def test_missing_source_does_not_become_success(self):
        d = endpoint_fixture()
        d.loc[0, "vision_event"] = np.nan
        self.assertTrue(pd.isna(derive(d).cure_failure.iloc[0]))

    def test_saved_disagreement_fails(self):
        d = endpoint_fixture()
        d["cure_failure"] = [1, 1, 1, 1]
        for c in ("access_high_burden", "control_high_burden", "recon_label"):
            d[c] = 0
        with self.assertRaises(ValidationError):
            validate_and_add(d)

    def test_developmental_score_maxima_and_boundaries(self):
        d = pd.DataFrame({"max_diameter_mm": [0, 5, 10, 15], "neck_width_mm": [0, 3, 5, 7],
                          "optic_compression": [0, 0, 0, 1], "opha_incorporation": [0, 0, 0, 1],
                          "neck_opha_involvement_pct": [0, 0, 0, 20], "intraluminal_thrombus": [0, 0, 0, 1],
                          "preoperative_visual_symptoms": [0, 0, 0, 1]})
        self.assertEqual(integers(d, "VISION").tolist(), [0, 4, 8, 31])
        self.assertEqual(integers(d, "CURE")[-1], 33)

    def test_score_does_not_default_missing_input_to_zero(self):
        d = pd.DataFrame({"max_diameter_mm": [np.nan]})
        with self.assertRaises(ValidationError):
            integers(d, "VISION")
        with self.assertRaises(ValidationError):
            probabilities([10])


class ConformalSoftwareTests(unittest.TestCase):
    def test_order_statistic_uses_rank_not_quantile_interpolation(self):
        p = np.arange(1, 21) / 100
        self.assertEqual(quantile(np.zeros(20), p, .1), .19)

    def test_small_calibration_sample_yields_universal_set(self):
        q = quantile([0, 1, 0], [.1, .8, .2], .1)
        self.assertTrue(np.isinf(q))
        self.assertTrue(sets([.01, .99], q).all())

    def test_empty_sets_preserved(self):
        self.assertEqual(sets([.5], .2).tolist(), [[False, False]])

    def test_invalid_scores_refused(self):
        with self.assertRaises(ValueError):
            quantile([0, 1], [.1, 1.1])
        with self.assertRaises(ValueError):
            sets([np.nan], .2)


class IsolationSoftwareTests(unittest.TestCase):
    def test_duplicate_patient_clusters_remain_in_one_role(self):
        groups = np.repeat(np.arange(60), 2)
        y = np.repeat(np.r_[np.zeros(30, int), np.ones(30, int)], 2)
        parts = split_roles(y, groups, 10)
        assert_disjoint(groups, parts)
        self.assertEqual(sum(len(x) for x in parts), len(y))

    def test_cross_role_overlap_is_rejected(self):
        with self.assertRaises(ValidationError):
            assert_disjoint(np.array(["A", "A", "B"]), [np.array([0]), np.array([1, 2])])

    def test_preprocessing_fits_only_training_values(self):
        train = pd.DataFrame({"size": [2., 4., 6.], "category": ["a", "b", "a"]})
        test = pd.DataFrame({"size": [np.nan, 1000.], "category": ["new", "a"]})
        p = processor(train, 42, "simple")
        p.fit(train)
        values = transform_frozen(p, test)
        self.assertAlmostEqual(values[0, 0], 0.)
        self.assertEqual(p.named_transformers_["numeric"].named_steps["impute"].statistics_[0], 4.)
        self.assertNotIn("new", p.named_transformers_["categorical"].named_steps["encode"].categories_[0])

    def test_stability_does_not_force_requested_number_of_features(self):
        x, y = np.zeros((40, 2)), np.r_[np.zeros(20, int), np.ones(20, int)]
        with self.assertRaises(ValidationError):
            stability_columns(x, y, np.arange(40), ["numeric__a", "numeric__b"], ["a", "b"],
                {"enabled": True, "resamples": 3, "threshold": .75, "max_raw_features": 6}, 42)

    def test_numeric_chained_imputation_and_frozen_transform(self):
        train = pd.DataFrame({"a": [2., 3., np.nan, 5., 6., 7.], "b": [3., 4., 5., np.nan, 7., 8.]})
        held = pd.DataFrame({"a": [np.nan, 1000.], "b": [5., np.nan]})
        partitions = imputed_partitions(train, held, {"imputation_backend": "iterative_numeric", "imputations": 2}, 42)
        self.assertEqual(len(partitions), 2)
        for pre, a, b in partitions:
            self.assertTrue(np.isfinite(a).all() and np.isfinite(b).all())
            np.testing.assert_allclose(b, transform_frozen(pre, held))
            np.testing.assert_allclose(b[::-1], transform_frozen(pre, held.iloc[::-1]))
            initial = pre.named_transformers_["numeric"].named_steps["impute"].initial_imputer_.statistics_
            np.testing.assert_allclose(initial, train.mean().to_numpy())

    def test_nonconverged_stability_fits_are_not_used(self):
        def warn(*args, **kwargs):
            warnings.warn("Artificial test warning", ConvergenceWarning)
        y = np.tile([0, 1], 20)
        with patch("ophaprism.preprocessing.LogisticRegression.fit", side_effect=warn):
            with self.assertRaises(ValidationError):
                stability_columns(np.zeros((40, 1)), y, np.arange(40), ["numeric__a"], ["a"],
                    {"enabled": True, "resamples": 1, "threshold": .75, "max_raw_features": 6}, 42)

    def test_stratified_bootstrap_preserves_event_count(self):
        y = np.r_[np.zeros(20, int), np.ones(7, int)]
        ix = stratified_bootstrap_indices(y, np.random.default_rng(42))
        self.assertEqual(len(ix), len(y))
        self.assertEqual(y[ix].sum(), y.sum())


class ReferenceFitSoftwareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Artificial fixture exercises plumbing only, not study performance.
        rng = np.random.default_rng(4)
        cls.x = pd.DataFrame({"a": rng.normal(size=120), "b": rng.normal(size=120)})
        cls.y = np.tile([0, 1], 60)
        cfg = configuration()
        cfg.update(imputation_backend="simple", imputations=1, optuna_trials=1, inner_folds=2)
        spec = {"family": "elastic_net", "parameters": {"C": .1, "l1_ratio": .5}, "selection": {"enabled": False}}
        cls.bundle, cls.metadata = fit_pipeline(cls.x, cls.y, np.arange(120), cfg, spec, 42)

    def test_new_reference_fit_returns_bounded_deterministic_probabilities(self):
        first, second = self.bundle.predict(self.x.iloc[:10]), self.bundle.predict(self.x.iloc[:10])
        np.testing.assert_allclose(first, second)
        self.assertTrue(((first >= 0) & (first <= 1)).all())
        self.assertEqual(sum(x["n"] for x in self.metadata["roles"]), 120)

    def test_serialized_bundle_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test_bundle.joblib"
            joblib.dump(self.bundle, path)
            recovered = joblib.load(path)
            np.testing.assert_allclose(self.bundle.predict(self.x.iloc[:5]), recovered.predict(self.x.iloc[:5]))

    def test_prediction_feature_order_checked(self):
        with self.assertRaises(ValidationError):
            self.bundle.predict(self.x[["b", "a"]])


class ReleaseSoftwareTests(unittest.TestCase):
    def test_patient_files_and_models_not_release_eligible(self):
        for path in ("runs/private/predictions.csv", "models/VISION/model.joblib", "data/private/patients.csv",
                     "models/VISION/oof_predictions.csv", "results/aggregate/unchecked.csv", "../secret.txt"):
            self.assertFalse(release_path_allowed(path), path)
        self.assertTrue(release_path_allowed("data/private/README.md"))
        self.assertTrue(release_path_allowed("ophaprism/training.py"))
        self.assertTrue(release_path_allowed("server_audit/audit_config.json"))
        self.assertFalse(release_path_allowed("models/patient_predictions.json"))

    def test_cli_help_and_missing_data_fail_without_fallback(self):
        help_result = subprocess.run([sys.executable, str(ROOT / "run_pipeline.py"), "--help"], capture_output=True, text=True)
        self.assertEqual(help_result.returncode, 0)
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([sys.executable, str(ROOT / "run_pipeline.py"), "validate", "--data-root", folder],
                                     capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("required_server_input_missing", result.stderr)
            self.assertFalse(list(Path(folder).iterdir()))

    def test_nonfinite_json_metrics_are_explicit_null(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "metric.json"
            save_json(path, {"metric": np.nan, "quantile": np.inf})
            self.assertEqual(json.loads(path.read_text()), {"metric": None, "quantile": None})

    def test_paired_difference_uses_matching_draws(self):
        y, p = np.array([0, 0, 1, 1]), np.array([.1, .4, .3, .9])
        result = paired_difference(y, p, p, repetitions=20)
        self.assertEqual(result["auc_difference_ci95"], [0., 0.])
        self.assertEqual(result["brier_difference_ci95"], [0., 0.])
        with self.assertRaises(ValueError):
            paired_difference(y, p, np.array([-1., .1, .2, .3]), repetitions=20)

    def test_figure_coordinates_exclude_individual_probability_thresholds(self):
        from ophaprism.figures import plot_predictions
        private = ROOT / "runs/private"
        private.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=private) as folder:
            folder = Path(folder)
            source = folder / "fixture.csv"
            pd.DataFrame({"patient_id": ["ARTIFICIAL_TEST_" + str(i) for i in range(40)],
                          "task": "VISION", "y_true": np.tile([0, 1], 20),
                          "probability": np.linspace(.05, .95, 40)}).to_csv(source, index=False)
            plot_predictions(source, folder / "figures", "VISION", bootstrap=10)
            exported = pd.read_csv(folder / "figures/VISION_roc_coordinates.csv")
            self.assertEqual(exported.columns.tolist(), ["fpr", "tpr"])
            self.assertTrue((folder / "figures/VISION_diagnostics.pdf").is_file())


if __name__ == "__main__":
    unittest.main()
