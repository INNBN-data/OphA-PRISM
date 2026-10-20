"""Software tests using artificial, nonclinical arrays; no study data are supplied."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server_audit import opha_prism_server_audit as audit


def configuration():
    return json.loads(Path(audit.__file__).with_name("audit_config.json").read_text())


def artificial_master():
    # Deliberately small, explicitly artificial records for joining/privacy tests.
    # These are never saved with, or presented as, clinical output.
    return pd.DataFrame({
        "patient_id": ["PRIVATE_TEST_A", "PRIVATE_TEST_B", "PRIVATE_TEST_C", "PRIVATE_TEST_D"],
        "cohort_flag": [0, 0, 0, 0], "surgery_year": [2000, 2010, 2020, 2021],
        "era": [1, 2, 3, 3], "max_diameter_mm": [3., 5., 11., 12.],
        "rroc_grade": [1, 2, 3, 1], "remnant_intentional_stable": [0, 1, 0, 0],
        "opha_patency_loss": [0, 0, 0, 1], "vision_event": [0, 0, 1, 0],
        "technical_success": [1, 1, 0, 0], "cure_failure": [0, 0, 1, 1],
        "access_high_burden": [0, 1, 1, 0], "control_high_burden": [1, 0, 0, 1],
        "recon_label": [0, 1, 1, 1]})


class MetricTests(unittest.TestCase):
    def test_auc_and_both_pr_definitions(self):
        y, p = [0, 0, 1, 1], [.1, .4, .35, .8]
        self.assertEqual(audit.auc_details(y, p)["auc"], .75)
        m = audit.pr_metrics(y, p)
        self.assertAlmostEqual(m["average_precision"], 5 / 6)
        self.assertAlmostEqual(m["pr_auc_trapezoidal"], 19 / 24)

    def test_ties_and_single_class(self):
        self.assertEqual(audit.auc_details([0, 1], [.5, .5])["auc"], .5)
        self.assertEqual(audit.auc_details([0, 1], [.5, .5])["tied_pairs"], 1)
        self.assertIsNone(audit.auc_details([0, 0], [.2, .8])["auc"])
        self.assertEqual(audit.pr_metrics([0, 1], [.5, .5])["average_precision"], .5)

    def test_documented_binary_rule_arithmetic(self):
        # Aggregate-count arithmetic test; not an attempted patient reconstruction.
        n, events, tp, fp = 261, 50, 39, 3
        sensitivity, specificity = tp / events, (n - events - fp) / (n - events)
        self.assertAlmostEqual((sensitivity + specificity) / 2, .8828909952606635)
        self.assertAlmostEqual((events - tp + fp) / n, .05363984674329502)
        self.assertAlmostEqual((tp - fp * .4 / .6) / n, .1417624521072797)

    def test_brier_and_decision_counts(self):
        m = audit.metric_summary([0, 0, 1, 1], [0, 1, 1, 1], [.5])
        self.assertEqual(m["brier"], .25)
        self.assertEqual(m["brier_skill"], 0.)
        self.assertEqual(m["decision_curve"][0]["tp"], 2)
        self.assertEqual(m["decision_curve"][0]["fp"], 1)

    def test_invalid_probabilities_not_repaired(self):
        with self.assertRaises(audit.AuditError):
            audit.metric_summary([0, 1], [-.1, 1.1], [.5])
        with self.assertRaises(audit.AuditError):
            audit.binary(pd.Series([0, "unknown"]))

    def test_conditional_bootstrap_is_reproducible(self):
        a = audit.metric_summary([0, 0, 1, 1], [.2, .4, .3, .9], [.5], bootstrap=30, seed=4)
        b = audit.metric_summary([0, 0, 1, 1], [.2, .4, .3, .9], [.5], bootstrap=30, seed=4)
        self.assertEqual(a["fixed_prediction_bootstrap"], b["fixed_prediction_bootstrap"])
        self.assertIn("no model refitting", a["fixed_prediction_bootstrap"]["scope"])


class EndpointTests(unittest.TestCase):
    def test_accepted_remnant_and_joint_distribution(self):
        df, cfg = artificial_master(), configuration()
        grade, loss, vision, favorable, technical, cure = audit.derived_endpoints(df, cfg["columns"])
        self.assertEqual(favorable.tolist(), [True, True, False, True])
        self.assertEqual(technical.tolist(), [1, 1, 0, 0])
        self.assertEqual(cure.tolist(), [0, 0, 1, 1])
        self.assertEqual(sum(x["n"] for x in audit.joint_counts(grade, loss, vision)), 4)

    def test_missing_is_not_a_negative(self):
        df, cfg = artificial_master(), configuration()
        df.loc[0, "vision_event"] = np.nan
        df.loc[1, "remnant_intentional_stable"] = np.nan
        result = audit.derived_endpoints(df, cfg["columns"])
        self.assertTrue(pd.isna(result[5].iloc[0]))
        self.assertTrue(pd.isna(result[5].iloc[1]))

    def test_unaccepted_remnant_is_exposed(self):
        df, cfg = artificial_master(), configuration()
        df.loc[1, "remnant_intentional_stable"] = 0
        out = audit.audit_cohort(df, cfg, 0)
        self.assertEqual(out["angiography"]["unaccepted_or_unknown_class_ii"], 1)
        self.assertEqual(out["cure_failure_saved_vs_derived"]["disagreements"], 1)

    def test_maneuver_count_and_construct_rules(self):
        df, cfg = artificial_master(), configuration()
        for i, name in enumerate(cfg["access_components"]):
            df[name] = [1, int(i < 5), int(i < 4), 0]
        df["access_count"] = [8, 5, 4, 0]
        df["clip_count"] = [1, 2, 1, 1]
        df["fenestrated_clip"] = [0, 0, 1, 0]
        df["tandem_clip"] = [0, 0, 0, 0]
        df["booster_clip"] = [0, 0, 0, 1]
        out = audit.operative_coding(df, cfg)
        self.assertEqual(out["access_count"]["count_ge5"], 2)
        self.assertEqual(out["access_count"]["saved_count_disagreements"], 0)
        self.assertEqual(out["recon_saved_vs_component_rule"]["disagreements"], 0)


class PredictionTests(unittest.TestCase):
    def test_equal_repeat_weight_after_imputation_averaging(self):
        d = pd.DataFrame({"id": ["A", "A", "A", "B", "B", "B"],
                          "repeat": [1, 1, 2, 1, 1, 2], "imp": [1, 2, 1, 1, 2, 1],
                          "y": [0, 0, 0, 1, 1, 1], "p": [.1, .3, .8, .3, .5, 1.]})
        pooled, _ = audit.pool_predictions(d, "id", d.y, d.p, "repeat", "imp")
        self.assertAlmostEqual(pooled.loc[pooled.id == "A", "p"].iloc[0], .5)
        self.assertAlmostEqual(pooled.loc[pooled.id == "B", "p"].iloc[0], .7)

    def test_duplicate_and_multiple_fold_predictions_refused(self):
        d = pd.DataFrame({"id": ["A", "A"], "y": [0, 0], "p": [.2, .3]})
        with self.assertRaises(audit.AuditError):
            audit.pool_predictions(d, "id", d.y, d.p)
        d["repeat"], d["imp"], d["fold"] = [1, 1], [1, 2], [1, 2]
        with self.assertRaises(audit.AuditError):
            audit.pool_predictions(d, "id", d.y, d.p, "repeat", "imp", "fold")

    def test_variants_kept_separate_and_training_rows_excluded(self):
        master, cfg = artificial_master(), configuration()
        base = pd.DataFrame({"patient_id": master.patient_id, "task": "VISION",
                             "probability": [.1, .2, .8, .3], "model": "elastic_net", "role": "test"})
        second = base.copy()
        second["model"] = "baseline"
        training = base.copy()
        training["role"] = "train"
        d = pd.concat([base, second, training], ignore_index=True)
        out = audit.audit_predictions(d, cfg["prediction_files"]["oof"], master, cfg, 0, False)
        self.assertEqual(out["excluded_non_test_rows"], 4)
        self.assertEqual(len(out["groups"]), 2)
        self.assertTrue(all(g["status"] == "calculated" for g in out["groups"]))
        self.assertTrue(all(g["result"]["metrics"]["n"] == 4 for g in out["groups"]))
        serialized = json.dumps(out)
        self.assertNotIn("PRIVATE_TEST_", serialized)

    def test_label_conflict_does_not_get_averaged(self):
        master, cfg = artificial_master(), configuration()
        d = pd.DataFrame({"patient_id": master.patient_id, "task": "VISION",
                          "probability": [.1, .2, .8, .3], "y_true": [0, 1, 1, 0]})
        out = audit.audit_predictions(d, cfg["prediction_files"]["oof"], master, cfg, 0, False)
        self.assertEqual(out["groups"][0]["code"], "saved_prediction_labels_disagree_with_master")

    def test_ambiguous_probability_columns_require_mapping(self):
        d = pd.DataFrame({"task": ["VISION"], "probability": [.1], "p_calibrated": [.2]})
        with self.assertRaises(audit.AuditError):
            list(audit.prediction_tasks(d, {}))
        self.assertEqual(list(audit.prediction_tasks(d, {"probability": "p_calibrated"}))[0][2], "p_calibrated")


class ConformalTests(unittest.TestCase):
    def test_empty_sets_remain_empty_and_mean_identity(self):
        out = audit.conformal_summary(pd.Series([0, 0, 1, 1]),
                                      pd.Series([1, 0, 1, 0]), pd.Series([0, 0, 1, 1]))
        self.assertEqual(out["empty"], 1)
        self.assertEqual(out["two_class"], 1)
        self.assertEqual(out["mean_set_size"], 1.)
        self.assertEqual(out["overall_coverage"]["numerator"], 3)

    def test_singleton_ppv_denominator(self):
        out = audit.conformal_summary(pd.Series([0, 1, 1]), pd.Series([0, 0, 0]), pd.Series([1, 1, 1]))
        self.assertEqual(out["positive_singleton_ppv"]["numerator"], 2)
        self.assertEqual(out["positive_singleton_ppv"]["denominator"], 3)


class ExecutionTests(unittest.TestCase):
    def test_end_to_end_report_has_no_identifiers_or_rows(self):
        cfg = configuration()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            df = artificial_master()
            df.to_csv(root / cfg["files"]["development"], index=False)
            original = (root / cfg["files"]["development"]).read_bytes()
            (root / "outputs").mkdir()
            pd.DataFrame({"patient_id": df.patient_id, "task": "VISION", "probability": [.1, .2, .8, .3]}).to_csv(
                root / cfg["prediction_files"]["oof"]["path"], index=False)
            out = audit.Audit(root, cfg).run()
            serialized = json.dumps(out, allow_nan=False)
            self.assertNotIn("PRIVATE_TEST_", serialized)
            self.assertEqual((root / cfg["files"]["development"]).read_bytes(), original)
            self.assertEqual(out["development"]["status"], "calculated")
            self.assertEqual(out["saved_predictions"]["oof"]["result"]["groups"][0]["status"], "calculated")

    def test_path_escape_and_duplicate_headers_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj = audit.Audit(tmp, configuration())
            with self.assertRaises(audit.AuditError):
                obj.path("../outside.csv")
            Path(tmp, "duplicate.csv").write_text("x,x\n1,2\n")
            with self.assertRaises(audit.AuditError):
                obj.read_csv("duplicate.csv")

    def test_output_never_overwrites_an_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp, "existing.json")
            target.write_text("keep this")
            process = subprocess.run([sys.executable, audit.__file__, "--root", tmp, "--output", str(target)],
                                     capture_output=True, text=True)
            self.assertEqual(process.returncode, 2)
            self.assertEqual(target.read_text(), "keep this")
            self.assertNotIn("Traceback", process.stderr)

    def test_bootstrap_variants_are_not_mixed(self):
        cfg = configuration()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "outputs").mkdir()
            pd.DataFrame({"task": ["CURE", "CURE"], "model": ["one", "two"],
                          "apparent_boot_auc": [.9, .8], "test_orig_auc": [.8, .7]}).to_csv(
                root / cfg["files"]["supporting"]["bootstrap"], index=False)
            with self.assertRaises(audit.AuditError):
                audit.Audit(root, cfg).bootstrap_audit()


if __name__ == "__main__":
    unittest.main()
