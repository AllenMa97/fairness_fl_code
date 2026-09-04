import json
import unittest
from pathlib import Path

from algorithm.fedfact_core import validate_fedfact_entrypoint
from main_SENT_CLF import fedfact_fraction_list, merge_fedfact_cli_overrides


class FedFACTConfigurationTest(unittest.TestCase):
    def test_json_names_metric_tolerances_and_all_algorithm_parameters(self):
        config = json.loads(Path("json/algorithm/FedFACT.json").read_text())
        self.assertEqual(config, {
            "fairness_metric": "DP",
            "global_constraint": 0.01,
            "local_constraint": 0.01,
            "dual_learning_rate": 0.03,
            "dual_bound": 5.0,
            "dual_init": 0.1,
            "ensemble_learning_rate": 0.3,
            "ensemble_weight_init": 0.5,
            "calibration_epsilon": 0.001,
            "FL_fraction": 1.0,
            "FL_drop_rate": 0.0,
            "checkpoint_keep_latest": 1,
        })
        self.assertNotIn("fairness_level", config)
        self.assertNotIn("eta_d", config)
        self.assertNotIn("eta_w", config)
        self.assertNotIn("w_init", config)

    def test_fedfact_fraction_is_all_clients_but_other_defaults_are_unchanged(self):
        self.assertEqual(fedfact_fraction_list("FedFACT"), [1.0])
        self.assertEqual(fedfact_fraction_list("FedAvg"), [.1])

    def test_non_null_cli_values_override_json_values(self):
        merged = merge_fedfact_cli_overrides(
            {"fairness_metric": "DP", "global_constraint": .01},
            {"fairness_metric": "EO", "global_constraint": .02,
             "local_constraint": None},
        )
        self.assertEqual(merged["fairness_metric"], "EO")
        self.assertEqual(merged["global_constraint"], .02)
        self.assertNotIn("local_constraint", merged)

    def test_image_and_tabular_entrypoints_are_rejected(self):
        validate_fedfact_entrypoint("FedFACT", "SENT_CLF")
        with self.assertRaisesRegex(ValueError, "only supports SENT_CLF"):
            validate_fedfact_entrypoint("FedFACT", "IMG_CLF")
        with self.assertRaisesRegex(ValueError, "only supports SENT_CLF"):
            validate_fedfact_entrypoint("FedFACT", "Tabular_CLF")
