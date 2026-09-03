import importlib.util
import sys
import types
import unittest
from unittest.mock import patch

import torch

if importlib.util.find_spec("tensorboard") is None:
    tensorboard_stub = types.ModuleType("tool.tensorboard_logger")
    tensorboard_stub.init_tensorboard_logger = lambda **_kwargs: None
    tensorboard_stub.log_test_metrics = lambda *_args, **_kwargs: None
    tensorboard_stub.log_system_metrics = lambda *_args, **_kwargs: None
    tensorboard_stub.flush = lambda: None
    tensorboard_stub.close = lambda: None
    tensorboard_stub.log_experiment_config = lambda *_args, **_kwargs: None
    tensorboard_stub.get_monitoring_config = lambda *_args, **_kwargs: {}
    tensorboard_stub.log_scalar = lambda *_args, **_kwargs: None
    tensorboard_stub.log_metrics = lambda *_args, **_kwargs: None
    tensorboard_stub.log_deep_metrics = lambda *_args, **_kwargs: None
    tensorboard_stub.update_step = lambda *_args, **_kwargs: None
    sys.modules["tool.tensorboard_logger"] = tensorboard_stub

import experiment
from algorithm.PraFFL import PraFFL
from main_SENT_CLF import Argparse
from tool.praffl_evaluation import evaluate_praffl


class ModelWithLargeUnusedHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = torch.nn.Linear(2, 2)
        self.bert.register_buffer("communicated_buffer", torch.ones(3))
        self.out = torch.nn.Linear(2, 200)


class PraFFLWiringTest(unittest.TestCase):
    def test_formula_counts_exact_selected_encoder_upload_and_download_only(self):
        model = ModelWithLargeUnusedHead()
        parameters = {
            "communication_round_I": 3,
            "num_clients_K": 4,
            "FL_fraction": 0.75,
            "FL_drop_rate": 0.5,
            "task": "SENT_CLF",
            "praffl_hypernetwork_hidden_dim": 5000,
        }
        encoder_mb = sum(
            tensor.numel() * tensor.element_size()
            for tensor in model.bert.state_dict().values()
        ) / (1024 * 1024)
        selected_before_drop = 3
        selected_after_drop = 2
        expected = 3 * selected_after_drop * 2 * encoder_mb

        actual = experiment.calculate_communication_cost("PraFFL", parameters, model)

        self.assertEqual(selected_before_drop, 3)
        self.assertAlmostEqual(actual, round(expected, 3))

    def test_praffl_dispatch_supplies_algorithm_specific_evaluator(self):
        with patch("experiment.Experiment_FL") as runner:
            experiment._run_praffl_experiment({"algorithm": "PraFFL"})
        runner.assert_called_once_with(
            PraFFL,
            {"algorithm": "PraFFL"},
            evaluator_function=evaluate_praffl,
        )

    def test_cli_parses_named_praffl_controls(self):
        argv = [
            "main_SENT_CLF.py",
            "-praffl_tau_c", "3",
            "-praffl_tau_p", "4",
            "-praffl_preference_batch_size", "9",
            "-praffl_hypernetwork_hidden_dim", "64",
            "-praffl_hypernetwork_learning_rate", "0.002",
            "-praffl_smooth_gamma", "2.5",
            "-praffl_report_preference", "0.4", "0.6",
            "-praffl_preference_points", "101",
            "-praffl_preference_chunk_size", "16",
        ]
        with patch.object(sys, "argv", argv):
            parsed = Argparse()
        self.assertEqual(parsed["praffl_tau_c"], 3)
        self.assertEqual(parsed["praffl_tau_p"], 4)
        self.assertEqual(parsed["praffl_preference_batch_size"], 9)
        self.assertEqual(parsed["praffl_hypernetwork_hidden_dim"], 64)
        self.assertAlmostEqual(parsed["praffl_hypernetwork_learning_rate"], 0.002)
        self.assertAlmostEqual(parsed["praffl_smooth_gamma"], 2.5)
        self.assertEqual(parsed["praffl_report_preference"], [0.4, 0.6])
        self.assertEqual(parsed["praffl_preference_points"], 101)
        self.assertEqual(parsed["praffl_preference_chunk_size"], 16)


if __name__ == "__main__":
    unittest.main()
