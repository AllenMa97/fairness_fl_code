import unittest
from types import SimpleNamespace

import torch

from algorithm.praffl_core import (
    PRAFFL_STATE_SCHEMA_VERSION,
    HyperNetwork,
    clone_state_dict_to_cpu,
)
from tool.praffl_evaluation import (
    PraFFLEvaluationError,
    build_preference_grid,
    evaluate_praffl,
    evaluate_praffl_report,
    evaluate_preference_grid,
    hypervolume_2d,
    metrics_from_predictions,
    pareto_front_2d,
)


class CountingBertModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = torch.nn.Linear(3, 4)
        self.drop = torch.nn.Identity()
        self.out = torch.nn.Linear(4, 2)
        self.encode_calls = 0

    def encode(self, input_ids, attention_mask):
        del attention_mask
        self.encode_calls += 1
        return self.bert(input_ids.float())


def make_batch():
    return {
        "input_ids": torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        "attention_mask": torch.ones(4, 3, dtype=torch.long),
        "labels": torch.tensor([0, 1, 1, 0]),
        "protected": torch.tensor([0, 0, 1, 1]),
    }


class PraFFLEvaluationMathTest(unittest.TestCase):
    def test_inclusive_preference_grid_is_deterministic(self):
        expected = torch.tensor(
            [[0.0, 1.0], [0.25, 0.75], [0.5, 0.5], [0.75, 0.25], [1.0, 0.0]]
        )
        self.assertTrue(torch.equal(build_preference_grid(5), expected))

    def test_paper_dp_disparity_is_group_deviation_from_overall_rate(self):
        result = metrics_from_predictions(
            predictions=torch.tensor([[0, 1, 1, 1]]),
            labels=torch.tensor([0, 1, 1, 0]),
            protected=torch.tensor([0, 0, 1, 1]),
            scope="fixture",
        )
        self.assertTrue(torch.allclose(result["ACC"], torch.tensor([0.75], dtype=torch.float64)))
        self.assertTrue(torch.allclose(result["DP"], torch.tensor([0.25], dtype=torch.float64)))
        self.assertTrue(torch.allclose(result["SPD"], torch.tensor([-0.5], dtype=torch.float64)))
        self.assertTrue(torch.allclose(result["DEO"], torch.tensor([0.0], dtype=torch.float64)))

    def test_missing_protected_group_is_diagnostic(self):
        with self.assertRaisesRegex(PraFFLEvaluationError, "client 7.*protected counts"):
            metrics_from_predictions(
                predictions=torch.tensor([[0, 1]]),
                labels=torch.tensor([0, 1]),
                protected=torch.tensor([0, 0]),
                scope="client 7",
            )

    def test_pareto_filter_and_minimization_hypervolume_match_hand_value(self):
        points = [[0.2, 0.8], [0.5, 0.3], [0.6, 0.9], [0.2, 0.8]]
        front = pareto_front_2d(points)
        self.assertEqual(front, [[0.2, 0.8], [0.5, 0.3]])
        self.assertAlmostEqual(hypervolume_2d(front, reference_point=(1.0, 1.0)), 0.41)

    def test_encoder_runs_once_per_batch_not_once_per_preference_chunk(self):
        torch.manual_seed(5)
        model = CountingBertModel()
        hypernetwork = HyperNetwork(2, 4, 2, 6)
        preferences = build_preference_grid(7)
        result = evaluate_preference_grid(
            model,
            hypernetwork,
            [make_batch(), make_batch()],
            preferences,
            device=torch.device("cpu"),
            use_amp=False,
            chunk_size=2,
            scope="feature reuse",
        )
        self.assertEqual(model.encode_calls, 2)
        self.assertEqual(result["ACC"].shape, (7,))
        self.assertEqual(result["DP"].shape, (7,))


class PraFFLEvaluatorHookTest(unittest.TestCase):
    def test_round_report_uses_every_private_head_on_common_test_loader(self):
        torch.manual_seed(11)
        model = CountingBertModel()
        template = HyperNetwork(2, 4, 2, 6)
        state = {
            "schema_version": PRAFFL_STATE_SCHEMA_VERSION,
            "hypernetwork_spec": {
                "preference_dim": 2,
                "feature_dim": 4,
                "num_classes": 2,
                "hidden_dim": 6,
            },
            "client_hypernetworks": {
                0: clone_state_dict_to_cpu(template),
                1: clone_state_dict_to_cpu(template),
            },
        }

        metrics = evaluate_praffl_report(
            model,
            {
                "device": "cpu",
                "use_amp": False,
                "num_clients_K": 2,
                "praffl_report_preference": [0.5, 0.5],
            },
            [make_batch()],
            state,
        )

        self.assertEqual(set(metrics), {"ACC", "DEO", "SPD"})
        self.assertTrue(all(isinstance(value, float) for value in metrics.values()))
        self.assertEqual(model.encode_calls, 2)

    def test_hook_uses_every_private_head_for_local_and_global_fronts(self):
        torch.manual_seed(11)
        model = CountingBertModel()
        template = HyperNetwork(2, 4, 2, 6)
        first_state = clone_state_dict_to_cpu(template)
        second_state = clone_state_dict_to_cpu(template)
        first_key = next(iter(second_state))
        second_state[first_key].add_(0.25)
        algorithm_state = {
            "schema_version": PRAFFL_STATE_SCHEMA_VERSION,
            "completed_round": 0,
            "round_boundary": True,
            "config": {},
            "hypernetwork_spec": {
                "preference_dim": 2,
                "feature_dim": 4,
                "num_classes": 2,
                "hidden_dim": 6,
            },
            "client_hypernetworks": {0: first_state, 1: second_state},
        }
        data_bundle = SimpleNamespace(
            client_testing_dataloaders=[[make_batch()], [make_batch()]],
            testing_dataloader=[make_batch()],
        )
        metrics = evaluate_praffl(
            model,
            {
                "device": "cpu",
                "use_amp": False,
                "num_clients_K": 2,
                "algorithm_epoch_T": 2,
                "learning_rate": 0.01,
                "praffl_tau_c": 1,
                "praffl_tau_p": 1,
                "praffl_hypernetwork_hidden_dim": 6,
                "praffl_preference_points": 3,
                "praffl_preference_chunk_size": 2,
                "praffl_report_preference": [0.5, 0.5],
            },
            data_bundle,
            algorithm_state,
        )

        self.assertEqual(set(metrics), {"ACC", "DEO", "SPD", "report_preference", "praffl"})
        self.assertEqual(set(metrics["praffl"]["local"]["clients"]), {"0", "1"})
        self.assertEqual(set(metrics["praffl"]["global"]["clients"]), {"0", "1"})
        self.assertEqual(len(metrics["praffl"]["preference_grid"]), 3)
        self.assertEqual(metrics["praffl"]["reference_point"], [1.0, 1.0])
        self.assertEqual(model.encode_calls, 4)
        self.assertIsInstance(metrics["ACC"], float)
        self.assertIsInstance(metrics["praffl"]["local"]["mean_hv"], float)
        self.assertIsInstance(metrics["praffl"]["global"]["mean_hv"], float)


if __name__ == "__main__":
    unittest.main()
