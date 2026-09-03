import copy
import tempfile
import unittest
from pathlib import Path

import torch

from algorithm.fedfact_core import build_support_statistics, initialize_fedfact_state
from algorithm.fedfact_evaluation import evaluate_fedfact
from module.experiment_setup import FederatedDataBundle
from tests.fedfact_test_utils import (
    TinyTextDataset, fedfact_params, make_datasets_and_loaders, seeded_model,
)


def make_bundle(datasets, loaders):
    return FederatedDataBundle(
        training_dataloaders=loaders,
        client_dataset_list=datasets,
        testing_dataloader=None,
        client_testing_dataloaders=loaders,
        client_testing_dataset_list=datasets,
        partition_fingerprint="fedfact-evaluator-fixture",
        partition_metadata={},
    )


class FedFACTEvaluatorTest(unittest.TestCase):
    def setUp(self):
        self.datasets, self.loaders = make_datasets_and_loaders(batch_size=2)
        self.bundle = make_bundle(self.datasets, self.loaders)
        self.global_model = seeded_model(41)
        with torch.no_grad():
            self.global_model.out.weight.zero_()
            self.global_model.out.bias.copy_(torch.tensor([2., -2.]))
        stats = build_support_statistics(self.datasets, "DP")
        self.state = initialize_fedfact_state(
            self.global_model, 2, "DP", stats, dual_init=.1,
            ensemble_weight_init=.5,
        )
        personal = copy.deepcopy(self.global_model)
        with torch.no_grad():
            # Personal classifier predicts class one exactly when feature > 0.
            personal.out.weight.copy_(torch.tensor([[-1.], [1.]]))
            personal.out.bias.zero_()
        personal_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in personal.state_dict().items()
        }
        self.state["personal_model_states"] = [
            copy.deepcopy(personal_state), copy.deepcopy(personal_state)
        ]

    def test_metrics_change_with_private_model_weight_and_are_not_global_only(self):
        with tempfile.TemporaryDirectory() as raw:
            params = fedfact_params(Path(raw))
            personal_dominant = copy.deepcopy(self.state)
            personal_dominant["ensemble_weights"] = torch.tensor([.01, .01], dtype=torch.float64)
            result = evaluate_fedfact(
                self.global_model, params, self.bundle, personal_dominant
            )
            self.assertLess(result["SPD"], -.9)
            self.assertGreater(result["global_fairness"], .9)
            self.assertEqual(len(result["local_fairness_by_client"]), 2)

            global_dominant = copy.deepcopy(self.state)
            global_dominant["ensemble_weights"] = torch.tensor([.99, .99], dtype=torch.float64)
            changed = evaluate_fedfact(
                self.global_model, params, self.bundle, global_dominant
            )
            self.assertAlmostEqual(changed["SPD"], 0.)
            self.assertAlmostEqual(changed["global_fairness"], 0.)
            self.assertNotEqual(result["SPD"], changed["SPD"])

    def test_client_id_selects_the_matching_private_model(self):
        with tempfile.TemporaryDirectory() as raw:
            params = fedfact_params(Path(raw))
            state = copy.deepcopy(self.state)
            reversed_personal = copy.deepcopy(self.global_model)
            with torch.no_grad():
                reversed_personal.out.weight.copy_(torch.tensor([[1.], [-1.]]))
                reversed_personal.out.bias.zero_()
            state["personal_model_states"][1] = {
                name: tensor.detach().cpu().clone()
                for name, tensor in reversed_personal.state_dict().items()
            }
            state["ensemble_weights"] = torch.tensor([.01, .01], dtype=torch.float64)
            result = evaluate_fedfact(self.global_model, params, self.bundle, state)
            self.assertGreater(result["local_signed_disparity_by_client"][0][0], 0.9)
            self.assertLess(result["local_signed_disparity_by_client"][1][0], -0.9)

    def test_missing_test_support_raises_instead_of_reporting_zero(self):
        bad = TinyTextDataset([(-1., 0, 0), (1., 1, 0)])
        bad_bundle = make_bundle([bad, self.datasets[1]], [
            torch.utils.data.DataLoader(bad, batch_size=2), self.loaders[1]
        ])
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaisesRegex(Exception, r"client 0.*protected=1"):
                evaluate_fedfact(
                    self.global_model, fedfact_params(Path(raw)), bad_bundle, self.state
                )
