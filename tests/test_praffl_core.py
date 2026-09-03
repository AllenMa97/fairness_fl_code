import unittest
from collections import OrderedDict

import torch

from algorithm.praffl_core import (
    HyperNetwork,
    PraFFLConfig,
    clone_state_dict_to_cpu,
    demographic_parity_surrogate,
    functional_linear_heads,
    smooth_tchebycheff,
    uniform_average_state_dicts,
)


class PraFFLCoreTest(unittest.TestCase):
    def test_hypernetwork_accepts_two_preferences_and_gradients_reach_it(self):
        hypernetwork = HyperNetwork(
            preference_dim=2,
            feature_dim=3,
            num_classes=2,
            hidden_dim=5,
        )
        preferences = torch.tensor([[0.5, 0.5], [0.2, 0.8]])
        features = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])

        weight, bias = hypernetwork(preferences)
        logits = functional_linear_heads(features, weight, bias)
        logits.square().mean().backward()

        self.assertEqual(weight.shape, (2, 2, 3))
        self.assertEqual(bias.shape, (2, 2))
        self.assertEqual(logits.shape, (2, 2, 2))
        self.assertTrue(
            all(parameter.grad is not None for parameter in hypernetwork.parameters())
        )

    def test_dp_surrogate_matches_hand_calculation_and_is_differentiable(self):
        logits = torch.tensor(
            [[[1.0, 3.0], [5.0, 7.0]]],
            requires_grad=True,
        )
        protected = torch.tensor([0.0, 1.0])

        loss = demographic_parity_surrogate(logits, protected)
        loss.sum().backward()

        self.assertTrue(torch.allclose(loss, torch.tensor([2.0])))
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_inverse_weighted_smooth_tchebycheff_matches_formula(self):
        accuracy_loss = torch.tensor([2.0])
        fairness_loss = torch.tensor([6.0])
        preference = torch.tensor([[0.25, 0.75]])

        actual = smooth_tchebycheff(
            accuracy_loss,
            fairness_loss,
            preference,
            gamma=2.0,
        )
        expected = torch.logsumexp(torch.tensor([[16.0, 16.0]]), dim=1) / 2.0

        self.assertTrue(torch.allclose(actual, expected))

    def test_scalarization_rejects_zero_training_preferences(self):
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            smooth_tchebycheff(
                torch.ones(1),
                torch.ones(1),
                torch.tensor([[0.0, 1.0]]),
                gamma=1.0,
            )

    def test_uniform_average_does_not_apply_dataset_size_weights(self):
        states = [
            OrderedDict(weight=torch.tensor([0.0, 2.0]), counter=torch.tensor(3)),
            OrderedDict(weight=torch.tensor([4.0, 6.0]), counter=torch.tensor(3)),
        ]

        averaged = uniform_average_state_dicts(states)

        self.assertTrue(torch.equal(averaged["weight"], torch.tensor([2.0, 4.0])))
        self.assertEqual(averaged["counter"].item(), 3)

    def test_non_float_buffers_must_match_before_aggregation(self):
        states = [
            OrderedDict(counter=torch.tensor(1)),
            OrderedDict(counter=torch.tensor(2)),
        ]
        with self.assertRaisesRegex(ValueError, "non-floating tensor"):
            uniform_average_state_dicts(states)

    def test_config_splits_total_epochs_and_validates_explicit_sum(self):
        config = PraFFLConfig.from_param_dict(
            {"learning_rate": 5e-5, "optimize_method": "adam"},
            algorithm_epoch_T=5,
        )
        self.assertEqual((config.tau_c, config.tau_p), (2, 3))
        self.assertEqual(config.report_preference, (0.5, 0.5))
        self.assertEqual(config.preference_points, 1000)

        with self.assertRaisesRegex(ValueError, "must equal algorithm_epoch_T"):
            PraFFLConfig.from_param_dict(
                {
                    "learning_rate": 5e-5,
                    "praffl_tau_c": 2,
                    "praffl_tau_p": 2,
                },
                algorithm_epoch_T=5,
            )

    def test_config_rejects_one_total_epoch(self):
        with self.assertRaisesRegex(ValueError, "at least 2"):
            PraFFLConfig.from_param_dict(
                {"learning_rate": 5e-5},
                algorithm_epoch_T=1,
            )

    def test_cpu_state_clone_load_roundtrip_is_independent(self):
        torch.manual_seed(17)
        original = HyperNetwork(
            preference_dim=2,
            feature_dim=3,
            num_classes=2,
            hidden_dim=5,
        )
        snapshot = clone_state_dict_to_cpu(original)
        expected = OrderedDict(
            (name, tensor.detach().cpu().clone())
            for name, tensor in original.state_dict().items()
        )

        with torch.no_grad():
            for parameter in original.parameters():
                parameter.add_(10.0)

        restored = HyperNetwork(
            preference_dim=2,
            feature_dim=3,
            num_classes=2,
            hidden_dim=5,
        )
        restored.load_state_dict(snapshot, strict=True)

        self.assertIsInstance(snapshot, OrderedDict)
        self.assertTrue(all(tensor.device.type == "cpu" for tensor in snapshot.values()))
        self.assertTrue(all(not tensor.requires_grad for tensor in snapshot.values()))
        for name, tensor in restored.state_dict().items():
            self.assertTrue(torch.equal(tensor.cpu(), expected[name]))
            self.assertFalse(torch.equal(tensor.cpu(), original.state_dict()[name].cpu()))


if __name__ == "__main__":
    unittest.main()
