import unittest

import torch

from algorithm.praffl_core import HyperNetwork, PraFFLConfig, clone_state_dict_to_cpu
from algorithm.praffl_training import (
    make_optimizer,
    train_communicated_phase,
    train_personalized_phase,
)


class TinyBertClassifier(torch.nn.Module):
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


class RecordingHyperNetwork(HyperNetwork):
    def __init__(self):
        super().__init__(2, 4, 2, 6)
        self.seen_preferences = []

    def forward(self, preferences):
        self.seen_preferences.append(preferences.detach().cpu().clone())
        return super().forward(preferences)


def make_config(tau_c=1, tau_p=1):
    return PraFFLConfig.from_param_dict(
        {
            "learning_rate": 0.05,
            "optimize_method": "sgd",
            "praffl_tau_c": tau_c,
            "praffl_tau_p": tau_p,
            "praffl_preference_batch_size": 3,
            "praffl_hypernetwork_hidden_dim": 6,
            "praffl_hypernetwork_learning_rate": 0.05,
        },
        algorithm_epoch_T=tau_c + tau_p,
    )


def make_batches():
    return [
        {
            "input_ids": torch.tensor(
                [[1.0, 0.0, 0.5], [0.0, 1.0, -0.5], [1.0, 1.0, 0.0], [0.5, 0.0, 1.0]]
            ),
            "attention_mask": torch.ones(4, 3, dtype=torch.long),
            "labels": torch.tensor([0, 1, 0, 1]),
            "protected": torch.tensor([0, 0, 1, 1]),
        },
        {
            "input_ids": torch.tensor(
                [[0.5, 1.0, 0.0], [1.0, -0.5, 1.0], [0.0, 0.5, 1.0], [1.0, 1.0, 1.0]]
            ),
            "attention_mask": torch.ones(4, 3, dtype=torch.long),
            "labels": torch.tensor([1, 0, 1, 0]),
            "protected": torch.tensor([0, 1, 0, 1]),
        },
    ]


class PraFFLTrainingPhaseTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.model = TinyBertClassifier()
        self.hypernetwork = RecordingHyperNetwork()
        self.config = make_config()
        self.device = torch.device("cpu")

    def test_communicated_phase_uses_balanced_frozen_head_and_updates_only_encoder(self):
        encoder_before = clone_state_dict_to_cpu(self.model.bert)
        classifier_before = clone_state_dict_to_cpu(self.model.out)
        hypernetwork_before = clone_state_dict_to_cpu(self.hypernetwork)
        optimizer = make_optimizer(
            self.model.bert.parameters(), self.config.optimizer_name, self.config.encoder_learning_rate
        )

        losses = train_communicated_phase(
            self.model,
            self.hypernetwork,
            make_batches(),
            epochs=self.config.tau_c,
            optimizer=optimizer,
            device=self.device,
            use_amp=False,
            scaler=None,
        )

        self.assertEqual(len(losses), 2)
        self.assertTrue(
            any(not torch.equal(encoder_before[name], value) for name, value in self.model.bert.state_dict().items())
        )
        self.assertTrue(
            all(torch.equal(classifier_before[name], value) for name, value in self.model.out.state_dict().items())
        )
        self.assertTrue(
            all(torch.equal(hypernetwork_before[name], value) for name, value in self.hypernetwork.state_dict().items())
        )
        self.assertTrue(
            all(torch.equal(preference, torch.tensor([[0.5, 0.5]])) for preference in self.hypernetwork.seen_preferences)
        )
        self.assertTrue(all(parameter.grad is None for parameter in self.hypernetwork.parameters()))

    def test_personalized_phase_updates_only_hypernetwork_and_encodes_once_per_batch(self):
        encoder_before = clone_state_dict_to_cpu(self.model.bert)
        classifier_before = clone_state_dict_to_cpu(self.model.out)
        hypernetwork_before = clone_state_dict_to_cpu(self.hypernetwork)
        optimizer = make_optimizer(
            self.hypernetwork.parameters(), "adam", self.config.hypernetwork_learning_rate
        )

        def fixed_preferences(count, device, dtype):
            self.assertEqual(count, 3)
            return torch.tensor(
                [[0.2, 0.8], [0.5, 0.5], [0.8, 0.2]],
                device=device,
                dtype=dtype,
            )

        losses = train_personalized_phase(
            self.model,
            self.hypernetwork,
            make_batches(),
            epochs=self.config.tau_p,
            preference_batch_size=self.config.preference_batch_size,
            smooth_gamma=self.config.smooth_gamma,
            optimizer=optimizer,
            device=self.device,
            use_amp=False,
            scaler=None,
            preference_sampler=fixed_preferences,
        )

        self.assertEqual(len(losses), 2)
        self.assertEqual(self.model.encode_calls, 2)
        self.assertTrue(
            all(torch.equal(encoder_before[name], value) for name, value in self.model.bert.state_dict().items())
        )
        self.assertTrue(
            all(torch.equal(classifier_before[name], value) for name, value in self.model.out.state_dict().items())
        )
        self.assertTrue(
            any(not torch.equal(hypernetwork_before[name], value) for name, value in self.hypernetwork.state_dict().items())
        )
        self.assertTrue(all(parameter.grad is None for parameter in self.model.bert.parameters()))

    def test_default_sampler_draws_two_dimensional_dirichlet_preferences(self):
        optimizer = make_optimizer(self.hypernetwork.parameters(), "adam", 0.01)
        train_personalized_phase(
            self.model,
            self.hypernetwork,
            make_batches()[:1],
            epochs=1,
            preference_batch_size=7,
            smooth_gamma=1.0,
            optimizer=optimizer,
            device=self.device,
            use_amp=False,
            scaler=None,
        )
        sampled = self.hypernetwork.seen_preferences[-1]
        self.assertEqual(sampled.shape, (7, 2))
        self.assertTrue(torch.all(sampled > 0))
        self.assertTrue(torch.allclose(sampled.sum(dim=1), torch.ones(7)))


if __name__ == "__main__":
    unittest.main()
