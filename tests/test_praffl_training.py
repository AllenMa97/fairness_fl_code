import copy
import tempfile
import unittest
from unittest.mock import patch

import torch

from algorithm.praffl_core import HyperNetwork, PraFFLConfig, clone_state_dict_to_cpu
from algorithm.PraFFL import PraFFL
from algorithm.praffl_training import (
    ClientTrainResult,
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


class PraFFLRoundTest(unittest.TestCase):
    def test_nonfinal_rounds_use_private_heads_for_evaluation(self):
        model = TinyBertClassifier()
        param_dict = {
            "task": "SENT_CLF",
            "learning_rate": 0.01,
            "optimize_method": "sgd",
            "use_amp": False,
            "repeat_seed": 101,
            "checkpoint_save_freq": 0,
            "communication_round_I": 2,
            "num_clients_K": 2,
        }

        def fake_train(
            global_model,
            hypernetwork_template,
            private_hypernetwork_state,
            dataloader,
            config,
            device,
            use_amp,
            scaler,
        ):
            del hypernetwork_template, dataloader, config, device, use_amp, scaler
            return ClientTrainResult(
                encoder_state=clone_state_dict_to_cpu(global_model.bert),
                hypernetwork_state=copy.deepcopy(private_hypernetwork_state),
                communicated_losses=(1.0,),
                personalized_losses=(2.0,),
                gpu_seconds=0.0,
            )

        round_metrics = {"ACC": 0.7, "DEO": 0.2, "SPD": -0.1}
        with (
            patch(
                "algorithm.PraFFL.client_selection",
                return_value=torch.tensor([0, 1]),
            ),
            patch(
                "algorithm.PraFFL.train_praffl_client",
                side_effect=fake_train,
            ),
            patch(
                "algorithm.PraFFL.evaluate_praffl_report",
                return_value=round_metrics,
            ) as evaluate_mock,
        ):
            result = PraFFL(
                torch.device("cpu"),
                model,
                2,
                2,
                2,
                1.0,
                0.0,
                [0, 1],
                list(range(4)),
                [[0, 1], [2, 3]],
                param_dict,
                ["global-test-batch"],
                1,
            )

        evaluate_mock.assert_called_once()
        self.assertIs(evaluate_mock.call_args.args[0], result.global_model)
        self.assertIs(evaluate_mock.call_args.args[1], param_dict)
        self.assertEqual(evaluate_mock.call_args.args[2], ["global-test-batch"])
        self.assertEqual(
            result.algorithm_state["round_metrics_history"],
            [{"round": 1, **round_metrics}],
        )

    def test_round_averages_only_selected_encoders_and_keeps_all_private_heads(self):
        torch.manual_seed(31)
        model = TinyBertClassifier()
        classifier_before = clone_state_dict_to_cpu(model.out)
        encoder_before = clone_state_dict_to_cpu(model.bert)
        recorded_initial_private_states = {}

        def fake_train(
            global_model,
            hypernetwork_template,
            private_hypernetwork_state,
            dataloader,
            config,
            device,
            use_amp,
            scaler,
        ):
            del hypernetwork_template, config, device, use_amp, scaler
            client_id = int(dataloader)
            encoder_state = clone_state_dict_to_cpu(global_model.bert)
            for _name, tensor in encoder_state.items():
                if tensor.is_floating_point():
                    tensor.add_(client_id + 1)
            private_state = copy.deepcopy(private_hypernetwork_state)
            recorded_initial_private_states.setdefault(
                client_id, copy.deepcopy(private_state)
            )
            first_name = next(iter(private_state))
            private_state[first_name] = private_state[first_name] + float(client_id + 1)
            return ClientTrainResult(
                encoder_state=encoder_state,
                hypernetwork_state=private_state,
                communicated_losses=(1.0,),
                personalized_losses=(2.0,),
                gpu_seconds=0.25,
            )

        with tempfile.TemporaryDirectory() as model_path:
            param_dict = {
                "task": "SENT_CLF",
                "learning_rate": 0.01,
                "optimize_method": "sgd",
                "use_amp": False,
                "repeat_seed": 101,
                "model_path": model_path,
                "checkpoint_save_freq": 1,
                "checkpoint_keep_latest": 1,
                "communication_round_I": 1,
                "num_clients_K": 3,
            }
            with (
                patch(
                    "algorithm.PraFFL.client_selection",
                    return_value=torch.tensor([0, 2]),
                ),
                patch(
                    "algorithm.PraFFL.train_praffl_client",
                    side_effect=fake_train,
                ),
                patch("algorithm.PraFFL.save_checkpoint") as save_mock,
                patch("algorithm.PraFFL.clean_old_checkpoints") as clean_mock,
            ):
                result = PraFFL(
                    torch.device("cpu"),
                    model,
                    2,
                    3,
                    1,
                    2 / 3,
                    0.0,
                    [0, 1, 2],
                    list(range(6)),
                    [[0, 1], [2, 3], [4, 5]],
                    param_dict,
                    [],
                    0,
                )

        for name, tensor in result.global_model.bert.state_dict().items():
            if tensor.is_floating_point():
                self.assertTrue(torch.allclose(tensor, encoder_before[name] + 2.0))
        self.assertTrue(
            all(
                torch.equal(classifier_before[name], value)
                for name, value in result.global_model.out.state_dict().items()
            )
        )
        private_states = result.algorithm_state["client_hypernetworks"]
        self.assertEqual(set(private_states), {0, 1, 2})
        first_name = next(iter(private_states[0]))
        self.assertFalse(
            torch.equal(private_states[0][first_name], private_states[2][first_name])
        )
        self.assertTrue(
            torch.equal(
                private_states[1][first_name],
                recorded_initial_private_states[0][first_name],
            )
        )
        self.assertEqual(result.client_selection_history, [[0, 2]])
        expected_encoder_mb = sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.bert.parameters()
        ) / (1024 * 1024)
        self.assertAlmostEqual(
            result.total_communication_cost, 2 * 2 * expected_encoder_mb
        )
        self.assertAlmostEqual(result.total_gpu_seconds, 0.5)
        save_mock.assert_called_once()
        self.assertNotIn("extra_state", save_mock.call_args.kwargs)
        self.assertEqual(
            set(
                save_mock.call_args.kwargs["algorithm_state"][
                    "client_hypernetworks"
                ]
            ),
            {0, 1, 2},
        )
        clean_mock.assert_called_once_with(param_dict, keep_latest=1)

    def test_non_bert_or_non_binary_model_fails_before_client_selection(self):
        with self.assertRaisesRegex(ValueError, "binary BERT"):
            PraFFL(
                torch.device("cpu"),
                torch.nn.Linear(2, 2),
                2,
                1,
                1,
                1.0,
                0.0,
                [0],
                [0],
                [[0]],
                {"task": "SENT_CLF", "learning_rate": 0.01},
                [],
                0,
            )


if __name__ == "__main__":
    unittest.main()
