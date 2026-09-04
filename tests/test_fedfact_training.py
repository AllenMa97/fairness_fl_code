import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from algorithm.FedFACT import (
    _audit_current_ensemble,
    _forward_text_logits,
    _train_theta_and_phi,
    StreamingModelAverage,
)
from algorithm.fedfact_core import (
    build_calibration_matrices, build_support_statistics, confusion_from_predictions,
)
from tests.fedfact_test_utils import (
    TinyTextDataset, cpu_state_dict, make_datasets_and_loaders, seeded_model,
)


class ClientRoundPrimitiveTest(unittest.TestCase):
    def setUp(self):
        self.datasets, self.loaders = make_datasets_and_loaders(batch_size=2)
        self.stats = build_support_statistics(self.datasets, "DP")
        self.matrix = build_calibration_matrices(
            0, self.stats.counts, "DP",
            torch.tensor([[.1, .1]], dtype=torch.float64),
            torch.tensor([[.1, .1]], dtype=torch.float64),
            .001,
        )

    def test_forward_adapter_extracts_exactly_two_logits(self):
        model = seeded_model()
        batch = next(iter(self.loaders[0]))
        logits = _forward_text_logits(model, batch, torch.device("cpu"))
        self.assertEqual(tuple(logits.shape), (2, 2))

    def test_audit_uses_current_probability_ensemble_and_sample_mean_losses(self):
        theta, phi = seeded_model(3), seeded_model(4)
        audit = _audit_current_ensemble(
            theta, phi, self.loaders[0], self.matrix, torch.tensor(.25),
            torch.device("cpu"), use_amp=False,
        )
        self.assertEqual(audit.sample_count, 4)
        self.assertEqual(tuple(audit.confusion.shape), (2, 2, 2))
        self.assertGreater(audit.theta_loss, 0)
        self.assertGreater(audit.phi_loss, 0)
        manual_predictions = []
        with torch.no_grad():
            for batch in self.loaders[0]:
                theta_logits = _forward_text_logits(theta, batch, torch.device("cpu"))
                phi_logits = _forward_text_logits(phi, batch, torch.device("cpu"))
                probabilities = .25 * theta_logits.softmax(1) + .75 * phi_logits.softmax(1)
                manual_predictions.append(probabilities.argmax(1))
        expected_confusion = confusion_from_predictions(
            torch.cat(manual_predictions),
            torch.tensor(self.datasets[0].labels),
            torch.tensor(self.datasets[0].protected),
        )
        torch.testing.assert_close(audit.confusion, expected_confusion)

    def test_same_batches_update_both_theta_and_phi(self):
        theta, phi = seeded_model(7), seeded_model(8)
        theta_before, phi_before = cpu_state_dict(theta), cpu_state_dict(phi)
        batch_trace = []
        _train_theta_and_phi(
            theta, phi, self.loaders[0], self.matrix,
            epochs=1, param_dict={"optimize_method": "sgd", "learning_rate": .05},
            device=torch.device("cpu"), use_amp=False, scaler=None,
            batch_trace=batch_trace,
        )
        self.assertEqual(batch_trace, [[-2., -1.], [1., 2.]])
        self.assertTrue(any(
            not torch.equal(theta_before[name], theta.state_dict()[name])
            for name in theta_before
        ))
        self.assertTrue(any(
            not torch.equal(phi_before[name], phi.state_dict()[name])
            for name in phi_before
        ))

    def test_streaming_average_accepts_only_unified_state(self):
        base = seeded_model(11)
        one, three = copy.deepcopy(base), copy.deepcopy(base)
        with torch.no_grad():
            for parameter in one.parameters():
                parameter.fill_(1)
            for parameter in three.parameters():
                parameter.fill_(3)
        average = StreamingModelAverage(base.state_dict(), total_weight=4)
        average.add(cpu_state_dict(one), sample_weight=1)
        average.add(cpu_state_dict(three), sample_weight=3)
        result = average.finish()
        for name, tensor in base.state_dict().items():
            if tensor.is_floating_point():
                torch.testing.assert_close(result[name], torch.full_like(tensor, 2.5))

from algorithm.FedFACT import FedFACT
from tool.experiment_state import AlgorithmRunResult
from tests.fedfact_test_utils import (
    assert_state_dict_equal, fedfact_params, seed_everything,
)


class FedFACTOrchestrationTest(unittest.TestCase):
    def test_nonfinal_round_uses_personalized_evaluator_and_records_history(self):
        from module.experiment_setup import FederatedDataBundle

        datasets, loaders = make_datasets_and_loaders(batch_size=4)
        bundle = FederatedDataBundle(
            training_dataloaders=loaders,
            client_dataset_list=datasets,
            testing_dataloader=loaders[0],
            client_testing_dataloaders=loaders,
            client_testing_dataset_list=datasets,
            partition_fingerprint="fedfact-round-eval",
            partition_metadata={},
        )
        metrics = {"ACC": .5, "DEO": .2, "SPD": -.1}
        with tempfile.TemporaryDirectory() as raw, patch(
            "algorithm.fedfact_evaluation.evaluate_fedfact",
            return_value=metrics,
        ) as evaluator:
            params = fedfact_params(Path(raw), rounds=2)
            result = FedFACT(
                "cpu", seeded_model(13), 1, 2, 2, 1.0, 0.0,
                loaders, datasets[0], datasets, params, None, 0,
                data_bundle=bundle,
            )

        evaluator.assert_called_once()
        self.assertEqual(
            result.algorithm_state["round_metrics_history"],
            [{"round": 1, **metrics}],
        )

    def test_support_failure_precedes_optimizer_or_model_copy(self):
        invalid = TinyTextDataset([(-1., 0, 0), (1., 1, 0)])
        with tempfile.TemporaryDirectory() as raw:
            params = fedfact_params(Path(raw))
            with patch("algorithm.FedFACT._make_optimizer") as optimizer:
                with self.assertRaisesRegex(Exception, r"client 0.*protected=1"):
                    FedFACT(
                        "cpu", seeded_model(), 1, 1, 1, 1.0, 0.0,
                        [torch.utils.data.DataLoader(invalid, batch_size=2)],
                        invalid, [invalid], dict(params, num_clients_K=1),
                        None, 0,
                    )
            optimizer.assert_not_called()

    def test_round_updates_persistent_phi_and_server_theta_and_records_all_clients(self):
        datasets, loaders = make_datasets_and_loaders(batch_size=2)
        with tempfile.TemporaryDirectory() as raw:
            params = fedfact_params(Path(raw), rounds=1)
            initial = seeded_model(19)
            initial_state = cpu_state_dict(initial)
            seed_everything()
            result = FedFACT(
                "cpu", initial, 1, 2, 1, 1.0, 0.0,
                loaders, datasets[0], datasets, params, None, 0,
            )
            self.assertIsInstance(result, AlgorithmRunResult)
            self.assertEqual(result.client_selection_history, [[0, 1]])
            self.assertEqual(len(result.algorithm_state["personal_model_states"]), 2)
            self.assertTrue(any(
                not torch.equal(initial_state[name], result.global_model.state_dict()[name])
                for name in initial_state
            ))
            for personal in result.algorithm_state["personal_model_states"]:
                self.assertTrue(any(
                    not torch.equal(initial_state[name], personal[name])
                    for name in initial_state
                ))
            self.assertFalse(torch.equal(
                result.algorithm_state["personal_model_states"][0]["out.weight"],
                result.algorithm_state["personal_model_states"][1]["out.weight"],
            ))

    def test_round_uses_old_weight_for_confusion_then_stores_new_weight(self):
        datasets, loaders = make_datasets_and_loaders(batch_size=4)
        with tempfile.TemporaryDirectory() as raw:
            params = fedfact_params(Path(raw), rounds=1)
            observed_weights = []
            real_audit = _audit_current_ensemble
            def recording_audit(*args, **kwargs):
                observed_weights.append(float(torch.as_tensor(args[4])))
                return real_audit(*args, **kwargs)
            with patch("algorithm.FedFACT._audit_current_ensemble", side_effect=recording_audit):
                result = FedFACT(
                    "cpu", seeded_model(23), 1, 2, 1, 1.0, 0.0,
                    loaders, datasets[0], datasets, params, None, 0,
                )
            self.assertEqual(observed_weights, [.5, .5])
            self.assertTrue(torch.isfinite(result.algorithm_state["ensemble_weights"]).all())
