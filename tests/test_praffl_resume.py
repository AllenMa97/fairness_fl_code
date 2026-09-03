import copy
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from algorithm.PraFFL import PraFFL
from module.experiment_setup import FederatedDataBundle
from tool.praffl_evaluation import evaluate_praffl


class TinyResumeBert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = torch.nn.Linear(3, 4)
        self.drop = torch.nn.Identity()
        self.out = torch.nn.Linear(4, 2)

    def encode(self, input_ids, attention_mask):
        del attention_mask
        return self.bert(input_ids.float())


def client_batch(offset):
    return {
        "input_ids": torch.tensor(
            [
                [1.0 + offset, 0.0, 0.5],
                [0.0, 1.0 + offset, -0.5],
                [1.0, 1.0, offset],
                [0.5, 0.0, 1.0 + offset],
            ]
        ),
        "attention_mask": torch.ones(4, 3, dtype=torch.long),
        "labels": torch.tensor([0, 1, 0, 1]),
        "protected": torch.tensor([0, 0, 1, 1]),
    }


def set_all_rng(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def assert_nested_close(test_case, left, right, path="root"):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0.0, atol=1e-7, msg=path)
    elif isinstance(left, dict):
        test_case.assertEqual(set(left), set(right), path)
        for key in left:
            assert_nested_close(test_case, left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        test_case.assertEqual(len(left), len(right), path)
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            assert_nested_close(test_case, left_item, right_item, f"{path}[{index}]")
    elif isinstance(left, float):
        test_case.assertAlmostEqual(left, right, places=7, msg=path)
    else:
        test_case.assertEqual(left, right, path)


class PlannedCrash(RuntimeError):
    pass


class PraFFLResumeTest(unittest.TestCase):
    def test_two_round_run_matches_round_one_checkpoint_plus_resume(self):
        set_all_rng(404)
        initial_model = TinyResumeBert()
        training_loaders = [[client_batch(0.0)], [client_batch(0.25)]]
        client_datasets = [[0, 1, 2, 3], [4, 5, 6, 7]]
        training_dataset = list(range(8))
        param_dict = {
            "task": "SENT_CLF",
            "device": "cpu",
            "learning_rate": 0.02,
            "optimize_method": "sgd",
            "use_amp": False,
            "repeat_seed": 7404,
            "model_path": "/tmp/praffl-resume-test",
            "checkpoint_save_freq": 1,
            "communication_round_I": 2,
            "num_clients_K": 2,
            "FL_fraction": 1.0,
            "FL_drop_rate": 0.0,
            "algorithm_epoch_T": 2,
            "praffl_tau_c": 1,
            "praffl_tau_p": 1,
            "praffl_preference_batch_size": 3,
            "praffl_hypernetwork_hidden_dim": 6,
            "praffl_hypernetwork_learning_rate": 0.01,
            "praffl_preference_points": 5,
            "praffl_preference_chunk_size": 2,
            "praffl_report_preference": [0.5, 0.5],
        }

        continuous_model = copy.deepcopy(initial_model)
        set_all_rng(909)
        with (
            patch("algorithm.PraFFL.save_checkpoint"),
            patch("algorithm.PraFFL.clean_old_checkpoints"),
        ):
            continuous = PraFFL(
                torch.device("cpu"), continuous_model, 2, 2, 2, 1.0, 0.0,
                training_loaders, training_dataset, client_datasets, param_dict, [], 0,
            )

        captured = {}

        def capture_checkpoint(
            checkpoint_param_dict,
            iter_t,
            checkpoint_model,
            **kwargs,
        ):
            self.assertIs(checkpoint_param_dict, param_dict)
            self.assertEqual(iter_t, 0)
            captured["model_state"] = copy.deepcopy(checkpoint_model.state_dict())
            captured["algorithm_state"] = copy.deepcopy(kwargs["algorithm_state"])
            captured["amp_scaler_state"] = None
            captured["total_gpu_seconds"] = kwargs["total_gpu_seconds"]
            captured["total_runtime_seconds"] = kwargs["total_runtime_seconds"]
            captured["total_communication_cost"] = kwargs["total_communication_cost"]
            captured["client_selection_history"] = copy.deepcopy(kwargs["client_selection_history"])
            captured["python_rng"] = random.getstate()
            captured["numpy_rng"] = np.random.get_state()
            captured["torch_rng"] = torch.get_rng_state().clone()
            raise PlannedCrash("stop after durable round one")

        resumed_model = copy.deepcopy(initial_model)
        set_all_rng(909)
        with (
            patch("algorithm.PraFFL.save_checkpoint", side_effect=capture_checkpoint),
            patch("algorithm.PraFFL.clean_old_checkpoints"),
        ):
            with self.assertRaisesRegex(PlannedCrash, "durable round one"):
                PraFFL(
                    torch.device("cpu"), resumed_model, 2, 2, 2, 1.0, 0.0,
                    training_loaders, training_dataset, client_datasets, param_dict, [], 0,
                )

        resumed_model.load_state_dict(captured["model_state"])
        checkpoint_state = SimpleNamespace(
            next_round=1,
            phase="train",
            algorithm_state=captured["algorithm_state"],
            amp_scaler_state=captured["amp_scaler_state"],
            total_gpu_seconds=captured["total_gpu_seconds"],
            total_runtime_seconds=captured["total_runtime_seconds"],
            total_communication_cost=captured["total_communication_cost"],
            client_selection_history=captured["client_selection_history"],
        )
        random.setstate(captured["python_rng"])
        np.random.set_state(captured["numpy_rng"])
        torch.set_rng_state(captured["torch_rng"])
        with (
            patch("algorithm.PraFFL.save_checkpoint"),
            patch("algorithm.PraFFL.clean_old_checkpoints"),
        ):
            resumed = PraFFL(
                torch.device("cpu"), resumed_model, 2, 2, 2, 1.0, 0.0,
                training_loaders, training_dataset, client_datasets, param_dict, [], 0,
                start_round=1,
                resume_state=checkpoint_state,
            )

        for name, tensor in continuous.global_model.state_dict().items():
            torch.testing.assert_close(tensor, resumed.global_model.state_dict()[name], rtol=0.0, atol=1e-7)
        assert_nested_close(self, continuous.algorithm_state, resumed.algorithm_state)
        self.assertEqual(continuous.client_selection_history, resumed.client_selection_history)
        self.assertAlmostEqual(
            continuous.total_communication_cost,
            resumed.total_communication_cost,
            places=12,
        )

        data_bundle = FederatedDataBundle(
            training_dataloaders=training_loaders,
            client_dataset_list=client_datasets,
            testing_dataloader=[client_batch(0.1)],
            client_testing_dataloaders=[[client_batch(0.0)], [client_batch(0.25)]],
            client_testing_dataset_list=client_datasets,
            partition_fingerprint="praffl-resume-fixture",
            partition_metadata={"fixture": True},
        )
        continuous_metrics = evaluate_praffl(
            continuous.global_model,
            param_dict,
            data_bundle,
            continuous.algorithm_state,
        )
        resumed_metrics = evaluate_praffl(
            resumed.global_model,
            param_dict,
            data_bundle,
            resumed.algorithm_state,
        )
        assert_nested_close(self, continuous_metrics, resumed_metrics)


    def test_evaluate_phase_checkpoint_does_not_train_again(self):
        model = TinyResumeBert()
        param_dict = {
            "task": "SENT_CLF",
            "learning_rate": 0.01,
            "optimize_method": "sgd",
            "use_amp": False,
            "repeat_seed": 12,
            "communication_round_I": 1,
            "praffl_tau_c": 1,
            "praffl_tau_p": 1,
            "praffl_hypernetwork_hidden_dim": 6,
        }
        with (
            patch("algorithm.PraFFL.save_checkpoint"),
            patch("algorithm.PraFFL.clean_old_checkpoints"),
        ):
            trained = PraFFL(
                torch.device("cpu"), model, 2, 2, 1, 1.0, 0.0,
                [[client_batch(0.0)], [client_batch(0.2)]],
                list(range(8)),
                [list(range(4)), list(range(4, 8))],
                param_dict, [], 0,
            )
        evaluate_state = SimpleNamespace(
            next_round=1,
            phase="evaluate",
            algorithm_state=copy.deepcopy(trained.algorithm_state),
            amp_scaler_state=None,
            total_gpu_seconds=trained.total_gpu_seconds,
            total_runtime_seconds=0.0,
            total_communication_cost=trained.total_communication_cost,
            client_selection_history=copy.deepcopy(trained.client_selection_history),
        )
        model_before = copy.deepcopy(trained.global_model.state_dict())
        with patch("algorithm.PraFFL.train_praffl_client") as train_mock:
            restored = PraFFL(
                torch.device("cpu"), trained.global_model, 2, 2, 1, 1.0, 0.0,
                [[client_batch(0.0)], [client_batch(0.2)]],
                list(range(8)),
                [list(range(4)), list(range(4, 8))],
                param_dict, [], 0,
                start_round=1,
                resume_state=evaluate_state,
            )
        train_mock.assert_not_called()
        for name, tensor in model_before.items():
            torch.testing.assert_close(tensor, restored.global_model.state_dict()[name], rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
