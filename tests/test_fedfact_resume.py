import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from algorithm.FedFACT import FedFACT
from algorithm.fedfact_evaluation import evaluate_fedfact
from experiment import _run_single_repeat
from module.experiment_setup import FederatedDataBundle
from tool.checkpoint import (
    build_experiment_config_hash, get_repeat_state_dir,
    load_checkpoint, save_checkpoint,
)
from tests.fedfact_test_utils import (
    TinyTextClassifier, TinyTextDataset, balanced_rows,
    fedfact_params, make_datasets_and_loaders,
)


PARTITION_FINGERPRINT = "fedfact-resume-partition"


def assert_nested_equal(testcase, left, right, path="state"):
    if torch.is_tensor(left):
        testcase.assertIsInstance(right, torch.Tensor, path)
        testcase.assertTrue(torch.equal(left, right), path)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right, err_msg=path)
    elif isinstance(left, dict):
        testcase.assertEqual(set(left), set(right), path)
        for key in left:
            assert_nested_equal(testcase, left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        testcase.assertEqual(type(left), type(right), path)
        testcase.assertEqual(len(left), len(right), path)
        for index, (a, b) in enumerate(zip(left, right)):
            assert_nested_equal(testcase, a, b, f"{path}[{index}]")
    else:
        testcase.assertEqual(left, right, path)


class InjectedRoundBoundaryCrash(RuntimeError):
    pass


class FedFACTResumeTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def params(self, name, resume):
        params = fedfact_params(self.root / name, rounds=2)
        params.update({
            "base_seed": 2026,
            "resume": resume,
            "exp_repeat_times": 1,
            "parallel_repeats": 1,
            "final_artifact_policy": "full_state",
        })
        params["experiment_config_hash"] = build_experiment_config_hash(params)
        return params

    @staticmethod
    def dataset_factory(params):
        del params
        train = TinyTextDataset(balanced_rows() + balanced_rows(.25))
        test = TinyTextDataset(balanced_rows() + balanced_rows(.25))
        return train, None, test

    @staticmethod
    def dataloader_factory(params, training_dataset, validation_dataset,
                           testing_dataset, split_strategy):
        del params, training_dataset, validation_dataset, testing_dataset, split_strategy
        datasets, loaders = make_datasets_and_loaders(batch_size=2)
        return FederatedDataBundle(
            training_dataloaders=loaders,
            client_dataset_list=datasets,
            testing_dataloader=loaders[0],
            client_testing_dataloaders=loaders,
            client_testing_dataset_list=datasets,
            partition_fingerprint=PARTITION_FINGERPRINT,
            partition_metadata={},
        )

    @staticmethod
    def model_factory(params):
        del params
        return TinyTextClassifier()

    def run_once(self, params):
        with mock.patch(
            "experiment.Experiment_Create_dataset", side_effect=self.dataset_factory
        ), mock.patch(
            "experiment.Experiment_Create_dataloader", side_effect=self.dataloader_factory
        ), mock.patch(
            "experiment.Experiment_Create_model", side_effect=self.model_factory
        ):
            return _run_single_repeat(0, FedFACT, evaluate_fedfact, params)

    @staticmethod
    def checkpoint_params(params):
        return dict(
            params,
            repeat_idx=0,
            repeat_seed=2026,
            partition_fingerprint=PARTITION_FINGERPRINT,
            partition_metadata={},
        )

    def test_two_round_continuous_equals_round_one_checkpoint_plus_resume(self):
        continuous_params = self.params("continuous", resume=False)
        continuous_result = self.run_once(continuous_params)

        resumed_params = self.params("resumed", resume=True)
        real_save = save_checkpoint

        def save_then_crash(*args, **kwargs):
            path = real_save(*args, **kwargs)
            if int(args[1]) == 0:
                raise InjectedRoundBoundaryCrash("planned FedFACT boundary crash")
            return path

        with mock.patch(
            "algorithm.FedFACT.save_checkpoint", side_effect=save_then_crash
        ):
            with self.assertRaisesRegex(
                InjectedRoundBoundaryCrash, "planned FedFACT boundary crash"
            ):
                self.run_once(resumed_params)

        boundary = load_checkpoint(self.checkpoint_params(resumed_params))
        self.assertEqual(boundary.next_round, 1)
        self.assertEqual(boundary.phase, "train")
        self.assertEqual(boundary.client_selection_history, [[0, 1]])
        self.assertEqual(len(boundary.algorithm_state["personal_model_states"]), 2)
        self.assertIn("global_dual", boundary.algorithm_state)
        self.assertIn("local_duals", boundary.algorithm_state)
        self.assertIn("ensemble_weights", boundary.algorithm_state)
        self.assertIsNone(boundary.amp_scaler_state)

        resumed_result = self.run_once(resumed_params)
        continuous = load_checkpoint(self.checkpoint_params(continuous_params))
        resumed = load_checkpoint(self.checkpoint_params(resumed_params))

        self.assertEqual(continuous.phase, "evaluate")
        self.assertEqual(resumed.phase, "evaluate")
        assert_nested_equal(
            self, continuous.global_model_state, resumed.global_model_state,
            "global_model_state",
        )
        assert_nested_equal(
            self, continuous.algorithm_state, resumed.algorithm_state,
            "algorithm_state",
        )
        assert_nested_equal(
            self, continuous.amp_scaler_state, resumed.amp_scaler_state,
            "amp_scaler_state",
        )
        assert_nested_equal(
            self, continuous.rng_state, resumed.rng_state, "rng_state"
        )
        self.assertEqual(
            continuous.client_selection_history,
            resumed.client_selection_history,
        )
        self.assertEqual(
            continuous.client_selection_history, [[0, 1], [0, 1]]
        )
        self.assertEqual(
            continuous.total_communication_cost,
            resumed.total_communication_cost,
        )
        self.assertEqual(continuous.total_gpu_seconds, 0.0)
        self.assertEqual(resumed.total_gpu_seconds, 0.0)
        assert_nested_equal(
            self, continuous_result.metrics, resumed_result.metrics, "metrics"
        )

    def test_final_round_checkpoint_without_metrics_resumes_at_evaluation(self):
        params = self.params("evaluate_only", resume=True)
        first_result = self.run_once(params)
        state_params = self.checkpoint_params(params)
        checkpoint = load_checkpoint(state_params)
        self.assertEqual(checkpoint.phase, "evaluate")

        metrics_path = get_repeat_state_dir(state_params) / "metrics.json"
        metrics_path.unlink()

        forbidden = mock.Mock(side_effect=AssertionError("training was rerun"))
        forbidden.__name__ = "FedFACT"
        with mock.patch(
            "experiment.Experiment_Create_dataset", side_effect=self.dataset_factory
        ), mock.patch(
            "experiment.Experiment_Create_dataloader", side_effect=self.dataloader_factory
        ), mock.patch(
            "experiment.Experiment_Create_model", side_effect=self.model_factory
        ):
            recovered = _run_single_repeat(
                0, forbidden, evaluate_fedfact, params
            )
        forbidden.assert_not_called()
        assert_nested_equal(self, first_result.metrics, recovered.metrics, "metrics")


if __name__ == "__main__":
    unittest.main()
