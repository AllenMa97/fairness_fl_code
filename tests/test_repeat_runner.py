import unittest

import torch

from tool.experiment_state import (
    AlgorithmRunResult,
    RepeatResult,
    aggregate_repeat_results,
    normalize_algorithm_result,
)


class ExperimentStateTest(unittest.TestCase):
    def test_normalizes_legacy_three_tuple(self):
        model = torch.nn.Linear(1, 1)
        result = normalize_algorithm_result((model, 1.5, 2.5))
        self.assertIs(result.global_model, model)
        self.assertEqual(result.total_gpu_seconds, 1.5)
        self.assertEqual(result.total_communication_cost, 2.5)
        self.assertEqual(result.algorithm_state, {})
        self.assertIsNone(result.amp_scaler_state)

    def test_rejects_unstructured_return_values(self):
        with self.assertRaisesRegex(TypeError, "AlgorithmRunResult or a three-item tuple"):
            normalize_algorithm_result((torch.nn.Linear(1, 1), 1.0))

    def test_aggregate_requires_each_repeat_exactly_once(self):
        rows = [
            RepeatResult(
                index,
                42 + 1000 * index,
                f"fp-{index}",
                {"ACC": 0.5 + index * 0.1, "DEO": 0.2},
                1.0,
                2.0,
            )
            for index in range(3)
        ]
        aggregate = aggregate_repeat_results(rows, expected_repeats=3)
        self.assertEqual(aggregate["repeat_seeds"], [42, 1042, 2042])
        self.assertAlmostEqual(aggregate["metrics"]["ACC"]["mean"], 0.6)
        with self.assertRaisesRegex(ValueError, "repeat indices"):
            aggregate_repeat_results(rows[:2], expected_repeats=3)


if __name__ == "__main__":
    unittest.main()

# The experiment module has optional corpus/TensorBoard dependencies.  Keep their
# replacements scoped to individual test cases: unittest discovery imports all test
# modules before it executes them, so module-global sys.modules stubs leak into
# unrelated tests.
import importlib
import random
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock
import numpy as np


def _experiment_dependency_stubs():
    setup = types.ModuleType("module.experiment_setup")
    setup.Experiment_Create_dataset = lambda *args, **kwargs: None
    setup.Experiment_Create_dataloader = lambda *args, **kwargs: None
    setup.Experiment_Create_model = lambda *args, **kwargs: None
    tensorboard = types.ModuleType("tool.tensorboard_logger")
    for name in ("init_tensorboard_logger", "log_test_metrics", "log_system_metrics",
                 "flush", "close", "log_scalar", "log_metrics", "update_step",
                 "log_deep_metrics", "get_monitoring_config"):
        setattr(tensorboard, name, lambda *args, **kwargs: None)
    return {
        "mat73": types.ModuleType("mat73"),
        "module.experiment_setup": setup,
        "tool.tensorboard_logger": tensorboard,
    }


_MISSING_MODULE = object()
_EXPERIMENT_DEPENDENCY_PREVIOUS = None


def setUpModule():
    """Load experiment once with optional dependencies only for this test module."""
    global _EXPERIMENT_DEPENDENCY_PREVIOUS
    stubs = _experiment_dependency_stubs()
    _EXPERIMENT_DEPENDENCY_PREVIOUS = {
        name: sys.modules.get(name, _MISSING_MODULE) for name in stubs
    }
    # Do not use a long-lived patch.dict(sys.modules, ...): stopping it restores
    # the entire module dictionary and removes modules imported by PyTorch while
    # the patch was active.  Re-importing those extension modules can then try to
    # register the same TORCH_LIBRARY namespace twice.
    sys.modules.update(stubs)
    sys.modules.pop("experiment", None)
    importlib.import_module("experiment")


def tearDownModule():
    sys.modules.pop("experiment", None)
    if _EXPERIMENT_DEPENDENCY_PREVIOUS is not None:
        for name, previous in _EXPERIMENT_DEPENDENCY_PREVIOUS.items():
            if previous is _MISSING_MODULE:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


class _ExperimentImportTestCase(unittest.TestCase):
    """Marker base: optional imports are isolated by module setup/teardown."""


class _ToyBundle:
    def __init__(self, fingerprint):
        self.partition_fingerprint = fingerprint
        self.partition_metadata = {"source": "test"}
        self.training_dataloaders = []
        self.client_dataset_list = []
        self.testing_dataloader = []
        self.client_testing_dataloaders = []
        self.client_testing_dataset_list = []


class RepeatRunnerTest(_ExperimentImportTestCase):
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.params = {
            "model_path": str(root / "models"),
            "result_path": str(root / "results.txt"),
            "log_path": str(root / "run.log"),
            "dataset_name": "toy", "dataset": "toy", "task": "Tabular_CLF",
            "algorithm": "Toy", "hypothesis": "TinyANN", "model_type": "ANN",
            "split_strategy": "Uniform", "num_clients_K": 2,
            "batch_size": 4, "test_batch_size": 4, "communication_round_I": 2,
            "algorithm_epoch_T": 1, "FL_fraction": 1.0, "FL_drop_rate": 0.0,
            "learning_rate": 0.1, "optimize_method": "sgd", "device": "cpu",
            "use_amp": False, "base_seed": 42, "exp_repeat_times": 3,
            "parallel_repeats": 1, "resume": False,
            "final_artifact_policy": "full_state",
        }

    def tearDown(self):
        self.temp_dir.cleanup()
        super().tearDown()

    def _patch_factories(self, events=None):
        import experiment

        def create_dataset(params):
            if events is not None:
                events.append(("dataset", random.random(), np.random.rand(), torch.rand(1).item()))
            return object(), None, []

        def create_loaders(params, train, validation, test, split):
            del train, validation, test, split
            if events is not None:
                events.append(("loader", random.random(), np.random.rand(), torch.rand(1).item()))
            bundle = _ToyBundle("partition-%d" % params["repeat_seed"])
            params["partition_fingerprint"] = bundle.partition_fingerprint
            params["partition_metadata"] = bundle.partition_metadata
            return bundle

        def create_model(params):
            del params
            if events is not None:
                events.append(("model", random.random(), np.random.rand(), torch.rand(1).item()))
            return torch.nn.Linear(1, 1)

        return mock.patch.multiple(
            experiment,
            Experiment_Create_dataset=mock.Mock(side_effect=create_dataset),
            Experiment_Create_dataloader=mock.Mock(side_effect=create_loaders),
            Experiment_Create_model=mock.Mock(side_effect=create_model),
            calculate_communication_cost=mock.Mock(return_value=0.0),
        )

    @staticmethod
    def _evaluator(model, params, bundle, algorithm_state):
        del model, params, bundle, algorithm_state
        return {"ACC": 1.0, "DEO": 0.0, "SPD": 0.0}

    def test_repeat_seed_precedes_dataset_loader_and_model_construction(self):
        from experiment import _run_single_repeat
        events = []

        def algorithm(*args, **kwargs):
            return AlgorithmRunResult(args[1], 0.0, 0.0)

        with self._patch_factories(events):
            _run_single_repeat(0, algorithm, self._evaluator, self.params)

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        expected = (random.random(), np.random.rand(), torch.rand(1).item())
        self.assertEqual(events[0][0], "dataset")
        self.assertEqual(events[0][1:], expected)
        self.assertEqual([event[0] for event in events], ["dataset", "loader", "model"])

    def test_scheduler_uses_same_single_repeat_worker_for_serial(self):
        from experiment import Experiment_FL
        seen = []

        def worker(repeat_idx, algorithm, evaluator, params):
            del algorithm, evaluator
            seen.append((repeat_idx, params["experiment_config_hash"]))
            return RepeatResult(repeat_idx, 42 + 1000 * repeat_idx, "p-%d" % repeat_idx,
                                {"ACC": 1.0}, 0.0, 0.0)

        with mock.patch("experiment._run_single_repeat", side_effect=worker), \
             mock.patch("experiment.save_aggregate_metrics"):
            aggregate = Experiment_FL(lambda: None, self.params)
        self.assertEqual([index for index, _ in seen], [0, 1, 2])
        self.assertEqual(aggregate["repeat_seeds"], [42, 1042, 2042])

    def test_fl_dispatch_defers_data_and_model_construction_to_repeat_worker(self):
        from experiment import Experiment, Fed_AVG
        params = dict(self.params, algorithm="FedAvg")
        with mock.patch("experiment.Experiment_Create_dataset") as dataset_factory, \
             mock.patch("experiment.Experiment_Create_dataloader") as loader_factory, \
             mock.patch("experiment.Experiment_Create_model") as model_factory, \
             mock.patch("experiment.Experiment_FL", return_value={}) as repeat_runner, \
             mock.patch("experiment.init_tensorboard_logger", return_value=None):
            Experiment(params)
        dataset_factory.assert_not_called()
        loader_factory.assert_not_called()
        model_factory.assert_not_called()
        repeat_runner.assert_called_once_with(Fed_AVG, params)

    def test_parallel_scheduler_maps_the_same_single_repeat_worker(self):
        from experiment import Experiment_FL
        params = dict(self.params, parallel_repeats=2)
        captured = {}

        class Pool:
            def __enter__(self):
                return self

            def __exit__(self, *unused):
                return False

            def starmap(self, function, arguments):
                captured["function"] = function
                captured["arguments"] = arguments
                return [function(*argument) for argument in arguments]

        class Context:
            def Pool(self, processes):
                captured["processes"] = processes
                return Pool()

        def worker(repeat_idx, algorithm, evaluator, run_params):
            del algorithm, evaluator, run_params
            return RepeatResult(repeat_idx, 42 + 1000 * repeat_idx, f"p-{repeat_idx}",
                                {"ACC": 1.0}, 0.0, 0.0)

        with mock.patch("experiment._run_single_repeat", side_effect=worker) as worker_mock, \
             mock.patch("experiment.mp.get_context", return_value=Context()), \
             mock.patch("experiment.save_aggregate_metrics"):
            Experiment_FL(lambda: None, params)
        self.assertIs(captured["function"], worker_mock)
        self.assertEqual(captured["processes"], 2)
        self.assertEqual([args[0] for args in captured["arguments"]], [0, 1, 2])

    def test_cuda_parallel_repeats_are_rejected(self):
        from experiment import Experiment_FL
        with self.assertRaisesRegex(ValueError, "CUDA repeats must run serially"):
            Experiment_FL(lambda: None, dict(self.params, device="cuda", parallel_repeats=2))

    def test_evaluation_phase_checkpoint_skips_algorithm_but_finishes_metrics(self):
        from experiment import _run_single_repeat
        from tool.checkpoint import build_experiment_config_hash, save_checkpoint
        calls = {"algorithm": 0, "evaluator": 0}
        params = dict(self.params, resume=True, exp_repeat_times=1)
        state_params = dict(params, repeat_idx=0, repeat_seed=42,
                            partition_fingerprint="partition-42")
        state_params["experiment_config_hash"] = build_experiment_config_hash(params)
        save_checkpoint(state_params, 1, torch.nn.Linear(1, 1),
                        algorithm_state={"value": 7})

        def algorithm(*args, **kwargs):
            calls["algorithm"] += 1
            return AlgorithmRunResult(args[1], 0.0, 0.0)

        def evaluator(model, repeat_params, bundle, algorithm_state):
            del model, repeat_params, bundle
            calls["evaluator"] += 1
            return {"ACC": float(algorithm_state["value"])}

        with self._patch_factories():
            result = _run_single_repeat(0, algorithm, evaluator, params)
        self.assertEqual(calls, {"algorithm": 0, "evaluator": 1})
        self.assertEqual(result.metrics["ACC"], 7.0)

class _TinyFedDataset(torch.utils.data.Dataset):
    def __init__(self, values):
        self.X = torch.as_tensor(values, dtype=torch.float32).reshape(-1, 1)
        self.labels = np.asarray([index % 2 for index in range(len(values))])
        self.protected = np.asarray([(index // 2) % 2 for index in range(len(values))])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "X": self.X[index],
            "labels": torch.tensor(self.labels[index]),
            "protected": torch.tensor(self.protected[index]),
        }


class TinyANN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1, bias=False)

    def forward(self, values):
        logits = self.linear(values.float())
        return torch.sigmoid(logits), values.float()


class FedAvgResumeTest(_ExperimentImportTestCase):
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()
        super().tearDown()

    def _params(self, name, resume):
        model_path = self.root / name / "models"
        for index in range(2):
            (model_path / f"client_{index + 1}").mkdir(parents=True, exist_ok=True)
        return {
            "model_path": str(model_path), "result_path": str(self.root / name / "results.txt"),
            "log_path": str(self.root / name / "run.log"), "dataset_name": "tiny",
            "dataset": "tiny", "task": "Tabular_CLF", "algorithm": "FedAvg",
            "hypothesis": "Tiny", "model_type": "ANN", "split_strategy": "Uniform",
            "num_clients_K": 2, "batch_size": 4, "test_batch_size": 4,
            "communication_round_I": 2, "algorithm_epoch_T": 1, "FL_fraction": 1.0,
            "FL_drop_rate": 0.0, "learning_rate": 0.05, "optimize_method": "sgd",
            "device": "cpu", "use_amp": False, "base_seed": 42, "resume": resume,
            "client_parallel": False, "checkpoint_save_freq": 1,
            "final_artifact_policy": "full_state", "tb_monitor": {"test": False, "gradient": False},
        }

    @staticmethod
    def _evaluator(model, params, bundle, state):
        del params, bundle, state
        return {"ACC": float(sum(value.sum() for value in model.state_dict().values()))}

    def _run(self, params):
        from algorithm.FederatedAverage import Fed_AVG
        from experiment import _run_single_repeat
        train = _TinyFedDataset(range(16))
        test = _TinyFedDataset(range(8))
        loaders = [torch.utils.data.DataLoader(
            torch.utils.data.Subset(train, list(range(i * 8, (i + 1) * 8))),
            batch_size=4, shuffle=True,
            generator=torch.Generator().manual_seed(100 + i),
        ) for i in range(2)]
        bundle = _ToyBundle("tiny-partition")
        bundle.training_dataloaders = loaders
        bundle.client_dataset_list = [loader.dataset for loader in loaders]
        bundle.testing_dataloader = torch.utils.data.DataLoader(test, batch_size=4, shuffle=False)

        with mock.patch("experiment.Experiment_Create_dataset", return_value=(train, None, test)), \
             mock.patch("experiment.Experiment_Create_dataloader", return_value=bundle), \
             mock.patch("experiment.Experiment_Create_model", side_effect=lambda p: TinyANN()), \
             mock.patch("experiment.calculate_communication_cost", return_value=0.0), \
             mock.patch("algorithm.FederatedAverage.log_deep_metrics"), \
             mock.patch("algorithm.FederatedAverage.log_system_metrics"), \
             mock.patch("algorithm.FederatedAverage.log_test_metrics"), \
             mock.patch("algorithm.FederatedAverage.flush"):
            params = dict(params, experiment_config_hash=__import__("tool.checkpoint", fromlist=["build_experiment_config_hash"]).build_experiment_config_hash(params))
            return _run_single_repeat(0, Fed_AVG, self._evaluator, params)

    def _checkpoint_state(self, params):
        from tool.checkpoint import build_experiment_config_hash, load_checkpoint
        return load_checkpoint(dict(
            params,
            repeat_idx=0,
            repeat_seed=42,
            partition_fingerprint="tiny-partition",
            experiment_config_hash=build_experiment_config_hash(params),
        ))

    def test_two_cpu_rounds_match_one_round_then_resume(self):
        continuous_params = self._params("continuous", resume=False)
        continuous = self._run(continuous_params)
        resumed_params = self._params("resumed", resume=True)
        from tool.checkpoint import save_checkpoint

        def crash_after_first_round(*args, **kwargs):
            path = save_checkpoint(*args, **kwargs)
            iter_t = kwargs["iter_t"] if "iter_t" in kwargs else args[1]
            if iter_t == 0:
                raise RuntimeError("planned crash")
            return path

        with mock.patch("algorithm.FederatedAverage.save_checkpoint", side_effect=crash_after_first_round):
            with self.assertRaisesRegex(RuntimeError, "planned crash"):
                self._run(resumed_params)
        resumed = self._run(resumed_params)
        self.assertEqual(continuous.metrics, resumed.metrics)
        continuous_state = self._checkpoint_state(continuous_params)
        resumed_state = self._checkpoint_state(resumed_params)
        self.assertEqual(continuous_state.client_selection_history,
                         resumed_state.client_selection_history)
        # Runtime measurements are intentionally nondeterministic.  Exact resume
        # applies to the model trajectory and RNG/loader state, not wall-clock
        # observations collected on two separate executions.
        continuous_algorithm_state = dict(continuous_state.algorithm_state)
        resumed_algorithm_state = dict(resumed_state.algorithm_state)
        continuous_timings = continuous_algorithm_state.pop("users_gpu_seconds_list")
        resumed_timings = resumed_algorithm_state.pop("users_gpu_seconds_list")
        self.assertEqual(len(continuous_timings), len(resumed_timings))
        self.assertTrue(all(value >= 0 for value in continuous_timings + resumed_timings))
        _assert_nested_state_equal(
            self, continuous_algorithm_state, resumed_algorithm_state
        )
        self.assertEqual(set(continuous_state.global_model_state),
                         set(resumed_state.global_model_state))
        for name in continuous_state.global_model_state:
            torch.testing.assert_close(
                continuous_state.global_model_state[name],
                resumed_state.global_model_state[name], rtol=0, atol=0,
            )

    def test_fedavg_counts_only_each_round_client_gpu_time_and_defers_terminal_checkpoint(self):
        from algorithm.FederatedAverage import Fed_AVG
        from experiment import _run_single_repeat

        train = _TinyFedDataset(range(16))
        test = _TinyFedDataset(range(8))
        loaders = [torch.utils.data.DataLoader(
            torch.utils.data.Subset(train, list(range(index * 8, (index + 1) * 8))),
            batch_size=4, shuffle=True,
            generator=torch.Generator().manual_seed(100 + index),
        ) for index in range(2)]
        bundle = _ToyBundle("tiny-partition")
        bundle.training_dataloaders = loaders
        bundle.client_dataset_list = [loader.dataset for loader in loaders]
        bundle.testing_dataloader = torch.utils.data.DataLoader(test, batch_size=4)
        params = self._params("counter", resume=False)
        params["experiment_config_hash"] = __import__("tool.checkpoint", fromlist=["build_experiment_config_hash"]).build_experiment_config_hash(params)
        algorithm_rounds, runner_rounds = [], []

        def deterministic_client(client_id, device, model, *args, **kwargs):
            del client_id, device, args, kwargs
            return {
                "gpu_seconds": 1.0,
                "state_dict": {name: value.detach().cpu().clone()
                               for name, value in model.state_dict().items()},
            }

        from tool.checkpoint import save_checkpoint as real_save_checkpoint

        def algorithm_checkpoint(*args, **kwargs):
            algorithm_rounds.append(kwargs["iter_t"])
            return real_save_checkpoint(*args, **kwargs)

        def runner_checkpoint(*args, **kwargs):
            runner_rounds.append(args[1])
            return real_save_checkpoint(*args, **kwargs)

        with mock.patch("experiment.Experiment_Create_dataset", return_value=(train, None, test)), \
             mock.patch("experiment.Experiment_Create_dataloader", return_value=bundle), \
             mock.patch("experiment.Experiment_Create_model", side_effect=lambda unused: TinyANN()), \
             mock.patch("algorithm.FederatedAverage._train_single_client_fedavg", side_effect=deterministic_client), \
             mock.patch("algorithm.FederatedAverage.save_checkpoint", side_effect=algorithm_checkpoint), \
             mock.patch("experiment.save_checkpoint", side_effect=runner_checkpoint), \
             mock.patch("algorithm.FederatedAverage.log_deep_metrics"), \
             mock.patch("algorithm.FederatedAverage.log_test_metrics"), \
             mock.patch("algorithm.FederatedAverage.log_system_metrics"), \
             mock.patch("algorithm.FederatedAverage.flush"), \
             mock.patch("experiment.log_test_metrics"), \
             mock.patch("experiment.flush"):
            result = _run_single_repeat(0, Fed_AVG, self._evaluator, params)
        self.assertEqual(result.total_gpu_seconds, 4.0)
        self.assertEqual(algorithm_rounds, [0])
        self.assertEqual(runner_rounds, [1])

# Review regressions for repeat completion, accounting, and resource artifacts.
# These are kept at module level so their expected pre-fix failures are explicit.

def _assert_nested_state_equal(testcase, actual, expected):
    if torch.is_tensor(actual):
        testcase.assertTrue(torch.is_tensor(expected))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        return
    if isinstance(actual, dict):
        testcase.assertIsInstance(expected, dict)
        testcase.assertEqual(set(actual), set(expected))
        for key in actual:
            _assert_nested_state_equal(testcase, actual[key], expected[key])
        return
    if isinstance(actual, (list, tuple)):
        testcase.assertIsInstance(expected, type(actual))
        testcase.assertEqual(len(actual), len(expected))
        for left, right in zip(actual, expected):
            _assert_nested_state_equal(testcase, left, right)
        return
    testcase.assertEqual(actual, expected)


class RepeatCompletionAndResourcesTest(_ExperimentImportTestCase):
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.params = {
            "model_path": str(root / "models"), "result_path": str(root / "result.txt"),
            "log_path": str(root / "run.log"), "dataset_name": "toy", "dataset": "toy",
            "task": "Tabular_CLF", "algorithm": "Toy", "hypothesis": "Tiny",
            "model_type": "ANN", "split_strategy": "Uniform", "num_clients_K": 2,
            "batch_size": 4, "test_batch_size": 4, "communication_round_I": 2,
            "algorithm_epoch_T": 1, "FL_fraction": 1.0, "FL_drop_rate": 0.0,
            "learning_rate": 0.1, "optimize_method": "sgd", "device": "cpu",
            "use_amp": False, "base_seed": 42, "exp_repeat_times": 3,
            "parallel_repeats": 1, "resume": False, "final_artifact_policy": "full_state",
        }

    def tearDown(self):
        self.temp_dir.cleanup()
        super().tearDown()

    def _factories(self, partition_for_repeat=None):
        import experiment

        def loaders(params, train, validation, test, split):
            del train, validation, test, split
            fingerprint = (partition_for_repeat or (lambda idx: f"partition-{params['repeat_seed']}"))(params["repeat_idx"])
            bundle = _ToyBundle(fingerprint)
            params["partition_fingerprint"] = fingerprint
            params["partition_metadata"] = {}
            return bundle

        return mock.patch.multiple(
            experiment,
            Experiment_Create_dataset=mock.Mock(return_value=(object(), None, [])),
            Experiment_Create_dataloader=mock.Mock(side_effect=loaders),
            Experiment_Create_model=mock.Mock(side_effect=lambda params: torch.nn.Linear(1, 1)),
            calculate_communication_cost=mock.Mock(return_value=0.0),
        )

    def test_final_evaluation_logs_terminal_round_once_and_flushes(self):
        from experiment import _run_single_repeat
        events = []

        def algorithm(*args, **kwargs):
            return AlgorithmRunResult(args[1], 3.0, 4.0)

        def evaluator(*args):
            events.append("evaluate")
            return {"ACC": 0.8, "DEO": 0.2, "SPD": -0.1, "FR": 0.8, "HM": 0.8}

        def log_metrics(**kwargs):
            self.assertEqual(events, ["evaluate"])
            events.append("log")
            self.assertEqual(kwargs["step"], 2)
            self.assertEqual(kwargs["communication_cost"], 4.0)
            self.assertEqual(kwargs["accuracy"], 0.8)

        with self._factories(), \
             mock.patch("experiment.log_test_metrics", side_effect=log_metrics) as log_mock, \
             mock.patch("experiment.flush", side_effect=lambda: events.append("flush")) as flush_mock:
            _run_single_repeat(0, algorithm, evaluator, dict(self.params, exp_repeat_times=1))
        log_mock.assert_called_once()
        flush_mock.assert_called_once()
        self.assertEqual(events, ["evaluate", "log", "flush"])

    def test_mixed_completed_resumed_and_fresh_repeats_all_enter_aggregate(self):
        from experiment import Experiment_FL
        from tool.checkpoint import build_experiment_config_hash, save_checkpoint, save_repeat_metrics
        params = dict(self.params, resume=True)
        config_hash = build_experiment_config_hash(params)
        calls = []
        for repeat_idx in (0, 1):
            state_params = dict(
                params, experiment_config_hash=config_hash, repeat_idx=repeat_idx,
                repeat_seed=42 + 1000 * repeat_idx,
                partition_fingerprint=f"partition-{repeat_idx}",
            )
            if repeat_idx == 0:
                save_repeat_metrics(
                    state_params, repeat_idx, config_hash, f"partition-{repeat_idx}",
                    {"ACC": 0.1}, repeat_seed=42, total_gpu_seconds=0.0,
                    total_communication_cost=0.0,
                    resource_usage={"peak_cuda_bytes": 0, "peak_rss_bytes": 1, "checkpoint_bytes": 1},
                )
            else:
                save_checkpoint(
                    state_params, 0, torch.nn.Linear(1, 1), algorithm_state={"step": 1},
                    total_gpu_seconds=0.0, total_runtime_seconds=0.0,
                    total_communication_cost=0.0, client_selection_history=[[0]],
                )

        def algorithm(*args, start_round=0, resume_state=None, **kwargs):
            del kwargs
            repeat_idx = args[10]["repeat_idx"]
            calls.append((repeat_idx, start_round, resume_state is not None))
            return AlgorithmRunResult(
                args[1], 0.0, 0.0,
                {} if resume_state is None else resume_state.algorithm_state,
                None, [] if resume_state is None else resume_state.client_selection_history,
            )

        with self._factories(lambda idx: f"partition-{idx}"):
            aggregate = Experiment_FL(algorithm, params, evaluator_function=lambda *args: {"ACC": 0.2})
        self.assertEqual(aggregate["repeat_indices"], [0, 1, 2])
        self.assertEqual(aggregate["repeat_seeds"], [42, 1042, 2042])
        self.assertEqual(calls, [(1, 1, True), (2, 0, False)])
        self.assertEqual(aggregate["resource_usage"][0]["checkpoint_bytes"], 1)


class ResourceSnapshotTest(unittest.TestCase):
    def test_resource_snapshot_records_peak_cuda_rss_and_checkpoint_bytes(self):
        from tool.experiment_state import capture_resource_snapshot
        with tempfile.NamedTemporaryFile() as stream:
            stream.write(b"checkpoint")
            stream.flush()
            snapshot = capture_resource_snapshot(Path(stream.name))
        self.assertEqual(snapshot["checkpoint_bytes"], 10)
        self.assertGreater(snapshot["peak_rss_bytes"], 0)
        self.assertGreaterEqual(snapshot["peak_cuda_bytes"], 0)
