import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from tool.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointCompatibilityError,
    CheckpointState,
    build_experiment_config_hash,
    clear_repeat_artifacts,
    finalize_repeat_artifacts,
    get_repeat_state_dir,
    load_checkpoint,
    load_repeat_metrics,
    restore_rng_state,
    save_aggregate_metrics,
    save_checkpoint,
    save_repeat_metrics,
)
from tool import checkpoint


class FakeScaler:
    def __init__(self, scale):
        self.scale = scale

    def state_dict(self):
        return {"scale": self.scale}


class CheckpointTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.params = {
            "model_path": self.temp_dir.name,
            "result_path": str(Path(self.temp_dir.name) / "result.txt"),
            "log_path": str(Path(self.temp_dir.name) / "run.log"),
            "dataset_name": "toy",
            "task": "Tabular_CLF",
            "algorithm": "FedAvg",
            "hypothesis": "Tiny",
            "split_strategy": "Dirichlet05",
            "num_clients_K": 2,
            "communication_round_I": 2,
            "algorithm_epoch_T": 1,
            "batch_size": 4,
            "learning_rate": 0.1,
            "base_seed": 42,
            "repeat_idx": 0,
            "repeat_seed": 42,
            "partition_fingerprint": "partition-a",
            "resume": True,
            "parallel_repeats": 1,
            "checkpoint_keep_latest": 9,
        }
        self.params["experiment_config_hash"] = build_experiment_config_hash(self.params)
        self.model = torch.nn.Linear(2, 1)

    def tearDown(self):
        self.temp_dir.cleanup()

    def save(self, round_index=0):
        return save_checkpoint(
            self.params,
            round_index,
            self.model,
            algorithm_state={"dual": torch.tensor([1.0])},
            amp_scaler=FakeScaler(128.0),
            total_gpu_seconds=2.0,
            total_runtime_seconds=3.0,
            total_communication_cost=4.0,
            client_selection_history=[[1]],
        )

    def test_config_hash_ignores_paths_and_resume_controls(self):
        changed = dict(
            self.params,
            model_path="elsewhere",
            result_path="elsewhere.txt",
            resume=False,
            checkpoint_keep_latest=1,
            parallel_repeats=3,
        )
        self.assertEqual(
            build_experiment_config_hash(self.params),
            build_experiment_config_hash(changed),
        )
        changed["learning_rate"] = 0.2
        self.assertNotEqual(
            build_experiment_config_hash(self.params),
            build_experiment_config_hash(changed),
        )

    def test_checkpoint_contains_complete_round_boundary_state(self):
        path = self.save(0)
        state = load_checkpoint(
            self.params,
            expected_config_hash=self.params["experiment_config_hash"],
            expected_partition_fingerprint="partition-a",
            expected_repeat_idx=0,
        )
        self.assertIsInstance(state, CheckpointState)
        self.assertEqual(state.schema_version, CHECKPOINT_SCHEMA_VERSION)
        self.assertEqual(state.next_round, 1)
        self.assertEqual(state.phase, "train")
        self.assertEqual(state.algorithm_state["dual"].item(), 1.0)
        self.assertEqual(state.amp_scaler_state, {"scale": 128.0})
        self.assertEqual(state.total_gpu_seconds, 2.0)
        self.assertEqual(state.total_runtime_seconds, 3.0)
        self.assertEqual(state.total_communication_cost, 4.0)
        self.assertEqual(state.client_selection_history, [[1]])
        self.assertTrue(Path(path).name == "checkpoint_latest.pt")

    def test_last_round_is_evaluation_phase(self):
        self.save(1)
        state = load_checkpoint(self.params)
        self.assertEqual(state.next_round, 2)
        self.assertEqual(state.phase, "evaluate")

    def test_only_latest_checkpoint_is_retained(self):
        self.save(0)
        self.save(1)
        repeat_dir = Path(self.temp_dir.name) / "experiment_state" / self.params[
            "experiment_config_hash"
        ] / "repeat_00"
        self.assertEqual(
            [path.name for path in repeat_dir.glob("checkpoint*.pt")],
            ["checkpoint_latest.pt"],
        )

    def test_mismatch_raises_before_rng_is_mutated(self):
        self.save(0)
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        expected = (random.random(), np.random.rand(), torch.rand(1).item())
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        with self.assertRaisesRegex(CheckpointCompatibilityError, "partition fingerprint"):
            load_checkpoint(self.params, expected_partition_fingerprint="wrong")
        observed = (random.random(), np.random.rand(), torch.rand(1).item())
        self.assertEqual(expected, observed)

    def test_rng_restore_round_trips_python_numpy_and_torch(self):
        random.seed(77)
        np.random.seed(77)
        torch.manual_seed(77)
        self.save(0)
        expected = (random.random(), np.random.rand(), torch.rand(2))
        state = load_checkpoint(self.params)
        restore_rng_state(state)
        observed = (random.random(), np.random.rand(), torch.rand(2))
        self.assertEqual(expected[0], observed[0])
        self.assertEqual(expected[1], observed[1])
        torch.testing.assert_close(expected[2], observed[2], rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA RNG requires CUDA")
    def test_cuda_rng_state_is_captured_for_every_visible_device(self):
        torch.cuda.manual_seed_all(91)
        self.save(0)
        state = load_checkpoint(self.params)
        self.assertEqual(len(state.rng_state["torch_cuda"]), torch.cuda.device_count())

    def test_v2_save_requires_explicit_zero_based_repeat_index(self):
        params = dict(self.params)
        params.pop("repeat_idx")
        params["Experiment_NO"] = 1
        with self.assertRaisesRegex(CheckpointCompatibilityError, "repeat_idx"):
            save_checkpoint(params, 0, self.model)

        params["repeat_idx"] = -1
        with self.assertRaisesRegex(CheckpointCompatibilityError, "repeat_idx"):
            save_checkpoint(params, 0, self.model)

    def test_v2_save_requires_nonempty_partition_fingerprint(self):
        params = dict(self.params)
        params["partition_fingerprint"] = "  "
        with self.assertRaisesRegex(CheckpointCompatibilityError, "partition_fingerprint"):
            save_checkpoint(params, 0, self.model)

        params.pop("partition_fingerprint")
        with self.assertRaisesRegex(CheckpointCompatibilityError, "partition_fingerprint"):
            save_checkpoint(params, 0, self.model)

    def test_checkpoint_state_preserves_dict_style_aliases(self):
        self.save(0)
        state = load_checkpoint(self.params)

        self.assertEqual(state["communication_round"], 0)
        self.assertIs(state["global_model_state"], state.global_model_state)
        self.assertIs(state["extra_state"], state.algorithm_state)
        self.assertIn("extra_state", state)
        self.assertEqual(state.get("unknown", "default"), "default")
        self.assertNotIn("_ALIASES", dict(state))
        with self.assertRaises(KeyError):
            state["unknown"]

    def test_load_rejects_malformed_rng_state_with_compatibility_error(self):
        path = self.save(0)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["rng_state"] = {"python": random.getstate()}
        torch.save(payload, path)

        with self.assertRaisesRegex(CheckpointCompatibilityError, "rng_state"):
            load_checkpoint(self.params)

    def test_load_rejects_cuda_rng_state_count_mismatch(self):
        path = self.save(0)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["rng_state"]["cuda_device_count"] = 0
        payload["rng_state"]["torch_cuda"] = [torch.get_rng_state()]
        torch.save(payload, path)

        with self.assertRaisesRegex(CheckpointCompatibilityError, "CUDA RNG"):
            load_checkpoint(self.params)

    def test_load_rejects_cuda_rng_state_with_invalid_tensor_type(self):
        path = self.save(0)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["rng_state"]["cuda_device_count"] = 1
        payload["rng_state"]["torch_cuda"] = [torch.tensor([1.0])]
        torch.save(payload, path)

        with mock.patch.object(checkpoint, "_current_cuda_device_count", return_value=1), self.assertRaisesRegex(
            CheckpointCompatibilityError, "torch_cuda"
        ):
            load_checkpoint(self.params)

    def test_load_wraps_invalid_checkpoint_field_types_as_compatibility_error(self):
        path = self.save(0)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["phase"] = []
        torch.save(payload, path)

        with self.assertRaises(CheckpointCompatibilityError):
            load_checkpoint(self.params)

    def test_restore_rejects_invalid_rng_state_with_compatibility_error(self):
        self.save(0)
        state = load_checkpoint(self.params)
        invalid_state = CheckpointState(
            **{**state.__dict__, "rng_state": {"python": "not-a-state"}}
        )

        with self.assertRaisesRegex(CheckpointCompatibilityError, "rng_state"):
            restore_rng_state(invalid_state)

    def test_atomic_writers_fsync_parent_directory_after_replace(self):
        calls = []
        original_replace = checkpoint.os.replace

        def replace_then_record(source, destination):
            original_replace(source, destination)
            calls.append("replace")

        with mock.patch.object(checkpoint.os, "replace", side_effect=replace_then_record), mock.patch.object(
            checkpoint, "_fsync_parent_directory", side_effect=lambda path: calls.append("fsync")
        ):
            checkpoint._atomic_json_save({"ok": True}, Path(self.temp_dir.name) / "atomic.json")

        self.assertEqual(calls, ["replace", "fsync"])


class RepeatMetricsTest(CheckpointTest):
    def test_v2_metrics_save_requires_explicit_partition_fingerprint(self):
        with self.assertRaisesRegex(CheckpointCompatibilityError, "partition_fingerprint"):
            save_repeat_metrics(
                self.params,
                0,
                self.params["experiment_config_hash"],
                "",
                {"ACC": 0.75},
                repeat_seed=42,
                total_gpu_seconds=0.0,
                total_communication_cost=0.0,
            )

    def test_checkpoint_alone_is_not_completion(self):
        self.save(1)
        self.assertIsNone(
            load_repeat_metrics(
                self.params, 0, self.params["experiment_config_hash"], "partition-a"
            )
        )

    def test_metrics_become_completion_only_after_atomic_write(self):
        path = save_repeat_metrics(
            self.params,
            0,
            self.params["experiment_config_hash"],
            "partition-a",
            {"ACC": 0.75, "DEO": 0.2, "SPD": 0.1},
            repeat_seed=42,
            total_gpu_seconds=2.0,
            total_communication_cost=4.0,
        )
        loaded = load_repeat_metrics(
            self.params, 0, self.params["experiment_config_hash"], "partition-a"
        )
        self.assertEqual(loaded["metrics"]["ACC"], 0.75)
        self.assertEqual(Path(path).name, "metrics.json")
        self.assertFalse(any(Path(path).parent.glob(".metrics.json.*")))

    def test_metrics_mismatch_fails_closed(self):
        save_repeat_metrics(
            self.params,
            0,
            self.params["experiment_config_hash"],
            "partition-a",
            {"ACC": 0.75},
            repeat_seed=42,
            total_gpu_seconds=0.0,
            total_communication_cost=0.0,
        )
        with self.assertRaisesRegex(CheckpointCompatibilityError, "partition fingerprint"):
            load_repeat_metrics(
                self.params, 0, self.params["experiment_config_hash"], "partition-b"
            )

    def test_metrics_only_policy_removes_completed_resume_state(self):
        self.save(1)
        checkpoint_path = get_repeat_state_dir(self.params) / "checkpoint_latest.pt"
        finalize_repeat_artifacts(self.params, 0, self.model, policy="metrics_only")
        self.assertFalse(checkpoint_path.exists())

    def test_fresh_run_clears_stale_repeat_state_and_metrics(self):
        self.save(0)
        save_repeat_metrics(
            self.params,
            0,
            self.params["experiment_config_hash"],
            "partition-a",
            {"ACC": 0.5},
            repeat_seed=42,
            total_gpu_seconds=0.0,
            total_communication_cost=0.0,
        )
        clear_repeat_artifacts(self.params, 0)
        repeat_dir = get_repeat_state_dir(self.params, 0)
        self.assertFalse((repeat_dir / "checkpoint_latest.pt").exists())
        self.assertFalse((repeat_dir / "metrics.json").exists())


if __name__ == "__main__":
    unittest.main()

class ResourceUsageMetricsTest(CheckpointTest):
    def test_resource_usage_round_trips_and_is_required_to_be_nonnegative_integers(self):
        save_repeat_metrics(
            self.params, 0, self.params["experiment_config_hash"], "partition-a",
            {"ACC": 0.5}, repeat_seed=42, total_gpu_seconds=0.0,
            total_communication_cost=0.0,
            resource_usage={"peak_cuda_bytes": 0, "peak_rss_bytes": 7, "checkpoint_bytes": 9},
        )
        loaded = load_repeat_metrics(
            self.params, 0, self.params["experiment_config_hash"], "partition-a"
        )
        self.assertEqual(loaded["resource_usage"], {
            "peak_cuda_bytes": 0, "peak_rss_bytes": 7, "checkpoint_bytes": 9,
        })
