import json
import os
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from module.partition import (
    DatasetView,
    PartitionDataError,
    dataset_fingerprint,
    extract_dataset_view,
)


class ToyDataset(Dataset):
    def __init__(self, sample_ids, labels, protected):
        self.sample_ids = list(sample_ids)
        self.labels = np.asarray(labels)
        self.protected = np.asarray(protected)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "x": torch.tensor(float(index)),
            "labels": torch.tensor(self.labels[index]),
            "protected": torch.tensor(self.protected[index]),
        }


class DatasetViewTest(unittest.TestCase):
    def test_extracts_subset_in_subset_order(self):
        source = ToyDataset(["a", "b", "c", "d"], [0, 1, 0, 1], [1, 0, 1, 0])
        view = extract_dataset_view(Subset(source, [3, 1, 2]))
        self.assertEqual(view.sample_ids, ("d", "b", "c"))
        np.testing.assert_array_equal(view.labels, np.array([1, 1, 0]))
        np.testing.assert_array_equal(view.protected, np.array([0, 0, 1]))

    def test_legacy_y_and_s1_are_supported(self):
        dataset = ToyDataset(["a", "b"], [0, 1], [1, 0])
        del dataset.labels
        dataset.y = np.array([0, 1])
        del dataset.protected
        dataset.s1 = np.array([1, 0])
        view = extract_dataset_view(dataset)
        np.testing.assert_array_equal(view.labels, np.array([0, 1]))
        np.testing.assert_array_equal(view.protected, np.array([1, 0]))

    def test_rejects_length_mismatch_and_non_scalar_labels(self):
        bad_length = ToyDataset(["a"], [0, 1], [0, 1])
        with self.assertRaisesRegex(PartitionDataError, "sample identity length"):
            extract_dataset_view(bad_length)
        bad_labels = ToyDataset(["a", "b"], [[0], [1]], [0, 1])
        with self.assertRaisesRegex(PartitionDataError, "one-dimensional"):
            extract_dataset_view(bad_labels)

    def test_order_and_label_changes_change_fingerprint(self):
        first = ToyDataset(["a", "b", "c"], [0, 1, 0], [0, 1, 1])
        reordered = ToyDataset(["b", "a", "c"], [1, 0, 0], [1, 0, 1])
        relabeled = ToyDataset(["a", "b", "c"], [1, 0, 0], [0, 1, 1])
        fp = dataset_fingerprint(first, dataset_name="toy", split="train", system_data_count=3)
        self.assertNotEqual(fp["sample_order_sha256"], dataset_fingerprint(
            reordered, dataset_name="toy", split="train", system_data_count=3
        )["sample_order_sha256"])
        self.assertNotEqual(fp["ordered_labels_sha256"], dataset_fingerprint(
            relabeled, dataset_name="toy", split="train", system_data_count=3
        )["ordered_labels_sha256"])

from module.partition import (
    PartitionSpec,
    build_label_dirichlet_partition,
    validate_indices,
)


class ScriptedGenerator:
    def __init__(self, profiles):
        self.profiles = iter(profiles)
        self.dirichlet_calls = 0

    def dirichlet(self, concentration):
        self.dirichlet_calls += 1
        return np.asarray(next(self.profiles), dtype=np.float64)

    def shuffle(self, values):
        values[:] = values[::-1]


class LabelDirichletTest(unittest.TestCase):
    def test_samples_one_profile_per_label_and_reuses_it_for_test(self):
        train_labels = np.array([0] * 6 + [1] * 6)
        test_labels = np.array([0] * 3 + [1] * 3)
        rng = ScriptedGenerator([[0.5, 0.3, 0.2], [0.1, 0.2, 0.7]])
        spec = PartitionSpec(
            strategy="Dirichlet05", alpha=0.5, num_clients=3, seed=7,
            min_samples_per_client=1, max_retries=1,
            repair_policy="minimum_move_v1",
        )
        result = build_label_dirichlet_partition(
            train_labels, test_labels, spec, rng=rng
        )
        self.assertEqual(rng.dirichlet_calls, 2)
        np.testing.assert_allclose(
            result.class_client_profile,
            np.array([[0.5, 0.3, 0.2], [0.1, 0.2, 0.7]]),
        )
        self.assertEqual(sum(map(len, result.train_indices.values())), 12)
        self.assertEqual(sum(map(len, result.test_indices.values())), 6)

    def test_same_seed_is_independent_of_global_rng_state(self):
        labels = np.repeat(np.arange(3), 30)
        spec = PartitionSpec("Dirichlet1", 1.0, 5, 42, 1, 20, "minimum_move_v1")
        random.seed(1)
        np.random.seed(1)
        first = build_label_dirichlet_partition(labels, labels, spec)
        random.seed(999)
        np.random.seed(999)
        second = build_label_dirichlet_partition(labels, labels, spec)
        for client_id in range(spec.num_clients):
            np.testing.assert_array_equal(
                first.train_indices[client_id], second.train_indices[client_id]
            )

    def test_repair_is_minimum_move_and_explicit(self):
        labels = np.array([0] * 50 + [1] * 50)
        rng = ScriptedGenerator([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        spec = PartitionSpec("Dirichlet01", 0.1, 4, 11, 3, 1, "minimum_move_v1")
        result = build_label_dirichlet_partition(labels, labels[:20], spec, rng=rng)
        self.assertTrue(result.repaired)
        self.assertEqual(len(result.repair_moves), 9)
        self.assertEqual(min(map(len, result.train_indices.values())), 3)
        self.assertEqual(result.partitioner, "label_dirichlet_repaired_v2")

    def test_impossible_minimum_fails_before_sampling(self):
        spec = PartitionSpec("Dirichlet01", 0.1, 40, 42, 51, 2, "minimum_move_v1")
        with self.assertRaisesRegex(PartitionDataError, "requires at least 2040"):
            build_label_dirichlet_partition(np.zeros(2000), np.zeros(100), spec)

    def test_alpha_point_one_forty_clients_finishes_with_valid_coverage(self):
        labels = np.tile(np.array([0, 1]), 1000)
        spec = PartitionSpec("Dirichlet01", 0.1, 40, 42, 1, 2, "minimum_move_v1")
        result = build_label_dirichlet_partition(labels, labels[:400], spec)
        validate_indices(result.train_indices, dataset_size=2000,
                         num_clients=40, min_size=1)
        self.assertLessEqual(result.attempts, 2)

from module.partition import (
    PartitionCacheError,
    build_or_load_partition,
    load_partition_artifact,
    partition_fingerprint,
)


class PartitionCacheTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache_root = Path(self.temp_dir.name) / "partition_cache"
        self.train = ToyDataset(
            [f"train-{index}" for index in range(120)],
            np.tile([0, 1], 60), np.tile([0, 1, 1, 0], 30),
        )
        self.test = ToyDataset(
            [f"test-{index}" for index in range(40)],
            np.tile([0, 1], 20), np.tile([0, 1, 1, 0], 10),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def make_params(self, algorithm="FedAvg", alpha_name="Dirichlet05", base_seed=42):
        return {
            "algorithm": algorithm, "hypothesis": "Tiny", "dataset_name": "toy",
            "task": "Tabular_CLF", "split_strategy": alpha_name,
            "num_clients_K": 4, "base_seed": base_seed,
            "partition_min_size": 2, "partition_max_retries": 3,
            "partition_repair_policy": "minimum_move_v1",
            "partition_cache_root": str(self.cache_root), "system_data_count": 120,
        }

    def test_algorithm_and_model_do_not_affect_cache_identity(self):
        first = build_or_load_partition(self.make_params("FedAvg"), self.train, self.test, 0)
        second = build_or_load_partition(self.make_params("PraFFL"), self.train, self.test, 0)
        self.assertEqual(first.fingerprint, second.fingerprint)
        for client_id in range(4):
            np.testing.assert_array_equal(first.train_indices[client_id], second.train_indices[client_id])

    def test_alpha_seed_and_data_order_each_change_identity(self):
        base = build_or_load_partition(self.make_params(), self.train, self.test, 0)
        alpha = build_or_load_partition(self.make_params(alpha_name="Dirichlet1"), self.train, self.test, 0)
        seed = build_or_load_partition(self.make_params(base_seed=43), self.train, self.test, 0)
        order = np.arange(len(self.train))[::-1]
        reordered = ToyDataset(
            [self.train.sample_ids[index] for index in order],
            self.train.labels[order], self.train.protected[order],
        )
        data = build_or_load_partition(self.make_params(), reordered, self.test, 0)
        self.assertEqual(len({base.fingerprint, alpha.fingerprint, seed.fingerprint, data.fingerprint}), 4)

    def test_metadata_records_profile_repair_and_fairness_cells(self):
        artifact = build_or_load_partition(self.make_params(), self.train, self.test, 0)
        self.assertIn(artifact.metadata["partitioner"],
                      {"label_dirichlet_v2", "label_dirichlet_repaired_v2"})
        self.assertIn("repair_count", artifact.metadata)
        self.assertIn("joint_counts", artifact.metadata["train_stats"]["0"])
        self.assertEqual(artifact.metadata["indices_sha256"], artifact.indices_sha256)

    def test_duplicate_missing_out_of_bounds_and_digest_corruption_fail_closed(self):
        artifact = build_or_load_partition(self.make_params(), self.train, self.test, 0)
        npz_path = artifact.cache_dir / "indices.npz"
        arrays = dict(np.load(npz_path, allow_pickle=False))
        arrays["train_0"] = np.append(arrays["train_0"], arrays["train_1"][0])
        with open(npz_path, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        with self.assertRaises(PartitionCacheError):
            load_partition_artifact(artifact.cache_dir, artifact.fingerprint,
                                    self.train, self.test, artifact.spec)

    def test_new_dirichlet_never_reads_legacy_algorithm_local_json(self):
        params = self.make_params()
        model_path = Path(self.temp_dir.name) / "model"
        params["model_path"] = str(model_path)
        legacy_dir = model_path / "split_info"
        legacy_dir.mkdir(parents=True)
        with open(legacy_dir / "split_indices.json", "w", encoding="utf-8") as stream:
            json.dump({"split_strategy": "Dirichlet05", "num_clients": 4,
                       "indices": {"0": list(range(120)), "1": [], "2": [], "3": []}}, stream)
        artifact = build_or_load_partition(params, self.train, self.test, 0)
        self.assertTrue(all(len(artifact.train_indices[index]) >= 2 for index in range(4)))


class StackedCachedTextLike(Dataset):
    """Minimal adapter shape used by CachedTextDataset's stacked_text format."""

    def __init__(self):
        self._stacked_cache = {
            "input_ids": torch.tensor([[11, 12, 0], [21, 22, 0], [31, 32, 0]]),
            "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 0], [1, 1, 0]]),
            "labels": torch.tensor([0, 1, 0]),
            "protected": torch.tensor([1, 0, 1]),
        }

    def __len__(self):
        return len(self._stacked_cache["input_ids"])


class PartitionReviewRegressionTest(PartitionCacheTest):
    def test_contract_inputs_change_identity_and_run_output_fields_do_not(self):
        baseline = build_or_load_partition(self.make_params(), self.train, self.test, 0)
        changed_inputs = []

        protected_changed = ToyDataset(
            self.train.sample_ids, self.train.labels, 1 - self.train.protected,
        )
        changed_inputs.append(build_or_load_partition(
            self.make_params(), protected_changed, self.test, 0
        ))
        changed_inputs.append(build_or_load_partition(self.make_params(), self.train, self.test, 1))

        for key, value in (
            ("partition_min_size", 1),
            ("partition_max_retries", 4),
            ("partition_repair_policy", "different_policy"),
            ("system_data_count", 119),
        ):
            params = self.make_params()
            params[key] = value
            changed_inputs.append(build_or_load_partition(params, self.train, self.test, 0))

        self.assertTrue(all(item.fingerprint != baseline.fingerprint for item in changed_inputs))

        output_only = self.make_params("DifferentAlgorithm")
        output_only.update({
            "hypothesis": "DifferentHypothesis",
            "model_path": str(Path(self.temp_dir.name) / "elsewhere"),
            "result_path": str(Path(self.temp_dir.name) / "results"),
        })
        same = build_or_load_partition(output_only, self.train, self.test, 0)
        self.assertEqual(same.fingerprint, baseline.fingerprint)

    def test_corrupt_cache_inputs_are_partition_cache_errors_with_context(self):
        cases = (
            ("truncated metadata JSON", "metadata.json", b'{"fingerprint":'),
            ("non-object metadata JSON", "metadata.json", b'[]'),
            ("truncated NPZ", "indices.npz", b'not an npz archive'),
        )
        for name, filename, payload in cases:
            with self.subTest(name=name):
                params = self.make_params()
                params["partition_cache_root"] = str(self.cache_root / name.replace(" ", "_"))
                artifact = build_or_load_partition(params, self.train, self.test, 0)
                (artifact.cache_dir / filename).write_bytes(payload)
                with self.assertRaisesRegex(PartitionCacheError, "partition cache"):
                    load_partition_artifact(
                        artifact.cache_dir, artifact.fingerprint,
                        self.train, self.test, artifact.spec,
                    )

    def test_missing_arrays_and_non_integer_arrays_fail_closed(self):
        for mode in ("missing", "float"):
            with self.subTest(mode=mode):
                params = self.make_params()
                params["partition_cache_root"] = str(self.cache_root / mode)
                artifact = build_or_load_partition(params, self.train, self.test, 0)
                npz_path = artifact.cache_dir / "indices.npz"
                arrays = dict(np.load(npz_path, allow_pickle=False))
                if mode == "missing":
                    del arrays["test_3"]
                else:
                    arrays["train_0"] = arrays["train_0"].astype(np.float64)
                with open(npz_path, "wb") as stream:
                    np.savez_compressed(stream, **arrays)
                with self.assertRaisesRegex(PartitionCacheError, "partition cache"):
                    load_partition_artifact(
                        artifact.cache_dir, artifact.fingerprint,
                        self.train, self.test, artifact.spec,
                    )

    def test_ready_and_metadata_incomplete_artifacts_fail_closed(self):
        artifact = build_or_load_partition(self.make_params(), self.train, self.test, 0)
        (artifact.cache_dir / "metadata.json").unlink()
        with self.assertRaisesRegex(PartitionCacheError, "partition artifact is incomplete"):
            load_partition_artifact(
                artifact.cache_dir, artifact.fingerprint,
                self.train, self.test, artifact.spec,
            )

        incomplete = artifact.cache_dir.parent / "incomplete"
        incomplete.mkdir()
        (incomplete / "metadata.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(PartitionCacheError, "partition artifact is incomplete"):
            load_partition_artifact(
                incomplete, artifact.fingerprint,
                self.train, self.test, artifact.spec,
            )

    def test_metadata_stats_tampering_fails_closed(self):
        artifact = build_or_load_partition(self.make_params(), self.train, self.test, 0)
        metadata_path = artifact.cache_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["train_stats"]["0"]["joint_counts"] = {"forged": 999}
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(PartitionCacheError, "partition metadata validation failed"):
            load_partition_artifact(
                artifact.cache_dir, artifact.fingerprint,
                self.train, self.test, artifact.spec,
            )

    def test_stacked_cached_text_adapter_uses_sequence_rows_as_stable_identities(self):
        dataset = StackedCachedTextLike()
        view = extract_dataset_view(dataset)
        self.assertEqual(len(view.sample_ids), 3)
        self.assertEqual(len(set(view.sample_ids)), 3)
        np.testing.assert_array_equal(view.labels, np.array([0, 1, 0]))
        np.testing.assert_array_equal(view.protected, np.array([1, 0, 1]))

    def test_explicit_legacy_alias_is_read_only_and_never_upgrades_to_v2_cache(self):
        params = self.make_params(alpha_name="LegacyQuantityDirichlet05")
        model_path = Path(self.temp_dir.name) / "legacy_model"
        params["model_path"] = str(model_path)
        legacy_dir = model_path / "split_info"
        legacy_dir.mkdir(parents=True)
        payload = {
            "split_strategy": "Dirichlet05", "num_clients": 4,
            "indices": {str(client): list(range(client * 30, (client + 1) * 30)) for client in range(4)},
        }
        legacy_path = legacy_dir / "split_indices.json"
        legacy_path.write_text(json.dumps(payload), encoding="utf-8")

        artifact = build_or_load_partition(params, self.train, self.test, repeat_idx=3)
        self.assertEqual(artifact.metadata["partitioner"], "legacy_quantity_dirichlet_v1")
        self.assertEqual(legacy_path.read_text(encoding="utf-8"), json.dumps(payload))
        self.assertFalse((self.cache_root / "v2").exists())
