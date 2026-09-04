import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch.utils.data import Dataset

from module.experiment_setup import Experiment_Create_dataloader, FederatedDataBundle
import module.dataloader as dataloader_module
from module.dataloader import get_FL_dataloader
from tool.experiment_state import (
    capture_training_loader_generator_states,
    restore_training_loader_generator_states,
)
from tool.seed_manager import seed_worker


class LoaderToyDataset(Dataset):
    def __init__(self, prefix, size):
        self.sample_ids = [f"{prefix}-{index}" for index in range(size)]
        self.labels = np.tile([0, 1], size // 2)
        self.protected = np.tile([0, 1, 1, 0], size // 4)
        self.X = torch.arange(size, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "X": self.X[index],
            "labels": torch.tensor(self.labels[index]),
            "protected": torch.tensor(self.protected[index]),
        }


class FederatedDataBundleTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.train = LoaderToyDataset("train", 120)
        self.test = LoaderToyDataset("test", 40)
        self.params = {
            "algorithm": "FedAvg",
            "dataset_name": "toy",
            "task": "Tabular_CLF",
            "split_strategy": "Dirichlet05",
            "num_clients_K": 4,
            "batch_size": 8,
            "test_batch_size": 10,
            "base_seed": 42,
            "partition_min_size": 2,
            "partition_max_retries": 3,
            "partition_repair_policy": "minimum_move_v1",
            "partition_cache_root": self.temp_dir.name,
            "dataloader_num_workers": 0,
            "system_data_count": 120,
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def build(self, repeat_idx=0, algorithm="FedAvg", **overrides):
        params = dict(self.params, repeat_idx=repeat_idx, algorithm=algorithm, **overrides)
        return Experiment_Create_dataloader(
            params, self.train, None, self.test, params["split_strategy"]
        )

    def test_returns_global_and_client_level_test_loaders(self):
        bundle = self.build()
        self.assertIsInstance(bundle, FederatedDataBundle)
        self.assertEqual(len(bundle.training_dataloaders), 4)
        self.assertEqual(len(bundle.client_dataset_list), 4)
        self.assertEqual(len(bundle.client_testing_dataloaders), 4)
        self.assertEqual(len(bundle.client_testing_dataset_list), 4)
        self.assertEqual(sum(map(len, bundle.client_dataset_list)), len(self.train))
        self.assertEqual(sum(map(len, bundle.client_testing_dataset_list)), len(self.test))
        self.assertEqual(len(bundle.testing_dataloader.dataset), len(self.test))

    def test_bundle_supports_legacy_tuple_unpacking(self):
        bundle = self.build()
        training_dataloaders, client_dataset_list, testing_dataloader = bundle
        self.assertEqual(training_dataloaders, bundle.training_dataloaders)
        self.assertEqual(client_dataset_list, bundle.client_dataset_list)
        self.assertEqual(testing_dataloader, bundle.testing_dataloader)

    def test_same_repeat_is_paired_across_algorithms(self):
        first = self.build(0, "FedAvg")
        second = self.build(0, "FedFACT")
        self.assertEqual(first.partition_fingerprint, second.partition_fingerprint)
        for left, right in zip(first.client_dataset_list, second.client_dataset_list):
            self.assertEqual(list(left.indices), list(right.indices))
        for left, right in zip(first.client_testing_dataset_list, second.client_testing_dataset_list):
            self.assertEqual(list(left.indices), list(right.indices))

    def test_next_repeat_has_distinct_partition_seed_and_fingerprint(self):
        first = self.build(0)
        second = self.build(1)
        self.assertNotEqual(first.partition_fingerprint, second.partition_fingerprint)
        self.assertEqual(first.partition_metadata["partition_seed"], 42)
        self.assertEqual(second.partition_metadata["partition_seed"], 1042)

    def test_global_and_client_test_loaders_do_not_shuffle(self):
        bundle = self.build()
        global_order = torch.cat([batch["X"][:, 0] for batch in bundle.testing_dataloader]).tolist()
        self.assertEqual(global_order, list(range(40)))
        for loader, subset in zip(bundle.client_testing_dataloaders, bundle.client_testing_dataset_list):
            observed = torch.cat([batch["X"][:, 0] for batch in loader]).tolist()
            expected = [float(index) for index in subset.indices]
            self.assertEqual(observed, expected)

    def test_train_loaders_have_explicit_repeat_scoped_generators(self):
        bundle = self.build(repeat_idx=1)
        seeds = [loader.generator.initial_seed() for loader in bundle.training_dataloaders]
        self.assertEqual(seeds, [1042, 1043, 1044, 1045])
        self.assertEqual(len({id(loader.generator) for loader in bundle.training_dataloaders}), 4)

    def test_dataloader_num_workers_and_worker_init_are_configurable(self):
        bundle = self.build(dataloader_num_workers=2)
        for loader in bundle.training_dataloaders:
            self.assertEqual(loader.num_workers, 2)
            self.assertIs(loader.worker_init_fn, seed_worker)
        self.assertEqual(bundle.testing_dataloader.num_workers, 2)
        self.assertIs(bundle.testing_dataloader.worker_init_fn, seed_worker)
        for loader in bundle.client_testing_dataloaders:
            self.assertEqual(loader.num_workers, 2)
            self.assertIs(loader.worker_init_fn, seed_worker)

    def test_formal_experiments_disable_persistent_workers(self):
        """Exact resume must not inherit memory_utils' persistent-worker suggestion."""
        with mock.patch.object(
            dataloader_module,
            "_DATALOADER_CONFIG",
            {"pin_memory": True, "num_workers": 2, "persistent_workers": True},
        ):
            bundle = self.build(dataloader_num_workers=2)

        loaders = (
            bundle.training_dataloaders
            + bundle.client_testing_dataloaders
            + [bundle.testing_dataloader]
        )
        self.assertTrue(all(not loader.persistent_workers for loader in loaders))

    def test_implicit_worker_count_is_capped_at_four(self):
        params = dict(self.params)
        params.pop("dataloader_num_workers")
        with mock.patch.object(
            dataloader_module,
            "_DATALOADER_CONFIG",
            {"pin_memory": False, "num_workers": 16, "persistent_workers": True},
        ):
            bundle = Experiment_Create_dataloader(
                params, self.train, None, self.test, params["split_strategy"]
            )

        loaders = bundle.training_dataloaders + bundle.client_testing_dataloaders
        self.assertLessEqual(bundle.testing_dataloader.num_workers, 4)
        self.assertTrue(all(loader.num_workers <= 4 for loader in loaders))

    def test_explicit_worker_count_is_preserved(self):
        with mock.patch.object(
            dataloader_module,
            "_DATALOADER_CONFIG",
            {"pin_memory": False, "num_workers": 16, "persistent_workers": True},
        ):
            bundle = self.build(dataloader_num_workers=7)

        loaders = (
            bundle.training_dataloaders
            + bundle.client_testing_dataloaders
            + [bundle.testing_dataloader]
        )
        self.assertTrue(all(loader.num_workers == 7 for loader in loaders))

    def test_global_defaults_cap_workers_and_disable_persistent_workers(self):
        with mock.patch.object(dataloader_module, "HAS_MEMORY_UTILS", True), mock.patch.object(
            dataloader_module,
            "get_dataloader_config",
            return_value={"pin_memory": True, "num_workers": 16, "persistent_workers": True},
        ), mock.patch.object(dataloader_module, "_DATALOADER_CONFIG", None):
            config = dataloader_module.get_global_dataloader_config()

        self.assertLessEqual(config["num_workers"], 4)
        self.assertFalse(config["persistent_workers"])

    def test_train_only_legacy_loader_request_fails_instead_of_fabricating_test_data(self):
        with self.assertRaisesRegex(ValueError, "testing_dataset"):
            get_FL_dataloader(
                dict(self.params, model_path=self.temp_dir.name),
                self.train,
                self.params["num_clients_K"],
                split_strategy="LegacyQuantityDirichlet05",
                do_train=True,
            )

    def test_training_loader_generator_states_can_be_restored_at_round_boundaries(self):
        bundle = self.build(repeat_idx=1)
        states = capture_training_loader_generator_states(bundle.training_dataloaders)
        for loader in bundle.training_dataloaders:
            torch.rand(3, generator=loader.generator)

        restore_training_loader_generator_states(bundle.training_dataloaders, states)

        for loader, state in zip(bundle.training_dataloaders, states):
            self.assertTrue(torch.equal(loader.generator.get_state(), state))

    def test_new_uniform_and_dirichlet_builds_ignore_legacy_split_cache(self):
        model_path = Path(self.temp_dir.name) / "legacy_model"
        split_dir = model_path / "split_info"
        split_dir.mkdir(parents=True)
        with open(split_dir / "split_indices.json", "w", encoding="utf-8") as stream:
            json.dump(
                {
                    "split_strategy": "Uniform",
                    "num_clients": 4,
                    "indices": {"0": list(range(120)), "1": [], "2": [], "3": []},
                },
                stream,
            )
        uniform = self.build(split_strategy="Uniform", model_path=str(model_path))
        self.assertEqual(sum(map(len, uniform.client_dataset_list)), len(self.train))
        self.assertTrue(all(len(dataset) > 0 for dataset in uniform.client_dataset_list))

        dirichlet = self.build(split_strategy="Dirichlet05", model_path=str(model_path))
        self.assertEqual(sum(map(len, dirichlet.client_dataset_list)), len(self.train))
        self.assertTrue(all(len(dataset) >= 2 for dataset in dirichlet.client_dataset_list))


if __name__ == "__main__":
    unittest.main()
