from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tool.checkpoint import build_experiment_config_hash


class TinyTextDataset(Dataset):
    def __init__(self, rows: list[tuple[float, int, int]]):
        self.rows = rows
        self.labels = np.asarray([row[1] for row in rows], dtype=np.int64)
        self.protected = np.asarray([row[2] for row in rows], dtype=np.int64)
        self.sample_ids = np.arange(len(rows), dtype=np.int64)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        feature, label, protected = self.rows[index]
        return {
            "input_ids": torch.tensor([feature], dtype=torch.float32),
            "attention_mask": torch.tensor([1], dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
            "protected": torch.tensor(protected, dtype=torch.long),
        }


class TinyTextClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.out = nn.Linear(1, 2)

    def forward(self, input_ids, attention_mask=None):
        features = input_ids.float().reshape(input_ids.shape[0], 1)
        return features, self.out(features)


def balanced_rows(offset=0.0):
    return [
        (-2.0 + offset, 0, 0), (-1.0 + offset, 1, 0),
        (1.0 + offset, 0, 1), (2.0 + offset, 1, 1),
    ]


def make_datasets_and_loaders(batch_size=4):
    datasets = [TinyTextDataset(balanced_rows()), TinyTextDataset(balanced_rows(0.25))]
    loaders = [DataLoader(ds, batch_size=batch_size, shuffle=False) for ds in datasets]
    return datasets, loaders


def seeded_model(seed=17):
    torch.manual_seed(seed)
    return TinyTextClassifier()


def cpu_state_dict(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def assert_state_dict_equal(testcase, left, right):
    testcase.assertEqual(set(left), set(right))
    for name in left:
        testcase.assertTrue(torch.equal(left[name], right[name]), name)


def seed_everything(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fedfact_params(root: Path, rounds=1):
    params = {
        "task": "SENT_CLF",
        "algorithm": "FedFACT",
        "dataset": "toy",
        "dataset_name": "toy",
        "hypothesis": "TinyTextClassifier",
        "split_strategy": "Uniform",
        "num_clients_K": 2,
        "communication_round_I": rounds,
        "algorithm_epoch_T": 1,
        "FL_fraction": 1.0,
        "FL_drop_rate": 0.0,
        "batch_size": 4,
        "learning_rate": 0.05,
        "optimize_method": "SGD",
        "fairness_metric": "DP",
        "global_constraint": 0.10,
        "local_constraint": 0.10,
        "dual_learning_rate": 0.5,
        "dual_bound": 5.0,
        "dual_init": 0.1,
        "ensemble_learning_rate": 0.3,
        "ensemble_weight_init": 0.5,
        "calibration_epsilon": 0.001,
        "device": "cpu",
        "use_amp": False,
        "checkpoint_save_freq": 1,
        "checkpoint_keep_latest": 1,
        "model_path": str(root / "models"),
        "result_path": str(root / "result.json"),
        "log_path": str(root / "run.log"),
        "checkpoint_dir": str(root / "checkpoints"),
        "base_seed": 77,
        "repeat_idx": 0,
        "repeat_seed": 77,
        "partition_fingerprint": "fedfact-test-partition",
        "partition_metadata": {},
        "resume": False,
        "parallel_repeats": 1,
    }
    params["experiment_config_hash"] = build_experiment_config_hash(params)
    return params
