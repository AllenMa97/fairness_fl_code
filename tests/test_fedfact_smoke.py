import math
import os
import resource
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from algorithm.FedFACT import FedFACT
from algorithm.fedfact_evaluation import evaluate_fedfact
from hypothesis.BERTCLASSIFIER import BertClassifier
from module.experiment_setup import FederatedDataBundle
from tool.checkpoint import build_experiment_config_hash, load_checkpoint
from tests.fedfact_test_utils import (
    fedfact_params,
    make_datasets_and_loaders,
    seeded_model,
)


class FedFACTSmokeTest(unittest.TestCase):
    def test_cpu_toy_one_round(self):
        datasets, loaders = make_datasets_and_loaders(batch_size=2)
        bundle = FederatedDataBundle(
            training_dataloaders=loaders,
            client_dataset_list=datasets,
            testing_dataloader=loaders[0],
            client_testing_dataloaders=loaders,
            client_testing_dataset_list=datasets,
            partition_fingerprint="fedfact-smoke-partition",
            partition_metadata={},
        )
        with tempfile.TemporaryDirectory() as raw:
            params = fedfact_params(Path(raw), rounds=1)
            params.update({
                "dataset_name": "toy", "dataset": "toy",
                "hypothesis": "TinyTextClassifier", "split_strategy": "Uniform",
                "base_seed": 77, "repeat_idx": 0, "repeat_seed": 77,
                "partition_fingerprint": bundle.partition_fingerprint,
                "partition_metadata": {}, "resume": False,
                "parallel_repeats": 1,
            })
            params["experiment_config_hash"] = build_experiment_config_hash(params)
            result = FedFACT(
                "cpu", seeded_model(), 1, 2, 1, 1.0, 0.0,
                loaders, datasets[0], datasets, params, loaders[0], 4,
            )
            metrics = evaluate_fedfact(
                result.global_model, params, bundle, result.algorithm_state
            )
            checkpoint = load_checkpoint(params)

        self.assertEqual(result.client_selection_history, [[0, 1]])
        for tensor in result.global_model.state_dict().values():
            self.assertTrue(torch.isfinite(tensor).all())
        state = result.algorithm_state
        for personal in state["personal_model_states"]:
            self.assertTrue(all(torch.isfinite(t).all() for t in personal.values()))
        self.assertTrue(torch.isfinite(state["global_dual"]).all())
        self.assertTrue(torch.isfinite(state["local_duals"]).all())
        self.assertTrue((state["global_dual"] >= 0).all())
        self.assertTrue((state["local_duals"] >= 0).all())
        self.assertLessEqual(state["global_dual"].sum().item(), params["dual_bound"])
        for dual in state["local_duals"]:
            self.assertLessEqual(dual.sum().item(), params["dual_bound"])
        self.assertTrue(((state["ensemble_weights"] > 0) & (state["ensemble_weights"] < 1)).all())
        for key in ("ACC", "SPD", "global_fairness", "mean_local_fairness", "max_local_fairness"):
            self.assertTrue(math.isfinite(metrics[key]), key)
        self.assertIsNotNone(checkpoint)
        self.assertEqual(len(checkpoint.algorithm_state["personal_model_states"]), 2)
        for key in ("global_dual", "local_duals", "ensemble_weights", "support_counts"):
            self.assertIn(key, checkpoint.algorithm_state)
        self.assertEqual(checkpoint.client_selection_history, [[0, 1]])
        self.assertTrue(checkpoint.rng_state)
        self.assertIsNone(checkpoint.amp_scaler_state)


class BalancedTokenDataset(torch.utils.data.Dataset):
    def __init__(self, client_id):
        self.labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
        self.protected = np.asarray([0, 0, 1, 1], dtype=np.int64)
        self.sample_ids = np.asarray([
            f"client-{client_id}-sample-{index}" for index in range(4)
        ])
        self.tokens = []
        for label, protected in zip(self.labels, self.protected):
            word = 1000 + 100 * client_id + 10 * int(protected) + int(label)
            self.tokens.append(torch.tensor(
                [101, word, 102, 0, 0, 0, 0, 0], dtype=torch.long
            ))

    def __len__(self):
        return 4

    def __getitem__(self, index):
        tokens = self.tokens[index]
        return {
            "input_ids": tokens,
            "attention_mask": (tokens != 0).long(),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
            "protected": torch.tensor(self.protected[index], dtype=torch.long),
        }


@unittest.skipUnless(
    os.environ.get("RUN_FEDFACT_BERT_SMOKE") == "1" and torch.cuda.is_available(),
    "set RUN_FEDFACT_BERT_SMOKE=1 on a CUDA host with a local BERT model",
)
class FedFACTBertSmokeTest(unittest.TestCase):
    def run_mode(self, use_amp):
        mode = "amp_on" if use_amp else "amp_off"
        root = Path(os.environ.get("FEDFACT_SMOKE_ROOT", "/tmp/fedfact-in-bert-smoke")) / mode
        if root.exists():
            shutil.rmtree(root)
        datasets = [BalancedTokenDataset(0), BalancedTokenDataset(1)]
        loaders = [torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False) for ds in datasets]
        bundle = FederatedDataBundle(
            training_dataloaders=loaders, client_dataset_list=datasets,
            testing_dataloader=loaders[0], client_testing_dataloaders=loaders,
            client_testing_dataset_list=datasets,
            partition_fingerprint=f"fedfact-bert-{mode}", partition_metadata={},
        )
        params = fedfact_params(root, rounds=1)
        params.update({
            "hypothesis": "BertClassifier", "batch_size": 2,
            "device": "cuda", "use_amp": use_amp, "fairness_metric": "EO",
            "partition_fingerprint": bundle.partition_fingerprint,
        })
        params["experiment_config_hash"] = build_experiment_config_hash(params)
        model_path = os.environ.get("BERT_MODEL_PATH", "bert-base-uncased")
        model = BertClassifier(n_classes=2, model_name_or_path=model_path)
        torch.cuda.set_device(0)
        _ = torch.empty(1, device="cuda")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        result = FedFACT(
            "cuda", model, 1, 2, 1, 1.0, 0.0,
            loaders, datasets[0], datasets, params, loaders[0], 4,
        )
        metrics = evaluate_fedfact(result.global_model, params, bundle, result.algorithm_state)
        checkpoint = load_checkpoint(params)
        cuda_mib = torch.cuda.max_memory_allocated() / 2**20
        rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        checkpoint_mib = checkpoint.path.stat().st_size / 2**20
        print(
            f"FEDFACT_BERT_SMOKE mode={mode} cuda_mib={cuda_mib:.1f} "
            f"rss_mib={rss_mib:.1f} checkpoint_mib={checkpoint_mib:.1f} "
            f"ACC={metrics['ACC']:.6f} global={metrics['global_fairness']:.6f} "
            f"max_local={metrics['max_local_fairness']:.6f}"
        )
        return result, metrics, checkpoint.path, cuda_mib, rss_mib

    def test_bert_one_round_amp_off(self):
        result, _, path, cuda_mib, rss_mib = self.run_mode(False)
        self.assertIsNone(result.amp_scaler_state)
        self.assertLess(cuda_mib, 12 * 1024)
        self.assertLess(rss_mib, 24 * 1024)
        self.assertLess(path.stat().st_size / 2**20, 2 * 1024)

    def test_bert_one_round_amp_on(self):
        result, _, path, cuda_mib, rss_mib = self.run_mode(True)
        self.assertIsNotNone(result.amp_scaler_state)
        self.assertLess(cuda_mib, 12 * 1024)
        self.assertLess(rss_mib, 24 * 1024)
        self.assertLess(path.stat().st_size / 2**20, 2 * 1024)


if __name__ == "__main__":
    unittest.main()
