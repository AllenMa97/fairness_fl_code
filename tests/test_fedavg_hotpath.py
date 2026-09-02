from collections import OrderedDict
import unittest
from unittest.mock import patch

import torch

from algorithm.FederatedAverage import (
    _aggregate_state_dicts,
    _build_client_updates,
    _needs_client_updates,
    _train_single_client_fedavg,
)


class FedAvgHotPathTest(unittest.TestCase):
    def test_aggregate_state_dicts_uses_weighted_torch_mean(self):
        states = [
            OrderedDict(weight=torch.tensor([1.0, 3.0]), counter=torch.tensor([2], dtype=torch.long)),
            OrderedDict(weight=torch.tensor([5.0, 7.0]), counter=torch.tensor([4], dtype=torch.long)),
        ]

        result = _aggregate_state_dicts(states, [1, 3])

        torch.testing.assert_close(result['weight'], torch.tensor([4.0, 6.0]))
        self.assertEqual(result['weight'].device.type, 'cpu')
        self.assertEqual(result['weight'].dtype, torch.float32)
        torch.testing.assert_close(result['counter'], torch.tensor([3], dtype=torch.long))

    def test_aggregate_state_dicts_rejects_invalid_inputs(self):
        state = OrderedDict(weight=torch.tensor([1.0]))
        with self.assertRaises(ValueError):
            _aggregate_state_dicts([], [])
        with self.assertRaises(ValueError):
            _aggregate_state_dicts([state], [0])
        with self.assertRaises(ValueError):
            _aggregate_state_dicts([state], [1, 2])

    def test_client_updates_follow_gradient_monitoring_frequency(self):
        cfg = {'gradient': True, 'gradient_freq': 2}
        self.assertFalse(_needs_client_updates(cfg, 1))
        self.assertTrue(_needs_client_updates(cfg, 2))
        self.assertFalse(_needs_client_updates({'gradient': False, 'gradient_freq': 1}, 1))



    def test_build_client_updates_matches_pre_aggregation_state(self):
        client_states = [
            OrderedDict(weight=torch.tensor([3.0]), counter=torch.tensor([2])),
            OrderedDict(weight=torch.tensor([5.0]), counter=torch.tensor([2])),
        ]
        reference = OrderedDict(weight=torch.tensor([1.0]), counter=torch.tensor([2]))

        updates = _build_client_updates(client_states, reference)

        torch.testing.assert_close(updates[0]['0'], torch.tensor([2.0]))
        torch.testing.assert_close(updates[1]['0'], torch.tensor([4.0]))
        torch.testing.assert_close(updates[0]['1'], torch.tensor([0]))

    def test_client_training_returns_cpu_state_without_disk_io_or_batch_gc(self):
        class TinyANN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 1)

            def forward(self, features):
                probabilities = torch.sigmoid(self.linear(features))
                return probabilities, features

        batches = [
            {'X': torch.tensor([[1.0, 0.0]]), 'labels': torch.tensor([1.0])},
            {'X': torch.tensor([[0.0, 1.0]]), 'labels': torch.tensor([0.0])},
        ]
        model = TinyANN()
        params = {
            'optimize_method': 'sgd',
            'learning_rate': 0.1,
            'task': 'Tabular_CLF',
        }

        with patch('algorithm.FederatedAverage.torch.save') as save_mock, \
             patch('algorithm.FederatedAverage.gc.collect') as collect_mock:
            result = _train_single_client_fedavg(
                client_id=0,
                device=torch.device('cpu'),
                model=model,
                param_dict=params,
                training_dataloaders=[batches],
                algorithm_epoch_T=1,
                accumulation_steps=1,
                use_amp=False,
                scaler=None,
                criterion=torch.nn.BCELoss(reduction='none'),
                iter_t=0,
                communication_round_I=1,
                num_clients_K=1,
            )

        save_mock.assert_not_called()
        collect_mock.assert_not_called()
        self.assertIn('state_dict', result)
        self.assertGreaterEqual(result['gpu_seconds'], 0)
        self.assertTrue(all(t.device.type == 'cpu' for t in result['state_dict'].values()))


if __name__ == '__main__':
    unittest.main()
