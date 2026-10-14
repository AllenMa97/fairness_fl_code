"""公共聚合模块的冒烟测试 / Smoke tests for tool.aggregation.

重点验证聚合会**完整搬运 buffer**（BatchNorm 的 running_mean/running_var 等），
这正是 FedRep / Scaffold 塌缩的根因：只聚合 parameters() 会漏掉这些 buffer。
"""

import unittest
from collections import OrderedDict

import torch
import torch.nn as nn

from tool.aggregation import (
    assign_state_dict,
    average_state_dicts,
    state_dict_keys_of_submodule,
)


class _ToyModel(nn.Module):
    """带 BatchNorm 的迷你模型，子模块命名对齐仓库（shared_base + head）。"""

    def __init__(self):
        super().__init__()
        self.shared_base = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.ReLU())
        self.head = nn.Linear(8, 1)

    def forward(self, x):
        return self.head(self.shared_base(x))


def _client(bn_mean, bn_var):
    model = _ToyModel()
    with torch.no_grad():
        model.shared_base[1].running_mean.fill_(bn_mean)
        model.shared_base[1].running_var.fill_(bn_var)
    return model


class AverageStateDictsTest(unittest.TestCase):
    def test_average_includes_batch_norm_buffers(self):
        states = [_client(1.0, 2.0).state_dict(), _client(3.0, 6.0).state_dict()]
        averaged = average_state_dicts(states, weights=[1.0, 1.0])

        mean_key = "shared_base.1.running_mean"
        var_key = "shared_base.1.running_var"
        self.assertIn(mean_key, averaged)
        self.assertIn(var_key, averaged)
        self.assertTrue(torch.allclose(averaged[mean_key], torch.full((8,), 2.0)))
        self.assertTrue(torch.allclose(averaged[var_key], torch.full((8,), 4.0)))

    def test_dataset_weighted_average(self):
        states = [_client(1.0, 1.0).state_dict(), _client(3.0, 1.0).state_dict()]
        averaged = average_state_dicts(states, weights=[3.0, 1.0])
        # (3 * 1 + 1 * 3) / 4 = 1.5
        self.assertTrue(
            torch.allclose(averaged["shared_base.1.running_mean"], torch.full((8,), 1.5))
        )

    def test_partial_submodule_aggregation_keeps_head_local(self):
        model = _ToyModel()
        head_before = model.head.weight.detach().clone()
        states = [_client(1.0, 1.0).state_dict(), _client(3.0, 3.0).state_dict()]

        keys = state_dict_keys_of_submodule(model, "shared_base")
        self.assertTrue(all(k.startswith("shared_base.") for k in keys))
        self.assertTrue(any("running_mean" in k for k in keys))

        averaged = average_state_dicts(states, keys=keys)
        assign_state_dict(model, averaged, strict=False)

        # base 层被写回，head 层（本地个性化）保持原值
        self.assertTrue(
            torch.allclose(model.shared_base[1].running_mean, torch.full((8,), 2.0))
        )
        self.assertTrue(torch.equal(model.head.weight, head_before))

    def test_parameters_only_omits_buffers_demonstrating_root_cause(self):
        model = _ToyModel()
        parameter_only_keys = [k for k, _ in model.named_parameters()]
        self.assertNotIn("shared_base.1.running_mean", parameter_only_keys)
        self.assertIn(
            "shared_base.1.running_mean",
            state_dict_keys_of_submodule(model, "shared_base"),
        )

    def test_mismatched_keys_raise(self):
        first = _ToyModel().state_dict()
        second = OrderedDict(_ToyModel().state_dict())
        second.pop("head.bias")
        with self.assertRaises(ValueError):
            average_state_dicts([first, second])

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            average_state_dicts([])

    def test_assign_partial_state_is_non_strict(self):
        model = _ToyModel()
        averaged = average_state_dicts([model.state_dict()], keys=["head.weight"])
        assign_state_dict(model, averaged, strict=False)
        self.assertEqual(model.head.weight.shape, (1, 8))


if __name__ == "__main__":
    unittest.main()
