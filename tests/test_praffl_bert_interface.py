import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from hypothesis.BERTCLASSIFIER import BertClassifier


class FakeBert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=3)

    def forward(self, input_ids, attention_mask, return_dict=False):
        del attention_mask, return_dict
        sequence = input_ids.float().unsqueeze(-1).expand(-1, -1, 3)
        pooled = sequence.mean(dim=1)
        return sequence, pooled


class BertClassifierBoundaryTest(unittest.TestCase):
    @patch("hypothesis.BERTCLASSIFIER.BertModel.from_pretrained", return_value=FakeBert())
    def test_model_source_is_configurable_for_offline_execution(self, factory):
        BertClassifier(n_classes=2, model_name_or_path="/models/bert-base-uncased")

        self.assertEqual(factory.call_args.args[0], "/models/bert-base-uncased")

    @patch("hypothesis.BERTCLASSIFIER.BertModel.from_pretrained", return_value=FakeBert())
    def test_forward_delegates_to_encode_and_classify_without_changing_legacy_api(self, _factory):
        model = BertClassifier(n_classes=2, pooled_output_flag=False)
        model.eval()
        input_ids = torch.tensor([[1, 2], [3, 4]])
        attention_mask = torch.ones_like(input_ids)

        encoded = model.encode(input_ids, attention_mask)
        classified = model.classify(encoded)
        legacy_encoded = model.only_PLM_forward(input_ids, attention_mask)
        legacy_feature, legacy_logits = model.only_clf_forward(encoded)
        forward_feature, forward_logits = model(input_ids, attention_mask)

        self.assertTrue(torch.equal(encoded, legacy_encoded))
        self.assertTrue(torch.equal(encoded, legacy_feature))
        self.assertTrue(torch.equal(encoded, forward_feature))
        self.assertTrue(torch.equal(classified, legacy_logits))
        self.assertTrue(torch.equal(classified, forward_logits))

    @patch("hypothesis.BERTCLASSIFIER.BertModel.from_pretrained", return_value=FakeBert())
    def test_pooled_flag_selects_pooled_feature(self, _factory):
        model = BertClassifier(n_classes=2, pooled_output_flag=True)
        input_ids = torch.tensor([[1, 3]])
        encoded = model.encode(input_ids, torch.ones_like(input_ids))
        self.assertTrue(torch.equal(encoded, torch.tensor([[2.0, 2.0, 2.0]])))


if __name__ == "__main__":
    unittest.main()
