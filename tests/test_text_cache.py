import tempfile
import unittest

import torch
from torch.utils.data import DataLoader

import module.dataset as dataset_module
from module.dataset import BiosDataset, CachedTextDataset, MoJiDataset


class TinyTokenizer:
    def __call__(
        self,
        texts,
        add_special_tokens,
        max_length,
        return_token_type_ids,
        padding,
        truncation,
        return_attention_mask,
        return_tensors,
    ):
        if isinstance(texts, str):
            texts = [texts]
        input_ids = torch.zeros((len(texts), max_length), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row, text in enumerate(texts):
            length = min(len(text), max_length)
            input_ids[row, :length] = row + 1
            attention_mask[row, :length] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class TextCacheTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_cache_dir = dataset_module.CACHE_DIR
        dataset_module.CACHE_DIR = self.temp_dir.name
        self.tokenizer = TinyTokenizer()

    def tearDown(self):
        dataset_module.CACHE_DIR = self.original_cache_dir
        self.temp_dir.cleanup()

    def make_dataset(self, dataset_class, size, cache_name):
        return dataset_class(
            texts=["x" * (index + 1) for index in range(size)],
            labels=list(range(size)),
            protected=[index % 2 for index in range(size)],
            tokenizer=self.tokenizer,
            max_len=8,
            cache_name=cache_name,
            cache_split="train",
        )

    def test_first_access_after_new_cache_returns_stacked_item(self):
        dataset = self.make_dataset(MoJiDataset, 3, "moji-first")
        sample = dataset[0]

        self.assertEqual(sample["input_ids"].shape, torch.Size([8]))
        self.assertEqual(sample["labels"].item(), 0)
        self.assertEqual(sample["protected"].item(), 0)

    def test_cache_is_complete_before_workers_start(self):
        dataset = self.make_dataset(MoJiDataset, 6, "moji-workers")

        self.assertTrue(dataset._use_cache)
        loader = DataLoader(dataset, batch_size=2, num_workers=2)
        labels = torch.cat([batch["labels"] for batch in loader]).tolist()

        self.assertEqual(labels, list(range(6)))

    def test_length_mismatch_rebuilds_cache(self):
        first = self.make_dataset(BiosDataset, 2, "bios-stale")
        self.assertEqual(first[1]["labels"].item(), 1)

        second = self.make_dataset(BiosDataset, 3, "bios-stale")
        self.assertEqual(second[2]["labels"].item(), 2)

        cached = CachedTextDataset(
            dataset_module._get_cache_dir("bios-stale", "train")
        )
        self.assertEqual(cached.meta["total_len"], 3)

    def test_moji_and_bios_share_eager_cache_behavior(self):
        for dataset_class, cache_name in (
            (MoJiDataset, "moji-shared"),
            (BiosDataset, "bios-shared"),
        ):
            with self.subTest(dataset_class=dataset_class.__name__):
                dataset = self.make_dataset(dataset_class, 2, cache_name)
                self.assertTrue(dataset._use_cache)
                self.assertTrue(dataset._cache_built)


if __name__ == "__main__":
    unittest.main()
