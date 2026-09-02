# AMP and Text Cache Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Make BERT text-classification training succeed with CUDA AMP and multi-worker data loading without manual cache prewarming.

**Architecture:** Keep BERTCLF_Optimizer as the public optimizer and explicitly delegate the PyTorch optimizer state needed by GradScaler. Build text caches in the parent process before DataLoader workers start, validate cache metadata against the dataset, and publish completed cache files atomically.

**Tech Stack:** Python 3.11, PyTorch 2.7.1 with CUDA 12.8, unittest, Hugging Face Transformers.

---

## File structure

- Create tests/test_amp_optimizer.py: focused CUDA regression for GradScaler and BERTCLF_Optimizer.
- Modify algorithm/Optimizers.py: expose the wrapped optimizer interface required by AMP.
- Create tests/test_text_cache.py: deterministic text-cache regressions with a small tokenizer.
- Modify module/dataset.py: validate, publish, build, load, and retrieve text caches safely.
- Create /home/ronnie/run_main_sent_amp.py outside Git: validation launcher only.
- Update this plan by checking completed steps as execution proceeds.

### Task 1: Add the failing CUDA AMP optimizer regression

**Files:**
- Create: tests/test_amp_optimizer.py
- Test: tests/test_amp_optimizer.py

- [ ] **Step 1: Create the regression test**

    import unittest

    import torch

    from algorithm.Optimizers import BERTCLF_Optimizer


    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for AMP regression")
    class BERTCLFOptimizerAMPTest(unittest.TestCase):
        def test_grad_scaler_steps_wrapper_and_updates_parameters(self):
            device = torch.device("cuda:0")
            model = torch.nn.Linear(4, 2).to(device)
            optimizer = BERTCLF_Optimizer(
                method="sgd",
                learning_rate=0.1,
                max_grad_norm=0,
            )
            optimizer.set_parameters(model.named_parameters())
            scaler = torch.amp.GradScaler("cuda")
            inputs = torch.randn(8, 4, device=device)
            targets = torch.randn(8, 2, device=device)
            before = [parameter.detach().clone() for parameter in model.parameters()]

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                loss = torch.nn.functional.mse_loss(model(inputs), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            self.assertIs(optimizer.param_groups, optimizer.optimizer.param_groups)
            self.assertIs(optimizer.state, optimizer.optimizer.state)
            self.assertTrue(
                any(
                    not torch.equal(old, new)
                    for old, new in zip(before, model.parameters())
                )
            )


    if __name__ == "__main__":
        unittest.main()

- [ ] **Step 2: Run the test and verify the original failure**

Run:

    cd /home/ronnie/.config/superpowers/worktrees/fairness_fl_code/fix-amp-text-cache
    /home/ronnie/anaconda3/envs/FL/bin/python -m unittest tests.test_amp_optimizer -v

Expected: ERROR in GradScaler with AttributeError stating that BERTCLF_Optimizer has no attribute param_groups.

### Task 2: Add the minimal AMP optimizer adapter

**Files:**
- Modify: algorithm/Optimizers.py:1-95
- Test: tests/test_amp_optimizer.py

- [ ] **Step 1: Import torch and add explicit delegated properties**

Add import torch before the existing torch.optim import, then add these properties immediately before zero_grad:

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def state(self):
        return self.optimizer.state

The wrapper remains responsible for step, gradient clipping, learning-rate scheduling, and pFedMe updates. Do not pass self.optimizer directly to GradScaler.

- [ ] **Step 2: Run the focused test**

Run:

    /home/ronnie/anaconda3/envs/FL/bin/python -m unittest tests.test_amp_optimizer -v

Expected: one test passes and the model parameters change.

- [ ] **Step 3: Compile the modified optimizer module**

Run:

    /home/ronnie/anaconda3/envs/FL/bin/python -m compileall -q algorithm/Optimizers.py

Expected: exit code 0 with no output.

- [ ] **Step 4: Commit the AMP fix**

Run:

    git add algorithm/Optimizers.py tests/test_amp_optimizer.py
    git commit -m "fix: make BERT optimizer compatible with AMP"

Expected: one commit containing only the optimizer and its regression test.

### Task 3: Add failing text-cache regressions

**Files:**
- Create: tests/test_text_cache.py
- Test: tests/test_text_cache.py

- [ ] **Step 1: Create a deterministic tokenizer and cache tests**

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

- [ ] **Step 2: Run the cache tests and verify failures**

Run:

    /home/ronnie/anaconda3/envs/FL/bin/python -m unittest tests.test_text_cache -v

Expected: failures include the new cache not being built in the constructor and KeyError when the newly built stacked dictionary is indexed by an integer.

### Task 4: Make text-cache validation and publication reliable

**Files:**
- Modify: module/dataset.py:39-87
- Test: tests/test_text_cache.py

- [ ] **Step 1: Replace _shard_exists with guarded metadata validation**

Use this signature and behavior:

    def _shard_exists(cache_dir, total_len, expected_format=None):
        if not os.path.exists(cache_dir):
            return False
        meta_path = os.path.join(cache_dir, "meta.pt")
        if not os.path.exists(meta_path):
            return False
        try:
            meta = torch.load(meta_path, weights_only=False)
            if meta.get("total_len") != total_len:
                return False
            if expected_format is not None and meta.get("format") != expected_format:
                return False
            num_shards = meta["num_shards"]
            if not isinstance(num_shards, int) or num_shards < 0:
                return False
            for index in range(num_shards):
                shard_path = os.path.join(cache_dir, f"shard_{index}.pt")
                if not os.path.exists(shard_path):
                    return False
            return True
        except (KeyError, OSError, RuntimeError, TypeError, ValueError, EOFError):
            return False

- [ ] **Step 2: Add atomic torch serialization**

Add immediately before _save_shards:

    def _atomic_torch_save(value, path):
        temp_path = f"{path}.tmp.{os.getpid()}"
        try:
            torch.save(value, temp_path)
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

- [ ] **Step 3: Publish stacked text cache metadata last**

At the start of _save_text_shards_stacked, remove an existing meta.pt if present. Save every shard with _atomic_torch_save. Save the new metadata with _atomic_torch_save only after every shard has been published.

    meta_path = os.path.join(cache_dir, "meta.pt")
    if os.path.exists(meta_path):
        os.remove(meta_path)

    total_len = len(input_ids_list)
    num_shards = math.ceil(total_len / MAX_SHARD_SIZE)
    for index in range(num_shards):
        start = index * MAX_SHARD_SIZE
        end = min(start + MAX_SHARD_SIZE, total_len)
        shard = {
            "input_ids": torch.stack(input_ids_list[start:end], dim=0),
            "attention_mask": torch.stack(attention_mask_list[start:end], dim=0),
            "labels": torch.tensor(labels_list[start:end], dtype=torch.long),
            "protected": torch.tensor(protected_list[start:end], dtype=torch.long),
        }
        _atomic_torch_save(
            shard,
            os.path.join(cache_dir, f"shard_{index}.pt"),
        )

    meta = {
        "total_len": total_len,
        "num_shards": num_shards,
        "shard_size": MAX_SHARD_SIZE,
        "format": "stacked_text",
    }
    _atomic_torch_save(meta, meta_path)

- [ ] **Step 4: Run the focused tests**

Run:

    /home/ronnie/anaconda3/envs/FL/bin/python -m unittest tests.test_text_cache -v

Expected: tests still fail because text datasets still defer cache construction until worker access; no new serialization error should appear.

### Task 5: Build and load text caches before DataLoader workers start

**Files:**
- Modify: module/dataset.py:465-672
- Test: tests/test_text_cache.py

- [ ] **Step 1: Add a shared cached-text base class**

Create _CachedTextClassificationDataset before MoJiDataset. Move the identical Moji and BIOS initialization, build, length, and item behavior into it.

The constructor must:

1. store texts, labels, protected, tokenizer, and max_len;
2. initialize _use_cache and _cache_built to False;
3. set _cache_dir when cache_name and cache_split are present;
4. call _shard_exists with expected_format="stacked_text";
5. load a valid cache or synchronously call _build_cache for a miss.

Use one shared item reader:

    def _cached_item(self, item):
        if isinstance(self._cached_items, dict):
            return {
                "input_ids": self._cached_items["input_ids"][item],
                "attention_mask": self._cached_items["attention_mask"][item],
                "labels": self._cached_items["labels"][item],
                "protected": self._cached_items["protected"][item],
            }
        return self._cached_items[item]

After _build_cache writes the shards, load them through one method:

    def _load_cache(self):
        cached = CachedTextDataset(self._cache_dir)
        self._cached_items = cached.items
        self._use_cache = True
        self._cache_built = True

The cached branch of __getitem__ must always return self._cached_item(item). The uncached branch is retained only when cache_name or cache_split was omitted.

- [ ] **Step 2: Make both dataset names inherit the shared implementation**

Replace the duplicated method bodies with:

    class MoJiDataset(_CachedTextClassificationDataset):
        pass

and:

    class BiosDataset(_CachedTextClassificationDataset):
        pass

Keep the dataset-size comments above the public class names.

- [ ] **Step 3: Run all new regression tests**

Run:

    /home/ronnie/anaconda3/envs/FL/bin/python -m unittest       tests.test_amp_optimizer       tests.test_text_cache -v

Expected: five tests pass on Ronnie, including the CUDA AMP and two-worker loader paths.

- [ ] **Step 4: Compile all affected packages**

Run:

    /home/ronnie/anaconda3/envs/FL/bin/python -m compileall -q       algorithm module tool tests

Expected: exit code 0 with no output.

- [ ] **Step 5: Commit the cache fix**

Run:

    git add --sparse module/dataset.py tests/test_text_cache.py
    git commit -m "fix: build text caches safely before worker startup"

Expected: one commit containing the cache implementation and its regression tests.

### Task 6: Verify the full AMP-enabled BERT path on Ronnie

**Files:**
- Create outside Git: /home/ronnie/run_main_sent_amp.py
- Use: /home/ronnie/.config/superpowers/worktrees/fairness_fl_code/fix-amp-text-cache
- Verify: result_path/moji/Uniform/FedAvg/BERTCLASSIFIER/2Clients/1.txt

- [ ] **Step 1: Create the AMP validation launcher**

    from main_SENT_CLF import Argparse, main
    import torch

    param_dict = Argparse()
    param_dict["use_amp"] = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    main(
        dataset_name=param_dict["dataset"],
        algorithm=param_dict["algorithm"],
        hypothesis="BERTCLASSIFIER",
        classifier_type="linear",
        device=device,
        param_dict=param_dict,
    )

- [ ] **Step 2: Link the offline BERT model into the worktree**

Run:

    ln -sfn /home/ronnie/fairness_fl_code/bert-base-uncased       /home/ronnie/.config/superpowers/worktrees/fairness_fl_code/fix-amp-text-cache/bert-base-uncased

Expected: the worktree path resolves to the existing local BERT model.

- [ ] **Step 3: Remove only the worktree smoke cache**

Run:

    rm -rf /home/ronnie/.config/superpowers/worktrees/fairness_fl_code/fix-amp-text-cache/dataset/cache/moji

Expected: the next run exercises fresh parent-process cache construction.

- [ ] **Step 4: Run the one-round AMP smoke experiment**

Run from the worktree with CUDA_VISIBLE_DEVICES=0, offline Hugging Face variables, TOKENIZERS_PARALLELISM=false, and PYTHONPATH set to the worktree:

    /home/ronnie/anaconda3/envs/FL/bin/python -u /home/ronnie/run_main_sent_amp.py       -algorithm FedAvg       -dataset moji       -task SENT_CLF       -batch_size 8       -test_batch_size 32       -cuda 0       -max_len 32       -system_data_count 128       -split_strategy Uniform       -communication_round_I 1       -algorithm_epoch_T 1       -num_clients_K 2       -exp_repeat_times 1       -parallel_repeats 1       -checkpoint_save_freq 0       -checkpoint_keep_latest 1       -tb_monitor '{"gradient":false,"embedding":false,"fisher":false,"sharpness":false,"activation":false,"update_stats":false,"client_divergence":false}'

Expected: exit code 0, one cache build per split before workers start, AMP enabled, and no param_groups or EOF error.

- [ ] **Step 5: Verify result and repository state**

Run:

    cat result_path/moji/Uniform/FedAvg/BERTCLASSIFIER/2Clients/1.txt
    git status --short --branch
    git log --oneline -4

Expected: result metrics are present; only ignored runtime outputs and the intentional model symlink may be untracked; the three documentation/implementation commits are visible.
