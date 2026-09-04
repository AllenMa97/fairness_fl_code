# FedAvg Hot-Path Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove avoidable garbage collection, model disk I/O, NumPy aggregation, and unconditional client-update construction from FedAvg.

**Architecture:** Client training returns CPU state dictionaries through the existing parallel executor. Focused pure helpers perform torch-native weighted aggregation and decide whether the current monitoring step needs client updates.

**Tech Stack:** Python, PyTorch, unittest, CUDA AMP, Hugging Face BERT.

---

### Task 1: Specify aggregation and monitoring behavior

**Files:**
- Create: `tests/test_fedavg_hotpath.py`
- Modify: `algorithm/FederatedAverage.py`

- [ ] Write tests importing `_aggregate_state_dicts` and `_needs_client_updates`.
- [ ] Verify the tests fail because the helpers do not exist.
- [ ] Implement `_aggregate_state_dicts(states, weights)` using CPU PyTorch tensors and `_needs_client_updates(config, step)` using gradient enable/frequency.
- [ ] Verify weighted floating tensors, integer buffers, invalid inputs, and monitoring frequency tests pass.

### Task 2: Remove per-batch collection and client model disk I/O

**Files:**
- Modify: `tests/test_fedavg_hotpath.py`
- Modify: `algorithm/FederatedAverage.py`

- [ ] Add a small two-batch client-training test that patches `gc.collect` and `torch.save` and expects a CPU state dictionary, zero calls to `torch.save`, and zero per-batch collection calls inside the train function.
- [ ] Verify the test fails against the existing implementation.
- [ ] Change `_train_single_client_fedavg` to return `state_dict` with `gpu_seconds` and remove its `basic_path` argument and `torch.save` call.
- [ ] Remove the initial per-client save loop and the aggregation-time `torch.load` loop.
- [ ] Verify the client-training test passes.

### Task 3: Wire torch-native aggregation and conditional updates

**Files:**
- Modify: `tests/test_fedavg_hotpath.py`
- Modify: `algorithm/FederatedAverage.py`

- [ ] Add a regression test exercising aggregation results with client result dictionaries and disabled monitoring.
- [ ] Verify it fails against the old NumPy/disk path.
- [ ] Build `client_states` from executor results, aggregate with client dataset weights, and load the averaged state directly.
- [ ] Snapshot the pre-aggregation state and build updates only when `_needs_client_updates` is true for the current step.
- [ ] Release client states after aggregation and retain round-boundary cleanup.

### Task 4: Verify resources and end-to-end behavior

**Files:**
- Modify: `algorithm/FederatedAverage.py` only if verification exposes a defect.

- [ ] Run `python -m unittest discover -s tests -v`; expect all tests to pass.
- [ ] Run `python -m compileall algorithm/FederatedAverage.py tests/test_fedavg_hotpath.py`; expect exit 0.
- [ ] Run a one-round, two-client CUDA AMP BERT smoke experiment with expensive monitoring and checkpoints disabled.
- [ ] Verify no per-client `model.pt` is written, result metrics exist, GPU peak remains below 30 GiB, process RAM remains below 10 GiB, and disk growth is limited to normal logs/results.
- [ ] Inspect `git diff --check` and commit the tested changes.
