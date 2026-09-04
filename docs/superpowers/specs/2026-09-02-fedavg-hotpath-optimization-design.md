# FedAvg Hot-Path Optimization Design

## Goal

Speed up the BERT + linear-head FedAvg path without changing its model architecture, client selection, local optimizer, aggregation weights, final metrics, checkpoint policy, or monitoring policy.

## Scope

Only `algorithm/FederatedAverage.py` and focused regression tests are changed. Other algorithms, evaluation frequency, TensorBoard defaults, checkpoint frequency, DataLoader behavior, AMP behavior, and client parallelism are out of scope.

## Design

1. Remove `gc.collect()` from the per-batch loop. Existing client/batch-of-clients and communication-round cleanup points remain.
2. Stop creating one on-disk model per client. A trained client returns a CPU `state_dict`; the existing executor still bounds GPU-resident model copies by its configured parallelism.
3. Aggregate state dictionaries directly with PyTorch tensors on CPU. Floating, complex, and integer tensors use dataset-size weighted means; integer results are cast back to the original buffer dtype, matching the previous load behavior. Boolean buffers retain the first client value.
4. Compute client update tensors only when gradient monitoring is enabled for the current communication step. The update tensors are CPU tensors and are released after deep metric logging.

## Resource Safety

For BERT-base, one FP32 state is about 0.42 GiB. Default text experiments select 10% of 20–40 clients, so returned client states consume about 0.84–1.68 GiB. A monitoring round may temporarily hold another copy of those updates plus one aggregate, keeping expected incremental RAM below 4 GiB against 118 GiB currently available. GPU residency is unchanged by this design. Disk use decreases because 20–40 persistent client model files, roughly 9–17 GiB total, are no longer created.

## Correctness and Tests

Regression tests must demonstrate weighted aggregation, integer-buffer handling, monitoring frequency gating, CPU state return without `torch.save`, and absence of per-batch garbage collection. Existing AMP and text-cache tests must remain green. A CUDA BERT smoke run must complete without client `model.pt` files and report peak RAM, GPU memory, and output metrics.
