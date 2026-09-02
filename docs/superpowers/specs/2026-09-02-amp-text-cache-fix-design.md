# AMP and Text Cache Reliability Fix

Date: 2026-09-02
Branch: fix/amp-text-cache

## Goal

Make BERT text-classification training work with AMP and multi-worker data loading without operational prewarming or disabling AMP.

## Scope

1. Make BERTCLF_Optimizer compatible with PyTorch GradScaler.
2. Prevent Moji and BIOS text-cache construction from racing across DataLoader workers.
3. Reject incomplete or stale caches whose recorded length does not match the dataset.
4. Fix first-access retrieval after a newly built stacked-text cache.
5. Add focused regression tests and run a CUDA AMP BERT smoke experiment on Ronnie.

Image-cache behavior and broad optimizer refactoring are out of scope.

## Design

### AMP optimizer compatibility

Keep BERTCLF_Optimizer as the public wrapper so its learning-rate schedule, pFedMe logic, and gradient clipping still execute.

Expose the standard optimizer attributes required by GradScaler through explicit properties delegated to the wrapped torch.optim.Optimizer, including param_groups and state. Do not pass the internal optimizer directly to GradScaler, because that would bypass the wrapper's step() behavior.

### Text-cache lifecycle

Text caches are built eagerly in the dataset constructor when no valid cache exists. Dataset construction occurs in the parent process before DataLoader workers are started, so workers only read a completed cache.

Cache validity requires:

- a readable meta.pt;
- meta["total_len"] equal to the requested dataset length;
- the expected number of shard files;
- a supported text-cache format.

The cache writer saves shard files first and publishes meta.pt last via a temporary file plus os.replace. Thus meta.pt remains the completion marker.

Moji and BIOS use shared helper behavior rather than maintaining divergent copies of cache initialization and item lookup. After building, item access goes through the same stacked-cache lookup path as an already existing cache, avoiding dictionary indexing by integer.

## Error handling

Unreadable metadata, missing shards, length mismatch, or unsupported structure is treated as a cache miss and triggers rebuilding. Temporary metadata is not accepted as a completed cache.

## Tests

Tests are written before implementation and must initially fail for the relevant reason.

1. CUDA AMP regression: a small model trained through GradScaler.step(BERTCLF_Optimizer) completes and updates parameters.
2. Cache first-build regression: requesting the first item from a missing cache returns a valid sample rather than raising KeyError.
3. Multi-worker regression: a fresh text dataset can be iterated with more than one DataLoader worker and produces the expected samples.
4. Stale-cache regression: a cache whose total_len differs from the requested dataset is rebuilt.
5. Existing compilation/import checks remain green.
6. Ronnie end-to-end BERT smoke run uses use_amp=True and multiple loader workers.

## Success criteria

- No AttributeError for param_groups under AMP.
- No duplicate cache builders or EOF/corrupt-shard errors under multi-worker loading.
- Cache length matches the current dataset.
- Ronnie's AMP-enabled BERT smoke experiment exits with code 0 and writes result metrics.
