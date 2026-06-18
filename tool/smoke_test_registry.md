# Smoke Test Registry / 冒烟测试登记
# 记录已通过的冒烟测试，避免重复测试

## Passed Tests / 已通过

| Date | Task | Dataset | Model | Algorithm | Status | Notes |
|------|------|---------|-------|-----------|--------|-------|
| 2026-06-18 | Tabular_CLF | DRUG | ANN | FedAvg | PASS | ACC=0.683, DEO=0.019, SPD=0.027, FR=0.981, HM=0.805 |
| 2026-06-18 | Tabular_CLF | DRUG | LogisticRegression | FedAvg | PASS | ACC=0.486, DEO=0.106, SPD=-0.077, FR=0.894, HM=0.63 |
| 2026-06-18 | SENT_CLF | moji (truncated 200) | BERTCLASSIFIER | FedAvg | PASS | ACC=0.74, DEO=0.0, SPD=0.0, FR=1.0, HM=0.851 |

## Pending Tests / 待测试

| Task | Dataset | Model | Algorithm | Status | Blocker |
|------|---------|-------|-----------|--------|--------|
| IMG_CLF | LFWA+ | CNNCLASSIFIER | FedAvg | PENDING | 内存不足（连750KB都分配不了），已添加懒加载机制。等内存充裕后用 `IMAGE_CACHE_LAZY=1 python main_IMG_CLF.py` 验证 |

## Bugs Fixed During Testing / 测试中修复的 Bug

1. experiment.py: `calculate_communication_cost` 对无 `shared_base` 的模型（LogisticRegression）做 `hasattr` 兼容
2. dataset.py: `_load_shards_stacked` 自动检测旧缓存格式（list）+ 中英双语 DeprecationWarning
3. dataset.py: `CachedImageDataset.__getitem__` 兼容 list/dict 两种缓存格式
4. dataset.py: `CachedImageDataset` 新增 `lazy=True` 懒加载模式，内存紧张时自动启用
5. dataset.py: `CustomizedImageDataset` 支持懒加载，通过 `IMAGE_CACHE_LAZY=1` 环境变量或自动检测

## Test Settings / 测试设置

- CPU only, `CUDA_VISIBLE_DEVICES=""`
- 2 clients, 1 communication round, 1 local epoch
- batch_size=16, test_batch_size=32
- learning_rate=1e-3, optimizer=sgd
- client_parallel=off
