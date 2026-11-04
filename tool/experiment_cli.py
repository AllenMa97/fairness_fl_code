from tool.amp_utils import DEFAULT_EFFECTIVE_BATCH_SIZE, resolve_batch_params


def add_client_parallel_argument(parser):
    """注册 -client_parallel（三种模态主线共用）。

    语义：N=整数表示同一通信轮内 N 个客户端多线程并发；off=串行；
    auto=按显存自动估算（默认）。实现见 tool/client_parallel.py。
    """
    parser.add_argument(
        "-client_parallel", default=None, type=str,
        help="同一通信轮内并发训练多少个客户端：N=整数表示 N 个客户端多线程并发；"
             "off=串行；auto=按显存自动估算（默认）。"
             "整数模式见 tool/client_parallel.py::_run_thread_parallel",
    )
    return parser


def normalize_client_parallel(param_dict):
    """把 -client_parallel 归一化：纯数字转 int，其余转小写字符串；None 直接剔除。"""
    value = param_dict.get("client_parallel")
    if value is None:
        param_dict.pop("client_parallel", None)
        return param_dict
    value = str(value).strip()
    param_dict["client_parallel"] = int(value) if value.isdigit() else value.lower()
    return param_dict


def add_batch_size_arguments(parser, legacy_default):
    """注册批量参数（五个入口共用）。

    新语义（New semantics）：
      -batch_size / batch_size                = 目标有效批 batch size（梯度累积后的等效批），默认 256
      -mini_batch / mini_batch                = 每次前向/反向的 micro-batch（实际读入内存的批）
      -effective_batch_size                   = 目标有效批的显式写法（默认 256，可覆盖）

    兼容（Compatibility）：旧的 `-batch_size` 语义是 micro-batch，因此把它保留为
    `-mini_batch` 的**已废弃别名**；只给 -batch_size 时行为与旧脚本一致。
    `legacy_default` 是该入口历史上的 -batch_size 默认值。
    """
    parser.add_argument(
        "-batch_size", type=int, default=None,
        help="[DEPRECATED alias of mini_batch / 已废弃，等价于 -mini_batch] "
             "旧参数按原语义映射到 micro-batch，保持旧脚本行为不变；"
             "新旧同时出现时以 -mini_batch 为准并告警。"
             f"未指定任何批量参数时按本入口历史默认值 {legacy_default} 作为 micro-batch。"
             "注意：解析后的配置字段 batch_size 表示**目标有效批**（默认 256），与这个旧 CLI 参数名不同。",
    )
    parser.add_argument(
        "-mini_batch", type=int, default=None,
        help="Micro-batch size used for each forward/backward pass (the batch actually read "
             "into memory). Accumulation steps = ceil(batch_size / mini_batch). "
             "每次读入内存做一次前向/反向的 micro-batch 大小；累积步数 = ceil(batch_size / mini_batch)。",
    )
    parser.add_argument(
        "-effective_batch_size", type=int, default=None,
        help="Target effective batch size (batch_size after gradient accumulation), default 256. "
             "It is recorded as `batch_size` in experiment_config; actual effective batch is "
             "mini_batch × accumulation_steps. "
             "算法层梯度累积的目标有效批（默认 256），落盘为 experiment_config 的 batch_size；"
             "真实有效批 = mini_batch × 累积步数。",
    )
    return parser


def normalize_batch_arguments(param_dict, legacy_default):
    """把 CLI 批量参数归一化为 {batch_size(有效批), mini_batch, accumulation_steps, effective_batch_size}。"""
    legacy_batch = param_dict.pop("batch_size", None)
    mini_batch = param_dict.pop("mini_batch", None)
    explicit_effective = param_dict.pop("effective_batch_size", None)

    if mini_batch is None:
        # 只给旧 -batch_size（或什么都没给）：旧语义即 micro-batch
        mini_batch = int(legacy_batch) if legacy_batch is not None else int(legacy_default)
    elif legacy_batch is not None and int(legacy_batch) != int(mini_batch):
        print(f"[WARNING] -batch_size={legacy_batch} 已废弃且与 -mini_batch={mini_batch} 冲突："
              f"以 -mini_batch 为准。")

    if explicit_effective is None:
        # 未显式指定目标有效批：保持旧行为（micro-batch > 256 时有效批 = micro-batch 自身）
        target = max(DEFAULT_EFFECTIVE_BATCH_SIZE, int(mini_batch))
    else:
        target = int(explicit_effective)

    param_dict["mini_batch"] = int(mini_batch)
    param_dict["batch_size"] = target
    return resolve_batch_params(param_dict)


def add_experiment_state_arguments(parser):
    parser.add_argument(
        "-resume",
        action="store_true",
        help="Resume only compatible schema-v2 repeat artifacts",
    )
    parser.add_argument("-exp_repeat_times", type=int, default=3)
    parser.add_argument("-parallel_repeats", type=int, default=1)
    parser.add_argument("-base_seed", type=int, default=42)
    parser.add_argument("-use_amp", choices=("auto", "true", "false"), default="auto")
    parser.add_argument("-partition_cache_root", default="./partition_cache")
    parser.add_argument("-partition_min_size", type=int, default=1)
    parser.add_argument("-partition_max_retries", type=int, default=100)
    parser.add_argument(
        "-partition_repair_policy",
        default="minimum_move_v1",
        choices=["minimum_move_v1"],
    )
    parser.add_argument("-dataloader_num_workers", type=int, default=None)
    # 检查点落盘频率（单位：通信轮）：默认 5。
    # 每轮落盘曾是纯 CPU 串行段（序列化整份 state_dict + fsync），在图像 50 轮 ×
    # 每卡多进程并发的口径下是 GPU 空转的主因之一；放宽到 5 轮一次可显著减少该
    # 开销，且该 key 已在 tool/checkpoint.py._CONFIG_EXCLUDED_KEYS 中，不影响
    # experiment_config_hash，resume 语义保持兼容。末轮另有 _run_single_repeat 的
    # 固定终态检查点兜底，崩溃最多回退到上一个 5 的倍数轮。
    # 注：落盘现已改为异步（tool/checkpoint.py 后台写盘，训练轮次不再等待
    # torch.save + fsync），该频率不再是 GPU 空转主因；保留默认值以控制磁盘 I/O
    # 与后台快照的显存占用。该频率只影响落盘时机，不改变任何指标。
    # 大模型任务（SENT_CLF 的 BERT 单份 state_dict 约 440MB，Scaffold 另存两份
    # 模型大小的控制变量）逐轮落盘会在 save_path 下堆积成百 GB，因此放宽到 10。
    parser.add_argument("-checkpoint_save_freq", type=int, default=10)
    parser.add_argument(
        "-checkpoint_keep_latest",
        type=int,
        default=1,
        help="Compatibility flag; schema v2 always retains one active checkpoint",
    )
    parser.add_argument(
        "-final_artifact_policy",
        default="metrics_only",
        choices=["metrics_only", "global_model", "full_state"],
    )
    parser.add_argument(
        "-redraw_partition_per_repeat",
        action="store_true",
        default=False,
        help="Legacy mode: redraw the data partition for every repeat "
             "(seed = base_seed + 1000*repeat_idx) instead of reusing the single "
             "base_seed partition. Off by default so every repeat and method runs "
             "on one fixed split (thesis protocol); enable only to reproduce or "
             "resume runs launched before the fixed-partition default.",
    )
    return parser
