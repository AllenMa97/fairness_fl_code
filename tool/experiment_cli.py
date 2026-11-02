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
