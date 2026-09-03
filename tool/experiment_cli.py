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
    parser.add_argument("-checkpoint_save_freq", type=int, default=1)
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
    return parser
