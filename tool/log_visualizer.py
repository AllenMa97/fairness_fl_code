"""
实验日志可视化工具
从 log_path 下的日志文件中解析训练过程数据，绘制曲线图。

用法：
    # 单日志文件画图
    python tool/log_visualizer.py -f log_path/celeba/Uniform/FedAvg/BERTCLF/20Clients/1.txt

    # 单日志文件画图 + 保存
    python tool/log_visualizer.py -f log_file.txt -o output.png

    # 对比多个实验的 ACC
    python tool/log_visualizer.py -d log_path/celeba/Uniform/FedAvg/ -k ACC --compare

    # 扫描整个 log_path 目录树，汇总所有实验
    python tool/log_visualizer.py --scan log_path/ -o summary.png

    # 只画 loss 曲线
    python tool/log_visualizer.py -f log_file.txt --metrics loss,ACC

    # 画不同数据集上同一算法的对比
    python tool/log_visualizer.py --scan log_path/ --filter-algo FedAvg --metrics ACC,DEO
"""

import re
import os
import sys
import argparse
import matplotlib
matplotlib.use('Agg')  # 无头模式，支持服务器端运行
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


# ─── 正则表达式库 ───

# 每轮 ACC/DEO/SPD（SENT_CLF 格式：无 FR/HM）
RE_METRIC_SENT = re.compile(
    r'ACC:\s*([\d.]+(?:[eE][+-]?\d+)?),\s*DEO:\s*([\d.]+(?:[eE][+-]?\d+)?),\s*SPD:\s*([\d.]+(?:[eE][+-]?\d+)?)'
)

# 每轮 ACC/DEO/SPD/FR/HM（IMG_CLF / Tabular_CLF 格式）
RE_METRIC_FULL = re.compile(
    r'ACC:\s*([\d.]+(?:[eE][+-]?\d+)?),\s*DEO:\s*([\d.]+(?:[eE][+-]?\d+)?),\s*SPD:\s*([\d.]+(?:[eE][+-]?\d+)?),\s*FR:\s*([\d.]+(?:[eE][+-]?\d+)?),\s*HM:\s*([\d.]+(?:[eE][+-]?\d+)?)'
)

# Communication Round: N
RE_ROUND_START = re.compile(
    r'Communication Round:\s*(\d+)(?:/\d+)?;?\s*Select clients:'
)

# Communication Round: N / M; Client: c / K; Epoch: E; Avg One Sample's Loss Over Epoch: loss
RE_CLIENT_LOSS = re.compile(
    r'Communication Round:\s*(\d+)\s*/\s*(\d+);\s*Client:\s*(\d+)\s*/\s*(\d+);\s*Epoch:\s*(\d+);\s*(?:Avg One Sample\'s Loss Over Epoch|Avg Loss Over Epoch):\s*([\d.]+(?:[eE][+-]?\d+)?)'
)

# Global Model testing at Communication N/M
RE_TEST_HEADER = re.compile(
    r'Global Model testing at Communication\s*(\d+)/?\s*(\d+)'
)

# 实验总结
RE_SUMMARY_METRIC = re.compile(
    r'\*{6}\s*(\w+)\s+(\w+)\s+Mean±STD:\s*([\d.]+(?:[eE][+-]?\d+)?)±([\d.]+(?:[eE][+-]?\d+)?)\s*\*{6}'
)

# Algorithm 名称
RE_ALGORITHM = re.compile(r'~{6}\s*Algorithm:\s*(.+?)\s*~{6}')

# Parameter 行
RE_PARAM = re.compile(r'\*{6}\s*(\S+)\s*:\s*(.+?)\s*\*{6}')


# ─── 解析函数 ───

def parse_experiment_name(filepath):
    """从文件路径推断实验名称。格式：.../dataset/split/algo/hypothesis/KClients/ExpNO.txt"""
    parts = filepath.replace('\\', '/').split('/')
    name_parts = []
    # 从后往前找有意义的部分
    idx = len(parts) - 1
    # 去掉文件名
    fname = parts[idx]
    idx -= 1
    if '.' in fname:
        name_parts.insert(0, fname.rsplit('.', 1)[0])
    # Clients
    if idx >= 0 and 'Clients' in parts[idx]:
        name_parts.insert(0, parts[idx])
        idx -= 1
    # hypothesis
    if idx >= 0:
        name_parts.insert(0, parts[idx])
        idx -= 1
    # algorithm
    if idx >= 0:
        name_parts.insert(0, parts[idx])
        idx -= 1
    # split
    if idx >= 0:
        name_parts.insert(0, parts[idx])
        idx -= 1
    # dataset
    if idx >= 0:
        name_parts.insert(0, parts[idx])
        idx -= 1
    return '/'.join(name_parts) if name_parts else os.path.basename(filepath)


def parse_log_file(filepath):
    """
    解析一个日志文件，提取逐轮指标和 client loss。

    Returns:
        dict: {
            'rounds': [0, 1, 2, ...],       # 通信轮次列表
            'loss': [0.5, 0.4, ...],        # 每轮平均 loss（所有 client 的均值）
            'ACC': [0.8, 0.82, ...],
            'DEO': [0.03, 0.028, ...],
            'SPD': [0.02, 0.018, ...],
            'FR': [0.85, 0.87, ...],        # 可能为空
            'HM': [0.84, 0.85, ...],        # 可能为空
            'algorithm': 'FedAvg',
            'params': {'dataset': 'celeba', ...},
            'summary': {'ACC': (mean, std), ...},
        }
    """
    data = {
        'rounds': [],
        'loss': [],
        'ACC': [],
        'DEO': [],
        'SPD': [],
        'FR': [],
        'HM': [],
        'algorithm': '',
        'params': {},
        'summary': {},
    }

    if not os.path.isfile(filepath):
        print(f"[Warning] File not found: {filepath}")
        return data

    current_round = None
    round_client_losses = defaultdict(list)  # round_idx -> [loss1, loss2, ...]

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()

            # --- 算法名称 ---
            m = RE_ALGORITHM.search(line)
            if m:
                data['algorithm'] = m.group(1).strip()

            # --- 参数 ---
            m = RE_PARAM.search(line)
            if m and m.group(1) not in ('now',):
                data['params'][m.group(1)] = m.group(2).strip()

            # --- 实验总结 ---
            m = RE_SUMMARY_METRIC.search(line)
            if m:
                alg, metric, mean_val, std_val = m.groups()
                try:
                    data['summary'][metric] = (float(mean_val), float(std_val))
                except ValueError:
                    pass

            # --- 通信轮开始 ---
            m = RE_ROUND_START.search(line)
            if m:
                current_round = int(m.group(1))

            # --- Client Loss ---
            m = RE_CLIENT_LOSS.search(line)
            if m:
                rnd = int(m.group(1))
                try:
                    loss_val = float(m.group(6))
                except ValueError:
                    loss_val = 0.0
                round_client_losses[rnd].append(loss_val)
                current_round = rnd

            # --- 全局测试结果 ---
            # 先尝试完整格式 (IMG/Tabular)
            m = RE_METRIC_FULL.search(line)
            if m:
                try:
                    ACC, DEO, SPD, FR, HM = map(float, m.groups())
                    # 确定当前 round（取最近匹配到的 round_start 或 test_header）
                    round_idx = current_round if current_round is not None else len(data['ACC'])
                    data['rounds'].append(round_idx)
                    data['ACC'].append(ACC)
                    data['DEO'].append(DEO)
                    data['SPD'].append(SPD)
                    data['FR'].append(FR)
                    data['HM'].append(HM)
                except ValueError:
                    pass
                continue

            # 再尝试 SENT 格式（无 FR/HM）
            m = RE_METRIC_SENT.search(line)
            if m:
                try:
                    ACC, DEO, SPD = map(float, m.groups())
                    round_idx = current_round if current_round is not None else len(data['ACC'])
                    data['rounds'].append(round_idx)
                    data['ACC'].append(ACC)
                    data['DEO'].append(DEO)
                    data['SPD'].append(SPD)
                except ValueError:
                    pass

    # --- 汇总每轮平均 loss ---
    if round_client_losses:
        # 注意：实验可能记录多次（多个 repeat），这里只取最早出现的一组 rounds
        seen_rounds = set(data['rounds'])
        for rnd in sorted(round_client_losses.keys()):
            if rnd in seen_rounds or not seen_rounds:
                pass
        # 按 rounds 对齐 loss
        # 简化处理：如果 rounds 列表和 round_client_losses 长度接近，直接对齐
        if data['rounds'] and round_client_losses:
            # 找到 loss 对应的 rounds
            loss_rounds = sorted(round_client_losses.keys())
            for r in loss_rounds:
                avg_loss = np.mean(round_client_losses[r]) if round_client_losses[r] else 0.0
                data['loss'].append(avg_loss)
            # 如果 loss 数量 ≠ rounds 数量，截断或填充
            target_len = len(data['rounds'])
            if target_len > 0 and len(data['loss']) > target_len:
                data['loss'] = data['loss'][:target_len]
            elif target_len > 0 and len(data['loss']) < target_len:
                data['loss'].extend([0.0] * (target_len - len(data['loss'])))

    return data


def scan_log_directory(root_dir, filter_algo=None, filter_dataset=None, filter_split=None):
    """递归扫描日志目录，返回所有 .txt 文件的解析结果列表。"""
    results = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in sorted(filenames):
            if not fname.endswith('.txt'):
                continue
            filepath = os.path.join(dirpath, fname)

            # 预过滤（通过路径判断）
            norm_path = filepath.replace('\\', '/')
            if filter_algo and filter_algo.lower() not in norm_path.lower():
                continue
            if filter_dataset and filter_dataset.lower() not in norm_path.lower():
                continue
            if filter_split and filter_split.lower() not in norm_path.lower():
                continue

            data = parse_log_file(filepath)
            if data['rounds']:  # 有有效数据才加入
                data['_filepath'] = filepath
                data['_name'] = parse_experiment_name(filepath)
                results.append(data)
                print(f"  Parsed: {data['_name']} ({len(data['rounds'])} rounds)")
    return results


# ─── 绘图函数 ───

def set_chinese_font():
    """尝试设置中文字体"""
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass


def plot_single_experiment(data, metrics=None, output_path=None, title=None):
    """
    画单个实验的曲线图。

    Args:
        data: parse_log_file 的返回值
        metrics: 要画的指标列表，默认 ['loss', 'ACC', 'DEO', 'SPD']
        output_path: 保存路径，None 则显示
        title: 图表标题
    """
    set_chinese_font()

    if metrics is None:
        metrics = ['loss', 'ACC', 'DEO', 'SPD']
    # 过滤掉没有数据的指标
    available_metrics = []
    for m in metrics:
        if m in data and len(data[m]) > 0:
            available_metrics.append(m)

    if not available_metrics:
        print("[Warning] No metrics to plot — this log may not contain per-round data (try summary-only mode)")
        return

    n_plots = len(available_metrics)
    # 如果有 loss，单独一行；其余一行一个
    has_loss = 'loss' in available_metrics
    n_loss_plots = 1 if has_loss else 0
    n_metric_plots = n_plots - n_loss_plots

    if has_loss and n_metric_plots > 0:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        ax_loss, ax_metric = axes[0], axes[1]
    elif has_loss:
        fig, axes = plt.subplots(1, 1, figsize=(12, 5))
        ax_loss = axes
        ax_metric = None
    elif n_metric_plots > 0:
        fig, axes = plt.subplots(1, 1, figsize=(12, 5))
        ax_loss = None
        ax_metric = axes
    else:
        return

    rounds = data['rounds']

    # --- Loss 子图 ---
    if has_loss and ax_loss is not None:
        loss_vals = data['loss']
        x_loss = list(range(1, len(loss_vals) + 1))
        ax_loss.plot(x_loss, loss_vals, 'b-', linewidth=1.2, alpha=0.8)
        ax_loss.set_xlabel('Communication Round')
        ax_loss.set_ylabel('Loss')
        ax_loss.set_title('Training Loss')
        ax_loss.grid(True, alpha=0.3)

    # --- 指标子图 ---
    if ax_metric is not None:
        metric_list = [m for m in available_metrics if m != 'loss']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for i, metric in enumerate(metric_list):
            vals = data[metric]
            x = list(range(1, len(vals) + 1))
            color = colors[i % len(colors)]
            ax_metric.plot(x, vals, '-', linewidth=1.5, color=color, label=metric.upper(), alpha=0.8)
        ax_metric.set_xlabel('Communication Round')
        ax_metric.set_ylabel('Score')
        ax_metric.set_title('Metrics over Rounds')
        ax_metric.legend(loc='best', fontsize=9)
        ax_metric.grid(True, alpha=0.3)

    # 总标题
    if title is None:
        algo = data.get('algorithm', 'Unknown')
        dataset = data.get('params', {}).get('dataset', '')
        title = f'{algo} on {dataset}' if dataset else algo
    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {output_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_comparison(all_data, metric='ACC', output_path=None, title=None):
    """
    对比多个实验的同一指标。

    Args:
        all_data: parse_log_file 返回值列表
        metric: 要对比的指标
        output_path: 保存路径
        title: 图表标题
    """
    set_chinese_font()
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_data)))

    for i, data in enumerate(all_data):
        if metric not in data or len(data[metric]) == 0:
            continue

        vals = data[metric]
        x = list(range(1, len(vals) + 1))
        label = data.get('_name', f'Experiment {i+1}')
        if data.get('algorithm'):
            label = f"{data['algorithm']}"
        # 截断过长的标签
        if len(label) > 40:
            label = label[:37] + '...'

        ax.plot(x, vals, '-', linewidth=1.5, color=colors[i], label=label, alpha=0.85)

    ax.set_xlabel('Communication Round', fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    if title is None:
        title = f'{metric.upper()} Comparison'
    ax.set_title(title, fontsize=13, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {output_path}")
    else:
        plt.show()
    plt.close(fig)


# ─── CLI ───

def main():
    parser = argparse.ArgumentParser(
        description='实验日志可视化工具 — 从 log_path 解析训练过程数据并画图',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tool/log_visualizer.py -f log_path/celeba/Uniform/FedAvg/.../1.txt
  python tool/log_visualizer.py -f log.txt -o result.png --metrics loss,ACC,DEO
  python tool/log_visualizer.py -d log_path/celeba/Uniform/FedAvg/ -k ACC --compare
  python tool/log_visualizer.py --scan log_path/ --filter-algo FedAvg -k ACC --compare
  python tool/log_visualizer.py --scan log_path/ --compare -k ACC,DEO -o summary.png
        """
    )
    # 输入源（三选一）
    parser.add_argument('-f', '--file', type=str, help='单个日志文件路径')
    parser.add_argument('-d', '--dir', type=str, help='单个目录（解析目录下所有 .txt）')
    parser.add_argument('--scan', type=str, help='递归扫描整个 log_path 目录树')

    # 指标选择
    parser.add_argument('--metrics', type=str, default='loss,ACC,DEO,SPD',
                        help='要画的指标，逗号分隔（默认: loss,ACC,DEO,SPD）')

    # 对比模式
    parser.add_argument('--compare', action='store_true', help='对比模式：多个实验画在同一张图上')
    parser.add_argument('-k', '--key', type=str, default='ACC',
                        help='对比模式下对比哪个指标（默认: ACC）')

    # 过滤
    parser.add_argument('--filter-algo', type=str, help='按算法名过滤')
    parser.add_argument('--filter-dataset', type=str, help='按数据集名过滤')
    parser.add_argument('--filter-split', type=str, help='按划分策略过滤')

    # 输出
    parser.add_argument('-o', '--output', type=str, default=None, help='输出图片路径')
    parser.add_argument('-t', '--title', type=str, default=None, help='图表标题')

    args = parser.parse_args()

    metrics = [m.strip() for m in args.metrics.split(',') if m.strip()]

    # 数据源
    if args.scan:
        print(f"[Scan] Scanning {args.scan} ...")
        all_data = scan_log_directory(
            args.scan,
            filter_algo=args.filter_algo,
            filter_dataset=args.filter_dataset,
            filter_split=args.filter_split,
        )
        print(f"[Scan] Found {len(all_data)} experiments with valid data")
    elif args.dir:
        all_data = scan_log_directory(args.dir)
    elif args.file:
        data = parse_log_file(args.file)
        data['_filepath'] = args.file
        data['_name'] = parse_experiment_name(args.file)
        all_data = [data]
    else:
        parser.print_help()
        sys.exit(1)

    if not all_data:
        print("[Error] No valid experiment data found")
        sys.exit(1)

    # 绘图
    if args.compare and len(all_data) > 0:
        plot_comparison(all_data, metric=args.key, output_path=args.output, title=args.title)
    else:
        for data in all_data:
            plot_single_experiment(data, metrics=metrics, output_path=args.output, title=args.title)
            break  # 非对比模式只画第一个


if __name__ == '__main__':
    main()
