#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 ACSIncome 的原始 PUMS CSV 压成仓库内可分发的小体积缓存。

Build a compact, redistributable cache for ACSIncome from the raw PUMS CSVs.

背景 / Background
    2018 ACS 1-Year PUMS 的 psam_pusa.csv + psam_pusb.csv 合计约 2.2 GB，
    无法入库分发；而过滤后的表格本身只有 10 列，压缩后约 20 MB。
    ``module/dataset.py::get_ACSIncome_dataset`` 会优先读这份 npz，
    因此 clone 仓库后无需下载原始 CSV 即可复现完全相同的样本顺序、
    预处理与 pickle。

用法 / Usage
    # 从原始 CSV 构建（需要先解压 csv_pus.zip）
    python scripts/build_acsincome_npz.py --csv-dir dataset/ACSIncome \
        --output dataset/ACSIncome/ACSIncome.npz

    # 校验已有 npz 与 CSV 是否逐位一致
    python scripts/build_acsincome_npz.py --csv-dir dataset/ACSIncome \
        --output dataset/ACSIncome/ACSIncome.npz --verify
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# 与 module/dataset.py::get_ACSIncome_dataset 保持完全一致的列与过滤条件
COLUMNS = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP',
           'SEX', 'RAC1P', 'PINCP', 'PWGTP']
CATEGORICAL = ['COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'SEX', 'RAC1P']
NUMERIC = ['AGEP', 'WKHP']


def load_filtered(csv_dir):
    """读取 psam_pus*.csv 并按论文 Appendix B.1 过滤，保持原始行序。

    Read every psam_pus*.csv shard and apply the Retiring Adult population
    filters, preserving the original row order.
    """
    shard_names = sorted(name for name in os.listdir(csv_dir)
                         if name.startswith('psam_pus') and name.endswith('.csv'))
    if not shard_names:
        raise FileNotFoundError(f"{csv_dir} 下未找到 psam_pus*.csv")
    frames = [pd.read_csv(os.path.join(csv_dir, name), usecols=COLUMNS, low_memory=False)
              for name in shard_names]
    df = pd.concat(frames, ignore_index=True)
    df = df[(df['AGEP'] > 16) & (df['PINCP'] > 100) & (df['WKHP'] > 0) & (df['PWGTP'] >= 1)]
    return df.reset_index(drop=True)


def to_arrays(df):
    """转成 dtype 尽量小的数组，压缩后体积接近信息量下限。

    Cast to the smallest safe dtypes so the compressed cache stays small.
    """
    arrays = {}
    for name in NUMERIC + ['PINCP']:
        arrays[name] = df[name].to_numpy(np.float32)
    for name in CATEGORICAL:
        arrays[name] = df[name].to_numpy(np.int16)
    arrays['y'] = (df['PINCP'].to_numpy() > 50000).astype(np.int8)
    arrays['s1'] = (df['SEX'].to_numpy() == 1).astype(np.int8)
    arrays['s2'] = (df['RAC1P'].to_numpy() == 1).astype(np.int8)
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--csv-dir', default='dataset/ACSIncome',
                        help='存放 psam_pus*.csv 的目录')
    parser.add_argument('--output', default='dataset/ACSIncome/ACSIncome.npz',
                        help='输出 npz 路径')
    parser.add_argument('--verify', action='store_true',
                        help='只校验已有 npz 与 CSV 是否一致，不写文件')
    args = parser.parse_args()

    df = load_filtered(args.csv_dir)
    arrays = to_arrays(df)
    print(f"[过滤后] {len(df)} 行, 正类 {arrays['y'].mean():.4f}, "
          f"s1(Male) {arrays['s1'].mean():.4f}, s2(White) {arrays['s2'].mean():.4f}")

    if args.verify:
        if not os.path.exists(args.output):
            print(f"[失败] 找不到 {args.output}")
            return 1
        cached = np.load(args.output)
        for key, value in arrays.items():
            if key not in cached:
                print(f"[失败] npz 缺少字段 {key}")
                return 1
            if not np.array_equal(cached[key], value):
                print(f"[失败] 字段 {key} 与 CSV 不一致")
                return 1
        print(f"[通过] {args.output} 与 CSV 逐位一致")
        return 0

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"[写入] {args.output} ({size_mb:.1f} MB, {len(df)} 行)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
