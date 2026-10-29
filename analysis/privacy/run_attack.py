#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
R2 属性推断攻击隐私实验 / R2 Property-Inference-Attack privacy experiment
===============================================================================

【威胁模型 Threat model】
诚实但好奇的服务器 (honest-but-curious server)。参与公平性联邦学习的客户端按
协议上传聚合信息，服务器试图从中推断每个客户端的敏感属性构成（COMPAS：受保护
组 African-American 占比 q 的回归 MAE，以及 "African-American 占多数?" 二分类
AUC）。攻击者可利用带真值标注的辅助客户端数据训练攻击模型（属性推断攻击的
标准假设）。

【上传通道 / Upload channels】
  A. group_stats       —— FedFB / FairFed / FedRenyi 类公平性基线上传的群组
                          统计量：各敏感群组样本量 n_s、群组正类率 r_s、
                          (类别×群组) 四格计数。解析攻击 q_hat = n_{s=1}/N
                          无需学习即可精确读出连续构成 → MAE=0, AUC=1。
  B. prototypes        —— PDFFed 上传的 (类别×群组) 类-组原型。两种嵌入来源：
                          local（各客户端随机初始化+本地短训练, 嵌入空间互异,
                          对攻击者最不利）与 shared（全体客户端共享同一在全量
                          数据上训练的 encoder, 嵌入跨客户端可比, 攻击者最有
                          利的理想化上界）。空槽位零填充的朴素实现会泄露
                          "槽位占用位"（极端偏斜时≈构成指示器）。
  B'. *_padded         —— 加固实现：空槽位用该客户端的类均值原型填充，消除
                          占用位泄露。
  C. shared_padded+DP  —— 对 B'(shared) 加高斯噪声（σ = 0.5/1/3 × 原型逐维
                          std），模拟 PDFFed_DP（在最不利于防御的攻击者设定
                          下评估）。效用四层衡量：
                            util_selfcos  单个原型的自余弦保持（个体保真）
                            util_paircorr 成对相似结构 Pearson r（结构保真）
                            util_aggcorr  跨客户端聚合均值原型的 Pearson r
                            util_disc     聚合类判别方向 (类1均值-类0均值) 的余弦
                          （公平性机制实际消费的是聚合量，噪声在聚合层互相抵消）。
  D. fairness_stats      —— LoGoFair 后处理机制必需的"每客户端群组条件预测
                          统计"：各群组正类预测率 pos_rate_g0/g1、TPR/FPR_g0/g1
                          及其平方差 dp_gap/eo_gap（对齐 LoGoFair.py::
                          find_optimal_thresholds_for_local_fairness 与
                          compute_fairness_metrics 实际消费的字段；部署形态下
                          客户端须上报这些比率）。两种模型来源：
                          local（客户端本地模型算自身指标）与 shared（全局模型
                          在客户端数据上评估, 即 LoGoFair 后处理的真实设定,
                          攻击者最有利）。
  E. fedrenyi_group_props —— FedRenyi 的 Renyi 公平性度量 Γ̂ 在初始化与每轮
                          都需要每客户端群组占比 r_bar_k(p) 与数据量权重
                          γ_k=n_k/N（FederatedRenyi.py::get_r_bar_k_p_list /
                          get_gamma_k_list）。r_bar_k(1) ≡ q 本身, 解析攻击
                          与通道 A 同为精确读出。
  参照 / references: raw_oracle（直接看原始样本，上限）、no_info_mean（常数
  预测，下限）。

【代码考古映射表 / Upload-content mapping (evidence: algorithm/*.py)】
  FedProto   → 每客户端 2 个类别原型(无群组结构)+模型参数
               (FederatedProto.py L345-357 原型上传聚合, L398 模型聚合)
               → 通道 c: class_prototypes(-_shared)。
  PraFFL     → 仅 encoder 参数均匀平均; 超网络(偏好/公平折衷所在)私有不上传
               (PraFFL.py L242-262 只收集 encoder_state; praffl_training.py
               L211-214 hypernetwork_state 私有) → 通道 e: 排除, 上传中无
               敏感属性显式信息。
  mFairFL    → 模型参数+n_k; 拉格朗日乘子 λ_k 与群组损失差严格留在本地
               (mFairFL.py L129/L202 本地更新, L316-329 仅聚合模型状态)
               → 通道 e: 排除。
  LoGoFair   → 模型参数(FedAvg)+后处理必需的每客户端群组条件预测统计
               (LoGoFair.py L118-165 群组 pos_rate/TPR/FPR, L571-629 每客户
               端公平指标) → 通道 d: loGo_fairness_stats(-_shared)。
  FedRenyi   → 每客户端群组占比 r_bar_k(p)、n_k/N 权重、(类别×群组)条件统计
               (FederatedRenyi.py L154-171 γ_k, L232-241 r_bar_k_p, L194-214
               Q̂ 所需 j/u/r 统计, L355-369 初始化消费) → 通道 a/E 行:
               fedrenyi_group_props。

【设置 Setup】
COMPAS (ProPublica compas-scores-two-years.csv)。预处理逐行对齐本仓库
module/dataset.py::get_COMPAS_dataset 的 mask_s1 分支（同样的 ProPublica 过滤、
random.seed(42) 取前 4800、OneHot(mask_s1_s2 字段)+追加 s2(性别)+PCA-64，
种族 s1 不进入特征）。两个异构 regime：Dirichlet(α=0.1) 极端群组偏斜 与
Dirichlet(α=1.0) 中等偏斜；每次 20 客户端 × 60 次独立划分（丢弃样本数 <10 的
客户端），每个客户端的本地 encoder 随机初始化并独立短训练以模拟异构嵌入空间。
攻击模型：Ridge（占比回归 MAE）+ LogisticRegression（多数判定 AUC），
5 次 80/20 随机划分（各通道共享同一组划分，可配对比较）→ mean±std。

【结论摘要 Conclusion summary】（2026-10-19 实测于 192.168.1.30，数值以
results/privacy_attack_results.csv 为准；MAE=敏感占比回归误差，AUC=多数组判定）
  1. 基线群组统计量（FedFB/FairFed 类上传的 n_s/r_s/四格计数）被解析攻击
     q_hat=n1/N 精确读出：两个 regime 均 MAE=0.000 / AUC=1.000 —— 计数一旦
     上传即构成完美泄露。
  2. 类-组原型泄露显著更低且取决于实现细节：
     - 中等偏斜(α=1.0)：AUC 0.517-0.613（接近随机 0.5），MAE 0.235-0.261
       ≈ 无信息参照 0.248；
     - 极端偏斜(α=0.1)：朴素零填充实现的泄露主要来自"槽位占用位"
       (AUC 0.938/0.945)，padded 实现即可消除（local: AUC 0.529）；
     - 攻击者最有利组合（共享 encoder + 极端偏斜 + padded）：AUC=0.862,
       MAE=0.261，仍显著劣于统计量的 1.000/0.000。
  3. PDFFed_DP（shared+padded+高斯噪声）：攻击随 σ 单调下降（极端偏斜下
     AUC 0.862→0.826/0.797/0.692，MAE 0.261→0.288/0.311/0.374，向无信息
     0.440 移动）；而公平性机制实际消费的聚合判别方向几乎无损
     （util_disc≥0.994），主要代价是个体原型保真（selfcos 0.96→0.50）。
  4. FedRenyi（扩展）：其 Renyi 公平性度量必需上传的每客户端群组占比
     r_bar_k(p) 中 r_bar_k(1)≡q 本身（FederatedRenyi.py get_r_bar_k_p_list,
     L232-241/L355-369），解析攻击与群组统计量同为精确读出：
     两 regime 均 MAE=0.000 / AUC=1.000 —— 与 FedFB/FairFed 同属完美泄露。
  5. FedProto 类原型（扩展，通道 c）：本地异构 encoder（FedProto 的真实
     设定，各客户端嵌入空间互异）下几乎不泄露——MAE 0.428/0.253 ≈ 无信息
     参照 0.440/0.248，AUC 0.577/0.500（α=0.1/1.0）；但在共享 encoder 的
     攻击者上界设定下泄露显著（MAE 0.141/0.095，AUC 0.984/0.961）——嵌入
     空间可比时，类别原型中携带的"客户端数据在特征空间的位置"即可反推群组
     构成。泄露程度完全由 encoder 可比性决定。
  6. LoGoFair 公平性统计（扩展，通道 d）：中等偏斜下接近随机（AUC
     0.580/0.552，MAE≈无信息 0.248）；极端偏斜下多数组判定泄露强（AUC
     0.914/0.943）而连续占比仍接近无信息（MAE 0.244/0.156 vs 0.440）——
     机理与零填充占用位同源：极端偏斜时统计量向量中"哪个群组为空/取默认
     值"本身构成占多数群的指示器。
  → 结论：公平性基线的群组统计量在信息论层面即等价于公开敏感构成；
     PDFFed 原型通道泄露低一个量级以上，配合 padded 实现 + 适度 DP 噪声
     可将属性推断压至接近随机水平，同时保持聚合效用。新扩展的五基线中：
     FedRenyi 的必需上传=q 本身（完美泄露）；FedProto 类原型与 LoGoFair
     公平性统计在真实（本地/异构）设定下泄露接近无信息下限或仅泄露多数
     判定，但在嵌入可比/极端偏斜的攻击者有利条件下仍显著泄露；
     PraFFL 与 mFairFL 的上传仅含模型参数(+n_k)，无敏感属性显式上传，
     不构成针对敏感属性的显式泄露通道（channel_mapping 见 summary.json）。

【范围声明 Scope】
本实验是与联邦训练解耦的"信息论层面"泄露对比：只针对"上传内容本身"的可推断
性做模拟（不跑联邦训练），不是端到端训练过程中的成员推断/属性推断 MIA，
请勿过度解读。基线统计量通道同样可以加 DP，但 FedFB 类机制的核心输入恰是
精确群组计数（后验阈值选择），加 DP 会直接损害公平性机制本身。
This is a training-decoupled, information-level leakage comparison on the
uploaded objects themselves (no FL training is run); it is NOT an end-to-end
training-time MIA.

运行 / Run:
  /root/miniconda3/envs/federated/bin/python /root/fl/privacy_attack/run_attack.py
产出 / Outputs:
  /root/fl/privacy_attack/results/privacy_attack_results.csv
  /root/fl/privacy_attack/results/privacy_attack_bars.png
  /root/fl/privacy_attack/results/summary.json
"""
import os

# 单线程 BLAS：避免 joblib threading 后端与 OpenBLAS 在多核机上互相争抢
# Single-thread BLAS so joblib threading workers do not fight over cores
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import warnings

import csv
import json
import random
import sys
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 绘图拆到 analysis/plot_privacy_bars.py：BAR_ORDER 只有一份，本脚本跑完实验
# 与事后从 CSV 重画走同一条代码路径。
# The rendering lives in analysis/plot_privacy_bars.py so that the experiment
# and a standalone redraw of the CSV share one BAR_ORDER and one layout.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plot_privacy_bars

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ------------------------------ 配置 / Config ------------------------------
DATA_CSV = "/root/fl/dataset/COMPAS/compas-scores-two-years.csv"
OUT_DIR = "/root/fl/privacy_attack/results"
GLOBAL_SEED = 20261019
NUM_CLIENTS = 20        # 每次划分的客户端数 / clients per partition
NUM_PARTITIONS = 60     # 每个 regime 的独立划分次数 / independent partitions per regime
REGIMES = [(0.1, "extreme"), (1.0, "moderate")]  # Dirichlet α → 异构程度
KEEP_MIN_SAMPLES = 10   # 丢弃过小客户端 / drop clients smaller than this
EMB_DIM = 16            # 本地 encoder 隐层维度 / local encoder hidden dim
MLP_MAX_ITER = 25       # encoder 短训练轮数 / short local training
N_ATTACK_SPLITS = 5     # 攻击模型 80/20 重复次数 / attack CV repeats
DP_SIGMA_MULTS = (0.5, 1.0, 3.0)  # DP 噪声强度（× 非空原型逐维 std）
N_JOBS = 8              # joblib threading workers


# ------------------ COMPAS 预处理（对齐仓库 mask_s1 分支） ------------------
def _mask_rows(row):
    """mask_s1_s2 特征字段切片，逐字对齐 dataset.py L2180 的 row_mask_s1_s2_copy。
    Field slicing identical to dataset.py L2180 (mask_s1_s2 variant, race/sex removed)."""
    return row[8:9] + row[10:11] + row[12:16] + row[22:23] + row[39:41] + row[48:49]


def load_compas(data_csv):
    """复现 module/dataset.py::get_COMPAS_dataset 的 mask_s1 分支预处理：
    同样的 ProPublica 过滤 → random.seed(42) 打乱取前 4800 →
    OneHot(训练+测试字段并集, 与仓库一致) + 追加 s2(性别)数值列 → PCA-64。
    s1(种族, African-American=1) 只作敏感属性/划分依据, 不进入特征。"""
    with open(data_csv) as f:
        raw_all = []
        for i, row in enumerate(csv.reader(f)):
            if i == 0:
                continue
            if row[15] != '' and row[24] != '' and row[22] != '' and row[40] != '':
                if 30 >= int(row[15]) >= -30:      # days_b_screening_arrest
                    if int(row[24]) != -1:         # is_recid
                        if row[22] != "0":         # c_charge_degree
                            if row[40] != 'N/A':   # score_text
                                if row[9] == "African-American" or row[9] == "Caucasian":
                                    raw_all.append(row)
    random.seed(42)
    random.shuffle(raw_all)
    train, test = raw_all[:4800], raw_all[4800:]   # 与仓库一致的训练/测试切分

    s1 = [1 if r[9] == "African-American" else 0 for r in train]
    s2 = [1 if r[5] == "Male" else 0 for r in train]
    y = [1 if int(r[24]) == 0 else 0 for r in train]  # is_recid 取反, 对齐仓库

    enc = OneHotEncoder()  # 仓库在 train+test 的字段并集上 fit
    enc.fit([_mask_rows(r) for r in train] + [_mask_rows(r) for r in test])
    X_oh = np.asarray(enc.transform([_mask_rows(r) for r in train]).todense())
    X = np.append(X_oh, np.asarray([s2], dtype=np.float64).T, axis=1)  # mask_s1: 追加性别
    n_comp = min(64, X.shape[1])
    pca = PCA(n_components=n_comp)
    X = pca.fit_transform(X)
    print(f"[data] filtered={len(raw_all)}, train={len(train)}, "
          f"onehot+sex dims={X_oh.shape[1] + 1}, pca={n_comp}")
    return np.asarray(X, dtype=np.float32), np.asarray(s1), np.asarray(y)


# ------------------- 敏感群组 Dirichlet 划分（简化版） -------------------
def partition_sgroup(s1, rng, alpha, n_clients=NUM_CLIENTS, retries=30):
    """按敏感属性分组做 Dirichlet(α) 划分（简化自 module/partition.py 的
    label-Dirichlet 逻辑：按类(此处=敏感组)抽取比例向量再分配索引，
    重试以尽量保证每个客户端不少于 KEEP_MIN_SAMPLES）。"""
    best, best_deficit = None, None
    for _ in range(retries):
        clients = [[] for _ in range(n_clients)]
        for sval in (0, 1):
            ids = np.flatnonzero(s1 == sval)
            rng.shuffle(ids)
            prof = rng.dirichlet(np.full(n_clients, alpha))
            prof = prof / prof.sum()
            cuts = np.cumsum(prof) * len(ids)
            start = 0
            for c in range(n_clients):
                if c == n_clients - 1:
                    end = len(ids)
                else:
                    end = min(len(ids), max(start, int(round(cuts[c]))))
                clients[c].extend(ids[start:end].tolist())
                start = end
        deficit = sum(max(0, KEEP_MIN_SAMPLES - len(v)) for v in clients)
        if best_deficit is None or deficit < best_deficit:
            best, best_deficit = clients, deficit
        if deficit == 0:
            break
    return [np.asarray(v) for v in best]


# ------------------------ 单客户端上传内容构造 ------------------------
def _fairness_stats_vector(preds, yl, sl):
    """LoGoFair 后处理消费的每客户端群组条件预测统计（LoGoFair.py L118-165 的
    pos_rate/TPR/FPR 与 L192 的 dp/eo 指标, 部署形态下客户端须上报的比率）:
    [pos_rate_g0, pos_rate_g1, tpr_g0, tpr_g1, fpr_g0, fpr_g1, dp_gap, eo_gap]。
    空群组的比率退化为本客户端整体预测正率/0（该组无样本时机制本就取默认值）。"""
    preds = np.asarray(preds, dtype=np.float64)
    overall = float(preds.mean()) if len(preds) else 0.0
    pos, neg = (yl == 1), (yl == 0)
    pr, tpr, fpr = {}, {}, {}
    for g in (0, 1):
        mg = (sl == g)
        pr[g] = float(preds[mg].mean()) if mg.any() else overall
        mp, mn = mg & pos, mg & neg
        tpr[g] = float(preds[mp].mean()) if mp.any() else 0.0
        fpr[g] = float(preds[mn].mean()) if mn.any() else 0.0
    dp_gap = (pr[0] - pr[1]) ** 2
    eo_gap = (tpr[0] - tpr[1]) ** 2 + (fpr[0] - fpr[1]) ** 2
    return np.asarray([pr[0], pr[1], tpr[0], tpr[1], fpr[0], fpr[1],
                       dp_gap, eo_gap], dtype=np.float64)


def build_client_payload(idx, X, s1, y, seed, shared=None):
    """构造一个客户端的上传内容（本实验核心对象）：
    - 群组统计量 n_s / r_s / 四格计数（基线通道, 攻击目标）
    - 类-组原型（4 槽位 × EMB_DIM, PDFFed 通道）：
        proto_fill / proto_sh_fill     : 空槽位零填充（朴素实现, 占用位可泄露）
        proto_pad  / proto_sh_pad      : 空槽位用本地类均值填充（加固, 无占用位）
    - 类原型（2 槽位 × EMB_DIM, FedProto 通道, 无群组结构）：
        class_proto / class_proto_sh
    - LoGoFair 后处理统计（8 维群组条件预测统计, 通道 D）：
        logo_stats（客户端本地模型）/ logo_stats_sh（全局模型, 后处理真实设定）
    真值 q = 该客户端受保护组占比。"""
    Xl, yl, sl = X[idx], y[idx], s1[idx]
    n0 = int((sl == 0).sum())
    n1 = int((sl == 1).sum())
    counts = [int(((yl == yy) & (sl == ss)).sum()) for yy in (0, 1) for ss in (0, 1)]
    r0 = float(yl[sl == 0].mean()) if n0 else 0.0
    r1 = float(yl[sl == 1].mean()) if n1 else 0.0
    q = n1 / float(n0 + n1)

    # 本地小 encoder：随机初始化 + 短训练（模拟各客户端独立训练的异构嵌入空间）
    mlp = MLPClassifier(hidden_layer_sizes=(EMB_DIM,), max_iter=MLP_MAX_ITER,
                        random_state=int(seed) % (2 ** 31 - 1)).fit(Xl, yl)

    def _protos(H):
        """给定嵌入 H, 计算 4 槽位 (y×s) 原型: 零填充版、类均值填充版与占用位。"""
        fill = np.zeros(4 * EMB_DIM)
        pad = np.zeros(4 * EMB_DIM)
        ok = []
        class_mean = {yy: (H[yl == yy].mean(axis=0) if (yl == yy).any() else H.mean(axis=0))
                      for yy in (0, 1)}
        gmean = H.mean(axis=0)
        k = 0
        for yy in (0, 1):
            for ss in (0, 1):
                m = (yl == yy) & (sl == ss)
                if m.any():
                    v = H[m].mean(axis=0)
                    ok.append(True)
                else:
                    v = None            # 空槽位: fill=0, pad=类均值(退化则全局均值)
                    v_pad = class_mean[yy] if (yl == yy).any() else gmean
                    ok.append(False)
                if v is not None:
                    pad[k * EMB_DIM:(k + 1) * EMB_DIM] = v
                    fill[k * EMB_DIM:(k + 1) * EMB_DIM] = v
                else:
                    fill[k * EMB_DIM:(k + 1) * EMB_DIM] = 0.0
                    pad[k * EMB_DIM:(k + 1) * EMB_DIM] = v_pad
                k += 1
        return fill, pad, ok

    def _class_protos(H):
        """FedProto 通道: 每客户端上传 2 个类别原型(无群组结构),
        对齐 FederatedProto.py L249/L256 的 label_*_prototype = 特征均值。"""
        out = np.zeros(2 * EMB_DIM)
        for j, yy in enumerate((0, 1)):
            m = (yl == yy)
            out[j * EMB_DIM:(j + 1) * EMB_DIM] = H[m].mean(axis=0) if m.any() else H.mean(axis=0)
        return out

    H_loc = np.maximum(Xl @ mlp.coefs_[0] + mlp.intercepts_[0], 0.0)  # 第一隐层=嵌入
    proto_fill, proto_pad, slot_ok = _protos(H_loc)
    class_proto = _class_protos(H_loc)
    # LoGoFair 通道: 客户端本地模型算自身群组条件预测统计
    logo_stats = _fairness_stats_vector(mlp.predict(Xl), yl, sl)
    if shared is not None:
        sh_mlp = shared
        W1, b1 = sh_mlp.coefs_[0], sh_mlp.intercepts_[0]
        sh_fill, sh_pad, _ = _protos(np.maximum(Xl @ W1 + b1, 0.0))
        class_proto_sh = _class_protos(np.maximum(Xl @ W1 + b1, 0.0))
        # LoGoFair 后处理的真实设定: 全局模型在客户端数据上的预测统计
        logo_stats_sh = _fairness_stats_vector(sh_mlp.predict(Xl), yl, sl)
    else:
        sh_fill = sh_pad = class_proto_sh = logo_stats_sh = None
    return dict(q=q, n0=n0, n1=n1, r0=r0, r1=r1, counts=counts, slot_ok=slot_ok,
                proto_fill=proto_fill, proto_pad=proto_pad,
                proto_sh_fill=sh_fill, proto_sh_pad=sh_pad,
                class_proto=class_proto, class_proto_sh=class_proto_sh,
                logo_stats=logo_stats, logo_stats_sh=logo_stats_sh)


# --------------------------- 攻击模型与评估 ---------------------------
def cv_attack(features, q, maj, n_splits=N_ATTACK_SPLITS, seed0=GLOBAL_SEED + 100):
    """重复 80/20 划分：Ridge 推断敏感占比(MAE) + LR 判定多数组(AUC)。
    各通道共享同一组划分 → 可配对比较。"""
    maes, aucs = [], []
    n = len(q)
    for k in range(n_splits):
        rng = np.random.default_rng(seed0 + k)
        perm = rng.permutation(n)
        n_te = n // 5
        te, tr = perm[:n_te], perm[n_te:]
        sc = StandardScaler().fit(features[tr])
        ftr, fte = sc.transform(features[tr]), sc.transform(features[te])
        ridge = Ridge(alpha=1.0).fit(ftr, q[tr])
        maes.append(mean_absolute_error(q[te], ridge.predict(fte)))
        if len(np.unique(maj[te])) < 2:
            aucs.append(np.nan)  # 测试折只有一个类时跳过 AUC
        else:
            lr = LogisticRegression(max_iter=2000).fit(ftr, maj[tr])
            aucs.append(roc_auc_score(maj[te], lr.predict_proba(fte)[:, 1]))
    return (float(np.mean(maes)), float(np.std(maes)),
            float(np.nanmean(aucs)), float(np.nanstd(aucs)))


def _row_mask(slot_ok, require_all=False):
    """参与的行：padded 通道所有槽位都有值 → 全部行；fill 通道 → 至少一个非空槽位。"""
    if require_all:
        return np.ones(slot_ok.shape[0], dtype=bool)
    return slot_ok.any(axis=1)


def proto_utility(clean, noisy, slot_ok, rng, require_all=False):
    """三层效用保持度（对非空/全部原型计算）：
    selfcos=自余弦, paircorr=成对相似结构 Pearson r, aggcorr=跨客户端聚合
    均值原型（服务器实际消费的统计量）的 Pearson r。"""
    rows = _row_mask(slot_ok, require_all)
    C, N = clean[rows], noisy[rows]

    def _cos_rows(A, B):
        num = (A * B).sum(axis=1)
        den = np.maximum(np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1), 1e-12)
        return num / den

    selfcos = float(_cos_rows(C, N).mean())

    idxs = rng.choice(len(C), size=min(200, len(C)), replace=False)
    Cc, Nc = C[idxs], N[idxs]
    Cn = Cc / np.maximum(np.linalg.norm(Cc, axis=1, keepdims=True), 1e-12)
    Nn = Nc / np.maximum(np.linalg.norm(Nc, axis=1, keepdims=True), 1e-12)
    iu = np.triu_indices(len(idxs), k=1)
    paircorr = float(np.corrcoef((Cn @ Cn.T)[iu], (Nn @ Nn.T)[iu])[0, 1])

    D = clean.shape[1] // 4
    agg_c = np.stack([C[:, k * D:(k + 1) * D].mean(axis=0) for k in range(4)])
    agg_n = np.stack([N[:, k * D:(k + 1) * D].mean(axis=0) for k in range(4)])
    aggcorr = float(np.corrcoef(agg_c.ravel(), agg_n.ravel())[0, 1])
    # 聚合类判别方向保持: (类1槽位均值 - 类0槽位均值), 公平性机制实际依赖的量
    disc_c = (agg_c[2] + agg_c[3]) / 2.0 - (agg_c[0] + agg_c[1]) / 2.0
    disc_n = (agg_n[2] + agg_n[3]) / 2.0 - (agg_n[0] + agg_n[1]) / 2.0
    disc = float(disc_c @ disc_n /
                 max(np.linalg.norm(disc_c) * np.linalg.norm(disc_n), 1e-12))
    return selfcos, paircorr, aggcorr, disc


def _sigma_base(protos, slot_ok, require_all):
    """噪声基准：非空槽位条目的逐维 std（padded 通道用全部条目）。"""
    D = protos.shape[1] // 4
    rows = _row_mask(slot_ok, require_all)
    sub = protos[rows]
    if not require_all:
        mask = np.repeat(slot_ok[rows], D, axis=1)
        sub = sub[mask]
    return float(sub.std())


# ------------------------------- 产出 --------------------------------
CSV_COLS = ["regime", "method", "mae_mean", "mae_std", "auc_mean", "auc_std",
            "util_selfcos", "util_paircorr", "util_aggcorr", "util_disc", "note"]


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLS)
        for r in rows:
            w.writerow([r.get(c, "") for c in CSV_COLS])


def read_csv_rows(path):
    """读取旧 CSV（dict 行列表）, 供追加合并; 不存在/损坏则返回空。"""
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [dict(zip(CSV_COLS, row)) for row in csv.reader(f)][1:]


def merge_and_write_csv(all_rows, path, append_methods):
    """保留旧行原样, 仅把 append_methods 中的新通道行追加/替换进去
    （旧通道行绝不重写; 同名新通道行在重跑时以本次结果为准）。"""
    old = [r for r in read_csv_rows(path) if r.get("method") not in append_methods]
    fresh = [r for r in all_rows if r.get("method") in append_methods]
    write_csv(old + fresh, path)


def run_regime(alpha, X, s1, y, shared):
    """单个异构 regime：划分 → 构造上传 → 三类攻击 + DP 防御 + 效用。"""
    # 1) 重复 Dirichlet 划分 → 攻击样本池
    jobs = []
    for p in range(NUM_PARTITIONS):
        rng = np.random.default_rng(GLOBAL_SEED + 1000 * int(alpha * 10) + p)
        clients = partition_sgroup(s1, rng, alpha)
        for c, idx in enumerate(clients):
            if len(idx) < KEEP_MIN_SAMPLES:
                continue
            jobs.append((idx, GLOBAL_SEED * 977 + p * 131 + c))
    print(f"[regime {alpha}] kept {len(jobs)} client records "
          f"({NUM_PARTITIONS} partitions x {NUM_CLIENTS} clients, alpha={alpha})")

    recs = Parallel(n_jobs=N_JOBS, backend="threading")(
        delayed(build_client_payload)(idx, X, s1, y, seed, shared)
        for idx, seed in jobs)

    q = np.asarray([r["q"] for r in recs])
    maj = (q > 0.5).astype(int)
    P_fill = np.stack([r["proto_fill"] for r in recs])
    P_pad = np.stack([r["proto_pad"] for r in recs])
    P_sh_fill = np.stack([r["proto_sh_fill"] for r in recs])
    P_sh_pad = np.stack([r["proto_sh_pad"] for r in recs])
    C_proto = np.stack([r["class_proto"] for r in recs])
    C_proto_sh = np.stack([r["class_proto_sh"] for r in recs])
    F_logo = np.stack([r["logo_stats"] for r in recs])
    F_logo_sh = np.stack([r["logo_stats_sh"] for r in recs])
    slot_ok = np.asarray([r["slot_ok"] for r in recs], dtype=bool)
    print(f"[regime {alpha}] q mean={q.mean():.3f} std={q.std():.3f}, "
          f"AA-majority rate={maj.mean():.3f}")

    rows, by_key = [], {}

    def add(key, note, mae_mean=None, mae_std=0.0, auc_mean=None, auc_std=0.0,
            util_selfcos="", util_paircorr="", util_aggcorr="", util_disc=""):
        r = dict(regime=alpha, method=key, mae_mean=mae_mean, mae_std=mae_std,
                 auc_mean=auc_mean, auc_std=auc_std, util_selfcos=util_selfcos,
                 util_paircorr=util_paircorr, util_aggcorr=util_aggcorr,
                 util_disc=util_disc, note=note)
        rows.append(r)
        by_key[key] = r

    # 2) 通道 A：群组统计量 → 解析攻击（无需学习, 精确读出）
    n0s = np.asarray([r["n0"] for r in recs], dtype=np.float64)
    n1s = np.asarray([r["n1"] for r in recs], dtype=np.float64)
    q_hat = n1s / (n0s + n1s)
    add("group_stats",
        "FedFB/FairFed 类基线上传 n_s/r_s/四格计数; 解析攻击 q_hat=n1/N "
        "(analytic attack on uploaded group statistics)",
        float(np.abs(q_hat - q).mean()), 0.0, float(roc_auc_score(maj, q_hat)), 0.0)

    # 2b) 通道 E：FedRenyi 上传的每客户端群组占比 r_bar_k(p)（=q 本身）→ 解析攻击
    add("fedrenyi_group_props",
        "FedRenyi 必需上传的每客户端群组占比 r_bar_k(p), r_bar_k(1)≡q "
        "(FederatedRenyi.py get_r_bar_k_p_list) → 解析攻击同 group_stats "
        "(analytic attack on uploaded per-client group proportions)",
        float(np.abs(q_hat - q).mean()), 0.0, float(roc_auc_score(maj, q_hat)), 0.0)

    # 3) 参照行：原始样本（上限）与无信息（下限）
    add("raw_oracle", "直接观测原始样本, 攻击上限 (oracle upper bound)", 0.0, 0.0, 1.0, 0.0)
    add("no_info_mean", "无上传信息的常数预测参照 (no-information reference)",
        float(np.abs(q - q.mean()).mean()), 0.0, 0.5, 0.0)

    # 4) 通道 B：类-组原型（local/shared 嵌入 × 零填充/padded）→ 训练攻击模型
    for key, P, require_all, note in [
            ("prototypes", P_fill, False,
             "PDFFed 原型(local 异构 encoder, 空槽位零填充) (local encoders, zero-fill)"),
            ("prototypes_padded", P_pad, True,
             "PDFFed 原型(local, 类均值填充, 无占用位泄露) (local encoders, padded)"),
            ("prototypes_shared", P_sh_fill, False,
             "共享 encoder 原型(攻击者最有利, 零填充) (shared encoder, zero-fill)"),
            ("prototypes_shared_padded", P_sh_pad, True,
             "共享 encoder 原型(攻击者最有利, 类均值填充) (shared encoder, padded)")]:
        m_, s_, a_, sa_ = cv_attack(P, q, maj)
        add(key, note, m_, s_, a_, sa_)

    # 4b) 通道 C（新）：FedProto 类别原型（无群组结构）
    for key, P, note in [
            ("class_prototypes", C_proto,
             "FedProto 类原型(local 异构 encoder, 2×EMB_DIM, 无群组结构) "
             "(FedProto class prototypes, local encoders)"),
            ("class_prototypes_shared", C_proto_sh,
             "FedProto 类原型(共享 encoder, 攻击者最有利) "
             "(FedProto class prototypes, shared encoder)")]:
        m_, s_, a_, sa_ = cv_attack(P, q, maj)
        add(key, note, m_, s_, a_, sa_)

    # 4c) 通道 D（新）：LoGoFair 后处理统计（每客户端群组条件预测统计）
    for key, F, note in [
            ("loGo_fairness_stats", F_logo,
             "LoGoFair 后处理必需的每客户端群组条件预测统计(本地模型计算) "
             "(LoGoFair post-processing stats, client-local model)"),
            ("loGo_fairness_stats_shared", F_logo_sh,
             "LoGoFair 后处理统计(全局模型在客户端数据上评估, 机制真实设定, "
             "攻击者最有利) (LoGoFair post-processing stats, global model)")]:
        m_, s_, a_, sa_ = cv_attack(F, q, maj)
        add(key, note, m_, s_, a_, sa_)

    # 5) 通道 C：shared+padded 原型 + 高斯噪声（模拟 PDFFed_DP, 最不利防御设定）
    sigma_base = _sigma_base(P_sh_pad, slot_ok, require_all=True)
    for i, mult in enumerate(DP_SIGMA_MULTS):
        rng = np.random.default_rng(GLOBAL_SEED + 555 + i + 17 * int(alpha * 10))
        noisy = P_sh_pad + rng.normal(0.0, mult * sigma_base, P_sh_pad.shape)
        m_dp, s_dp, a_dp, sa_dp = cv_attack(noisy, q, maj)
        sc, pc, ac, dc = proto_utility(P_sh_pad, noisy, slot_ok, rng, require_all=True)
        add(f"proto_shared_dp_{mult}",
            f"shared+padded 原型+高斯噪声 sigma={mult}x std_p (PDFFed_DP simulation)",
            m_dp, s_dp, a_dp, sa_dp, f"{sc:.4f}", f"{pc:.4f}", f"{ac:.4f}", f"{dc:.4f}")
        print(f"[regime {alpha}][dp sigma={mult}x] attack MAE={m_dp:.4f} AUC={a_dp:.4f} | "
              f"selfcos={sc:.4f} paircorr={pc:.4f} aggcorr={ac:.4f} disc={dc:.4f}")

    print(f"[regime {alpha}] group_stats: MAE={by_key['group_stats']['mae_mean']:.4f} "
          f"AUC={by_key['group_stats']['auc_mean']:.4f}; fedrenyi_group_props: "
          f"MAE={by_key['fedrenyi_group_props']['mae_mean']:.4f} "
          f"AUC={by_key['fedrenyi_group_props']['auc_mean']:.4f}")
    for key in ("prototypes", "prototypes_padded", "prototypes_shared",
                "prototypes_shared_padded", "class_prototypes",
                "class_prototypes_shared", "loGo_fairness_stats",
                "loGo_fairness_stats_shared"):
        print(f"[regime {alpha}] {key}: MAE={by_key[key]['mae_mean']:.4f} "
              f"AUC={by_key[key]['auc_mean']:.4f}")
    return rows, by_key, dict(n=len(recs), q_std=float(q.std()))


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    X, s1, y = load_compas(DATA_CSV)
    print(f"[data] X={X.shape}, s1(AA) mean={s1.mean():.3f}, y mean={y.mean():.3f}")

    # 共享 encoder（攻击者最有利设定: 全体客户端嵌入空间可比的上界情形）
    shared_mlp = MLPClassifier(hidden_layer_sizes=(EMB_DIM,), max_iter=MLP_MAX_ITER,
                               random_state=GLOBAL_SEED % (2 ** 31 - 1)).fit(X, y)
    shared = shared_mlp

    # 新增通道（重跑时替换 CSV 中同名行, 旧通道行保持原样不动）
    NEW_METHODS = {"fedrenyi_group_props", "class_prototypes",
                   "class_prototypes_shared", "loGo_fairness_stats",
                   "loGo_fairness_stats_shared"}

    all_rows, summary = [], {}
    # 保留旧 summary 的已有键, 追加新通道键
    sum_path = os.path.join(OUT_DIR, "summary.json")
    if os.path.exists(sum_path):
        try:
            with open(sum_path, encoding="utf-8") as f:
                summary = json.load(f)
        except Exception:
            summary = {}
    if not isinstance(summary, dict):
        summary = {}

    for alpha, name in REGIMES:
        rows, by_key, meta = run_regime(alpha, X, s1, y, shared)
        all_rows.extend(rows)
        entry = dict(summary.get(name) or {})
        entry.update(
            dict(alpha=alpha, n_clients=meta["n"],
                 group_stats_mae=by_key["group_stats"]["mae_mean"],
                 group_stats_auc=by_key["group_stats"]["auc_mean"],
                 fedrenyi_props_mae=by_key["fedrenyi_group_props"]["mae_mean"],
                 fedrenyi_props_auc=by_key["fedrenyi_group_props"]["auc_mean"],
                 proto_local_mae=by_key["prototypes"]["mae_mean"],
                 proto_local_auc=by_key["prototypes"]["auc_mean"],
                 proto_local_pad_mae=by_key["prototypes_padded"]["mae_mean"],
                 proto_local_pad_auc=by_key["prototypes_padded"]["auc_mean"],
                 proto_shared_mae=by_key["prototypes_shared"]["mae_mean"],
                 proto_shared_auc=by_key["prototypes_shared"]["auc_mean"],
                 proto_shared_pad_mae=by_key["prototypes_shared_padded"]["mae_mean"],
                 proto_shared_pad_auc=by_key["prototypes_shared_padded"]["auc_mean"],
                 class_proto_mae=by_key["class_prototypes"]["mae_mean"],
                 class_proto_auc=by_key["class_prototypes"]["auc_mean"],
                 class_proto_shared_mae=by_key["class_prototypes_shared"]["mae_mean"],
                 class_proto_shared_auc=by_key["class_prototypes_shared"]["auc_mean"],
                 logo_stats_mae=by_key["loGo_fairness_stats"]["mae_mean"],
                 logo_stats_auc=by_key["loGo_fairness_stats"]["auc_mean"],
                 logo_stats_shared_mae=by_key["loGo_fairness_stats_shared"]["mae_mean"],
                 logo_stats_shared_auc=by_key["loGo_fairness_stats_shared"]["auc_mean"],
                 dp={str(m): dict(
                     mae=by_key[f"proto_shared_dp_{m}"]["mae_mean"],
                     auc=by_key[f"proto_shared_dp_{m}"]["auc_mean"],
                     selfcos=by_key[f"proto_shared_dp_{m}"]["util_selfcos"],
                     aggcorr=by_key[f"proto_shared_dp_{m}"]["util_aggcorr"],
                     disc=by_key[f"proto_shared_dp_{m}"]["util_disc"])
                     for m in DP_SIGMA_MULTS},
                 no_info_mae=by_key["no_info_mean"]["mae_mean"]))
        summary[name] = entry

    # 通道映射与排除说明（代码考古结论, 供汇报与后续引用）
    summary["channel_mapping"] = {
        "FedProto": {
            "upload": "per-class prototypes (no group structure) + model params",
            "evidence": "algorithm/FederatedProto.py L345-357 (prototype upload/agg), L398 (model agg)",
            "channel": "c class_prototypes(-_shared)"},
        "PraFFL": {
            "upload": "encoder params only (uniform average); hypernetwork (preference/fairness trade-off) stays private",
            "evidence": "algorithm/PraFFL.py L242-262 (only encoder_state uploaded); algorithm/praffl_training.py L211-214",
            "channel": "e EXCLUDED (no explicit sensitive-attribute upload)"},
        "mFairFL": {
            "upload": "model params + n_k; Lagrangian lambda_k and group loss gap strictly local",
            "evidence": "algorithm/mFairFL.py L129/L202 (lambda local), L316-329 (only model states communicated)",
            "channel": "e EXCLUDED (no explicit sensitive-attribute upload)"},
        "LoGoFair": {
            "upload": "model params (FedAvg) + per-client group-conditional prediction stats consumed by post-processing",
            "evidence": "algorithm/LoGoFair.py L118-165 (per-group pos_rate/TPR/FPR), L571-629 (per-client fairness metrics)",
            "channel": "d loGo_fairness_stats(-_shared)"},
        "FedRenyi": {
            "upload": "per-client group proportions r_bar_k(p) (r_bar_k(1)=q), n_k/N weights, (class x group) conditionals",
            "evidence": "algorithm/FederatedRenyi.py L154-171 (gamma_k), L232-241 (r_bar_k_p), L194-214 (Q-hat), L355-369 (init)",
            "channel": "a/E fedrenyi_group_props (analytic, equivalent to group_stats)"},
    }

    csv_path = os.path.join(OUT_DIR, "privacy_attack_results.csv")
    merge_and_write_csv(all_rows, csv_path, NEW_METHODS)
    # 图与 CSV 严格一致：从合并后的 CSV 读回
    merged = []
    for r in read_csv_rows(csv_path):
        try:
            merged.append(dict(regime=r.get("regime", ""), method=r.get("method", ""),
                               mae_mean=float(r["mae_mean"]), mae_std=float(r["mae_std"] or 0),
                               auc_mean=float(r["auc_mean"]), auc_std=float(r["auc_std"] or 0)))
        except (KeyError, ValueError, TypeError):
            continue
    # 每个异构 regime 单独一张图（通道画在 y 轴），见 plot_privacy_bars
    figures = plot_privacy_bars.make_split_figures(merged, OUT_DIR)
    summary["runtime_sec"] = round(time.time() - t0, 1)
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[summary] " + json.dumps(summary, ensure_ascii=False))
    print(f"[done] {time.time() - t0:.1f}s -> {csv_path}, {', '.join(figures)}")


if __name__ == "__main__":
    main()
