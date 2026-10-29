# https://proceedings.mlr.press/v286/ma25a.html
# Federated Rényi Fair Inference in Federated Heterogeneous System
# 核心思想：利用Rényi散度进行联邦学习中的公平性推断，在异构联邦系统中实现公平推理
#
# ===== 机制说明：公平项如何进入梯度（2026-10-23 按论文口径修正）=====
# 论文（PMLR v286, ma25a）对「Ĥ 对 θ 可导」的要求是硬性而非可选的，四处直接依据如下
# （页码为论文 PDF 物理页）：
#   (1) Eq.(6)（p.4）：min_θ [ L̂(θ) + λ·max_v Ĥ(θ,v) ]，其中 Ĥ(θ,v) = vᵀ Q̂_θᵀ Q̂_θ v；
#   (2) Algorithm 1 第 7 行（p.4）：本地更新 θ ← θ − η·∂θ( L̂(θ) + λ·Ĥ(θ,v) )，
#       λ·Ĥ 被 ∂θ 求导，因此 ∂θĤ 必须非平凡；
#   (3) Assumption 2（p.6）：P̂[f_θ(X_k,S_k)=c] 关于 θ 是 L-Lipschitz 连续的，
#       只有软概率（模型输出的 P(f=1|x)）才满足，硬 0/1 阈值几乎处处常值；
#   (4) Proposition 1（p.5）及其证明（附录 B.1.6, p.19）：收敛定理显式假设
#       ‖∂Q̂_θ/∂θ‖²、‖∂Q_θ/∂θ‖²、‖∂v_θ/∂θ‖² 有界，并直接写出
#       ∂H_k(θ_k,v)/∂θ_k = Q_θk·v_θ·[ ∂Q_θk/∂θ_k·v_θ + ∂v_θ/∂θ·Q_θk ]。
# 由 (2)(3)(4) 可知：「梯度不可传导」（硬阈值使 ∂θĤ ≡ 0）是错误实现，不是论文特性。
# 需澄清的是 Eq.(5) 里经验概率 P̂[X∈D] := (1/n_k)Σᵢ ⊮[x_ki∈D] 的指示函数写法——
# 它只在「θ 固定、估 Q̂_θ 的统计误差」时成立（附录 Theorem 1 的 Hoeffding/Chernoff
# 集中不等式都按硬事件 f_θ(x)=c 做），一旦进入优化就必须换成软预测 P_θ(f=1|x)。
# 因此本实现：
#   - get_positive_class_probability：取模型的 P(f=1|x)（文本任务 softmax[:,1]，
#     图像/音频/表格任务模型已过 sigmoid，直接 reshape）；
#   - get_soft_fairness_components：以软概率均值估计 ĵ(1,p)/û(1)，c=0 侧取 1−ĵ(1,p)；
#   - assemble_Q：按 q̂_cp = ĵ(c,p)·r̂(p)/√(û(c)·r̂(p)) 用 torch.stack 组装 Q̂，保留计算图；
#   - v 由服务端 get_argmax_v 对 Q̂ 做 SVD 取第二奇异向量，本地训练当常数（与论文
#     「固定 v 更新 θ」一致）；get_argmax_v 用同一套软估计，保证 max 与 min 的 Ĥ 同目标。
# 注意：λ 现在真正生效，量级不同于旧硬阈值实现（旧实现 λ 恒无效），
# 历史 FedRényi 结果与之不可比。

import copy
import os
import gc
import time
import torch
import math
import numpy as np
from tool.logger import *
from tool.aggregation import average_state_dicts
from algorithm.Optimizers import BERTCLF_Optimizer
from tool.utils import (get_parameters, set_parameters, communication_cost_simulated_by_beta_distribution, get_HM_by_two_value,
                        FL_fairness_and_accuracy_test, FL_fairness_and_accuracy_test_4_IMG_CLF, FL_fairness_and_accuracy_test_4_Tabular_CLF)
from tool.checkpoint import save_checkpoint, clean_old_checkpoints
from tool.amp_utils import autocast_context, get_scaler, scale_backward, scaler_step, AutocastSafeBCELoss
from tool.tensorboard_logger import log_scalar, log_metrics, log_test_metrics, log_system_metrics, update_step, flush, log_deep_metrics, get_monitoring_config
from tool.client_parallel import ClientParallelExecutor
from tool.experiment_state import (
    AlgorithmRunResult,
    capture_training_loader_generator_states,
    restore_training_loader_generator_states,
)


def _protected_tensor(training_dataset, indices):
    """读取敏感属性向量。

    优先走 Dataset 的 protected 缓存：图像任务下 ``training_dataset[idx]['protected']``
    会触发整张图解码（读 zip 分片），而为取一个标量白读一遍数据集会把初始化拖到
    几十分钟。两条路径取到的值逐位相同。
    """
    protected = getattr(training_dataset, 'protected', None)
    if protected is not None:
        return torch.tensor([int(protected[idx]) for idx in indices])
    return torch.tensor([int(training_dataset[idx]['protected']) for idx in indices])


def get_argmax_v(param_dict, idxs_users, client_model_path_list, mask_s1_flag, training_dataset, client_dataset_list,
                 r_hat_p0, r_hat_p1, device, γ_k_style):

    training_dataset_size = len(training_dataset)

    # 按论文 Eq.(5) 的 Σ_k γ_k 形式累积软预测统计量（c=1 一侧，c=0 由 1−ĵ 得到）
    j_hat_1_0, j_hat_1_1, u_hat_1 = 0.0, 0.0, 0.0

    with torch.no_grad():
        for index, id in enumerate(idxs_users):
            selected_model = torch.load(client_model_path_list[index], weights_only=False).to(device)  # 持久化
            selected_model.eval()
            if "SENT_CLF" in param_dict["task"]:
                # TODO 这里还没有调过，不知道有没有bug
                # 有可能一次性推理多个句子，会爆显存，最稳妥还是逐个逐个推理然后再合并
                client_input_ids = torch.stack([training_dataset[idx]['input_ids'] for idx in client_dataset_list[id].indices]).to(device)
                client_attention_mask = torch.stack([training_dataset[idx]['attention_mask'] for idx in client_dataset_list[id].indices]).to(device)
                s = _protected_tensor(training_dataset, client_dataset_list[id].indices)
                __, client_logits = selected_model(
                    input_ids=client_input_ids,
                    attention_mask=client_attention_mask
                )
                # probs 尺寸 [样本数]，取模型预测为正类的概率（软预测）
                probs = get_positive_class_probability(client_logits, param_dict["task"])

            elif "IMG_CLF" in param_dict["task"] or "AUD_CLF" in param_dict["task"]:
                s = _protected_tensor(training_dataset, client_dataset_list[id].indices)
                # 分块批量推理：该函数每轮要对全部客户端的全部样本重新前向一遍，
                # 逐样本调用在 FairFace（8.7 万张 224x224）上占掉整轮约 2/3 的时间。
                # 此处已脱离训练图（外层 no_grad + eval，BN 用 running stats），
                # 分块前向与逐样本前向逐位一致，只是把每样本的调度开销摊到 chunk 上。
                client_indices = client_dataset_list[id].indices
                chunk = max(1, int(param_dict.get('get_v_batch_size', 128)))
                tmp_list = []
                for start in range(0, len(client_indices), chunk):
                    batch_indices = client_indices[start:start + chunk]
                    client_imgs = torch.stack(
                        [training_dataset[idx]['img'] for idx in batch_indices]).to(device)
                    tmp_list.append(get_positive_class_probability(
                        selected_model(client_imgs)[0], param_dict["task"]))
                probs = torch.cat(tmp_list).reshape(-1).to(device)

            elif "Tabular_CLF" in param_dict["task"]:
                # if mask_s1_flag:
                #     # Sensitive attribute 2
                #     s = torch.tensor([training_dataset[idx]['s2'] for idx in client_dataset_list[id].indices])
                # else:
                #     # Sensitive attribute 1
                #     s = torch.tensor([training_dataset[idx]['s1'] for idx in client_dataset_list[id].indices])

                s = _protected_tensor(training_dataset, client_dataset_list[id].indices)

                client_X = torch.tensor(np.array([training_dataset[idx]['X'] for idx in client_dataset_list[id].indices])).to(device)
                # local_prediction尺寸 [batch_size, 1]
                if "ANN" in str(type(selected_model)):
                    local_prediction, __ = selected_model(client_X)
                elif "LogisticRegression" in str(type(selected_model)):
                    local_prediction = selected_model(client_X)
                else:
                    local_prediction = selected_model(client_X)
                probs = get_positive_class_probability(local_prediction, param_dict["task"])

            if "client" in γ_k_style: # uniform over client
                γ_k = float(1 / len(idxs_users))
            else: # uniform over distribution
                γ_k = float(len(probs) / training_dataset_size)

            j_bar_1_0, j_bar_1_1, u_bar_1, __, __ = get_soft_fairness_components(probs, s)

            j_hat_1_0 += γ_k * float(j_bar_1_0)
            j_hat_1_1 += γ_k * float(j_bar_1_1)
            u_hat_1 += γ_k * float(u_bar_1)

        # 服务端的 v 是「被最大化的 Ĥ」的 argmax，必须与本地最小化的 Ĥ 用同一套软估计；
        # r̂(p) 用全局统计量（initialization 里按客户端加权算出的 r_hat_p0/r_hat_p1）。
        Q_hat = assemble_Q(torch.tensor(j_hat_1_0, device=device),
                           torch.tensor(j_hat_1_1, device=device),
                           torch.tensor(u_hat_1, device=device),
                           r_hat_p0, r_hat_p1, device)

        u, s, v = torch.linalg.svd(Q_hat)

        second_singular_vector_of_Q_hat = v[1].reshape(-1, 1).to(device)
    return second_singular_vector_of_Q_hat


def get_communication_idxs_list(num_clients_K, straggler_rate_α, descending_order_list):
    idxs_users = [i for i in range(num_clients_K)]
    straggler_ids = []
    if straggler_rate_α != 0:
        straggle_count = math.ceil(straggler_rate_α * num_clients_K)  # Round up the straggler count
        for tmp in range(straggle_count):
            straggler_id = descending_order_list[tmp]
            straggler_ids.append(straggler_id)
            # Remove stragglers
            idxs_users.remove(straggler_id)  # .remove(内容) or .pop(索引)
    return idxs_users, straggler_ids


def get_gamma_k_list(γ_k_style, client_datasets_size_list, num_clients_K):
    if "uniform_distribution" in γ_k_style:
        # uniform over distribution, γ_k =  n_k / n
        γ_denominator = sum(client_datasets_size_list)
    else:
        # uniform over client, γ_k = 1 / K
        γ_denominator = num_clients_K

    γ_k_list = []
    for i in range(num_clients_K):
        if "uniform_distribution" in γ_k_style:
            γ_numerator = client_datasets_size_list[i]
        else:
            γ_numerator = 1
        γ_k = γ_numerator / γ_denominator
        γ_k_list.append(float(γ_k))

    return γ_k_list


def get_positive_class_probability(preds, task):
    """把模型前向输出统一成 P(f=1|x)，即论文里的软预测。

    文本任务用 CrossEntropyLoss，模型输出 [B, num_classes] 的 logits；
    图像/音频/表格任务在模型内部已过 sigmoid，输出本身就是概率。
    """
    if "SENT_CLF" in task:
        return torch.softmax(preds.float(), dim=1)[:, 1]
    return preds.reshape(-1).float()


def get_soft_fairness_components(probs, s, eps=1e-6):
    """按论文 Eq.(5) 估计 c=1 一侧的 ĵ(1,p)、û(1) 与 r̂(p)，全程保留计算图。

    与硬标签版本的区别：ĵ(c,p) 由「组内硬标签的占比」改为
    「组内 P_θ(f=1|x) 的均值」，û(c) 同理改为全体软概率均值，
    这样 Ĥ(θ,v) 才对 θ 可导（Algorithm 1 第 7 行要求 ∂θĤ）。
    c=0 一侧由 1−ĵ(1,p) 得到，空组（r̂(p)=0）的分母用 eps 兜底，
    再由 assemble_Q 里的 √r̂ 自然把该组压成 0。
    """
    probs = probs.reshape(-1).float().clamp(0.0, 1.0)
    s = s.reshape(-1)
    one = (s == 1).to(probs.dtype)
    zero = 1.0 - one
    j_1_0 = (probs * zero).sum() / zero.sum().clamp_min(eps)
    j_1_1 = (probs * one).sum() / one.sum().clamp_min(eps)
    u_1 = probs.mean()
    n = float(max(probs.numel(), 1))
    r_0 = zero.sum() / n
    r_1 = one.sum() / n
    return j_1_0, j_1_1, u_1, r_0, r_1


def assemble_Q(j_1_0, j_1_1, u_1, r_0, r_1, device, eps=1e-6):
    """按 q̂_cp = ĵ(c,p)·r̂(p)/√(û(c)·r̂(p)) 组装 Q̂，保留计算图。

    分母写成 √r̂(p)·√û(c)：r̂(p)=0 的组直接得 0（等价于「空组 q=0」的兜底），
    同时不产生 nan。r̂(p) 是常数（不参与求导），统一搬到 device 上。
    û(c) 的下限取 1e-3 而非机器精度：模型输出饱和到 0/1 时 û(c) 会趋近 0，
    若按 1e-6 兜底会把 q̂ 放大 1000 倍（梯度同步爆掉），1e-3 只放大 31 倍。
    """
    def _const(x):
        if torch.is_tensor(x):
            return x.detach().to(device).float().reshape(())
        return torch.tensor(float(x), device=device)

    u_floor = 1e-3
    sqrt_r0 = torch.sqrt(_const(r_0))
    sqrt_r1 = torch.sqrt(_const(r_1))
    sqrt_u0 = torch.sqrt((1.0 - u_1).clamp(u_floor, 1.0))
    sqrt_u1 = torch.sqrt(u_1.clamp(u_floor, 1.0))

    q_00 = (1.0 - j_1_0) * sqrt_r0 / sqrt_u0
    q_01 = (1.0 - j_1_1) * sqrt_r1 / sqrt_u0
    q_10 = j_1_0 * sqrt_r0 / sqrt_u1
    q_11 = j_1_1 * sqrt_r1 / sqrt_u1

    Q = torch.stack([torch.stack([q_00, q_01]),
                     torch.stack([q_10, q_11])]).to(device)
    return Q


def get_Q_hat_theta_soft(probs, s, device):
    """由软预测得到参与本地训练的 Q̂（可导）。"""
    j_1_0, j_1_1, u_1, r_0, r_1 = get_soft_fairness_components(probs, s)
    return assemble_Q(j_1_0, j_1_1, u_1, r_0, r_1, device)


def get_G_hat_θ_hat_v(Q, v, device):
    Q = Q.to(device)
    v = v.reshape(-1, 1).to(device)
    result = v.T.matmul(Q.T).matmul(Q).matmul(v)
    result = result[0][0].to(device)
    result = torch.where(torch.isnan(result), torch.full_like(result, 0), result)
    return result


def get_r_bar_k_p_list(param_dict, num_clients_K, mask_s1_flag, training_dataset, client_dataset_list, p):
    r_bar_k_p_list = []
    for k in range(num_clients_K):
        sensitive_attribute = _protected_tensor(training_dataset, client_dataset_list[k].indices)

        r_bar_k_p = sum(sensitive_attribute == p) / len(sensitive_attribute)

        r_bar_k_p_list.append(r_bar_k_p)

    return r_bar_k_p_list


def get_statistical_distance(tuple_a, tuple_b):
    # Tuple:  {j_bar_0_0, j_bar_0_1, j_bar_1_0, j_bar_1_1, u_bar_0, u_bar_1, get_parameters(model)}
    j_a = np.array([
        [tuple_a[0].cpu(), tuple_a[1].cpu()],
        [tuple_a[2].cpu(), tuple_a[3].cpu()]
    ])
    j_b = np.array([
        [tuple_b[0].cpu(), tuple_b[1].cpu()],
        [tuple_b[2].cpu(), tuple_b[3].cpu()]
    ])

    u_a = np.array([tuple_a[4].cpu(), tuple_a[5].cpu()])
    u_b = np.array([tuple_b[4].cpu(), tuple_b[5].cpu()])

    θ_a, θ_b = np.array(tuple_a[-1]), np.array(tuple_b[-1])

    j_distance = np.linalg.norm(j_a - j_b)
    u_distance = np.linalg.norm(u_a - u_b)
    try:
        θ_distance = np.linalg.norm(θ_a - θ_b)
    except ValueError:
        differ = (θ_a - θ_b)
        θ_distance = sum([np.linalg.norm(differ[i]) for i in range(len(differ))])
    return j_distance, u_distance, θ_distance


def get_statistical_similarity(tuple_a, tuple_b, λ, ρ):
    j_distance, u_distance, θ_distance = get_statistical_distance(tuple_a, tuple_b)
    # DSim_a_b = θ_distance \
    #            + (-0.5 + (math.sqrt(4 * λ + 1) / 2)) * u_distance \
    #            + (λ + 1 / 2 - (math.sqrt(4 * λ + 1) / 2)) * j_distance
    if θ_distance != 0:
        W_θ_a_b = math.exp(-θ_distance / ρ)
    else:
        W_θ_a_b = 0

    if u_distance != 0:
        W_u_a_b = math.exp(-u_distance / ρ)
    else:
        W_u_a_b = 0

    if j_distance != 0:
        W_j_a_b = math.exp(-j_distance / ρ)
    else:
        W_j_a_b = 0

    return (W_θ_a_b, W_u_a_b, W_j_a_b)


def get_similarity_matrix(tuple_list, λ, ρ):
    K = len(tuple_list)
    similarity_matrix = []
    # 此处可以利用矩阵的对称性进行优化，降低运算次数
    for i in range(K):
        similarity_matrix.append([])
        for j in range(K):
            if i == j:
                similarity_matrix[i].append((1, 1, 1, 1))
            else:
                similarity_matrix[i].append(  # i用户 对 j用户的相似性
                    get_statistical_similarity(tuple_list[i], tuple_list[j], λ, ρ)
                )
    return similarity_matrix


def initialization(param_dict, client_dataset_list, num_clients_K, mask_s1_flag, training_dataset, γ_k_style):
    client_datasets_size_list = [len(item) for item in client_dataset_list]

    global_v = torch.rand(2, 1)

    r_bar_k_p0_list = get_r_bar_k_p_list(param_dict, num_clients_K, mask_s1_flag, training_dataset, client_dataset_list, p=0)
    r_bar_k_p1_list = get_r_bar_k_p_list(param_dict, num_clients_K, mask_s1_flag, training_dataset, client_dataset_list, p=1)

    γ_k_list = get_gamma_k_list(γ_k_style, client_datasets_size_list, num_clients_K)

    r_hat_p0 = sum([r_bar_k_p0_list[i] * γ_k_list[i] for i in range(len(γ_k_list))])
    r_hat_p1 = sum([r_bar_k_p1_list[i] * γ_k_list[i] for i in range(len(γ_k_list))])

    v_hat_1 = [math.sqrt(r_hat_p0), math.sqrt(r_hat_p1)]
    return client_datasets_size_list, global_v, r_bar_k_p0_list, r_bar_k_p1_list, γ_k_list, r_hat_p0, r_hat_p1, v_hat_1


def localized_approximation(j_bar_0_0_list, j_bar_0_1_list, j_bar_1_0_list, j_bar_1_1_list,
                            u_bar_0_list, u_bar_1_list,
                            i, local_model_list, similarity_matrix):
    θ_tilde_i = 0
    j_tilde_i_0_0, j_tilde_i_0_1, j_tilde_i_1_0, j_tilde_i_1_1 = 0, 0, 0, 0
    u_tilde_i_0, u_tilde_i_1 = 0, 0
    W_θ_sum, W_u_sum, W_j_sum = 0, 0, 0
    for j in range(len(local_model_list)):
        if j == i:
            continue
        else:
            W_θ, W_u, W_j = similarity_matrix[j][i][0], similarity_matrix[j][i][1], similarity_matrix[j][i][2]
            W_θ_sum += W_θ
            W_u_sum += W_u
            W_j_sum += W_j

            θ_tilde_i += W_θ * np.array(get_parameters(local_model_list[j]))

            j_bar_0_0, j_bar_0_1, j_bar_1_0, j_bar_1_1 = j_bar_0_0_list[j], j_bar_0_1_list[j], j_bar_1_0_list[j], \
                                                         j_bar_1_1_list[j]

            u_bar_0, u_bar_1 = u_bar_0_list[j], u_bar_1_list[j]

            j_tilde_i_0_0 += W_j * j_bar_0_0
            j_tilde_i_0_1 += W_j * j_bar_0_1
            j_tilde_i_1_0 += W_j * j_bar_1_0
            j_tilde_i_1_1 += W_j * j_bar_1_1

            u_tilde_i_0 += W_u * u_bar_0
            u_tilde_i_1 += W_u * u_bar_1

    θ_tilde_i = θ_tilde_i / W_θ_sum

    if W_j_sum == 0:
        j_tilde_i_0_0, j_tilde_i_0_1, j_tilde_i_1_0, j_tilde_i_1_1 = 0,0,0,0
    else:
        j_tilde_i_0_0 = j_tilde_i_0_0 / W_j_sum
        j_tilde_i_0_1 = j_tilde_i_0_1 / W_j_sum
        j_tilde_i_1_0 = j_tilde_i_1_0 / W_j_sum
        j_tilde_i_1_1 = j_tilde_i_1_1 / W_j_sum

    if W_u_sum == 0:
        u_tilde_i_0, u_tilde_i_1 = 0, 0
    else:
        u_tilde_i_0 = u_tilde_i_0 / W_u_sum
        u_tilde_i_1 = u_tilde_i_1 / W_u_sum

    return (j_tilde_i_0_0, j_tilde_i_0_1, j_tilde_i_1_0, j_tilde_i_1_1, u_tilde_i_0, u_tilde_i_1, θ_tilde_i)


def staleness_function_beta_β(zeta_ζ, phi_φ=0.1, function_style="exponential"):
    if "exponential" in function_style:
        compensation = math.exp(-zeta_ζ * phi_φ)
    elif "polynomial" in function_style:
        compensation = math.pow(zeta_ζ, -phi_φ)
    elif "linear" in function_style:
        compensation = phi_φ / zeta_ζ
    else:
        compensation = math.exp(-zeta_ζ * phi_φ)
    return compensation


def Fed_Renyi(device,
            global_model,
            algorithm_epoch_T, num_clients_K, communication_round_I, FL_fraction, FL_drop_rate,
            training_dataloaders,
            training_dataset,
            client_dataset_list,
            param_dict,
            testing_dataloader,
            testing_dataset_len,
            start_round=0,
            resume_state=None,
            ):
    # Initialization
    accumulation_steps = max(1, int(256 / param_dict['batch_size']))
    # AMP 初始化
    use_amp = param_dict.get('use_amp', False)
    scaler = get_scaler(device, use_amp)
    if resume_state is not None and resume_state.amp_scaler_state is not None:
        if scaler is None:
            raise RuntimeError("checkpoint contains AMP scaler state but AMP is disabled")
        scaler.load_state_dict(resume_state.amp_scaler_state)


    mask_s1_flag = False  # 表格型的数据集里面会有2个敏感属性s1和s2，默认是用s1，掩码s2。所以这里默认填False
    logger.info(f'mask_s1_flag: {mask_s1_flag}')

    logger.info("FedRenyi's Initialization")

    try:
        γ_k_style = param_dict['γ_k_style']
        tolerance_τ = param_dict['tolerance_τ']
        lamda = param_dict['FedRenyi_λ']
    except Exception:
        γ_k_style = "uniform_distribution"
        tolerance_τ = communication_round_I * algorithm_epoch_T
        lamda = 1

    client_datasets_size_list, global_v, r_bar_k_p0_list, r_bar_k_p1_list, \
    γ_k_list, r_hat_p0, r_hat_p1, v_hat_1 = initialization(param_dict, client_dataset_list,  num_clients_K,
                                                           mask_s1_flag, training_dataset, γ_k_style)

    training_dataset_size = len(training_dataset.labels)
    client_datasets_size_list = [len(_) for _ in client_dataset_list]

    # basic_path = os.path.join("./save_path", param_dict['dataset_name'],
    #                           param_dict['split_strategy'],
    #                           param_dict['algorithm'],
    #                           param_dict['hypothesis'],
    #                           str(num_clients_K) + "Clients")
    basic_path = param_dict['model_path']

    # Parameter Initialization
    for k in range(param_dict["num_clients_K"]):  # 持久化
        full_path = os.path.join(basic_path, "client_" + str(k + 1), 'model.pt')
        torch.save(global_model, full_path)
    # local_model_list = [copy.deepcopy(global_model) for _ in range(num_clients_K)] # 内存化

    # Training process
    logger.info("Training process begin!")
    logger.info(f'Training Dataset Size: {training_dataset_size}; Client Datasets Size:{client_datasets_size_list}')
    if "SENT_CLF" in param_dict["task"]:
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
    elif "IMG_CLF" in param_dict["task"] or "AUD_CLF" in param_dict["task"]:
        criterion = AutocastSafeBCELoss(reduction='none').to(device)
    elif "Tabular_CLF" in param_dict["task"]:
        criterion = AutocastSafeBCELoss(reduction='none').to(device)

    total_gpu_seconds = 0.0 if resume_state is None else float(resume_state.total_gpu_seconds)
    previous_state = {} if resume_state is None else dict(resume_state.algorithm_state)
    if resume_state is not None:
        restore_training_loader_generator_states(
            training_dataloaders, previous_state.get("training_loader_generator_states")
        )
    users_gpu_seconds_list = list(previous_state.get("users_gpu_seconds_list", [0.0] * num_clients_K))
    if len(users_gpu_seconds_list) != num_clients_K:
        raise RuntimeError("FedRenyi checkpoint has incompatible users_gpu_seconds_list")
    total_communication_cost = 0.0 if resume_state is None else float(resume_state.total_communication_cost)
    client_selection_history = [] if resume_state is None else [list(row) for row in resume_state.client_selection_history]
    runtime_offset = 0.0 if resume_state is None else float(resume_state.total_runtime_seconds)
    runtime_start = time.monotonic()

    # 断点恢复：global_v 在循环中以张量参与计算，需要从 CPU 列表搬回 device
    if resume_state is not None:
        saved_global_v = previous_state.get("global_v")
        if saved_global_v is not None:
            global_v = torch.tensor(saved_global_v, dtype=global_v.dtype, device=device).reshape(-1, 1)

    # model_MB_size = sys.getsizeof(global_model.state_dict()) / (1024 ** 2)
    model_MB_size = sum(p.numel() for p in global_model.parameters()) * 4 / (1024 * 1024)
    # logger.info(f"Model's Communication Cost: {model_MB_size} MB")

    # Simulate Client Parallel
    # TODO:改了迭代的架构，现在有三个for 最外层的for通信轮次 第二层是for每个通信轮次中的客户端训练epoch 第三层是for batch
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    for iter_t in range(start_round, communication_round_I):
        # Client Selection
        # 先选客户端，只对选中的客戶下发模型
        # FedRenyi需要与所有客户端通信
        idxs_users = [i for i in range(num_clients_K)]

        selected_clients = idxs_users.tolist() if hasattr(idxs_users, "tolist") else list(idxs_users)
        client_selection_history.append([int(client_id) for client_id in selected_clients])

        logger.info(f"Communication Round: {iter_t + 1}; Select clients: {idxs_users}; Start Local Training!")

        def _train_single_client_fedrenyi(client_id, device, model, param_dict, training_dataloaders, algorithm_epoch_T, 
                                        use_amp, scaler, criterion, basic_path, iter_t, communication_round_I, 
                                        num_clients_K, accumulation_steps, global_v, lamda):
            # Local Initialization
            # 下发模型
            logger.info(f"Client {client_id}: Copy From Global Model")
            model = copy.deepcopy(model)
            model.train()
            model.to(device)
            optimizer = BERTCLF_Optimizer(
                method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
            optimizer.set_parameters(list(model.named_parameters()))
            client_i_dataloader = training_dataloaders[client_id]

            gpu_seconds_for_client = 0

            # Local Training
            for epoch in range(algorithm_epoch_T):
                # 设置状态变量
                epoch_total_loss = 0
                epoch_total_size = 0

                # 注意：mini-batch gradient descent一般是把整个batch的损失累加起来，然后除以batch内的样本数目
                # FedAvg算法中，一个batch就更新一次参数
                for batch_id, batch in enumerate(client_i_dataloader):
                # for batch in client_i_dataloader:
                    if "SENT_CLF" in param_dict["task"]:
                        # input_ids尺寸 [batch_size, max_len]
                        input_ids = batch["input_ids"].to(device)
                        attention_mask = batch["attention_mask"].to(device)
                        protected = batch["protected"].to(device)
                    elif "IMG_CLF" in param_dict["task"] or "AUD_CLF" in param_dict["task"]:
                        imgs = batch["img"].to(device)
                        protected = batch["protected"].to(device)
                    elif "Tabular_CLF" in param_dict["task"]:
                        X = batch["X"].to(device)
                        # s = batch["s2"] if mask_s1_flag else batch["s1"]
                        # protected = s
                        protected = batch['protected'].to(device)
                    # labels尺寸 [batch_size]
                    labels = batch["labels"].to(device)

                    # 考虑到有可能没取满一整个batch，所以动态获取一下实际batch_size
                    true_batch_size = labels.size()[0]
                    epoch_total_size += true_batch_size

                    # 记录GPU计算开始时间
                    gpu_start_time = time.time()

                    with autocast_context(device, use_amp):
                        if "SENT_CLF" in param_dict["task"]:
                            # features尺寸 [batch_size, emb_dim]
                            # logits尺寸 [batch_size, category]
                            features, logits = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask
                            )
                            # activated_preds = logits.softmax(dim=1)
                            activated_preds = logits  # 由于我们采用了torch.nn.CrossEntropyLoss，在Pytorch里面这个函数是已经加了softmax的，所以我们不需要再手动加softmax
                            _, preds = torch.max(activated_preds, dim=1)
                            # batch_loss尺寸 [batch_size]
                            batch_loss = criterion(activated_preds, labels)

                        elif "IMG_CLF" in param_dict["task"] or "AUD_CLF" in param_dict["task"]:
                            # preds尺寸 [batch_size, 1]
                            # features尺寸 [batch_size, emb_dim]
                            preds, features = model(imgs)
                            batch_loss = criterion(preds[:, 0], labels.float())

                        elif "Tabular_CLF" in param_dict["task"]:
                            # local_prediction尺寸 [batch_size, 1]
                            if "ANN" in str(type(model)):
                                preds, features = model(X)
                            elif "LogisticRegression" in str(type(model)):
                                preds = model(X)
                            else:
                                preds = model(X)
                            batch_loss = criterion(preds[:, 0], labels.float())

                        # 软预测 P_θ(f=1|x)：ĵ/û 必须由它估计，Ĥ 才对 θ 可导，
                        # λ·Ĥ 才能按 Algorithm 1 第 7 行进入本地更新（见文件头说明）。
                        if "SENT_CLF" in param_dict["task"]:
                            # 文本任务的 preds 是 argmax 出来的类别索引，概率要从 logits 取
                            probs = get_positive_class_probability(logits, param_dict["task"])
                        else:
                            probs = get_positive_class_probability(preds, param_dict["task"])

                        Q = get_Q_hat_theta_soft(probs, protected, device)
                        G = get_G_hat_θ_hat_v(Q, global_v, device).to(device)
                        regularization_term = lamda * G

                        loss = torch.sum(batch_loss) / true_batch_size
                        loss += regularization_term


                    # 梯度累积：micro-batch 损失先除以窗口长度，窗口内各次 backward
                    # 的梯度之和才等于「等效大 batch」的平均梯度（batch_size=256 时
                    # accumulation_steps=1，除以 1 行为不变）。
                    scale_backward(loss / accumulation_steps, scaler)


                    # 记录GPU计算结束时间
                    gpu_end_time = time.time()

                    gpu_seconds_for_client += (gpu_end_time - gpu_start_time)

                    # FedAvg算法一个batch就做一次更新；累积模式下每个累积窗口更新一次。
                    # 最后一个不满窗口的 tail-step 也要先 step，否则残余梯度永远不生效。
                    if (batch_id + 1) % accumulation_steps == 0 or (batch_id + 1) == len(client_i_dataloader):
                        scaler_step(scaler, optimizer)
                        # 只在更新之后清空梯度，否则窗口内已累积的梯度会被逐个 micro-batch 抹掉
                        model.zero_grad()
                    # 记录状态信息
                    epoch_total_loss += float(loss.detach())
                    # average_one_sample_loss_in_epoch += average_one_sample_loss_in_batch / math.ceil(
                    #     client_datasets_size_list[client_id] / param_dict['batch_size'])

                    if "SENT_CLF" in param_dict["task"]:
                        del input_ids, attention_mask, labels
                    elif "IMG_CLF" in param_dict["task"] or "AUD_CLF" in param_dict["task"]:
                        del imgs, labels

                    gc.collect()

                    # if param_dict["one_batch_per_Epoch"]:
                    #     break

                average_one_sample_loss_in_epoch = epoch_total_loss / epoch_total_size
                logger.info(f"Communication Round: {iter_t + 1} / {communication_round_I}; "
                            f"Client: {client_id} / {num_clients_K}; "
                            f"Epoch: {epoch + 1}; Avg One Sample's Loss Over Epoch: {average_one_sample_loss_in_epoch}")

            # Upgrade the local model list
            client_model_path = os.path.join(basic_path, "client_" + str(client_id + 1), 'model.pt')
            # local_model_list[client_id] = model.cpu()  # 内存化
            torch.save(model.cpu(), client_model_path)  # 持久化

            del model
            gc.collect()
            # torch.cuda.empty_cache()
            
            return gpu_seconds_for_client


        # 使用并行执行器处理客户端训练
        parallel_executor = ClientParallelExecutor(device=device, global_model=global_model, param_dict=param_dict, needs_global_model_during_training=False)
        
        results = parallel_executor.run_clients(
            idxs_users,
            _train_single_client_fedrenyi,
            param_dict=param_dict,
            training_dataloaders=training_dataloaders,
            algorithm_epoch_T=algorithm_epoch_T,
            use_amp=use_amp,
            scaler=scaler,
            criterion=criterion,
            basic_path=basic_path,
            iter_t=iter_t,
            communication_round_I=communication_round_I,
            num_clients_K=num_clients_K,
            accumulation_steps=accumulation_steps,
            global_v=global_v,
            lamda=lamda
        )
        
        # 收集每个客户端的GPU时间
        for i, client_id in enumerate(idxs_users):
            users_gpu_seconds_list[client_id] += results[i]

        # Communicate
        total_gpu_seconds += sum(users_gpu_seconds_list)
        total_communication_cost += len(idxs_users) * 2 * model_MB_size
        logger.info(
            f"Communication Round {(iter_t + 1)} 's Communication Cost: {(iter_t + 1) * len(idxs_users) * 2 * model_MB_size} MB")

        # ── 收集客户端模型更新（用于梯度监控）──
        pre_agg_params = get_parameters(global_model)
        client_model_updates = []

        # Global operation
        logger.info("Parameter aggregation")
        client_states = []
        client_model_path_list = []
        for id in idxs_users:
            client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
            selected_model = torch.load(client_model_path, weights_only=False)  # 持久化
            client_model_path_list.append(client_model_path)
            client_params = get_parameters(selected_model)
            client_states.append(selected_model.state_dict())

            # 计算该客户端的更新量
            updates = {}
            for j, (p_local, p_global) in enumerate(zip(client_params, pre_agg_params)):
                updates[str(j)] = torch.tensor(p_local) - torch.tensor(p_global)
            client_model_updates.append(updates)

            del selected_model
            gc.collect()

        aggregated_state = average_state_dicts(client_states, weights=[client_datasets_size_list[j] for j in idxs_users])

        # 检查theta_avg的结果是否存在非np.ndarray的项，如有则转换
        # checked_theta_avg = []
        # for item in theta_avg:
        #     if isinstance(item, np.ndarray):
        #         checked_theta_avg.append(item)
        #     elif isinstance(item, np.float64):
        #         checked_theta_avg.append(np.array([item]))
        # theta_avg = checked_theta_avg

        logger.info("Update Global Model")
        global_model.load_state_dict(aggregated_state, strict=True)
        del client_states, aggregated_state

        logger.info("********** Global v update **********")
        backup_v = global_v
        try:
            global_v = get_argmax_v(param_dict, [i for i in range(num_clients_K)], client_model_path_list, mask_s1_flag,
                                    training_dataset, client_dataset_list, r_hat_p0, r_hat_p1, device, γ_k_style)
        except Exception:
            global_v = backup_v


        # 当前消耗的总GPU秒，平均GPU秒
        avg_gpu_seconds = (total_gpu_seconds / num_clients_K)
        logger.info(
            f"Global Model testing at Communication {(iter_t + 1)}/ {communication_round_I}")
        logger.info(
            f"Total GPU seconds: {total_gpu_seconds}, Avg GPU seconds over client: {avg_gpu_seconds}")

        gc.collect()

        # 没有到达最后一次通信轮次之前，都要做测试
        if (iter_t + 1) != param_dict['communication_round_I'] and (iter_t + 1) % param_dict.get('test_eval_freq', 1) == 0:
            if "SENT_CLF" in param_dict["task"]:
                evaluation = FL_fairness_and_accuracy_test(global_model, param_dict, testing_dataloader,
                                                                   testing_dataset_len)
                accuracy, DEO, SPD, ACC_gap = evaluation[0], evaluation[1], evaluation[2], evaluation.acc_gap
                logger.info(
                    f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)}, ACC_gap: {round(float(ACC_gap), 3)}")
                # ===== TensorBoard logging =====
                log_test_metrics(
                    ACC_gap=float(ACC_gap), accuracy=float(accuracy), DEO=float(DEO), SPD=float(SPD),
                    step=iter_t+1, gpu_seconds=total_gpu_seconds, avg_gpu_seconds=avg_gpu_seconds
                )
                log_system_metrics(step=iter_t+1, gpu_seconds=total_gpu_seconds, 
                                   selected_client_count=len(idxs_users), selected_clients=idxs_users.tolist() if hasattr(idxs_users, 'tolist') else idxs_users)
                flush()


            elif "IMG_CLF" in param_dict["task"] or "AUD_CLF" in param_dict["task"]:
                evaluation = FL_fairness_and_accuracy_test_4_IMG_CLF(global_model, param_dict,
                                                                             testing_dataloader, testing_dataset_len)
                accuracy, DEO, SPD, ACC_gap = evaluation[0], evaluation[1], evaluation[2], evaluation.acc_gap
                FR = 1 - DEO
                HM = get_HM_by_two_value(accuracy, FR)
                logger.info(
                    f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)},"
                    f" FR: {round(float(FR), 3)}, HM: {round(float(HM), 3)}, ACC_gap: {round(float(ACC_gap), 3)}")
                # ===== TensorBoard logging =====
                log_test_metrics(
                    ACC_gap=float(ACC_gap), accuracy=float(accuracy), DEO=float(DEO), SPD=float(SPD),
                    FR=float(FR), HM=float(HM),
                    step=iter_t+1, gpu_seconds=total_gpu_seconds, avg_gpu_seconds=avg_gpu_seconds
                )
                log_system_metrics(step=iter_t+1, gpu_seconds=total_gpu_seconds, 
                                   selected_client_count=len(idxs_users), selected_clients=idxs_users.tolist() if hasattr(idxs_users, 'tolist') else idxs_users)
                flush()
            elif "Tabular_CLF" in param_dict["task"]:
                evaluation = FL_fairness_and_accuracy_test_4_Tabular_CLF(global_model, param_dict,
                                                                                 testing_dataloader,
                                                                                 testing_dataset_len)
                accuracy, DEO, SPD, ACC_gap = evaluation[0], evaluation[1], evaluation[2], evaluation.acc_gap
                FR = 1 - DEO
                HM = get_HM_by_two_value(accuracy, FR)
                logger.info(
                    f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)},"
                    f" FR: {round(float(FR), 3)}, HM: {round(float(HM), 3)}, ACC_gap: {round(float(ACC_gap), 3)}")
                # ===== TensorBoard logging =====
                log_test_metrics(
                    ACC_gap=float(ACC_gap), accuracy=float(accuracy), DEO=float(DEO), SPD=float(SPD),
                    FR=float(FR), HM=float(HM),
                    step=iter_t+1, gpu_seconds=total_gpu_seconds, avg_gpu_seconds=avg_gpu_seconds
                )
                log_system_metrics(step=iter_t+1, gpu_seconds=total_gpu_seconds, 
                                   selected_client_count=len(idxs_users), selected_clients=idxs_users.tolist() if hasattr(idxs_users, 'tolist') else idxs_users)
                flush()


        # ===== 深度监控（每轮都执行，包括最后一轮）=====
        cfg_deep = get_monitoring_config(param_dict)
        log_deep_metrics(global_model, param_dict, testing_dataloader, 
                         iter_t + 1, client_model_updates=client_model_updates)

        # 保存检查点（按 checkpoint_save_freq 间隔，含 global_v）
        if (param_dict.get('checkpoint_save_freq', 1) > 0
                and iter_t % param_dict.get('checkpoint_save_freq', 1) == 0
                and (iter_t + 1) < communication_round_I):
            save_checkpoint(
                param_dict=param_dict,
                iter_t=iter_t,
                global_model=global_model,
                algorithm_state={
                    "users_gpu_seconds_list": users_gpu_seconds_list,
                    "training_loader_generator_states": capture_training_loader_generator_states(training_dataloaders),
                    "global_v": global_v.detach().cpu().tolist(),
                },
                amp_scaler=scaler,
                total_gpu_seconds=total_gpu_seconds,
                total_runtime_seconds=runtime_offset + (time.monotonic() - runtime_start),
                total_communication_cost=total_communication_cost,
                client_selection_history=client_selection_history,
                start_time=start_time,
            )

            # 清理旧检查点，保留最近 N 个
            clean_old_checkpoints(param_dict, keep_latest=param_dict.get('checkpoint_keep_latest', 5))


    logger.info("Training finish, save and return the global model.")
    # Save global model
    save_dir = f'./save_path/'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"global_FedRenyi_"+γ_k_style+".pt")
    torch.save(global_model, save_path)
    return AlgorithmRunResult(
        global_model=global_model,
        total_gpu_seconds=total_gpu_seconds,
        total_communication_cost=total_communication_cost,
        algorithm_state={
            "users_gpu_seconds_list": users_gpu_seconds_list,
            "training_loader_generator_states": capture_training_loader_generator_states(training_dataloaders),
            "global_v": global_v.detach().cpu().tolist(),
        },
        amp_scaler_state=None if scaler is None else scaler.state_dict(),
        client_selection_history=client_selection_history,
    )

