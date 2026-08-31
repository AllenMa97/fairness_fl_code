# AFL: A Single-Round Analytic Approach for Federated Learning with Pre-trained Models
# https://arxiv.org/abs/2405.16240 (CVPR 2025)
# Git Repo: https://github.com/ZHUANGHP/Analytic-federated-learning
# 核心思想: 操作空间为模型空间（闭式解）。冻结预训练骨干（PLM/CNN/ANN），只在分类头部分用岭回归
#           闭式解：客户端计算局部统计量 A_i = Σ φφ^T, B_i = Σ φt^T（φ 为骨干特征，t 为 one-hot
#           目标），服务器端求和后解 W = (A + λI)^{-1} B，一步得到全局最优分类头。
#           彻底绕开梯度下降与知识蒸馏，单次前向 + 单轮聚合即收敛。
# Core Idea: Operates in model space (closed-form solution). Freeze the pre-trained backbone
#            (PLM/CNN/ANN) and solve the classifier head via ridge regression in closed form:
#            clients compute local statistics A_i = Σ φφ^T, B_i = Σ φt^T (φ: backbone features,
#            t: one-hot targets); the server sums them and solves W = (A + λI)^{-1} B in one step.
#            Completely bypasses gradient descent and knowledge distillation -- a single forward
#            pass plus one aggregation round suffices.
# 框架适配说明: 每轮通信等价于一次解析聚合（幂等）：客户端一次前向统计 A_i/B_i 并上传，
#               服务器求和求逆并直接写入最后一个 Linear 层权重，无任何本地梯度训练。

import copy
import os
import gc
import time
import torch
import numpy as np
from tool.logger import *
from tool.utils import get_parameters, set_parameters
from algorithm.client_selection import client_selection
from tool.utils import FL_fairness_and_accuracy_test, FL_fairness_and_accuracy_test_4_IMG_CLF, FL_fairness_and_accuracy_test_4_Tabular_CLF, get_HM_by_two_value
from tool.checkpoint import save_checkpoint, clean_old_checkpoints
from tool.tensorboard_logger import log_test_metrics, log_system_metrics, flush, log_deep_metrics, get_monitoring_config
from tool.client_parallel import ClientParallelExecutor


def _forward_features(model, param_dict, batch, device):
    """冻结骨干提取特征（不经过分类头）。"""
    with torch.no_grad():
        if "SENT_CLF" in param_dict["task"]:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            return model.only_PLM_forward(input_ids=input_ids, attention_mask=attention_mask)
        elif "IMG_CLF" in param_dict["task"]:
            imgs = batch["img"].to(device)
            return model.only_backbone_forward(imgs)
        else:
            X = batch["X"].to(device)
            if "LogisticRegression" in str(type(model)):
                return X
            return model.only_backbone_forward(X)


def _get_last_linear(model):
    """获取模型中最后一个 Linear 层（即分类头）。"""
    last_linear = None
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            last_linear = m
    return last_linear


def _train_single_client_afl(client_id, device, model, param_dict,
                             training_dataloaders, algorithm_epoch_T,
                             accumulation_steps, use_amp, scaler, criterion,
                             basic_path, iter_t, communication_round_I, num_clients_K,
                             num_outputs):
    """AFL 单客户端：冻结骨干单次前向，统计 A_i = Σ φ̃φ̃^T, B_i = Σ φ̃t^T（φ̃ 为增广特征）。"""
    model.eval()
    model.to(device)

    gpu_start_time = time.time()
    feat_dim = None
    A_acc, B_acc = None, None
    n_samples = 0

    client_i_dataloader = training_dataloaders[client_id]
    for batch in client_i_dataloader:
        feats = _forward_features(model, param_dict, batch, device).detach()
        labels = batch["labels"].to(device)
        n = feats.size(0)
        if feat_dim is None:
            feat_dim = feats.size(1)
            # 增广一维 bias 特征
            A_acc = torch.zeros(feat_dim + 1, feat_dim + 1, device=device)
            B_acc = torch.zeros(feat_dim + 1, num_outputs, device=device)

        # 目标矩阵：SENT -> one-hot；二分类 -> 单列 0/1
        if "SENT_CLF" in param_dict["task"]:
            t = torch.nn.functional.one_hot(labels.long(), num_classes=num_outputs).float()
        else:
            t = labels.float().unsqueeze(1)

        phi = torch.cat([feats, torch.ones(n, 1, device=device)], dim=1)  # [n, d+1]
        A_acc += phi.T @ phi
        B_acc += phi.T @ t
        n_samples += n

    gpu_seconds = time.time() - gpu_start_time

    client_model_path = os.path.join(basic_path, "client_" + str(client_id + 1), 'model.pt')
    torch.save(model.cpu(), client_model_path)
    return {'gpu_seconds': gpu_seconds, 'A': A_acc.cpu().numpy(), 'B': B_acc.cpu().numpy(),
            'n_samples': n_samples, 'feat_dim': feat_dim}


def AFL(device,
        global_model,
        algorithm_epoch_T, num_clients_K, communication_round_I, FL_fraction, FL_drop_rate,
        training_dataloaders,
        training_dataset,
        client_dataset_list,
        param_dict,
        testing_dataloader,
        testing_dataset_len,
        start_round=0):
    use_amp = False  # 解析法无梯度训练，无需 AMP
    scaler = None

    training_dataset_size = len(training_dataset.labels) if hasattr(training_dataset, 'labels') else len(training_dataset)
    client_datasets_size_list = [len(_) for _ in client_dataset_list]
    del training_dataset, client_dataset_list
    gc.collect()

    # AFL 超参数
    afl_lambda = float(param_dict.get('AFL_lambda', 0.1))  # 岭回归正则系数

    # 输出维度：SENT 为类别数；二分类任务为 1
    if "SENT_CLF" in param_dict["task"]:
        head_probe = _get_last_linear(global_model)
        num_outputs = head_probe.out_features if head_probe is not None else int(param_dict.get('num_classes', 2))
    else:
        num_outputs = 1

    basic_path = param_dict['model_path']
    for k in range(param_dict["num_clients_K"]):
        full_path = os.path.join(basic_path, "client_" + str(k + 1), 'model.pt')
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        torch.save(global_model, full_path)

    total_gpu_seconds = 0
    users_gpu_seconds_list = [0] * num_clients_K
    model_MB_size = sum(p.numel() for p in global_model.parameters()) * 4 / (1024 * 1024)
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    parallel_executor = ClientParallelExecutor(
        device=device, global_model=global_model, param_dict=param_dict, needs_global_model_during_training=False)

    for iter_t in range(start_round, communication_round_I):
        idxs_users = client_selection(
            client_num=num_clients_K, fraction=FL_fraction,
            dataset_size=training_dataset_size, client_dataset_size_list=client_datasets_size_list,
            drop_rate=FL_drop_rate, style="FedAvg")

        logger.info(f"Round {iter_t + 1}; Select clients: {idxs_users}; Start Analytic Statistics (AFL)")

        # 注意：AFL 无本地梯度训练（algorithm_epoch_T 不生效，单次前向统计）
        results = parallel_executor.run_clients(
            idxs_users, _train_single_client_afl,
            param_dict=param_dict, training_dataloaders=training_dataloaders,
            algorithm_epoch_T=0, accumulation_steps=1,
            use_amp=False, scaler=None, criterion=None, basic_path=basic_path,
            iter_t=iter_t, communication_round_I=communication_round_I, num_clients_K=num_clients_K,
            num_outputs=num_outputs)

        for i, client_id in enumerate(idxs_users):
            users_gpu_seconds_list[client_id] += results[i]['gpu_seconds']
        total_gpu_seconds += sum(users_gpu_seconds_list)

        # ===== AFL 核心：闭式解聚合 W = (A + λI)^{-1} B =====
        logger.info("AFL: Closed-form ridge regression aggregation")
        pre_agg_params = get_parameters(global_model)
        client_model_updates = []
        A_sum, B_sum = None, None

        for i, id in enumerate(idxs_users):
            A_i, B_i = results[i]['A'], results[i]['B']
            A_sum = A_i if A_sum is None else A_sum + A_i
            B_sum = B_i if B_sum is None else B_sum + B_i

            client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
            selected_model = torch.load(client_model_path, weights_only=False)
            client_params = get_parameters(selected_model)
            updates = {}
            for j, (p_local, p_global) in enumerate(zip(client_params, pre_agg_params)):
                updates[str(j)] = torch.tensor(p_local) - torch.tensor(p_global)
            client_model_updates.append(updates)
            del selected_model
            gc.collect()

        gpu_s = time.time()
        # 求解岭回归闭式解：Theta = (A + λI)^{-1} B
        A_reg = A_sum + afl_lambda * np.eye(A_sum.shape[0])
        try:
            Theta = np.linalg.solve(A_reg, B_sum)  # [d+1, c]
        except np.linalg.LinAlgError:
            Theta = np.linalg.lstsq(A_reg, B_sum, rcond=None)[0]

        # 将闭式解写入最后一个 Linear 层
        global_model.to(device)
        head = _get_last_linear(global_model)
        with torch.no_grad():
            head.weight.data = torch.tensor(Theta[:-1, :].T, dtype=head.weight.dtype, device=device)
            head.bias.data = torch.tensor(Theta[-1, :], dtype=head.bias.dtype, device=device)
        logger.info(f"AFL: analytic head solved (lambda={afl_lambda}, "
                    f"feat_dim={Theta.shape[0]-1}, out_dim={Theta.shape[1]})")
        total_gpu_seconds += time.time() - gpu_s
        global_model.to("cpu")
        gc.collect()

        avg_gpu_seconds = total_gpu_seconds / num_clients_K
        if (iter_t + 1) != param_dict['communication_round_I']:
            if "SENT_CLF" in param_dict["task"]:
                acc, DEO, SPD = FL_fairness_and_accuracy_test(global_model, param_dict, testing_dataloader, testing_dataset_len)
                logger.info(f"ACC: {round(float(acc),3)}, DEO: {round(float(DEO),3)}, SPD: {round(float(SPD),3)}")
                log_test_metrics(accuracy=float(acc), DEO=float(DEO), SPD=float(SPD),
                    step=iter_t+1, gpu_seconds=total_gpu_seconds, avg_gpu_seconds=avg_gpu_seconds,
                    communication_cost=(iter_t+1)*len(idxs_users)*2*model_MB_size)
                log_system_metrics(step=iter_t+1, gpu_seconds=total_gpu_seconds,
                    communication_cost=(iter_t+1)*len(idxs_users)*2*model_MB_size,
                    selected_client_count=len(idxs_users), selected_clients=idxs_users.tolist(), model_mb_size=model_MB_size)
                flush()
            elif "IMG_CLF" in param_dict["task"]:
                acc, DEO, SPD = FL_fairness_and_accuracy_test_4_IMG_CLF(global_model, param_dict, testing_dataloader, testing_dataset_len)
                FR = 1 - DEO
                HM = get_HM_by_two_value(acc, FR)
                logger.info(f"ACC: {round(float(acc),3)}, DEO: {round(float(DEO),3)}, SPD: {round(float(SPD),3)}, FR: {round(float(FR),3)}, HM: {round(float(HM),3)}")
                log_test_metrics(accuracy=float(acc), DEO=float(DEO), SPD=float(SPD), FR=float(FR), HM=float(HM),
                    step=iter_t+1, gpu_seconds=total_gpu_seconds, avg_gpu_seconds=avg_gpu_seconds,
                    communication_cost=(iter_t+1)*len(idxs_users)*2*model_MB_size)
                log_system_metrics(step=iter_t+1, gpu_seconds=total_gpu_seconds,
                    communication_cost=(iter_t+1)*len(idxs_users)*2*model_MB_size,
                    selected_client_count=len(idxs_users), selected_clients=idxs_users.tolist(), model_mb_size=model_MB_size)
                flush()
            elif "Tabular_CLF" in param_dict["task"]:
                acc, DEO, SPD = FL_fairness_and_accuracy_test_4_Tabular_CLF(global_model, param_dict, testing_dataloader, testing_dataset_len)
                FR = 1 - DEO
                HM = get_HM_by_two_value(acc, FR)
                logger.info(f"ACC: {round(float(acc),3)}, DEO: {round(float(DEO),3)}, SPD: {round(float(SPD),3)}, FR: {round(float(FR),3)}, HM: {round(float(HM),3)}")
                log_test_metrics(accuracy=float(acc), DEO=float(DEO), SPD=float(SPD), FR=float(FR), HM=float(HM),
                    step=iter_t+1, gpu_seconds=total_gpu_seconds, avg_gpu_seconds=avg_gpu_seconds,
                    communication_cost=(iter_t+1)*len(idxs_users)*2*model_MB_size)
                log_system_metrics(step=iter_t+1, gpu_seconds=total_gpu_seconds,
                    communication_cost=(iter_t+1)*len(idxs_users)*2*model_MB_size,
                    selected_client_count=len(idxs_users), selected_clients=idxs_users.tolist(), model_mb_size=model_MB_size)
                flush()

        cfg_deep = get_monitoring_config(param_dict)
        log_deep_metrics(global_model, param_dict, testing_dataloader, iter_t + 1, client_model_updates=client_model_updates)

        if param_dict.get('checkpoint_save_freq', 1) > 0 and iter_t % param_dict.get('checkpoint_save_freq', 1) == 0:
            save_checkpoint(param_dict=param_dict, iter_t=iter_t, global_model=global_model,
                total_gpu_seconds=total_gpu_seconds,
                client_selection_history=[idxs_users.tolist()] if hasattr(idxs_users, 'tolist') else [idxs_users],
                start_time=start_time)
            clean_old_checkpoints(param_dict, keep_latest=param_dict.get('checkpoint_keep_latest', 5))

    logger.info("AFL training finished.")
    save_dir = './save_path/'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(global_model, os.path.join(save_dir, "global_AFL.pt"))
    # A/B 统计量远小于模型参数，通信成本近似为模型大小的 1.1x（上传 A,B + 必要的模型分发）
    total_comm_cost = communication_round_I * num_clients_K * FL_fraction * 1.1 * model_MB_size
    return global_model, total_gpu_seconds, total_comm_cost
