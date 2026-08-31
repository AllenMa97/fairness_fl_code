# FedSum: Data-Efficient Federated Learning with Semantic Portraits
# 核心机制：
#   1. Semantic Portrait（语义画像）：客户端类别级特征原型 [label_0_proto, label_1_proto]，作为跨客户端分布知识载体
#   2. PDT（画像距离表）：语义画像之间的 tanh(L2) 距离，刻画客户端分布相似度
#   3. Sample Skipping（样本跳过）：基于标签占比 Q 与损失 epsilon 的全局统计量，将样本划分为 Prime/Normal，
#      对 Normal 样本按训练进度递增的概率跳过，实现数据高效联邦学习
#   4. Semantic-guided Inter-client Knowledge Transfer（语义引导跨客户端知识迁移，L_pm）：
#      用上一轮其他客户端的分类头（分发预测头）处理本地表征，按画像距离加权学习
#   5. Prototype Loss（原型损失，L_po）：全局-本地类原型对齐 + 拉大预测原型间距
#   6. CLF Perturbation（分类头扰动）：对收集的分类头加噪，防止过拟合

import copy
import os
import gc
import time
import torch
import math
import random
import numpy as np
from tool.logger import *
from algorithm.Optimizers import BERTCLF_Optimizer
from algorithm.client_selection import client_selection
from tool.utils import get_parameters, set_parameters, get_HM_by_two_value
from tool.utils import FL_fairness_and_accuracy_test, FL_fairness_and_accuracy_test_4_IMG_CLF, FL_fairness_and_accuracy_test_4_Tabular_CLF
from tool.checkpoint import save_checkpoint, clean_old_checkpoints, load_checkpoint
from tool.amp_utils import autocast_context, get_scaler, scale_backward, scaler_step
from tool.tensorboard_logger import log_test_metrics, log_system_metrics, flush, log_deep_metrics, get_monitoring_config
from tool.client_parallel import ClientParallelExecutor


# ==================== 通用工具函数 ====================

def mask_input_with_mask_rate(input_tensor: torch.Tensor, mask_rate: float, use_rescale: bool, mask_strategy: str):
    """
    按 mask_rate 随机遮蔽参数（用于分类头扰动）。
    :param input_tensor: Tensor, 输入张量
    :param mask_rate: float, 遮蔽比例 [0.0, 1.0]
    :param use_rescale: boolean, 是否按 1 / (1 - mask_rate) 缩放
    :param mask_strategy: str, "random" 或 "magnitude"
    """
    assert 0.0 <= mask_rate <= 1.0, f"wrong range of mask_rate {mask_rate}, should be [0.0, 1.0]!"
    if mask_strategy == "random":
        mask = torch.bernoulli(torch.full_like(input=input_tensor, fill_value=mask_rate)).to(input_tensor.device)
        masked_input_tensor = input_tensor * (1 - mask)
    else:
        assert mask_strategy == "magnitude", f"wrong setting for mask_strategy {mask_strategy}!"
        original_shape = input_tensor.shape
        input_tensor = input_tensor.flatten()
        num_mask_params = int(len(input_tensor) * mask_rate)
        kth_values, _ = input_tensor.abs().kthvalue(k=num_mask_params, dim=0, keepdim=True)
        mask = input_tensor.abs() <= kth_values
        masked_input_tensor = input_tensor * (~mask)
        masked_input_tensor = masked_input_tensor.reshape(original_shape)
    if use_rescale and mask_rate != 1.0:
        masked_input_tensor = torch.div(input=masked_input_tensor, other=1 - mask_rate)
    return masked_input_tensor


def model_perturb(operated_model, mask_rate):
    """对分类头参数随机加噪（mask_rate 比例置 0），并冻结梯度。"""
    with torch.no_grad():
        for param_name, param_value in operated_model.named_parameters():
            param_value.data.copy_(
                mask_input_with_mask_rate(param_value, mask_rate, use_rescale=False, mask_strategy="random")
            )
    for param in operated_model.parameters():
        param.requires_grad = False
    return operated_model


def get_PDT(num_clients_K, semantic_portrait_list):
    """计算客户端语义画像距离表：portrait_distance_table[i][j] = tanh(||portrait_i - portrait_j||_2)"""
    portrait_distance_table = [[0.0 for _ in range(num_clients_K)] for _ in range(num_clients_K)]
    for i in range(num_clients_K):
        portrait_i = semantic_portrait_list[i]
        for j in range(i, num_clients_K):
            portrait_j = semantic_portrait_list[j]
            dist = torch.dist(portrait_i, portrait_j)  # L2 距离
            portrait_distance_table[i][j] = math.tanh(float(dist))
            portrait_distance_table[j][i] = math.tanh(float(dist))
    return portrait_distance_table


# ==================== 任务适配辅助函数 ====================

def _get_clf_module(model, param_dict):
    """返回模型的分类头模块（SENT: out; IMG/Tabular: out_layer; LR: layer）"""
    if "SENT_CLF" in param_dict["task"]:
        return model.out
    elif "IMG_CLF" in param_dict["task"]:
        return model.out_layer
    else:  # Tabular_CLF
        if "LogisticRegression" in str(type(model)):
            return model.layer
        return model.out_layer


def _clf_forward(clf, feature, param_dict):
    """分类头前向：SENT 返回 logits（配合 CrossEntropyLoss），其余返回 sigmoid 概率（配合 BCELoss）"""
    if "SENT_CLF" in param_dict["task"]:
        return clf(feature)
    else:
        return torch.sigmoid(clf(feature))


def _extract_features(model, input_ids=None, attention_mask=None, imgs=None, X=None, param_dict=None):
    """提取表征特征（SENT: PLM; IMG/ANN: backbone; LR: 输入本身）"""
    if "SENT_CLF" in param_dict["task"]:
        return model.only_PLM_forward(input_ids=input_ids, attention_mask=attention_mask)
    elif "IMG_CLF" in param_dict["task"]:
        return model.only_backbone_forward(imgs)
    else:  # Tabular_CLF
        if "ANN" in str(type(model)):
            return model.only_backbone_forward(X)
        else:
            return X  # LogisticRegression 无 backbone，特征即输入


def _compute_client_semantic_portrait(model, client_dataloader, param_dict, device):
    """计算单个客户端的初始语义画像 = [label_0_proto, label_1_proto]（使用全局模型、无梯度）"""
    model.eval()
    model.to(device)
    label_0_feature_list, label_1_feature_list = [], []
    with torch.no_grad():
        for batch in client_dataloader:
            if "SENT_CLF" in param_dict["task"]:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                features = model.only_PLM_forward(input_ids=input_ids, attention_mask=attention_mask)
            elif "IMG_CLF" in param_dict["task"]:
                imgs = batch["img"].to(device)
                features = model.only_backbone_forward(imgs)
            else:  # Tabular_CLF
                X = batch["X"].to(device)
                if "ANN" in str(type(model)):
                    features = model.only_backbone_forward(X)
                else:
                    features = X
            labels = batch["labels"]
            for feat, lab in zip(features.detach().cpu(), labels):
                if float(lab) > 0.5:
                    label_1_feature_list.append(feat)
                else:
                    label_0_feature_list.append(feat)
            if len(label_0_feature_list) != 0 and len(label_1_feature_list) != 0:
                break
    model.to("cpu")
    if len(label_0_feature_list) != 0:
        label_0_prototype = torch.stack(label_0_feature_list, dim=0).mean(dim=0)
    else:
        label_0_prototype = torch.zeros_like(label_1_feature_list[0]) if label_1_feature_list else torch.zeros(768)
    if len(label_1_feature_list) != 0:
        label_1_prototype = torch.stack(label_1_feature_list, dim=0).mean(dim=0)
    else:
        label_1_prototype = torch.zeros_like(label_0_feature_list[0]) if label_0_feature_list else torch.zeros(768)
    return torch.stack([label_0_prototype, label_1_prototype], dim=0)


# ==================== 单客户端训练 ====================

def _train_single_client_fedsum(client_id, device, model, param_dict,
                                training_dataloaders, algorithm_epoch_T,
                                accumulation_steps, use_amp, scaler, criterion,
                                basic_path, iter_t, communication_round_I, num_clients_K,
                                semantic_portrait_list, portrait_distance_table,
                                clf_tuple, global_label_0_prototype_list, global_label_1_prototype_list,
                                Q_bar, epsilon_bar, local_update_times_list,
                                gamma_mask, lambda_lpo):
    """FedSum 单客户端训练（可被 ClientParallelExecutor 调度）。
    返回：{'gpu_seconds', 'label_0_prototype', 'label_1_prototype', 'epsilon_sum', 'Q_sum'}
    """
    model.train()
    model.to(device)
    optimizer = BERTCLF_Optimizer(
        method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
    optimizer.set_parameters(list(model.named_parameters()))
    client_i_dataloader = training_dataloaders[client_id]

    local_update_times = local_update_times_list[client_id]
    gpu_seconds_for_client = 0
    epsilon_i_j_list = []
    Q_i_j_list = []

    client_i_label_0_feature_list = []
    client_i_label_1_feature_list = []
    client_i_predict_0_feature_list = []
    client_i_predict_1_feature_list = []

    last_idxs_users, last_clf_list = clf_tuple if len(clf_tuple) != 0 else ([], [])

    for epoch in range(algorithm_epoch_T):
        epoch_total_loss = 0.0    # 含 L_po 的总损失（detach 后按 float 累计）
        epoch_task_loss = 0.0     # 纯任务损失（CE/BCE），用于监测收敛
        epoch_total_size = 0

        for batch_id, batch in enumerate(client_i_dataloader):
            if "SENT_CLF" in param_dict["task"]:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
            elif "IMG_CLF" in param_dict["task"]:
                imgs = batch["img"].to(device)
            else:  # Tabular_CLF
                X = batch["X"].to(device)
            labels = batch["labels"].to(device)

            true_batch_size = labels.size(0)
            epoch_total_size += true_batch_size
            gpu_start_time = time.time()

            with autocast_context(device, use_amp):
                if "SENT_CLF" in param_dict["task"]:
                    features, logits = model(input_ids=input_ids, attention_mask=attention_mask)
                    batch_loss = criterion(logits, labels)
                    pred_flag = logits.argmax(dim=1)
                elif "IMG_CLF" in param_dict["task"]:
                    preds, features = model(imgs)
                    batch_loss = criterion(preds[:, 0], labels.float())
                    pred_flag = (preds >= 0.5).reshape(-1)
                else:  # Tabular_CLF
                    if "ANN" in str(type(model)):
                        preds, features = model(X)
                    else:  # LogisticRegression
                        preds = model(X)
                        features = X
                    batch_loss = criterion(preds[:, 0], labels.float())
                    pred_flag = (preds >= 0.5).reshape(-1)

                loss = torch.sum(batch_loss) / true_batch_size
                task_loss_val = float(loss.detach())  # 记录纯任务损失（L_po 加入前的值）

                # ---- 样本跳过机制的统计量（detach，不参与梯度） ----
                epsilon_i_j = float(loss.detach())
                Q_i_j = float((labels > 0.5).float().mean().detach())
                epsilon_i_j_list.append(epsilon_i_j / (algorithm_epoch_T * local_update_times))
                Q_i_j_list.append(Q_i_j / (algorithm_epoch_T * local_update_times))

                # ---- Prime / Normal 判定：Normal 样本按训练进度递增的概率跳过 ----
                skip_this_batch = False
                if (Q_i_j <= Q_bar) and (epsilon_i_j <= epsilon_bar):
                    rho = (((iter_t + 1) / communication_round_I)
                           * ((epoch + 1) / algorithm_epoch_T)
                           * ((batch_id + 1) / local_update_times))
                    if rho >= random.random():
                        skip_this_batch = True

                # ---- 原型损失（L_po）：全局-本地类原型对齐 + 拉大预测原型间距（带梯度） ----
                if not skip_this_batch and lambda_lpo > 0:
                    lpo_loss = 0
                    if len(global_label_0_prototype_list) != 0:
                        mask_0 = (labels <= 0.5)
                        if mask_0.any():
                            local_0_proto = features[mask_0].mean(dim=0)
                            lpo_loss = lpo_loss + torch.norm(
                                local_0_proto - global_label_0_prototype_list[-1].to(device), p=2)
                    if len(global_label_1_prototype_list) != 0:
                        mask_1 = (labels > 0.5)
                        if mask_1.any():
                            local_1_proto = features[mask_1].mean(dim=0)
                            lpo_loss = lpo_loss + torch.norm(
                                local_1_proto - global_label_1_prototype_list[-1].to(device), p=2)
                    pred_mask_0 = (pred_flag <= 0.5)
                    pred_mask_1 = (pred_flag > 0.5)
                    if pred_mask_0.any() and pred_mask_1.any():
                        predict_0_proto = features[pred_mask_0].mean(dim=0)
                        predict_1_proto = features[pred_mask_1].mean(dim=0)
                        lpo_loss = lpo_loss - torch.norm(predict_1_proto - predict_0_proto, p=2)
                    loss = loss + lambda_lpo * lpo_loss

            # ---- 收集类原型素材（detach，用于上传语义画像） ----
            features_cpu = features.detach().cpu()
            for idx, feat in enumerate(features_cpu):
                if float(labels[idx]) > 0.5:
                    client_i_label_1_feature_list.append(feat)
                else:
                    client_i_label_0_feature_list.append(feat)

            # ---- 反向传播（Normal 且被跳过则不更新） ----
            if not skip_this_batch:
                scale_backward(loss, scaler)
                if (batch_id + 1) % accumulation_steps == 0:
                    scaler_step(scaler, optimizer)
                    model.zero_grad()
            else:
                model.zero_grad()

            gpu_end_time = time.time()
            gpu_seconds_for_client += (gpu_end_time - gpu_start_time)
            epoch_total_loss += float(loss.detach())
            epoch_task_loss += task_loss_val

            if "SENT_CLF" in param_dict["task"]:
                del input_ids, attention_mask, logits
            elif "IMG_CLF" in param_dict["task"]:
                del imgs, preds
            gc.collect()

        # flush 尾部梯度
        if (batch_id + 1) % accumulation_steps != 0:
            scaler_step(scaler, optimizer)
            model.zero_grad()

        # ---- 语义引导的跨客户端知识迁移（L_pm）：分发预测头（其他客户端分类头） ----
        if len(last_clf_list) != 0:
            try:
                batch = next(iter(client_i_dataloader))
                if "SENT_CLF" in param_dict["task"]:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                elif "IMG_CLF" in param_dict["task"]:
                    imgs = batch["img"].to(device)
                else:
                    X = batch["X"].to(device)
                labels = batch["labels"].to(device)
                true_batch_size = labels.size(0)

                with autocast_context(device, use_amp):
                    feature = _extract_features(model,
                                                input_ids=input_ids if "SENT_CLF" in param_dict["task"] else None,
                                                attention_mask=attention_mask if "SENT_CLF" in param_dict["task"] else None,
                                                imgs=imgs if "IMG_CLF" in param_dict["task"] else None,
                                                X=X if "Tabular_CLF" in param_dict["task"] else None,
                                                param_dict=param_dict)
                    L_pm = 0
                    for num, clf in enumerate(last_clf_list):
                        other_id = last_idxs_users[num]
                        distance = float(portrait_distance_table[client_id][other_id])
                        if distance <= 1e-8:  # 语义完全相同的客户端不贡献知识
                            continue
                        clf.to(device)
                        out = _clf_forward(clf, feature, param_dict)
                        if "SENT_CLF" in param_dict["task"]:
                            sub_batch_loss = criterion(out, labels)
                        else:
                            sub_batch_loss = criterion(out[:, 0], labels.float())
                        L_pm = L_pm + distance * torch.sum(sub_batch_loss) / (len(last_clf_list) * true_batch_size)
                        clf.cpu()
                scale_backward(L_pm, scaler)
                scaler_step(scaler, optimizer)
                model.zero_grad()
            except Exception as e:
                logger.warning(f"Client {client_id} L_pm knowledge transfer skipped: {e}")

        avg_task_loss = epoch_task_loss / max(epoch_total_size, 1)
        avg_total_loss = epoch_total_loss / max(epoch_total_size, 1)
        logger.info(f"Communication Round: {iter_t + 1} / {communication_round_I}; "
                    f"Client: {client_id} / {num_clients_K}; "
                    f"Epoch: {epoch + 1}; Avg Task Loss: {avg_task_loss:.4f}; "
                    f"Avg Total Loss (w/ L_po): {avg_total_loss:.4f}")

    # ---- 计算本地类原型与预测原型（上传给服务器做全局聚合 / 画像更新） ----
    def _safe_prototype(feature_list):
        if len(feature_list) != 0:
            return torch.stack(feature_list, dim=0).mean(dim=0)
        return None

    client_i_label_0_prototype = _safe_prototype(client_i_label_0_feature_list)
    client_i_label_1_prototype = _safe_prototype(client_i_label_1_feature_list)

    # 保存模型（持久化）
    client_model_path = os.path.join(basic_path, "client_" + str(client_id + 1), 'model.pt')
    os.makedirs(os.path.dirname(client_model_path), exist_ok=True)
    torch.save(model.cpu(), client_model_path)

    del model
    gc.collect()

    return {
        'gpu_seconds': gpu_seconds_for_client,
        'label_0_prototype': client_i_label_0_prototype,
        'label_1_prototype': client_i_label_1_prototype,
        'predict_0_prototype': client_i_predict_0_prototype,
        'predict_1_prototype': client_i_predict_1_prototype,
        'epsilon_sum': float(sum(epsilon_i_j_list)),
        'Q_sum': float(sum(Q_i_j_list)),
    }


# ==================== FedSum 主训练循环 ====================

def Fed_Sum(device,
            global_model,
            algorithm_epoch_T, num_clients_K, communication_round_I, FL_fraction, FL_drop_rate,
            training_dataloaders,
            training_dataset,
            client_dataset_list,
            param_dict,
            testing_dataloader,
            testing_dataset_len,
            start_round=0
            ):
    accumulation_steps = max(1, int(256 / param_dict['batch_size']))

    # AMP 初始化
    use_amp = param_dict.get('use_amp', False)
    scaler = get_scaler(device, use_amp)

    # ---- 数据统计 ----
    training_dataset_size = len(training_dataset.labels) if hasattr(training_dataset, 'labels') else len(training_dataset)
    client_datasets_size_list = [len(_) for _ in client_dataset_list]
    del training_dataset, client_dataset_list
    gc.collect()
    local_update_times_list = [max(1, int(math.ceil(size / param_dict['batch_size']))) for size in client_datasets_size_list]
    average_weight = np.array([float(size / training_dataset_size) for size in client_datasets_size_list])

    # ---- FedSum 超参 ----
    gamma_mask = float(param_dict.get('FedSum_mask_rate', 0.3))  # 分类头扰动遮蔽比例
    lambda_lpo = float(param_dict.get('FedSum_lpo_λ', 1.0))   # 原型损失权重

    basic_path = param_dict['model_path']
    for k in range(param_dict["num_clients_K"]):  # 持久化
        full_path = os.path.join(basic_path, "client_" + str(k + 1), 'model.pt')
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        torch.save(global_model, full_path)

    # 损失函数
    if "SENT_CLF" in param_dict["task"]:
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
    else:
        criterion = torch.nn.BCELoss(reduction='none').to(device)

    total_gpu_seconds = 0
    users_gpu_seconds_list = [0] * num_clients_K
    model_MB_size = sum(p.numel() for p in global_model.parameters()) * 4 / (1024 * 1024)
    portrait_MB_size = torch.rand([2, param_dict.get('emb_dim', 768)]).numel() * 4 / (1024 ** 2)

    # ---- 语义画像初始化（使用全局初始模型） ----
    semantic_portrait_list = []
    for id in range(num_clients_K):
        portrait = _compute_client_semantic_portrait(global_model, training_dataloaders[id], param_dict, device)
        semantic_portrait_list.append(portrait)
    portrait_distance_table = get_PDT(num_clients_K, semantic_portrait_list)

    # ---- 全局状态 ----
    Q_bar = 0.0        # 全局标签占比统计量（随训练累积）
    epsilon_bar = 0.0  # 全局损失统计量（随训练累积）
    clf_tuple = ()     # (idxs_users, clf_list)：上一轮选中的客户端分类头列表（分发预测头）
    global_label_0_prototype_list = []  # 全局 0 类原型历史
    global_label_1_prototype_list = []  # 全局 1 类原型历史

    # 断点恢复：恢复画像 / 原型 / 统计量
    if start_round > 0:
        ckpt = load_checkpoint(param_dict)
        if ckpt is not None:
            es = ckpt.get('extra_state', {})
            try:
                Q_bar = float(es.get('Q_bar', 0.0))
                epsilon_bar = float(es.get('epsilon_bar', 0.0))
                saved_portraits = es.get('semantic_portrait_list', [])
                if len(saved_portraits) == num_clients_K:
                    semantic_portrait_list = [torch.tensor(p) for p in saved_portraits]
                    portrait_distance_table = get_PDT(num_clients_K, semantic_portrait_list)
                global_label_0_prototype_list = [torch.tensor(p) for p in es.get('global_label_0_prototype_list', [])]
                global_label_1_prototype_list = [torch.tensor(p) for p in es.get('global_label_1_prototype_list', [])]
            except Exception as e:
                logger.warning(f"FedSum checkpoint extra_state restore failed: {e}")

    logger.info("Training process begin!")
    logger.info(f'Training Dataset Size: {training_dataset_size}; Client Datasets Size:{client_datasets_size_list}')
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    parallel_executor = ClientParallelExecutor(
        device=device,
        global_model=global_model,
        param_dict=param_dict,
        needs_global_model_during_training=False,
    )

    for iter_t in range(start_round, communication_round_I):
        # Client Selection（FedAvg 风格，drop_rate 表示掉队者，掉队者不参与本轮）
        idxs_users = client_selection(
            client_num=num_clients_K,
            fraction=FL_fraction,
            dataset_size=training_dataset_size,
            client_dataset_size_list=client_datasets_size_list,
            drop_rate=FL_drop_rate,
            style="FedAvg",
        )

        logger.info(f"Communication Round: {iter_t + 1}; Select clients: {idxs_users}; Start Local Training!")

        results = parallel_executor.run_clients(
            idxs_users,
            _train_single_client_fedsum,
            param_dict=param_dict,
            training_dataloaders=training_dataloaders,
            algorithm_epoch_T=algorithm_epoch_T,
            accumulation_steps=accumulation_steps,
            use_amp=use_amp,
            scaler=scaler,
            criterion=criterion,
            basic_path=basic_path,
            iter_t=iter_t,
            communication_round_I=communication_round_I,
            num_clients_K=num_clients_K,
            semantic_portrait_list=semantic_portrait_list,
            portrait_distance_table=portrait_distance_table,
            clf_tuple=clf_tuple,
            global_label_0_prototype_list=global_label_0_prototype_list,
            global_label_1_prototype_list=global_label_1_prototype_list,
            Q_bar=Q_bar,
            epsilon_bar=epsilon_bar,
            local_update_times_list=local_update_times_list,
            gamma_mask=gamma_mask,
            lambda_lpo=lambda_lpo,
        )

        # 收集 GPU 计时
        for i, client_id in enumerate(idxs_users):
            users_gpu_seconds_list[client_id] += results[i]['gpu_seconds']
        total_gpu_seconds += sum(users_gpu_seconds_list)

        # ---- 聚合 ----
        logger.info("Parameter aggregation")
        pre_agg_params = get_parameters(global_model)
        client_model_updates = []
        theta_list = []
        clf_list = []
        for id in idxs_users:
            client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
            selected_model = torch.load(client_model_path, weights_only=False)
            client_params = get_parameters(selected_model)
            theta_list.append(client_params)

            updates = {}
            for j, (p_local, p_global) in enumerate(zip(client_params, pre_agg_params)):
                updates[str(j)] = torch.tensor(p_local) - torch.tensor(p_global)
            client_model_updates.append(updates)

            # 收集分类头（分发预测头：用于下一轮 L_pm 与扰动）
            clf_list.append(copy.deepcopy(_get_clf_module(selected_model, param_dict)))

            del selected_model
            gc.collect()

        theta_list = np.array(theta_list, dtype=object)
        theta_avg = np.average(theta_list, axis=0, weights=[client_datasets_size_list[j] for j in idxs_users]).tolist()
        logger.info("Update Global Model")
        set_parameters(global_model, theta_avg)

        # ---- 全局原型聚合（数据量加权） ----
        global_label_0_prototype, global_label_1_prototype = None, None
        for i, client_id in enumerate(idxs_users):
            w = float(average_weight[client_id])
            r = results[i]
            if r['label_0_prototype'] is not None:
                global_label_0_prototype = (w * r['label_0_prototype']
                                            if global_label_0_prototype is None
                                            else global_label_0_prototype + w * r['label_0_prototype'])
                semantic_portrait_list[client_id][0] = r['label_0_prototype']
            if r['label_1_prototype'] is not None:
                global_label_1_prototype = (w * r['label_1_prototype']
                                            if global_label_1_prototype is None
                                            else global_label_1_prototype + w * r['label_1_prototype'])
                semantic_portrait_list[client_id][1] = r['label_1_prototype']
        if global_label_0_prototype is not None:
            global_label_0_prototype_list.append(global_label_0_prototype)
        if global_label_1_prototype is not None:
            global_label_1_prototype_list.append(global_label_1_prototype)

        # ---- 更新样本跳过机制的全局统计量 ----
        for i, client_id in enumerate(idxs_users):
            epsilon_bar += float(average_weight[client_id]) * results[i]['epsilon_sum']
            Q_bar += float(average_weight[client_id]) * results[i]['Q_sum']

        # ---- 更新画像距离表 ----
        portrait_distance_table = get_PDT(num_clients_K, semantic_portrait_list)

        # ---- 分类头扰动（防过拟合），更新分发预测头元组 ----
        logger.info("Perturb CLF module")
        for index, clf in enumerate(clf_list):
            clf_list[index] = model_perturb(clf, gamma_mask)
        clf_tuple = (idxs_users.tolist(), clf_list)

        avg_gpu_seconds = (total_gpu_seconds / num_clients_K)
        logger.info(f"Global Model testing at Communication {(iter_t + 1)}/ {communication_round_I}")
        logger.info(f"Total GPU seconds: {total_gpu_seconds}, Avg GPU seconds over client: {avg_gpu_seconds}")

        del theta_list
        gc.collect()

        # ---- 非最后一轮测试（最后一轮由外层统一测试） ----
        if (iter_t + 1) != param_dict['communication_round_I']:
            if "SENT_CLF" in param_dict["task"]:
                accuracy, DEO, SPD = FL_fairness_and_accuracy_test(global_model, param_dict, testing_dataloader, testing_dataset_len)
                logger.info(f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)}")
                log_test_metrics(accuracy=float(accuracy), DEO=float(DEO), SPD=float(SPD),
                                 step=iter_t + 1, gpu_seconds=total_gpu_seconds, avg_gpu_seconds=avg_gpu_seconds)
                log_system_metrics(step=iter_t + 1, gpu_seconds=total_gpu_seconds,
                                   selected_client_count=len(idxs_users), selected_clients=idxs_users.tolist())
                flush()
            elif "IMG_CLF" in param_dict["task"]:
                accuracy, DEO, SPD = FL_fairness_and_accuracy_test_4_IMG_CLF(global_model, param_dict, testing_dataloader, testing_dataset_len)
                FR = 1 - DEO
                HM = get_HM_by_two_value(accuracy, FR)
                logger.info(f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)},"
                            f" FR: {round(float(FR), 3)}, HM: {round(float(HM), 3)}")
                log_test_metrics(accuracy=float(accuracy), DEO=float(DEO), SPD=float(SPD),
                                 FR=float(FR), HM=float(HM),
                                 step=iter_t + 1, gpu_seconds=total_gpu_seconds, avg_gpu_seconds=avg_gpu_seconds)
                log_system_metrics(step=iter_t + 1, gpu_seconds=total_gpu_seconds,
                                   selected_client_count=len(idxs_users), selected_clients=idxs_users.tolist())
                flush()
            elif "Tabular_CLF" in param_dict["task"]:
                accuracy, DEO, SPD = FL_fairness_and_accuracy_test_4_Tabular_CLF(global_model, param_dict, testing_dataloader, testing_dataset_len)
                FR = 1 - DEO
                HM = get_HM_by_two_value(accuracy, FR)
                logger.info(f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)},"
                            f" FR: {round(float(FR), 3)}, HM: {round(float(HM), 3)}")
                log_test_metrics(accuracy=float(accuracy), DEO=float(DEO), SPD=float(SPD),
                                 FR=float(FR), HM=float(HM),
                                 step=iter_t + 1, gpu_seconds=total_gpu_seconds, avg_gpu_seconds=avg_gpu_seconds)
                log_system_metrics(step=iter_t + 1, gpu_seconds=total_gpu_seconds,
                                   selected_client_count=len(idxs_users), selected_clients=idxs_users.tolist())
                flush()

        # ---- 深度监控（每轮执行） ----
        cfg_deep = get_monitoring_config(param_dict)
        log_deep_metrics(global_model, param_dict, testing_dataloader,
                         iter_t + 1, client_model_updates=client_model_updates)

        # ---- 保存检查点 ----
        if param_dict.get('checkpoint_save_freq', 1) > 0 and iter_t % param_dict.get('checkpoint_save_freq', 1) == 0:
            save_checkpoint(
                param_dict=param_dict,
                iter_t=iter_t,
                global_model=global_model,
                total_gpu_seconds=total_gpu_seconds,
                client_selection_history=[idxs_users.tolist()] if hasattr(idxs_users, 'tolist') else [idxs_users],
                start_time=start_time,
                extra_state={
                    'Q_bar': Q_bar,
                    'epsilon_bar': epsilon_bar,
                    'semantic_portrait_list': [p.cpu().tolist() for p in semantic_portrait_list],
                    'global_label_0_prototype_list': [p.cpu().tolist() for p in global_label_0_prototype_list],
                    'global_label_1_prototype_list': [p.cpu().tolist() for p in global_label_1_prototype_list],
                }
            )
            clean_old_checkpoints(param_dict, keep_latest=param_dict.get('checkpoint_keep_latest', 5))

    logger.info("Training finish, save and return the global model.")
    save_dir = f'./save_path/'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "global_FedSum.pt")
    torch.save(global_model, save_path)
    clf_MB_size = sum(p.numel() for p in _get_clf_module(global_model, param_dict).parameters()) * 4 / (1024 * 1024)
    total_communication_cost = communication_round_I * num_clients_K * FL_fraction * (2 * model_MB_size + portrait_MB_size + clf_MB_size)
    return global_model, total_gpu_seconds, total_communication_cost
    return global_model, total_gpu_seconds, total_communication_cost
