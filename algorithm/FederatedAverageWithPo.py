# -*- coding: utf-8 -*-
# FederatedAverageWithPo (Fed_AVG_Po): FedAvg + Class Prototype + L_Po
# 论文链接 Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/34129/36284 (FedSum, AAAI 2025)
# 核心思想 Core Idea:
#   本算法是引入了FedSum的L_Po机制：在标准 FedAvg 之上仅引入"类别原型作为全局知识载体"，
#   客户端在本地训练中以 L_Po 将本地类原型锚定到全局类原型（并拉大预测原型间距），
#   服务器按样本量加权聚合模型参数与类原型并回传。不含 FedSum 的样本跳过、语义画像距离、
#   跨客户端知识迁移与分类头扰动等互补机制，用于隔离验证"原型载体 + L_Po 锚定"本身的贡献。

# 与 FedSum 的机制对应关系 Mechanism mapping to FedSum:
#   L_Po = 全局-本地类原型对齐 (global-local class prototype alignment)
#        + 拉大预测原型间距 (prediction-prototype repulsion)
#   通信内容 Communication: 模型参数 + 2 个类别原型（无敏感属性统计量）

import os
import gc
import time
import math
import torch
import numpy as np
from tool.logger import *
from algorithm.Optimizers import BERTCLF_Optimizer
from algorithm.client_selection import client_selection
from tool.utils import get_parameters, set_parameters, get_HM_by_two_value
from tool.utils import FL_fairness_and_accuracy_test, FL_fairness_and_accuracy_test_4_IMG_CLF, FL_fairness_and_accuracy_test_4_Tabular_CLF
from tool.checkpoint import save_checkpoint, clean_old_checkpoints, load_checkpoint
from tool.amp_utils import autocast_context, get_scaler, scale_backward, scaler_step
from tool.tensorboard_logger import log_test_metrics, log_system_metrics, flush, get_monitoring_config
from tool.client_parallel import ClientParallelExecutor


# ==================== 任务适配辅助函数 Task-adaptive helpers ====================

def _extract_features(model, input_ids=None, attention_mask=None, imgs=None, X=None, param_dict=None):
    """提取表征特征 Extract representation features
    （SENT: PLM 输出; IMG/ANN: backbone 输出; LogisticRegression: 输入本身）
    """
    if "SENT_CLF" in param_dict["task"]:
        return model.only_PLM_forward(input_ids=input_ids, attention_mask=attention_mask)
    elif "IMG_CLF" in param_dict["task"]:
        return model.only_backbone_forward(imgs)
    else:  # Tabular_CLF
        if "ANN" in str(type(model)):
            return model.only_backbone_forward(X)
        else:
            return X  # LogisticRegression 无 backbone，特征即输入 no backbone, features are inputs


# ==================== 单客户端训练 Single-client training ====================

def _train_single_client_avg_po(client_id, device, model, param_dict,
                                training_dataloaders, algorithm_epoch_T,
                                accumulation_steps, use_amp, scaler, criterion,
                                basic_path, iter_t, communication_round_I, num_clients_K,
                                global_label_0_prototype, global_label_1_prototype,
                                lambda_lpo):
    """Fed_AVG_Po 单客户端训练（可被 ClientParallelExecutor 调度）。
    Single-client local training for Fed_AVG_Po (schedulable by ClientParallelExecutor).
    返回 Returns: {'gpu_seconds', 'label_0_prototype', 'label_1_prototype'}
    """
    model.train()
    model.to(device)
    optimizer = BERTCLF_Optimizer(
        method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
    optimizer.set_parameters(list(model.named_parameters()))
    client_i_dataloader = training_dataloaders[client_id]

    gpu_seconds_for_client = 0
    client_i_label_0_feature_list = []
    client_i_label_1_feature_list = []

    for epoch in range(algorithm_epoch_T):
        epoch_total_loss = 0.0    # 含 L_Po 的总损失 total loss with L_Po
        epoch_task_loss = 0.0     # 纯任务损失 pure task loss
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
                elif "IMG_CLF" in param_dict["task"]:
                    preds, features = model(imgs)
                    batch_loss = criterion(preds[:, 0], labels.float())
                else:  # Tabular_CLF
                    if "ANN" in str(type(model)):
                        preds, features = model(X)
                    else:  # LogisticRegression
                        preds = model(X)
                        features = X
                    batch_loss = criterion(preds[:, 0], labels.float())

                loss = torch.sum(batch_loss) / true_batch_size
                task_loss_val = float(loss.detach())

                # ---- 原型损失 L_Po：全局-本地类原型对齐 + 拉大预测原型间距 ----
                # ---- Prototype loss L_Po: global-local alignment + prediction-prototype repulsion ----
                lpo_loss = 0
                mask_0 = (labels <= 0.5)
                mask_1 = (labels > 0.5)
                if global_label_0_prototype is not None and mask_0.any():
                    local_0_proto = features[mask_0].mean(dim=0)
                    lpo_loss = lpo_loss + torch.norm(
                        local_0_proto - global_label_0_prototype.to(device), p=2)
                if global_label_1_prototype is not None and mask_1.any():
                    local_1_proto = features[mask_1].mean(dim=0)
                    lpo_loss = lpo_loss + torch.norm(
                        local_1_proto - global_label_1_prototype.to(device), p=2)
                if mask_0.any() and mask_1.any():
                    predict_0_proto = features[mask_0].mean(dim=0)
                    predict_1_proto = features[mask_1].mean(dim=0)
                    lpo_loss = lpo_loss - torch.norm(predict_1_proto - predict_0_proto, p=2)
                loss = loss + lambda_lpo * lpo_loss

            # ---- 收集类原型素材（detach，用于上传） Collect features for prototype upload (detached) ----
            features_cpu = features.detach().cpu()
            for idx, feat in enumerate(features_cpu):
                if float(labels[idx]) > 0.5:
                    client_i_label_1_feature_list.append(feat)
                else:
                    client_i_label_0_feature_list.append(feat)

            # ---- 反向传播 Backward ----
            scale_backward(loss, scaler)
            if (batch_id + 1) % accumulation_steps == 0:
                scaler_step(scaler, optimizer)
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

        # flush 尾部梯度 flush tail gradients
        if (batch_id + 1) % accumulation_steps != 0:
            scaler_step(scaler, optimizer)
            model.zero_grad()

        avg_task_loss = epoch_task_loss / max(epoch_total_size, 1)
        avg_total_loss = epoch_total_loss / max(epoch_total_size, 1)
        logger.info(f"Communication Round: {iter_t + 1} / {communication_round_I}; "
                    f"Client: {client_id} / {num_clients_K}; "
                    f"Epoch: {epoch + 1}; Avg Task Loss: {avg_task_loss:.4f}; "
                    f"Avg Total Loss (w/ L_Po): {avg_total_loss:.4f}")

    # ---- 本地类原型（上传给服务器聚合） Local class prototypes for server aggregation ----
    def _safe_prototype(feature_list):
        if len(feature_list) != 0:
            return torch.stack(feature_list, dim=0).mean(dim=0)
        return None

    client_i_label_0_prototype = _safe_prototype(client_i_label_0_feature_list)
    client_i_label_1_prototype = _safe_prototype(client_i_label_1_feature_list)

    # 保存模型（持久化） persist client model
    client_model_path = os.path.join(basic_path, "client_" + str(client_id + 1), 'model.pt')
    os.makedirs(os.path.dirname(client_model_path), exist_ok=True)
    torch.save(model.cpu(), client_model_path)

    del model
    gc.collect()

    return {
        'gpu_seconds': gpu_seconds_for_client,
        'label_0_prototype': client_i_label_0_prototype,
        'label_1_prototype': client_i_label_1_prototype,
    }


# ==================== Fed_AVG_Po 主训练循环 Main training loop ====================

def Fed_AVG_Po(device,
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

    # AMP 初始化 AMP init
    use_amp = param_dict.get('use_amp', False)
    scaler = get_scaler(device, use_amp)

    # ---- 数据统计 Data statistics ----
    training_dataset_size = len(training_dataset.labels) if hasattr(training_dataset, 'labels') else len(training_dataset)
    client_datasets_size_list = [len(_) for _ in client_dataset_list]
    del training_dataset, client_dataset_list
    gc.collect()
    average_weight = np.array([float(size / training_dataset_size) for size in client_datasets_size_list])

    # ---- Fed_AVG_Po 超参 Hyper-parameter ----
    lambda_lpo = float(param_dict.get('FedAvgPo_lpo_λ', 1.0))  # L_Po 权重 weight of L_Po

    basic_path = param_dict['model_path']
    for k in range(param_dict["num_clients_K"]):  # 持久化 persist
        full_path = os.path.join(basic_path, "client_" + str(k + 1), 'model.pt')
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        torch.save(global_model, full_path)

    # 损失函数 Loss functions
    if "SENT_CLF" in param_dict["task"]:
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
    else:
        criterion = torch.nn.BCELoss(reduction='none').to(device)

    total_gpu_seconds = 0
    users_gpu_seconds_list = [0] * num_clients_K
    model_MB_size = sum(p.numel() for p in global_model.parameters()) * 4 / (1024 * 1024)
    prototype_MB_size = (2 * param_dict.get('emb_dim', 768)) * 4 / (1024 ** 2)

    # ---- 全局状态 Global states ----
    global_label_0_prototype = None  # 全局 0 类原型 global class-0 prototype
    global_label_1_prototype = None  # 全局 1 类原型 global class-1 prototype

    # 断点恢复 Resume from checkpoint
    if start_round > 0:
        ckpt = load_checkpoint(param_dict)
        if ckpt is not None:
            es = ckpt.get('extra_state', {})
            try:
                saved_0 = es.get('global_label_0_prototype', None)
                saved_1 = es.get('global_label_1_prototype', None)
                if saved_0 is not None:
                    global_label_0_prototype = torch.tensor(saved_0)
                if saved_1 is not None:
                    global_label_1_prototype = torch.tensor(saved_1)
            except Exception as e:
                logger.warning(f"Fed_AVG_Po checkpoint extra_state restore failed: {e}")

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
        # Client Selection（FedAvg 风格 FedAvg-style selection）
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
            _train_single_client_avg_po,
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
            global_label_0_prototype=global_label_0_prototype,
            global_label_1_prototype=global_label_1_prototype,
            lambda_lpo=lambda_lpo,
        )

        # 收集 GPU 计时 Collect GPU timing
        for i, client_id in enumerate(idxs_users):
            users_gpu_seconds_list[client_id] += results[i]['gpu_seconds']
        total_gpu_seconds += sum(users_gpu_seconds_list)

        # ---- 参数聚合（样本量加权）Parameter aggregation (sample-size weighted) ----
        logger.info("Parameter aggregation")
        pre_agg_params = get_parameters(global_model)
        theta_list = []
        for id in idxs_users:
            client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
            selected_model = torch.load(client_model_path, weights_only=False)
            theta_list.append(get_parameters(selected_model))
            del selected_model
            gc.collect()

        theta_list = np.array(theta_list, dtype=object)
        theta_avg = np.average(theta_list, axis=0, weights=[client_datasets_size_list[j] for j in idxs_users]).tolist()
        logger.info("Update Global Model")
        set_parameters(global_model, theta_avg)

        # ---- 全局类原型聚合（样本量加权，仅对被选中客户端求和后归一化） ----
        # ---- Aggregate global class prototypes (weighted by client sample size) ----
        agg_weight_sum_0, agg_weight_sum_1 = 0.0, 0.0
        new_global_0, new_global_1 = None, None
        for i, client_id in enumerate(idxs_users):
            w = float(average_weight[client_id])
            r = results[i]
            if r['label_0_prototype'] is not None:
                new_global_0 = w * r['label_0_prototype'] if new_global_0 is None else new_global_0 + w * r['label_0_prototype']
                agg_weight_sum_0 += w
            if r['label_1_prototype'] is not None:
                new_global_1 = w * r['label_1_prototype'] if new_global_1 is None else new_global_1 + w * r['label_1_prototype']
                agg_weight_sum_1 += w
        if new_global_0 is not None and agg_weight_sum_0 > 0:
            global_label_0_prototype = new_global_0 / agg_weight_sum_0
        if new_global_1 is not None and agg_weight_sum_1 > 0:
            global_label_1_prototype = new_global_1 / agg_weight_sum_1

        avg_gpu_seconds = (total_gpu_seconds / num_clients_K)
        logger.info(f"Global Model testing at Communication {(iter_t + 1)}/ {communication_round_I}")
        logger.info(f"Total GPU seconds: {total_gpu_seconds}, Avg GPU seconds over client: {avg_gpu_seconds}")

        del theta_list
        gc.collect()

        # ---- 非最后一轮测试（最后一轮由外层统一测试） ----
        # ---- Test at non-final rounds (final round is tested by outer loop) ----
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

        # ---- 保存检查点 Save checkpoint ----
        if param_dict.get('checkpoint_save_freq', 1) > 0 and iter_t % param_dict.get('checkpoint_save_freq', 1) == 0:
            save_checkpoint(
                param_dict=param_dict,
                iter_t=iter_t,
                global_model=global_model,
                total_gpu_seconds=total_gpu_seconds,
                client_selection_history=[idxs_users.tolist()] if hasattr(idxs_users, 'tolist') else [idxs_users],
                start_time=start_time,
                extra_state={
                    'global_label_0_prototype': global_label_0_prototype.cpu().tolist() if global_label_0_prototype is not None else None,
                    'global_label_1_prototype': global_label_1_prototype.cpu().tolist() if global_label_1_prototype is not None else None,
                }
            )
            clean_old_checkpoints(param_dict, keep_latest=param_dict.get('checkpoint_keep_latest', 5))

    logger.info("Training finish, save and return the global model.")
    save_dir = f'./save_path/'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "global_Fed_AVG_Po.pt")
    torch.save(global_model, save_path)
    # 通信成本：上传模型+2类原型，下载模型 Up: model + 2 prototypes, Down: model
    total_communication_cost = communication_round_I * num_clients_K * FL_fraction * (2 * model_MB_size + prototype_MB_size)
    return global_model, total_gpu_seconds, total_communication_cost
