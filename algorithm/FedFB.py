# https://arxiv.org/pdf/2110.15545
# IMPROVING FAIRNESS VIA FEDERATED LEARNING
# 核心思想：将FairBatch技术扩展到联邦学习中，通过公平批处理采样来缓解数据偏差

import copy
import os
import gc
import time
import torch
import numpy as np
from tool.logger import *
from tool.aggregation import average_state_dicts
from tool.utils import get_parameters
from algorithm.Optimizers import BERTCLF_Optimizer
from algorithm.client_selection import client_selection
from tool.utils import FL_fairness_and_accuracy_test, FL_fairness_and_accuracy_test_4_IMG_CLF, FL_fairness_and_accuracy_test_4_Tabular_CLF, get_HM_by_two_value
from tool.amp_utils import autocast_context, get_scaler, scale_backward, scaler_step, AutocastSafeBCELoss
from tool.client_parallel import ClientParallelExecutor
from tool.tensorboard_logger import log_scalar, log_metrics, log_test_metrics, log_system_metrics, update_step, flush, log_deep_metrics, get_monitoring_config
from tool.checkpoint import save_checkpoint, clean_old_checkpoints
from tool.experiment_state import (
    AlgorithmRunResult,
    capture_training_loader_generator_states,
    restore_training_loader_generator_states,
)


def weighted_loss(criterion, logits, targets, weights, mean=True):
    acc_loss = criterion(logits, targets)
    if mean:
        weights_sum = weights.sum().item()
        acc_loss = torch.sum(acc_loss * weights / weights_sum)
    else:
        acc_loss = torch.sum(acc_loss * weights)
    return acc_loss


def get_logits_from_logistic(p):
    logits = torch.log(p / (1 - p))
    return logits


def FedFB_style_inference(param_dict, device, model, inference_dataloader, bits=False, truem_yz=None):
    """
    Returns the inference accuracy,
                            loss,
                            N(sensitive group, pos),
                            N(non-sensitive group, pos),
                            N(sensitive group),
                            N(non-sensitive group),
                            acc_loss,
                            fair_loss
    """

    model.eval()
    model = model.to(device)
    loss, total, correct, fair_loss, acc_loss, num_batch = 0.0, 0.0, 0.0, 0.0, 0.0, 0
    n_yz, loss_yz, m_yz, f_z = {}, {}, {}, {}

    if "SENT_CLF" in param_dict["task"]:
        criterion = torch.nn.CrossEntropyLoss(reduction='sum').to(device)
    elif "IMG_CLF" in param_dict["task"] or "AUD_CLF" in param_dict["task"] or "Tabular_CLF" in param_dict["task"]:
        criterion = AutocastSafeBCELoss(reduction='sum').to(device)

    for y in [0, 1]:
        for z in range(2):
            loss_yz[(y, z)] = 0
            n_yz[(y, z)] = 0
            m_yz[(y, z)] = 0

    with torch.no_grad():
        for batch in inference_dataloader:
            try:
                if "SENT_CLF" in param_dict["task"]:
                    # input_ids尺寸 [batch_size, max_len]
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                elif "IMG_CLF" in param_dict["task"] or "AUD_CLF" in param_dict["task"]:
                    imgs = batch["img"].to(device)
                elif "Tabular_CLF" in param_dict["task"]:
                    X = batch["X"].to(device)
                # labels尺寸 [batch_size]
                labels = batch["labels"].to(device)

                sensitive = batch["protected"].to(device)
                # Inference & Prediction
                if "SENT_CLF" in param_dict["task"]:
                    # features尺寸 [batch_size, emb_dim]
                    # logits尺寸 [batch_size, category]
                    _, logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    # activated_preds = logits.softmax(dim=1)
                    activated_preds = logits  # 由于我们采用了torch.nn.CrossEntropyLoss，在Pytorch里面这个函数是已经加了softmax的，所以我们不需要再手动加softmax
                    _, pred_labels = torch.max(activated_preds, dim=1)

                elif "IMG_CLF" in param_dict["task"] or "AUD_CLF" in param_dict["task"]:
                    # preds尺寸 [batch_size, 1]，里面只有0和1
                    # features尺寸 [batch_size, emb_dim]
                    # tmp_logits尺寸 [batch_size, 1]，里面是预测为1的概率
                    tmp_logits, preds, _ = model(imgs, return_logit=True)
                    # RegularCNN给出来的结果没有经过归一化，所以得手动处理下
                    tmp_logits_min = tmp_logits.min()
                    tmp_logits_max = tmp_logits.max()
                    tmp_logits_range = tmp_logits_max - tmp_logits_min
                    # logits尺寸 [batch_size, 2]
                    logits = torch.tensor([[(tmp_logits_range-item+tmp_logits_min)/tmp_logits_range, (item-tmp_logits_min)/tmp_logits_range] for item in tmp_logits]).to(device)
                    # pred_labels尺寸 [batch_size, ]，里面只有0和1
                    pred_labels = preds.squeeze(1)

                elif "Tabular_CLF" in param_dict["task"]:
                    # local_prediction尺寸 [batch_size, 1]，里面是预测为1的概率
                    if "ANN" in str(type(model)):
                        local_prediction, features = model(X)
                    elif "LogisticRegression" in str(type(model)):
                        local_prediction = model(X)
                    else:
                        local_prediction = model(X)
                    # pred_labels尺寸 [batch_size, 1]，里面只有True和False
                    pred_labels = (local_prediction >= 0.5).squeeze(1)
                    # 逻辑回归给出来的结果没有经过归一化，有可能inf或者nan，所以得手动处理下
                    tmp_logits_min = local_prediction.min()
                    tmp_logits_max = local_prediction.max()
                    tmp_logits_range = tmp_logits_max - tmp_logits_min
                    # logits尺寸 [batch_size, 2]
                    logits = torch.tensor([[(tmp_logits_range - item + tmp_logits_min) / tmp_logits_range,
                                            (item - tmp_logits_min) / tmp_logits_range] for item in local_prediction]).to(device)


                correct += sum(pred_labels.eq(labels)).item()
                total += len(labels)
                num_batch += 1

                group_boolean_idx = {}

                for yz in n_yz:
                    group_boolean_idx[yz] = (labels == yz[0]) & (sensitive == yz[1])
                    n_yz[yz] += torch.sum((pred_labels == yz[0]) & (sensitive == yz[1])).item()
                    m_yz[yz] += torch.sum((labels == yz[0]) & (sensitive == yz[1])).item()

                    try:
                        if group_boolean_idx[yz].sum() != 0:
                            # the objective function have no lagrangian term
                            if "SENT_CLF" in param_dict["task"]:
                                acc_loss = criterion(activated_preds[group_boolean_idx[yz]], labels[group_boolean_idx[yz]])
                            elif "IMG_CLF" in param_dict["task"] or "AUD_CLF" in param_dict["task"]:
                                if torch.isnan(preds[group_boolean_idx[yz],0]).any():
                                    acc_loss = torch.tensor(0).to(device)
                                else:
                                    acc_loss = criterion(preds[group_boolean_idx[yz],0], labels[group_boolean_idx[yz]].float())
                            elif "Tabular_CLF" in param_dict["task"]:
                                if torch.isnan(local_prediction[group_boolean_idx[yz],0]).any():
                                    acc_loss = torch.tensor(0).to(device)
                                else:
                                    acc_loss = criterion(local_prediction[group_boolean_idx[yz],0], labels[group_boolean_idx[yz]].float())
                            loss_yz[yz] += acc_loss.item()

                    except Exception as e:
                        logger.info(f"*** Some error Happen during calculating the acc_loss！： {e} ***")


                # logits_1尺寸 [batch_size, 2]
                logits_1 = logits
                logits_0 = 1-logits_1

                fair_loss0 = torch.mul(sensitive - sensitive.type(torch.FloatTensor).mean(),
                                       torch.max(logits_0 - torch.mean(logits_0), dim=1)[1])
                fair_loss0 = torch.mean(torch.mul(fair_loss0, fair_loss0))
                fair_loss1 = torch.mul(sensitive - sensitive.type(torch.FloatTensor).mean(),
                                       torch.max(logits_1 - torch.mean(logits_1), dim=1)[1])
                fair_loss1 = torch.mean(torch.mul(fair_loss1, fair_loss1))
                fair_loss = fair_loss0 + fair_loss1
                if "SENT_CLF" in param_dict["task"]:
                    acc_loss = criterion(activated_preds, labels)
                elif "IMG_CLF" in param_dict["task"] or "AUD_CLF" in param_dict["task"]:
                    acc_loss = criterion(preds[:,0], labels.float())
                elif "Tabular_CLF" in param_dict["task"]:
                    acc_loss = criterion(local_prediction[:,0], labels.float())

                batch_loss, batch_acc_loss, batch_fair_loss = acc_loss, acc_loss, fair_loss

                loss, acc_loss, fair_loss = (loss + batch_loss.item(),
                                             acc_loss + batch_acc_loss.item(),
                                             fair_loss + batch_fair_loss)

            except Exception as e:
                continue

        try:
            accuracy = correct / total
        except ZeroDivisionError as zde:
            accuracy = 0
        for z in range(1, 2):
            f_z[z] = - loss_yz[(0, 0)] / (truem_yz[(0, 0)] + truem_yz[(1, 0)])
            + loss_yz[(1, 0)] / (truem_yz[(0, 0)] + truem_yz[(1, 0)])
            + loss_yz[(0, z)] / (truem_yz[(0, z)] + truem_yz[(1, z)])
            - loss_yz[(1, z)] / (truem_yz[(0, z)] + truem_yz[(1, z)])

        # return accuracy, loss, n_yz, acc_loss / num_batch, fair_loss / num_batch, f_z
        # 检查代码发现acc_loss / num_batch 和 fair_loss / num_batch是没有用到的，所以可以直接反0
        return accuracy, loss, n_yz, 0, 0, f_z


def _train_single_client_fedfb(client_id, device, model, param_dict, training_dataloaders,
                               algorithm_epoch_T, client_datasets_size_list, num_clients_K,
                               basic_path, iter_t, communication_round_I, lbd, m_yz, use_amp, scaler, criterion):
    """FedFB 单客户端训练函数（供 ClientParallelExecutor 调用）"""
    logger.info("Copy From Global Model")
    model.train()
    model.to(device)
    optimizer = BERTCLF_Optimizer(
        method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
    optimizer.set_parameters(list(model.named_parameters()))
    client_i_dataloader = training_dataloaders[client_id]

    nc = 0  # New Param in FedFB

    for epoch in range(algorithm_epoch_T):
        epoch_total_loss = 0
        epoch_total_size = 0

        nc = 0

        gpu_start_time = time.time()

        for batch in client_i_dataloader:
            if "SENT_CLF" in param_dict["task"]:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
            elif "IMG_CLF" in param_dict["task"] or "AUD_CLF" in param_dict["task"]:
                imgs = batch["img"].to(device)
            elif "Tabular_CLF" in param_dict["task"]:
                X = batch["X"].to(device)

            labels = batch["labels"].to(device)
            protected = batch["protected"].to(device)

            true_batch_size = labels.size()[0]
            epoch_total_size += true_batch_size

            v = torch.ones(true_batch_size).type(torch.DoubleTensor).to(device)
            group_idx = {}
            for y, z in lbd:
                group_idx[(y, z)] = torch.where((labels == y) & (protected == z))[0].to(device)
                v[group_idx[(y, z)]] = lbd[(y, z)] / (m_yz[(1, z)] + m_yz[(0, z)])
                nc += v[group_idx[(y, z)]].sum().item()

            with autocast_context(device, use_amp):
                if "SENT_CLF" in param_dict["task"]:
                    features, logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    activated_preds = logits
                    _, preds = torch.max(activated_preds, dim=1)
                    batch_loss_sum = weighted_loss(criterion, activated_preds, labels, v, mean=False)

                elif "IMG_CLF" in param_dict["task"] or "AUD_CLF" in param_dict["task"]:
                    preds, features = model(imgs)
                    batch_loss_sum = weighted_loss(criterion, preds[:, 0], labels.float(), v, mean=False)

                elif "Tabular_CLF" in param_dict["task"]:
                    if "ANN" in str(type(model)):
                        local_prediction, features = model(X)
                    elif "LogisticRegression" in str(type(model)):
                        local_prediction = model(X)
                    else:
                        local_prediction = model(X)
                    batch_loss_sum = weighted_loss(criterion, local_prediction[:, 0], labels.float(), v, mean=False)

            if batch_loss_sum.item() != 0:
                loss = batch_loss_sum / true_batch_size
                scale_backward(loss, scaler)
                scaler_step(scaler, optimizer)
            else:
                loss = 0

            model.zero_grad()
            epoch_total_loss += float(loss.detach()) if torch.is_tensor(loss) else float(loss)

            if "SENT_CLF" in param_dict["task"]:
                del input_ids, attention_mask, labels
            elif "IMG_CLF" in param_dict["task"] or "AUD_CLF" in param_dict["task"]:
                del imgs, labels

            gc.collect()

        gpu_end_time = time.time()

        epoch_total_size = max(epoch_total_size, client_datasets_size_list[client_id])

        average_one_sample_loss_in_epoch = epoch_total_loss / epoch_total_size
        logger.info(f"Communication Round: {iter_t + 1} / {communication_round_I}; "
                    f"Client: {client_id} / {num_clients_K}; "
                    f"Epoch: {epoch + 1}; Avg One Sample's Loss Over Epoch: {average_one_sample_loss_in_epoch}")

    # Upgrade the local model list
    client_model_path = os.path.join(basic_path, "client_" + str(client_id + 1), 'model.pt')
    torch.save(model.cpu(), client_model_path)

    del model
    gc.collect()

    return {'nc': nc, 'gpu_seconds': gpu_end_time - gpu_start_time}


def FedFB(device,
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
    training_dataset_size = len(training_dataset.labels)
    client_datasets_size_list = [len(_) for _ in client_dataset_list]
    # AMP 初始化
    use_amp = param_dict.get('use_amp', False)
    scaler = get_scaler(device, use_amp)
    if resume_state is not None and resume_state.amp_scaler_state is not None:
        if scaler is None:
            raise RuntimeError("checkpoint contains AMP scaler state but AMP is disabled")
        scaler.load_state_dict(resume_state.amp_scaler_state)
    # average_weight = np.array([float(i / training_dataset_size) for i in client_datasets_size_list])

    basic_path = os.path.join("./save_path", param_dict['dataset_name'],
                              param_dict['split_strategy'],
                              param_dict['algorithm'],
                              param_dict['hypothesis'],
                              str(num_clients_K) + "Clients")

    # Operation in FedFB
    # the number of samples whose label is y and sensitive attribute is z
    # 原实现写成 `sum(list_a and list_b)`，而 Python 的 `and` 在两个非空 list 上直接
    # 返回第二个 list，于是 m_yz 实际只统计了「敏感属性 == z」的样本、完全忽略标签，
    # 使公平项彻底失效：群组比例修正项恒为 0，f_z 被整体减半，λ 的夹逼上界被放大 2 倍。
    # 这里改为逐元素联合条件求和。
    # The original `sum(list_a and list_b)` relies on Python's `and`, which returns the
    # second non-empty list, so m_yz counted only `sensitive == z` and ignored the label,
    # which silently disabled the fairness term. Count the joint condition elementwise.
    m_yz, lbd = {}, {}
    _fedfb_labels = np.asarray(training_dataset.labels).reshape(-1)
    _fedfb_sensitive = np.asarray(
        getattr(training_dataset, "protected", getattr(training_dataset, "s1", None))
    ).reshape(-1)
    for y in [0, 1]:
        for z in range(2):
            m_yz[(y, z)] = int(np.sum((_fedfb_labels == y) & (_fedfb_sensitive == z)))
    del _fedfb_labels, _fedfb_sensitive
    for y in [0, 1]:
        for z in range(2):
            lbd[(y, z)] = (m_yz[(1, z)] + m_yz[(0, z)]) / len(training_dataset)
    # New Params in FedFB
    alpha = 0.3


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
    elif "IMG_CLF" in param_dict["task"] or "AUD_CLF" in param_dict["task"] or "Tabular_CLF" in param_dict["task"]:
        criterion = AutocastSafeBCELoss(reduction='none').to(device)

    # 断点恢复：恢复计数器、AMP、客户端选择历史与 DataLoader 生成器状态
    total_gpu_seconds = 0.0 if resume_state is None else float(resume_state.total_gpu_seconds)
    previous_state = {} if resume_state is None else dict(resume_state.algorithm_state)
    if resume_state is not None:
        restore_training_loader_generator_states(
            training_dataloaders, previous_state.get("training_loader_generator_states")
        )
    users_gpu_seconds_list = list(previous_state.get("users_gpu_seconds_list", [0.0] * num_clients_K))
    if len(users_gpu_seconds_list) != num_clients_K:
        raise RuntimeError("FedFB checkpoint has incompatible users_gpu_seconds_list")
    total_communication_cost = 0.0 if resume_state is None else float(resume_state.total_communication_cost)
    client_selection_history = [] if resume_state is None else [list(row) for row in resume_state.client_selection_history]
    runtime_offset = 0.0 if resume_state is None else float(resume_state.total_runtime_seconds)
    runtime_start = time.monotonic()
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    # FedFB 的拉格朗日乘子 lbd 为跨轮维护状态，续跑时覆盖为检查点中的值
    if previous_state.get("lbd") is not None:
        lbd = {(int(y), int(z)): float(value) for y, z, value in previous_state["lbd"]}

    # model_MB_size = sys.getsizeof(global_model.state_dict()) / (1024 ** 2)
    model_MB_size = sum(p.numel() for p in global_model.parameters()) * 4 / (1024*1024)
    # logger.info(f"Model's Communication Cost: {model_MB_size} MB")

    parallel_executor = ClientParallelExecutor(device=device, global_model=global_model, param_dict=param_dict, needs_global_model_during_training=False)

    # Simulate Client Parallel
    # TODO:改了迭代的架构，现在有三个for 最外层的for通信轮次 第二层是for每个通信轮次中的客户端训练epoch 第三层是for batch
    for iter_t in range(start_round, communication_round_I):
        round_nc = []  # 本轮选中客户端的 nc（聚合权重）
        # Client Selection
        # 先选客户端，只对选中的客戶下发模型
        idxs_users = client_selection(
            client_num=num_clients_K,
            fraction=FL_fraction,
            dataset_size=training_dataset_size,
            client_dataset_size_list=client_datasets_size_list,
            drop_rate=FL_drop_rate,
            style="FedAvg",
        )
        selected_clients = idxs_users.tolist() if hasattr(idxs_users, "tolist") else list(idxs_users)
        client_selection_history.append([int(client_id) for client_id in selected_clients])

        logger.info(f"Communication Round: {iter_t + 1}; Select clients: {idxs_users}; Start Local Training!")

        # Simulate Client Parallel
        client_results = parallel_executor.run_clients(
            idxs_users, _train_single_client_fedfb,
            param_dict=param_dict,
            training_dataloaders=training_dataloaders,
            algorithm_epoch_T=algorithm_epoch_T,
            client_datasets_size_list=client_datasets_size_list,
            num_clients_K=num_clients_K,
            basic_path=basic_path,
            iter_t=iter_t,
            communication_round_I=communication_round_I,
            lbd=lbd, m_yz=m_yz,
            use_amp=use_amp, scaler=scaler, criterion=criterion,
        )
        # users_gpu_seconds_list 是每个客户端的「累计」用时，若每轮对它整体求和，
        # total_gpu_seconds 会随轮次平方级膨胀。与 FedAvg 口径一致：只累加本轮选中
        # 客户端新产生的 GPU 秒数。
        # users_gpu_seconds_list is cumulative per client; summing the whole list every
        # round makes total_gpu_seconds grow quadratically. Match the FedAvg convention
        # and add only this round's per-client GPU seconds.
        round_gpu_seconds = 0.0
        for i, result in enumerate(client_results):
            id = idxs_users[i]
            users_gpu_seconds_list[id] += result['gpu_seconds']
            round_gpu_seconds += float(result['gpu_seconds'])
            round_nc.append(result['nc'])

        # Communicate
        total_gpu_seconds += round_gpu_seconds
        total_communication_cost += len(idxs_users) * 2 * model_MB_size
        logger.info(f"Communication Round {(iter_t + 1)} 's Communication Cost: {(iter_t + 1) * len(idxs_users) * 2 * model_MB_size} MB")

        # ── 收集客户端模型更新（用于梯度监控）──
        pre_agg_params = get_parameters(global_model)
        client_model_updates = []

        # Global operation
        logger.info("Parameter aggregation")
        client_states = []
        for id in idxs_users:
            client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
            selected_model = torch.load(client_model_path, weights_only=False)  # 持久化
            client_states.append(selected_model.state_dict())  # Operation in FedFB

            # 计算该客户端的更新量
            client_params = get_parameters(selected_model)
            updates = {}
            for j, (p_local, p_global) in enumerate(zip(client_params, pre_agg_params)):
                updates[str(j)] = torch.tensor(p_local) - torch.tensor(p_global)
            client_model_updates.append(updates)

            del selected_model
            gc.collect()

        # 记录GPU计算开始时间
        gpu_start_time = time.time()
        # 只用本轮选中客户端的 nc 做加权：global_nc 跨轮累积会导致分母
        # 随轮次膨胀、分子索引错位，全局权重被过度缩小。
        # 本轮 nc 全为 0 时退回算术平均（weights=None）。
        _nc_weights = round_nc if sum(round_nc) > 0 else None
        aggregated_state = average_state_dicts(client_states, weights=_nc_weights)  # Operation in FedFB
        logger.info("Update Global Model")
        global_model.load_state_dict(aggregated_state, strict=True)  # Operation in FedFB
        # 记录GPU计算结束时间
        gpu_end_time = time.time()
        total_gpu_seconds += (gpu_end_time - gpu_start_time)


        # 记录GPU计算开始时间
        gpu_start_time = time.time()
        # Calculate avg training accuracy over all clients at every round
        list_acc = []
        # the number of samples which are assigned to class y and belong to the sensitive group z
        n_yz, f_z = {}, {}
        for z in range(2):
            f_z[z] = 0
            for y in [0, 1]:
                n_yz[(y, z)] = 0
        global_model.eval()
        for i in range(num_clients_K):
            client_i_dataloader = training_dataloaders[i]
            acc, loss, n_yz_c, acc_loss, fair_loss, f_z_c = FedFB_style_inference(param_dict, device, global_model,
                                                                                  client_i_dataloader, False, m_yz)
            list_acc.append(acc)

            for yz in n_yz:
                n_yz[yz] += n_yz_c[yz]

            for z in range(1, 2):
                f_z[z] += f_z_c[z]
                tmp_0 =  m_yz[(0, 0)] + m_yz[(1, 0)]
                if tmp_0 != 0:
                    f_z[z] += m_yz[(0, 0)] / tmp_0
                tmp_1 =  m_yz[(0, z)] + m_yz[(1, z)]
                if tmp_1 != 0:
                    f_z[z] += -m_yz[(0, z)] / tmp_1

        for z in range(2):
            if z == 0:
                lbd[(0, z)] -= alpha / (iter_t + 1) ** .5 * sum([f_z[z] for z in range(1, 2)])
                # lbd[(0, z)] = lbd[(0, z)].item()
                lbd[(0, z)] = max(0, min(lbd[(0, z)], 2 * (m_yz[(1, 0)] + m_yz[(0, 0)]) / len(training_dataset)))
                lbd[(1, z)] = 2 * (m_yz[(1, 0)] + m_yz[(0, 0)]) / len(training_dataset) - lbd[(0, z)]
            else:
                lbd[(0, z)] += alpha / (iter_t + 1) ** .5 * f_z[z]
                # lbd[(0, z)] = lbd[(0, z)].item()
                lbd[(0, z)] = max(0, min(lbd[(0, z)], 2 * (m_yz[(1, 0)] + m_yz[(0, 0)]) / len(training_dataset)))
                lbd[(1, z)] = 2 * (m_yz[(1, 0)] + m_yz[(0, 0)]) / len(training_dataset) - lbd[(0, z)]
        # 记录GPU计算结束时间
        gpu_end_time = time.time()
        total_gpu_seconds += (gpu_end_time - gpu_start_time)

        # 当前消耗的总GPU秒，平均GPU秒
        avg_gpu_seconds = (total_gpu_seconds / num_clients_K)
        logger.info(
            f"Global Model testing at Communication {(iter_t + 1)}/ {communication_round_I}")
        logger.info(
            f"Total GPU seconds: {total_gpu_seconds}, Avg GPU seconds over client: {avg_gpu_seconds}")

        del client_states
        gc.collect()

        # 没有到达最后一次通信轮次之前，都要做测试
        if (iter_t + 1) != param_dict['communication_round_I'] and (iter_t + 1) % param_dict.get('test_eval_freq', 1) == 0:
            try:
                if "SENT_CLF" in param_dict["task"]:
                    evaluation = FL_fairness_and_accuracy_test(global_model, param_dict, testing_dataloader,
                                                                       testing_dataset_len)
                    accuracy, DEO, SPD, ACC_gap = evaluation[0], evaluation[1], evaluation[2], evaluation.acc_gap
                    logger.info(
                        f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)}, ACC_gap: {round(float(ACC_gap), 3)}")
                    # ===== TensorBoard logging =====
                    log_test_metrics(
                        ACC_gap=float(ACC_gap), accuracy=float(accuracy), DEO=float(DEO), SPD=float(SPD),
                        step=iter_t+1, gpu_seconds=total_gpu_seconds, avg_gpu_seconds=avg_gpu_seconds,
                        communication_cost=(iter_t + 1) * len(idxs_users) * 2 * model_MB_size
                    )
                    log_system_metrics(step=iter_t+1, gpu_seconds=total_gpu_seconds, 
                                       communication_cost=(iter_t + 1) * len(idxs_users) * 2 * model_MB_size,
                                       selected_client_count=len(idxs_users), model_mb_size=model_MB_size)
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
                        step=iter_t+1, gpu_seconds=total_gpu_seconds, avg_gpu_seconds=avg_gpu_seconds,
                        communication_cost=(iter_t + 1) * len(idxs_users) * 2 * model_MB_size
                    )
                    log_system_metrics(step=iter_t+1, gpu_seconds=total_gpu_seconds, 
                                       communication_cost=(iter_t + 1) * len(idxs_users) * 2 * model_MB_size,
                                       selected_client_count=len(idxs_users), model_mb_size=model_MB_size)
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
                        step=iter_t+1, gpu_seconds=total_gpu_seconds, avg_gpu_seconds=avg_gpu_seconds,
                        communication_cost=(iter_t + 1) * len(idxs_users) * 2 * model_MB_size
                    )
                    log_system_metrics(step=iter_t+1, gpu_seconds=total_gpu_seconds, 
                                       communication_cost=(iter_t + 1) * len(idxs_users) * 2 * model_MB_size,
                                       selected_client_count=len(idxs_users), model_mb_size=model_MB_size)
                    flush()


            except Exception as e:
                logger.info("Exception: {}".format(e))
                logger.info("!!!!!!!!!!!!! Skipping the middle test in communication round {} !!!!!!!!!".format(iter_t + 1))

        # ===== 深度监控（每轮都执行，包括最后一轮）=====
        cfg_deep = get_monitoring_config(param_dict)
        log_deep_metrics(global_model, param_dict, testing_dataloader, 
                         iter_t + 1, client_model_updates=client_model_updates)

        # 保存检查点（按 checkpoint_save_freq 间隔）
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
                    "lbd": [[y, z, lbd[(y, z)]] for y in [0, 1] for z in range(2)],
                },
                amp_scaler=scaler,
                total_gpu_seconds=total_gpu_seconds,
                total_runtime_seconds=runtime_offset + (time.monotonic() - runtime_start),
                total_communication_cost=total_communication_cost,
                client_selection_history=client_selection_history,
                start_time=start_time
            )

            # 清理旧检查点，保留最近 N 个
            clean_old_checkpoints(param_dict, keep_latest=param_dict.get('checkpoint_keep_latest', 5))

    logger.info("Training finish, save and return the global model.")
    # Save global model
    save_dir = f'./save_path/'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"global_FedFB.pt")
    torch.save(global_model, save_path)
    return AlgorithmRunResult(
        global_model=global_model,
        total_gpu_seconds=total_gpu_seconds,
        total_communication_cost=total_communication_cost,
        algorithm_state={
            "users_gpu_seconds_list": users_gpu_seconds_list,
            "training_loader_generator_states": capture_training_loader_generator_states(training_dataloaders),
            "lbd": [[y, z, lbd[(y, z)]] for y in [0, 1] for z in range(2)],
        },
        amp_scaler_state=None if scaler is None else scaler.state_dict(),
        client_selection_history=client_selection_history,
    )