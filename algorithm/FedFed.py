# FedFed: Feature Distillation against Data Heterogeneity in Federated Learning
# https://arxiv.org/abs/2310.05077 (NeurIPS 2023) | https://github.com/visitworld123/FedFed
# 核心思想: 操作空间为中间特征空间（activation 蒸馏对齐）。
# 客户端上传中间层 activation 统计量，服务器聚合后通过特征蒸馏指导本地训练。
# Core Idea: Operates in intermediate feature space (activation distillation alignment).
# Clients upload intermediate-layer activation statistics; the server aggregates them
# and guides local training via feature distillation.

import copy
import os
import gc
import time
import torch
import torch.nn.functional as F
import numpy as np
from tool.logger import *
from tool.utils import get_parameters, set_parameters
from algorithm.Optimizers import BERTCLF_Optimizer
from algorithm.client_selection import client_selection
from tool.utils import FL_fairness_and_accuracy_test, FL_fairness_and_accuracy_test_4_IMG_CLF, FL_fairness_and_accuracy_test_4_Tabular_CLF, get_HM_by_two_value
from tool.checkpoint import save_checkpoint, clean_old_checkpoints
from tool.amp_utils import autocast_context, get_scaler, scale_backward, scaler_step
from tool.tensorboard_logger import log_test_metrics, log_system_metrics, flush, log_deep_metrics, get_monitoring_config
from tool.client_parallel import ClientParallelExecutor


def _extract_intermediate_features(model, param_dict, batch, device):
    """提取 backbone 的中间特征 activation（最终 penultimate + 1 层返回的 features）。"""
    if "SENT_CLF" in param_dict["task"]:
        ids = batch["input_ids"].to(device); am = batch["attention_mask"].to(device)
        features, _ = model(input_ids=ids, attention_mask=am)
        return features  # [B, D]
    elif "IMG_CLF" in param_dict["task"]:
        im = batch["img"].to(device)
        _, features = model(im)
        return features  # [B, D]
    else:
        X = batch["X"].to(device)
        if "ANN" in str(type(model)):
            _, features = model(X)
        else:
            features = X  # LogisticRegression 无中间特征
        return features


def _train_single_client_fedfed(client_id, device, model, param_dict,
                                 training_dataloaders, algorithm_epoch_T,
                                 accumulation_steps, use_amp, scaler, criterion,
                                 basic_path, iter_t, communication_round_I, num_clients_K,
                                 global_feat_stats):
    """FedFed 单客户端：本地任务训练 + 特征蒸馏（对齐全局 activation 统计量）。"""
    model.train()
    model.to(device)
    optimizer = BERTCLF_Optimizer(
        method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
    optimizer.set_parameters(list(model.named_parameters()))
    client_i_dataloader = training_dataloaders[client_id]
    gpu_seconds = 0

    fd_lambda = float(param_dict.get('FedFed_lambda', 0.5))

    # 本地特征统计量收集
    local_feat_mean = None
    local_feat_var = None
    local_feat_count = 0

    for epoch in range(algorithm_epoch_T):
        el, es = 0, 0
        for batch_id, batch in enumerate(client_i_dataloader):
            if "SENT_CLF" in param_dict["task"]:
                ids = batch["input_ids"].to(device); am = batch["attention_mask"].to(device)
                lb = batch["labels"].to(device)
                features, logits = model(input_ids=ids, attention_mask=am)
                bl = criterion(logits, lb)
            elif "IMG_CLF" in param_dict["task"]:
                im = batch["img"].to(device); lb = batch["labels"].to(device)
                preds, features = model(im)
                bl = criterion(preds[:, 0], lb.float())
            else:
                X = batch["X"].to(device); lb = batch["labels"].to(device)
                if "ANN" in str(type(model)):
                    preds, features = model(X)
                else:
                    preds = model(X); features = X
                bl = criterion(preds[:, 0], lb.float())

            bs = lb.size(0); es += bs
            gs = time.time()
            with autocast_context(device, use_amp):
                loss_task = torch.sum(bl) / bs

                # 特征蒸馏：对齐全局 mean / var → 让本地 activation 统计分布匹配全局
                loss_feat = 0.0
                if global_feat_stats is not None and fd_lambda > 0:
                    gm = global_feat_stats['mean'].to(device)
                    gv = global_feat_stats['var'].to(device)
                    if features.dim() >= 2:
                        fm = features.mean(dim=0)
                        fv = features.var(dim=0, unbiased=False)
                        # Mean alignment + Var alignment
                        loss_mean = F.mse_loss(fm, gm)
                        loss_var = F.mse_loss(torch.sqrt(fv + 1e-8), torch.sqrt(gv + 1e-8))
                        loss_feat = (loss_mean + loss_var) * 0.5
                    else:
                        loss_feat = 0.0

                loss = loss_task + fd_lambda * loss_feat

            scale_backward(loss, scaler)
            if (batch_id + 1) % accumulation_steps == 0:
                scaler_step(scaler, optimizer); model.zero_grad()
            gpu_seconds += (time.time() - gs)
            el += loss_task

            # 累积本地特征统计（最后一 epoch，用于上传）
            if epoch == algorithm_epoch_T - 1:
                with torch.no_grad():
                    f = features.detach().float()
                    cur_mean = f.mean(dim=0)
                    cur_var = f.var(dim=0, unbiased=False)
                    if local_feat_mean is None:
                        local_feat_mean = cur_mean * bs
                        local_feat_var = cur_var * bs
                    else:
                        local_feat_mean += cur_mean * bs
                        local_feat_var += cur_var * bs
                    local_feat_count += bs

            gc.collect()

        if (batch_id + 1) % accumulation_steps != 0:
            scaler_step(scaler, optimizer); model.zero_grad()
        logger.info(f"Round {iter_t+1}/{communication_round_I}; Client {client_id}; Epoch {epoch+1}; Loss: {el/max(es,1):.4f}")

    # 打包上传本地特征统计
    stats_out = None
    if local_feat_count > 0 and local_feat_mean is not None:
        stats_out = {
            'mean': (local_feat_mean / local_feat_count).cpu().numpy(),
            'var': (local_feat_var / local_feat_count).cpu().numpy(),
            'count': int(local_feat_count)
        }

    client_model_path = os.path.join(basic_path, "client_" + str(client_id + 1), 'model.pt')
    torch.save(model.cpu(), client_model_path)
    return {'gpu_seconds': gpu_seconds, 'feat_stats': stats_out}


def Fed_Fed(device,
            global_model,
            algorithm_epoch_T, num_clients_K, communication_round_I, FL_fraction, FL_drop_rate,
            training_dataloaders,
            training_dataset,
            client_dataset_list,
            param_dict,
            testing_dataloader,
            testing_dataset_len,
            start_round=0):
    accumulation_steps = max(1, int(256 / param_dict['batch_size']))
    use_amp = param_dict.get('use_amp', False)
    scaler = get_scaler(device, use_amp)

    training_dataset_size = len(training_dataset.labels) if hasattr(training_dataset, 'labels') else len(training_dataset)
    client_datasets_size_list = [len(_) for _ in client_dataset_list]
    del training_dataset, client_dataset_list
    gc.collect()

    basic_path = param_dict['model_path']
    for k in range(param_dict["num_clients_K"]):
        full_path = os.path.join(basic_path, "client_" + str(k + 1), 'model.pt')
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        torch.save(global_model, full_path)

    if "SENT_CLF" in param_dict["task"]:
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
    else:
        criterion = torch.nn.BCELoss(reduction='none').to(device)

    total_gpu_seconds = 0
    users_gpu_seconds_list = [0] * num_clients_K
    model_MB_size = sum(p.numel() for p in global_model.parameters()) * 4 / (1024 * 1024)
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # 全局特征统计量（第 0 轮为空）
    global_feat_stats = None

    parallel_executor = ClientParallelExecutor(
        device=device, global_model=global_model, param_dict=param_dict, needs_global_model_during_training=False)

    for iter_t in range(start_round, communication_round_I):
        idxs_users = client_selection(
            client_num=num_clients_K, fraction=FL_fraction,
            dataset_size=training_dataset_size, client_dataset_size_list=client_datasets_size_list,
            drop_rate=FL_drop_rate, style="FedAvg")

        logger.info(f"Round {iter_t+1}; Select clients: {idxs_users}; Start FedFed (Feature Distillation)")

        results = parallel_executor.run_clients(
            idxs_users, _train_single_client_fedfed,
            param_dict=param_dict, training_dataloaders=training_dataloaders,
            algorithm_epoch_T=algorithm_epoch_T, accumulation_steps=accumulation_steps,
            use_amp=use_amp, scaler=scaler, criterion=criterion, basic_path=basic_path,
            iter_t=iter_t, communication_round_I=communication_round_I, num_clients_K=num_clients_K,
            global_feat_stats=global_feat_stats)

        for i, cid in enumerate(idxs_users):
            users_gpu_seconds_list[cid] += results[i]['gpu_seconds']
        total_gpu_seconds += sum(users_gpu_seconds_list)

        # 聚合模型
        logger.info("FedFed: Aggregate + Aggregate Feature Stats for next round")
        pre_agg_params = get_parameters(global_model)
        client_model_updates = []
        theta_list = []
        all_stats = []
        all_counts = []
        for i, cid in enumerate(idxs_users):
            path = os.path.join(basic_path, "client_" + str(cid + 1), 'model.pt')
            sm = torch.load(path, weights_only=False)
            cp = get_parameters(sm)
            theta_list.append(cp)
            if results[i]['feat_stats'] is not None:
                all_stats.append(results[i]['feat_stats'])
                all_counts.append(results[i]['feat_stats']['count'])

            updates = {}
            for j, (pl, pg) in enumerate(zip(cp, pre_agg_params)):
                updates[str(j)] = torch.tensor(pl) - torch.tensor(pg)
            client_model_updates.append(updates)
            del sm
            gc.collect()

        weights = [client_datasets_size_list[j] for j in idxs_users]
        w_arr = np.array(weights, dtype=np.float64); w_arr = w_arr / w_arr.sum()
        theta_arr = np.array(theta_list, dtype=object)
        theta_avg = np.average(theta_arr, axis=0, weights=w_arr).tolist()
        set_parameters(global_model, theta_avg)

        # ===== FedFed 核心：聚合全局特征统计量（按样本数加权） =====
        gpu_s = time.time()
        if len(all_stats) > 0:
            total_n = sum(all_counts)
            agg_mean = np.zeros_like(all_stats[0]['mean'], dtype=np.float64)
            agg_var = np.zeros_like(all_stats[0]['var'], dtype=np.float64)
            for s, n in zip(all_stats, all_counts):
                w = n / max(total_n, 1)
                agg_mean += w * s['mean'].astype(np.float64)
                agg_var += w * s['var'].astype(np.float64)
            global_feat_stats = {
                'mean': torch.from_numpy(agg_mean.astype(np.float32)),
                'var': torch.from_numpy(agg_var.astype(np.float32))
            }
        total_gpu_seconds += (time.time() - gpu_s)

        del theta_arr, theta_list
        gc.collect()

        avg_gpu_seconds = total_gpu_seconds / num_clients_K
        logger.info(f"Round {iter_t+1} Test; GPU: {total_gpu_seconds:.1f}s, Avg: {avg_gpu_seconds:.1f}s")

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
                FR = 1 - DEO; HM = get_HM_by_two_value(acc, FR)
                logger.info(f"ACC: {round(float(acc),3)}, DEO: {round(float(DEO),3)}, SPD: {round(float(SPD),3)}, FR: {round(float(FR),3)}, HM: {round(float(HM),3)}")
                log_test_metrics(accuracy=float(acc), DEO=float(DEO), SPD=float(SPD), FR=float(FR), HM=float(HM),
                    step=iter_t+1, gpu_seconds=total_gpu_seconds, avg_gpu_seconds=avg_gpu_seconds,
                    communication_cost=(iter_t+1)*len(idxs_users)*2*model_MB_size)
                log_system_metrics(step=iter_t+1, gpu_seconds=total_gpu_seconds,
                    communication_cost=(iter_t+1)*len(idxs_users)*2*model_MB_size,
                    selected_client_count=len(idxs_users), selected_clients=idxs_users.tolist(), model_mb_size=model_MB_size)
                flush()
            else:
                acc, DEO, SPD = FL_fairness_and_accuracy_test_4_Tabular_CLF(global_model, param_dict, testing_dataloader, testing_dataset_len)
                FR = 1 - DEO; HM = get_HM_by_two_value(acc, FR)
                logger.info(f"ACC: {round(float(acc),3)}, DEO: {round(float(DEO),3)}, SPD: {round(float(SPD),3)}, FR: {round(float(FR),3)}, HM: {round(float(HM),3)}")
                log_test_metrics(accuracy=float(acc), DEO=float(DEO), SPD=float(SPD), FR=float(FR), HM=float(HM),
                    step=iter_t+1, gpu_seconds=total_gpu_seconds, avg_gpu_seconds=avg_gpu_seconds,
                    communication_cost=(iter_t+1)*len(idxs_users)*2*model_MB_size)
                log_system_metrics(step=iter_t+1, gpu_seconds=total_gpu_seconds,
                    communication_cost=(iter_t+1)*len(idxs_users)*2*model_MB_size,
                    selected_client_count=len(idxs_users), selected_clients=idxs_users.tolist(), model_mb_size=model_MB_size)
                flush()

        log_deep_metrics(global_model, param_dict, testing_dataloader, iter_t+1, client_model_updates=client_model_updates)

        if param_dict.get('checkpoint_save_freq', 1) > 0 and iter_t % param_dict.get('checkpoint_save_freq', 1) == 0:
            save_checkpoint(param_dict=param_dict, iter_t=iter_t, global_model=global_model,
                total_gpu_seconds=total_gpu_seconds,
                client_selection_history=[idxs_users.tolist()] if hasattr(idxs_users, 'tolist') else [idxs_users],
                start_time=start_time)
            clean_old_checkpoints(param_dict, keep_latest=param_dict.get('checkpoint_keep_latest', 5))

    logger.info("FedFed training finished.")
    save_dir = './save_path/'; os.makedirs(save_dir, exist_ok=True)
    torch.save(global_model, os.path.join(save_dir, "global_FedFed.pt"))
    total_comm_cost = communication_round_I * num_clients_K * FL_fraction * 2 * model_MB_size
    return global_model, total_gpu_seconds, total_comm_cost
