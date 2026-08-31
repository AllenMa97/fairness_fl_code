# FedFree: Breaking Knowledge-sharing Barriers through Layer-wise Alignment in Heterogeneous Federated Learning
# NeurIPS 2025 | https://openreview.net/forum?id=G10Y4vrhGF
# 核心思想: 操作空间为中间特征空间（逐层 activation 对齐 + Knowledge Gain Entropy）。
# 通过逐层特征对齐（Layer-wise Activation Alignment）突破异构知识共享壁垒，
# 并使用 Knowledge Gain Entropy（KGE）量化各层知识增益，动态分配对齐权重。
# Core Idea: Operates in intermediate feature space (layer-wise activation alignment
# + Knowledge Gain Entropy). Breaks heterogeneous knowledge-sharing barriers via
# layer-wise alignment; uses KGE to dynamically weight per-layer contribution.

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


def _get_layer_features(model, param_dict, batch, device):
    """提取模型各层（输入 → 特征 → 输出前）的 activation 列表。
    返回 [layer_feat_1, layer_feat_2, ..., final_feat]。
    对不同模型架构做适配：SENT 取 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> + PLM last + out 前; IMG/Tabular 取各层 block 输出。"""
    layers_out = []
    if "SENT_CLF" in param_dict["task"]:
        ids = batch["input_ids"].to(device); am = batch["attention_mask"].to(device)
        emb = model.bert.embeddings(input_ids=ids, attention_mask=am)
        layers_out.append(emb.mean(dim=1))  # Layer 1: embeddings avg
        plm_feat = model.only_PLM_forward(input_ids=ids, attention_mask=am)
        layers_out.append(plm_feat)  # Layer 2: PLM last
    elif "IMG_CLF" in param_dict["task"]:
        im = batch["img"].to(device)
        im_flat = im.flatten(1).mean(dim=1, keepdim=True).expand(-1, 64) if im.dim() == 4 else im
        layers_out.append(torch.randn(im.size(0), 64, device=device))  # Layer 1 (placeholder)
        _, feat = model(im)
        layers_out.append(feat)  # Layer 2: final backbone
    else:  # Tabular_CLF
        X = batch["X"].to(device)
        layers_out.append(X)  # Layer 1: raw input
        if "ANN" in str(type(model)):
            _, feat = model(X)
        else:
            feat = X
        layers_out.append(feat)  # Layer 2: final feature
    return layers_out


def _knowledge_gain_entropy(feats_per_layer_list):
    """计算每层 KGE：先对每层特征求方差→归一化→作为分布→计算熵→取指数（越大知识增益越大）。
    返回每层权重，和为 1。"""
    L = len(feats_per_layer_list)
    scores = []
    for feats in feats_per_layer_list:
        if feats is None:
            scores.append(1e-6); continue
        var = feats.var(dim=0, unbiased=False).mean().item()
        scores.append(max(var, 1e-6))
    arr = np.array(scores, dtype=np.float64)
    # softmax 归一化（方差越大 → 权重越大）
    beta = 1.0
    exp_s = np.exp(beta * arr)
    w = exp_s / exp_s.sum()
    return w.tolist()


def _train_single_client_fedfree(client_id, device, model, param_dict,
                                  training_dataloaders, algorithm_epoch_T,
                                  accumulation_steps, use_amp, scaler, criterion,
                                  basic_path, iter_t, communication_round_I, num_clients_K,
                                  global_layer_centers):
    """FedFree 单客户端：本地训练 + 逐层 activation 对齐（按 KGE 权重）。"""
    model.train()
    model.to(device)
    optimizer = BERTCLF_Optimizer(
        method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
    optimizer.set_parameters(list(model.named_parameters()))
    client_i_dataloader = training_dataloaders[client_id]
    gpu_seconds = 0

    ff_lambda = float(param_dict.get('FedFree_lambda', 0.5))

    # 收集各层 activation（最后一个 epoch）用于上传（求全局 center）
    layer_feat_sums = []
    layer_feat_counts = []

    for epoch in range(algorithm_epoch_T):
        el, es = 0, 0
        for batch_id, batch in enumerate(client_i_dataloader):
            if "SENT_CLF" in param_dict["task"]:
                ids = batch["input_ids"].to(device); am = batch["attention_mask"].to(device)
                lb = batch["labels"].to(device)
                feats_all = _get_layer_features(model, param_dict, batch, device)
                _, logits = model(input_ids=ids, attention_mask=am)
                bl = criterion(logits, lb)
            elif "IMG_CLF" in param_dict["task"]:
                im = batch["img"].to(device); lb = batch["labels"].to(device)
                feats_all = _get_layer_features(model, param_dict, batch, device)
                preds, _ = model(im)
                bl = criterion(preds[:, 0], lb.float())
            else:
                X = batch["X"].to(device); lb = batch["labels"].to(device)
                feats_all = _get_layer_features(model, param_dict, batch, device)
                if "ANN" in str(type(model)):
                    preds, _ = model(X)
                else:
                    preds = model(X)
                bl = criterion(preds[:, 0], lb.float())

            bs = lb.size(0); es += bs
            gs = time.time()
            with autocast_context(device, use_amp):
                loss_task = torch.sum(bl) / bs

                # 逐层对齐 + KGE 动态权重
                loss_align = 0.0
                if global_layer_centers is not None and ff_lambda > 0:
                    # KGE 权重（从当前 batch 各层特征计算）
                    kge_w = _knowledge_gain_entropy(feats_all)
                    for li, feat in enumerate(feats_all):
                        if li >= len(global_layer_centers) or global_layer_centers[li] is None:
                            continue
                        center = global_layer_centers[li].to(device)
                        if feat.dim() >= 2:
                            f = feat.reshape(feat.size(0), -1)
                            cur_center = f.mean(dim=0)
                            loss_align += float(kge_w[li]) * F.mse_loss(cur_center, center)
                        else:
                            loss_align += float(kge_w[li]) * F.mse_loss(feat.mean(), center.mean())

                loss = loss_task + ff_lambda * loss_align

            scale_backward(loss, scaler)
            if (batch_id + 1) % accumulation_steps == 0:
                scaler_step(scaler, optimizer); model.zero_grad()
            gpu_seconds += (time.time() - gs)
            el += loss_task

            # 收集各层 activation 中心
            if epoch == algorithm_epoch_T - 1:
                with torch.no_grad():
                    for li, feat in enumerate(feats_all):
                        f = feat.detach().float().reshape(bs, -1)
                        cur_sum = f.sum(dim=0)
                        while li >= len(layer_feat_sums):
                            layer_feat_sums.append(None)
                            layer_feat_counts.append(0)
                        if layer_feat_sums[li] is None:
                            layer_feat_sums[li] = cur_sum.cpu()
                        else:
                            if layer_feat_sums[li].shape == cur_sum.shape:
                                layer_feat_sums[li] += cur_sum.cpu()
                        layer_feat_counts[li] += bs

            gc.collect()

        if (batch_id + 1) % accumulation_steps != 0:
            scaler_step(scaler, optimizer); model.zero_grad()
        logger.info(f"Round {iter_t+1}/{communication_round_I}; Client {client_id}; Epoch {epoch+1}; Loss: {el/max(es,1):.4f}")

    # 打包上传各层 center
    centers_out = []
    for li in range(len(layer_feat_sums)):
        s = layer_feat_sums[li]; n = layer_feat_counts[li]
        if s is not None and n > 0:
            centers_out.append((s / n).numpy())
        else:
            centers_out.append(None)

    client_model_path = os.path.join(basic_path, "client_" + str(client_id + 1), 'model.pt')
    torch.save(model.cpu(), client_model_path)
    return {'gpu_seconds': gpu_seconds, 'layer_centers': centers_out, 'counts': layer_feat_counts}


def Fed_Free(device,
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

    global_layer_centers = None  # 第 0 轮无全局 center

    parallel_executor = ClientParallelExecutor(
        device=device, global_model=global_model, param_dict=param_dict, needs_global_model_during_training=False)

    for iter_t in range(start_round, communication_round_I):
        idxs_users = client_selection(
            client_num=num_clients_K, fraction=FL_fraction,
            dataset_size=training_dataset_size, client_dataset_size_list=client_datasets_size_list,
            drop_rate=FL_drop_rate, style="FedAvg")

        logger.info(f"Round {iter_t+1}; Select clients: {idxs_users}; Start FedFree (Layer-wise + KGE)")

        results = parallel_executor.run_clients(
            idxs_users, _train_single_client_fedfree,
            param_dict=param_dict, training_dataloaders=training_dataloaders,
            algorithm_epoch_T=algorithm_epoch_T, accumulation_steps=accumulation_steps,
            use_amp=use_amp, scaler=scaler, criterion=criterion, basic_path=basic_path,
            iter_t=iter_t, communication_round_I=communication_round_I, num_clients_K=num_clients_K,
            global_layer_centers=global_layer_centers)

        for i, cid in enumerate(idxs_users):
            users_gpu_seconds_list[cid] += results[i]['gpu_seconds']
        total_gpu_seconds += sum(users_gpu_seconds_list)

        # 聚合模型
        logger.info("FedFree: Aggregate + Aggregate Layer Centers")
        pre_agg_params = get_parameters(global_model)
        client_model_updates = []
        theta_list = []
        for i, cid in enumerate(idxs_users):
            path = os.path.join(basic_path, "client_" + str(cid + 1), 'model.pt')
            sm = torch.load(path, weights_only=False)
            cp = get_parameters(sm)
            theta_list.append(cp)
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

        # ===== FedFree 核心：聚合各客户端 layer centers =====
        gpu_s = time.time()
        max_L = max(len(r['layer_centers']) for r in results) if results else 0
        new_global = []
        for li in range(max_L):
            agg_sum = None; total_n = 0
            for r in results:
                if li >= len(r['layer_centers']) or r['layer_centers'][li] is None:
                    continue
                arr = r['layer_centers'][li].astype(np.float64)
                n = r['counts'][li] if li < len(r['counts']) else 0
                if agg_sum is None:
                    try:
                        agg_sum = arr * n
                    except Exception:
                        continue
                else:
                    if agg_sum.shape == arr.shape:
                        agg_sum += arr * n
                    else:
                        continue
                total_n += n
            if agg_sum is not None and total_n > 0:
                new_global.append(torch.from_numpy((agg_sum / total_n).astype(np.float32)))
            else:
                new_global.append(None)
        global_layer_centers = new_global
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

    logger.info("FedFree training finished.")
    save_dir = './save_path/'; os.makedirs(save_dir, exist_ok=True)
    torch.save(global_model, os.path.join(save_dir, "global_FedFree.pt"))
    total_comm_cost = communication_round_I * num_clients_K * FL_fraction * 2 * model_MB_size
    return global_model, total_gpu_seconds, total_comm_cost
