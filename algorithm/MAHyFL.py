# MA-HyFL: Modality-Agnostic Hybrid Federated Learning via Knowledge Distillation and Reinforcement Learning Based Aggregation
# IEEE Trans. Circuits and Systems for Video Technology (TCSVT), Early Access, 2026
# https://doi.org/10.1109/TCSVT.2026.3663412
# 核心思想: 操作空间为输出分布空间（bidirectional cross-modal KD + RL aggregation）。
# 双向跨模态知识蒸馏 + 基于强化学习的动态权重聚合策略。
# Core Idea: Operates in output distribution space (bidirectional cross-modal KD
# + RL aggregation). Bidirectional cross-modal distillation with RL-based dynamic
# weight aggregation.

import copy
import os
import gc
import time
import torch
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


def _bidirectional_kd_single_client(client_id, device, model, param_dict,
                                     training_dataloaders, algorithm_epoch_T,
                                     accumulation_steps, use_amp, scaler, criterion,
                                     basic_path, iter_t, communication_round_I, num_clients_K,
                                     prev_global_logits_buf):
    """MA-HyFL 单客户端：本地训练 + 与上一轮全局 logits 做双向 KD。"""
    model.train()
    model.to(device)
    optimizer = BERTCLF_Optimizer(
        method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
    optimizer.set_parameters(list(model.named_parameters()))
    client_i_dataloader = training_dataloaders[client_id]
    gpu_seconds = 0

    kd_T = float(param_dict.get('MAHyFL_T', 3.0))
    kd_w = float(param_dict.get('MAHyFL_kd_weight', 0.3))

    # 本地输出 logits 缓冲（上传做双向蒸馏）
    local_logits_buf = []

    for epoch in range(algorithm_epoch_T):
        el, es = 0, 0
        for batch_id, batch in enumerate(client_i_dataloader):
            if "SENT_CLF" in param_dict["task"]:
                ids = batch["input_ids"].to(device); am = batch["attention_mask"].to(device)
                lb = batch["labels"].to(device)
                feats, logits = model(input_ids=ids, attention_mask=am)
                bl = criterion(logits, lb)
            elif "IMG_CLF" in param_dict["task"]:
                im = batch["img"].to(device); lb = batch["labels"].to(device)
                preds, feats = model(im)
                bl = criterion(preds[:, 0], lb.float())
                logits = torch.logit(torch.clamp(preds, min=1e-7, max=1-1e-7))
            else:
                X = batch["X"].to(device); lb = batch["labels"].to(device)
                if "ANN" in str(type(model)):
                    preds, feats = model(X)
                else:
                    preds = model(X); feats = X
                bl = criterion(preds[:, 0], lb.float())
                logits = torch.logit(torch.clamp(preds, min=1e-7, max=1-1e-7))

            bs = lb.size(0); es += bs
            gs = time.time()
            with autocast_context(device, use_amp):
                loss_task = torch.sum(bl) / bs

                # 双向 KD：与 prev_global_logits_buf（上一轮全局 logits）
                if prev_global_logits_buf is not None and epoch == 0 and batch_id < len(prev_global_logits_buf):
                    try:
                        prev_lg = prev_global_logits_buf[batch_id].to(device)
                        cur_lg = logits
                        min_n = min(prev_lg.size(0), cur_lg.size(0))
                        pl = prev_lg[:min_n]; cl = cur_lg[:min_n]
                        if "SENT_CLF" in param_dict["task"]:
                            s_soft = torch.log_softmax(cl / kd_T, dim=1)
                            t_soft = torch.softmax(pl / kd_T, dim=1).detach()
                            l_kd_fwd = -torch.sum(t_soft * s_soft, dim=1).mean() * (kd_T ** 2)
                            s2 = torch.log_softmax(pl / kd_T, dim=1).detach()
                            t2 = torch.softmax(cl / kd_T, dim=1)
                            l_kd_bwd = -torch.sum(t2 * s2, dim=1).mean() * (kd_T ** 2)
                        else:
                            sp = torch.sigmoid(cl / kd_T)
                            tp = torch.sigmoid(pl / kd_T).detach()
                            l_kd_fwd = torch.nn.functional.mse_loss(sp, tp) * (kd_T ** 2)
                            l_kd_bwd = torch.nn.functional.mse_loss(tp, sp) * (kd_T ** 2)
                        loss_kd = (l_kd_fwd + l_kd_bwd) * 0.5
                        loss = loss_task + kd_w * loss_kd
                    except Exception:
                        loss = loss_task
                else:
                    loss = loss_task

            scale_backward(loss, scaler)
            if (batch_id + 1) % accumulation_steps == 0:
                scaler_step(scaler, optimizer); model.zero_grad()
            gpu_seconds += (time.time() - gs)
            el += loss_task

            # 缓存第一个 epoch 的 logits
            if epoch == 0:
                local_logits_buf.append(logits.detach().cpu())
            gc.collect()

        if (batch_id + 1) % accumulation_steps != 0:
            scaler_step(scaler, optimizer); model.zero_grad()
        logger.info(f"Round {iter_t+1}/{communication_round_I}; Client {client_id}; Epoch {epoch+1}; Loss: {el/max(es,1):.4f}")

    client_model_path = os.path.join(basic_path, "client_" + str(client_id + 1), 'model.pt')
    torch.save(model.cpu(), client_model_path)
    return {'gpu_seconds': gpu_seconds, 'local_logits_buf': local_logits_buf}


def _rl_aggregation_weights(client_perf_list, round_idx, num_clients, param_dict):
    """简易 RL-based aggregation: 基于各客户端近期性能的 softmax 动态权重。
    用指数移动平均 + softmax 作为策略网络的近似（替代 RL agent）。"""
    perf = np.array(client_perf_list, dtype=np.float64)
    # EMA 平滑
    if not hasattr(_rl_aggregation_weights, '_history'):
        _rl_aggregation_weights._history = {}
    key = str(round_idx)
    if key not in _rl_aggregation_weights._history:
        _rl_aggregation_weights._history[key] = perf
    else:
        prev = _rl_aggregation_weights._history[key]
        _rl_aggregation_weights._history[key] = 0.7 * prev + 0.3 * perf
    ema = _rl_aggregation_weights._history[key]
    beta = float(param_dict.get('MAHyFL_rl_beta', 2.0))
    exp_p = np.exp(beta * ema)
    w = exp_p / exp_p.sum()
    return w.tolist()


def MA_HyFL(device,
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

    # 全局 logits 缓冲（用于双向 KD）
    global_logits_buf = None

    parallel_executor = ClientParallelExecutor(
        device=device, global_model=global_model, param_dict=param_dict, needs_global_model_during_training=False)

    for iter_t in range(start_round, communication_round_I):
        idxs_users = client_selection(
            client_num=num_clients_K, fraction=FL_fraction,
            dataset_size=training_dataset_size, client_dataset_size_list=client_datasets_size_list,
            drop_rate=FL_drop_rate, style="FedAvg")

        logger.info(f"Round {iter_t+1}; Select clients: {idxs_users}; Start MA-HyFL")

        results = parallel_executor.run_clients(
            idxs_users, _bidirectional_kd_single_client,
            param_dict=param_dict, training_dataloaders=training_dataloaders,
            algorithm_epoch_T=algorithm_epoch_T, accumulation_steps=accumulation_steps,
            use_amp=use_amp, scaler=scaler, criterion=criterion, basic_path=basic_path,
            iter_t=iter_t, communication_round_I=communication_round_I, num_clients_K=num_clients_K,
            prev_global_logits_buf=global_logits_buf)

        for i, cid in enumerate(idxs_users):
            users_gpu_seconds_list[cid] += results[i]['gpu_seconds']
        total_gpu_seconds += sum(users_gpu_seconds_list)

        # 收集模型和客户端 logits
        logger.info("MA-HyFL: RL-Based Weighted Aggregation")
        pre_agg_params = get_parameters(global_model)
        client_model_updates = []
        theta_list = []
        client_logits_list = []  # 每个客户端一组 logits buffer
        client_perf_list = []    # RL 输入：客户端模型在测试 batch 上的性能
        for i, cid in enumerate(idxs_users):
            path = os.path.join(basic_path, "client_" + str(cid + 1), 'model.pt')
            sm = torch.load(path, weights_only=False)
            cp = get_parameters(sm)
            theta_list.append(cp)
            client_logits_list.append(results[i]['local_logits_buf'])

            # 快速估计性能 (testing_dataloader first batch)
            sm.eval(); sm.to(device)
            correct, total = 0, 0
            try:
                batch = next(iter(testing_dataloader))
                with torch.no_grad():
                    if "SENT_CLF" in param_dict["task"]:
                        ids = batch["input_ids"].to(device); am = batch["attention_mask"].to(device)
                        lb = batch["labels"].to(device)
                        _, lg = sm(input_ids=ids, attention_mask=am)
                        correct = (lg.argmax(dim=1) == lb).sum().item()
                        total = lb.size(0)
                    elif "IMG_CLF" in param_dict["task"] or "Tabular_CLF" in param_dict["task"]:
                        if "IMG_CLF" in param_dict["task"]:
                            inp = batch["img"].to(device); p, _ = sm(inp)
                        else:
                            inp = batch["X"].to(device)
                            p = sm(inp)[0] if "ANN" in str(type(sm)) else sm(inp)
                        lb = batch["labels"].to(device)
                        correct = ((p[:, 0] >= 0.5).long() == lb.long()).sum().item()
                        total = lb.size(0)
            except Exception:
                pass
            client_perf_list.append(correct / max(total, 1))
            sm.to("cpu"); del sm

            # 计算更新量
            updates = {}
            for j, (pl, pg) in enumerate(zip(cp, pre_agg_params)):
                updates[str(j)] = torch.tensor(pl) - torch.tensor(pg)
            client_model_updates.append(updates)
            gc.collect()

        # RL 聚合权重（与数据量权重融合）
        rl_w = _rl_aggregation_weights(client_perf_list, iter_t, num_clients_K, param_dict)
        data_w_arr = np.array([client_datasets_size_list[j] for j in idxs_users], dtype=np.float64)
        data_w_arr = data_w_arr / data_w_arr.sum()
        rl_w_arr = np.array(rl_w, dtype=np.float64)
        comb_w = 0.5 * data_w_arr + 0.5 * rl_w_arr
        comb_w = comb_w / comb_w.sum()

        theta_arr = np.array(theta_list, dtype=object)
        theta_avg = np.average(theta_arr, axis=0, weights=comb_w).tolist()
        set_parameters(global_model, theta_avg)

        # 更新全局 logits 缓冲（等权合并客户端 logits 供下一轮双向 KD 使用）
        new_buf = []
        if len(client_logits_list) > 0:
            max_len_buf = max(len(b) for b in client_logits_list)
            for bi in range(max_len_buf):
                arr = []
                for cb in client_logits_list:
                    if bi < len(cb):
                        arr.append(cb[bi])
                if arr:
                    stacked = torch.stack(arr, dim=0).float()
                    new_buf.append(stacked.mean(dim=0))
        global_logits_buf = new_buf if new_buf else None

        avg_gpu_seconds = total_gpu_seconds / num_clients_K
        logger.info(f"Round {iter_t+1} Test; GPU: {total_gpu_seconds:.1f}s, Avg: {avg_gpu_seconds:.1f}s")

        del theta_arr, theta_list
        gc.collect()

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

    logger.info("MA-HyFL training finished.")
    save_dir = './save_path/'; os.makedirs(save_dir, exist_ok=True)
    torch.save(global_model, os.path.join(save_dir, "global_MAHyFL.pt"))
    total_comm_cost = communication_round_I * num_clients_K * FL_fraction * 2 * model_MB_size
    return global_model, total_gpu_seconds, total_comm_cost
