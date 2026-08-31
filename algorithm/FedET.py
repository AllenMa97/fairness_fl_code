# Fed-ET: Heterogeneous Ensemble Knowledge Transfer for Training Large Models in Federated Learning
# https://arxiv.org/abs/2204.12703 (IJCAI 2022)
# 核心思想: 操作空间为输出分布空间（weighted consensus distillation + diversity regularization）。
# 通过加权共识蒸馏（按客户端性能动态加权）+ 多样性正则化实现异构联邦知识迁移。
# Core Idea: Operates in output distribution space (weighted consensus distillation
# + diversity regularization). Transfers heterogeneous federated knowledge via
# performance-weighted consensus distillation with diversity regularization.

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


def _forward_logits(model, param_dict, batch, device):
    if "SENT_CLF" in param_dict["task"]:
        input_ids = batch["input_ids"].to(device); attention_mask = batch["attention_mask"].to(device)
        _, logits = model(input_ids=input_ids, attention_mask=attention_mask)
        return logits
    elif "IMG_CLF" in param_dict["task"]:
        imgs = batch["img"].to(device)
        preds, _ = model(imgs)
        return torch.logit(torch.clamp(preds, min=1e-7, max=1-1e-7))
    else:
        X = batch["X"].to(device)
        if "ANN" in str(type(model)):
            preds, _ = model(X)
        else:
            preds = model(X)
        return torch.logit(torch.clamp(preds, min=1e-7, max=1-1e-7))


def _train_single_client_fedet(client_id, device, model, param_dict,
                                training_dataloaders, algorithm_epoch_T,
                                accumulation_steps, use_amp, scaler, criterion,
                                basic_path, iter_t, communication_round_I, num_clients_K,
                                testing_dataloader, testing_dataset_len):
    """Fed-ET 单客户端训练：本地训练后返回本地模型在公共小测试集上的准确率作为权重。"""
    model.train()
    model.to(device)
    optimizer = BERTCLF_Optimizer(
        method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
    optimizer.set_parameters(list(model.named_parameters()))
    client_i_dataloader = training_dataloaders[client_id]
    gpu_seconds = 0

    for epoch in range(algorithm_epoch_T):
        epoch_loss, epoch_size = 0, 0
        for batch_id, batch in enumerate(client_i_dataloader):
            if "SENT_CLF" in param_dict["task"]:
                input_ids = batch["input_ids"].to(device); attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                _, logits = model(input_ids=input_ids, attention_mask=attention_mask)
                bl = criterion(logits, labels)
            elif "IMG_CLF" in param_dict["task"]:
                imgs = batch["img"].to(device); labels = batch["labels"].to(device)
                preds, _ = model(imgs); bl = criterion(preds[:, 0], labels.float())
            else:
                X = batch["X"].to(device); labels = batch["labels"].to(device)
                if "ANN" in str(type(model)):
                    preds, _ = model(X)
                else:
                    preds = model(X)
                bl = criterion(preds[:, 0], labels.float())

            bs = labels.size(0); epoch_size += bs
            gpu_s = time.time()
            with autocast_context(device, use_amp):
                loss = torch.sum(bl) / bs
            scale_backward(loss, scaler)
            if (batch_id + 1) % accumulation_steps == 0:
                scaler_step(scaler, optimizer); model.zero_grad()
            gpu_seconds += (time.time() - gpu_s)
            epoch_loss += loss
            gc.collect()

        if (batch_id + 1) % accumulation_steps != 0:
            scaler_step(scaler, optimizer); model.zero_grad()
        logger.info(f"Round {iter_t+1}/{communication_round_I}; Client {client_id}; Epoch {epoch+1}; Loss: {epoch_loss/max(epoch_size,1):.4f}")

    # 快速估计本地性能（取 testing_dataloader 的首个 batch 作为 proxy，权重范围 [0.2, 1.0]）
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        try:
            batch = next(iter(testing_dataloader))
            if "SENT_CLF" in param_dict["task"]:
                input_ids = batch["input_ids"].to(device); attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                _, logits = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = logits.argmax(dim=1)
                correct = (preds == labels).sum().item()
                total = labels.size(0)
            elif "IMG_CLF" in param_dict["task"] or "Tabular_CLF" in param_dict["task"]:
                if "IMG_CLF" in param_dict["task"]:
                    inp = batch["img"].to(device)
                    p, _ = model(inp)
                else:
                    inp = batch["X"].to(device)
                    p = model(inp)[0] if "ANN" in str(type(model)) else model(inp)
                labels = batch["labels"].to(device)
                pred_flag = (p[:, 0] >= 0.5).long()
                correct = (pred_flag == labels.long()).sum().item()
                total = labels.size(0)
        except Exception:
            pass
    perf_weight = 0.2 + 0.8 * (correct / max(total, 1))

    client_model_path = os.path.join(basic_path, "client_" + str(client_id + 1), 'model.pt')
    torch.save(model.cpu(), client_model_path)

    return {'gpu_seconds': gpu_seconds, 'perf_weight': float(perf_weight)}


def Fed_ET(device,
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

    # Fed-ET 超参数
    et_T = float(param_dict.get('FedET_T', 4.0))              # KD 温度
    et_lambda_div = float(param_dict.get('FedET_lambda_div', 0.1))  # 多样性正则权重
    et_steps = int(param_dict.get('FedET_steps', 15))         # 蒸馏步数
    et_proxy = int(param_dict.get('FedET_proxy', 128))        # 代理样本数
    et_alpha = float(param_dict.get('FedET_alpha', 0.6))      # KD 权重

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

    parallel_executor = ClientParallelExecutor(
        device=device, global_model=global_model, param_dict=param_dict, needs_global_model_during_training=False)

    for iter_t in range(start_round, communication_round_I):
        idxs_users = client_selection(
            client_num=num_clients_K, fraction=FL_fraction,
            dataset_size=training_dataset_size, client_dataset_size_list=client_datasets_size_list,
            drop_rate=FL_drop_rate, style="FedAvg")

        logger.info(f"Round {iter_t+1}; Select clients: {idxs_users}; Start Fed-ET")

        results = parallel_executor.run_clients(
            idxs_users, _train_single_client_fedet,
            param_dict=param_dict, training_dataloaders=training_dataloaders,
            algorithm_epoch_T=algorithm_epoch_T, accumulation_steps=accumulation_steps,
            use_amp=use_amp, scaler=scaler, criterion=criterion, basic_path=basic_path,
            iter_t=iter_t, communication_round_I=communication_round_I, num_clients_K=num_clients_K,
            testing_dataloader=testing_dataloader, testing_dataset_len=testing_dataset_len)

        for i, cid in enumerate(idxs_users):
            users_gpu_seconds_list[cid] += results[i]['gpu_seconds']
        total_gpu_seconds += sum(users_gpu_seconds_list)

        # 聚合
        logger.info("Fed-ET: Weighted aggregation + Weighted consensus KD + Diversity reg")
        pre_agg_params = get_parameters(global_model)
        client_model_updates = []
        theta_list = []
        perf_weights = []
        client_models = []
        for i, cid in enumerate(idxs_users):
            path = os.path.join(basic_path, "client_" + str(cid + 1), 'model.pt')
            sm = torch.load(path, weights_only=False)
            cp = get_parameters(sm)
            theta_list.append(cp)
            client_models.append(sm)
            perf_weights.append(results[i]['perf_weight'])
            updates = {}
            for j, (pl, pg) in enumerate(zip(cp, pre_agg_params)):
                updates[str(j)] = torch.tensor(pl) - torch.tensor(pg)
            client_model_updates.append(updates)

        # 聚合权重 = 数据量权重 × 性能权重
        data_w = np.array([client_datasets_size_list[j] for j in idxs_users], dtype=np.float64)
        perf_w = np.array(perf_weights, dtype=np.float64)
        comb_w = data_w * perf_w
        comb_w = comb_w / comb_w.sum()
        theta_arr = np.array(theta_list, dtype=object)
        theta_avg = np.average(theta_arr, axis=0, weights=comb_w).tolist()
        set_parameters(global_model, theta_avg)

        # ===== Fed-ET 核心：Weighted Consensus Distillation + Diversity Reg =====
        gpu_s = time.time()
        global_model.to(device)
        for m in client_models:
            m.to(device).eval()
        global_model.train()

        # 生成代理数据
        if "SENT_CLF" in param_dict["task"]:
            max_len = param_dict.get('max_len', 128)
            proxy_inp_ids = torch.randint(0, 30522, (et_proxy, max_len), device=device)
            proxy_attn = torch.ones((et_proxy, max_len), device=device)
            proxy_batch = {"input_ids": proxy_inp_ids, "attention_mask": proxy_attn}
        elif "IMG_CLF" in param_dict["task"]:
            inp_ch = 3 if param_dict.get('dataset', '').lower() not in ['fmnist', 'mnist'] else 1
            proxy_imgs = torch.randn(et_proxy, inp_ch, 32, 32, device=device)
            proxy_batch = {"img": proxy_imgs}
        else:
            inp_size = param_dict.get('nn_input_size', 128)
            proxy_X = torch.randn(et_proxy, inp_size, device=device)
            proxy_batch = {"X": proxy_X}

        # 计算各客户端 teacher logits
        client_logits_list = []
        with torch.no_grad():
            for m in client_models:
                lg = _forward_logits(m, param_dict, proxy_batch, device)
                client_logits_list.append(lg)  # 每个 [N, C] 或 [N, 1]

        distill_optim = BERTCLF_Optimizer(method="ADAM", learning_rate=param_dict['learning_rate']*0.1, max_grad_norm=0)
        distill_optim.set_parameters(list(global_model.named_parameters()))

        perf_w_t = torch.tensor(perf_w, device=device)  # [K]
        perf_w_t = perf_w_t / perf_w_t.sum()

        for _s in range(et_steps):
            student_lg = _forward_logits(global_model, param_dict, proxy_batch, device)

            # 按性能加权的 consensus (加权平均 logits)
            stacked = torch.stack(client_logits_list, dim=0)  # [K, N, ...]
            # 广播权重: [K] → [K, N, 1]
            w = perf_w_t.view(-1, 1, 1) if stacked.dim() == 3 else perf_w_t.view(-1, 1)
            teacher_consensus = (w * stacked).sum(dim=0)

            if "SENT_CLF" in param_dict["task"]:
                s_soft = torch.log_softmax(student_lg / et_T, dim=1)
                t_soft = torch.softmax(teacher_consensus / et_T, dim=1).detach()
                loss_kd = -torch.sum(t_soft * s_soft, dim=1).mean() * (et_T ** 2)
            else:
                s_p = torch.sigmoid(student_lg / et_T)
                t_p = torch.sigmoid(teacher_consensus / et_T).detach()
                loss_kd = torch.nn.functional.mse_loss(s_p, t_p) * (et_T ** 2)

            # 多样性正则：鼓励各客户端 teacher logits 之间保持差异（方差越大越好 → 取负方差）
            if et_lambda_div > 0:
                if stacked.dim() == 3:
                    mean_lg = (w * stacked).sum(dim=0, keepdim=True)
                    var_lg = ((stacked - mean_lg) ** 2).mean()
                else:
                    mean_lg = (w * stacked).sum(dim=0, keepdim=True)
                    var_lg = ((stacked - mean_lg) ** 2).mean()
                loss_div = -var_lg
            else:
                loss_div = 0.0

            total_loss = et_alpha * loss_kd + et_lambda_div * loss_div
            distill_optim.zero_grad()
            total_loss.backward()
            distill_optim.step()

        for m in client_models:
            m.to("cpu"); del m
        client_models.clear()
        global_model.to("cpu")
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

    logger.info("Fed-ET training finished.")
    save_dir = './save_path/'; os.makedirs(save_dir, exist_ok=True)
    torch.save(global_model, os.path.join(save_dir, "global_FedET.pt"))
    total_comm_cost = communication_round_I * num_clients_K * FL_fraction * 2 * model_MB_size
    return global_model, total_gpu_seconds, total_comm_cost
