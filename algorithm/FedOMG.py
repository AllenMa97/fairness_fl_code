# FedOMG: Federated Domain Generalization with Data-free On-server Matching Gradient
# ICLR 2025 (Poster) | https://github.com/skydvn/fedomg
# 核心思想: 操作空间为梯度空间（gradient inner product maximization）。
# 服务器端在无数据场景下，通过最大化与客户端上传梯度的内积（即梯度对齐）
# 来直接优化全局模型参数，从而提升跨域泛化能力。
# Core Idea: Operates in gradient space (gradient inner product maximization).
# Data-free on-server: maximizes inner product with uploaded client gradients to
# directly optimize global parameters, improving cross-domain generalization.

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


def _train_single_client_fedomg(client_id, device, model, param_dict,
                                 training_dataloaders, algorithm_epoch_T,
                                 accumulation_steps, use_amp, scaler, criterion,
                                 basic_path, iter_t, communication_round_I, num_clients_K):
    """FedOMG 本地训练后返回 flatten 梯度向量（目标梯度方向）。"""
    model.train()
    model.to(device)
    optimizer = BERTCLF_Optimizer(
        method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
    optimizer.set_parameters(list(model.named_parameters()))
    client_i_dataloader = training_dataloaders[client_id]
    gpu_seconds = 0
    init_params = [p.data.clone() for p in model.parameters()]

    for epoch in range(algorithm_epoch_T):
        el, es = 0, 0
        for batch_id, batch in enumerate(client_i_dataloader):
            if "SENT_CLF" in param_dict["task"]:
                ids = batch["input_ids"].to(device); am = batch["attention_mask"].to(device)
                lb = batch["labels"].to(device)
                _, logits = model(input_ids=ids, attention_mask=am)
                bl = criterion(logits, lb)
            elif "IMG_CLF" in param_dict["task"]:
                im = batch["img"].to(device); lb = batch["labels"].to(device)
                preds, _ = model(im); bl = criterion(preds[:, 0], lb.float())
            else:
                X = batch["X"].to(device); lb = batch["labels"].to(device)
                if "ANN" in str(type(model)):
                    preds, _ = model(X)
                else:
                    preds = model(X)
                bl = criterion(preds[:, 0], lb.float())
            bs = lb.size(0); es += bs
            gs = time.time()
            with autocast_context(device, use_amp):
                loss = torch.sum(bl) / bs
            scale_backward(loss, scaler)
            if (batch_id + 1) % accumulation_steps == 0:
                scaler_step(scaler, optimizer); model.zero_grad()
            gpu_seconds += (time.time() - gs)
            el += loss
            gc.collect()
        if (batch_id + 1) % accumulation_steps != 0:
            scaler_step(scaler, optimizer); model.zero_grad()
        logger.info(f"Round {iter_t+1}/{communication_round_I}; Client {client_id}; Epoch {epoch+1}; Loss: {el/max(es,1):.4f}")

    # 上传目标梯度: delta = w_final - w_initial (flatten 向量)
    with torch.no_grad():
        grad_vec = []
        for i, (name, p) in enumerate(model.named_parameters()):
            delta = (p.data.detach() - init_params[i].to(p.device)).cpu()
            grad_vec.append(delta.view(-1))
        target_grad = torch.cat(grad_vec, dim=0)

    client_model_path = os.path.join(basic_path, "client_" + str(client_id + 1), 'model.pt')
    torch.save(model.cpu(), client_model_path)
    return {'gpu_seconds': gpu_seconds, 'target_grad': target_grad.numpy()}


def Fed_OMG(device,
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

    # FedOMG 超参数
    omg_steps = int(param_dict.get('FedOMG_steps', 10))       # 服务器端梯度匹配步数
    omg_lr = float(param_dict.get('FedOMG_lr', 0.005))         # 服务器端学习率
    omg_lambda = float(param_dict.get('FedOMG_lambda', 1.0))   # 内积最大化权重
    omg_proxy = int(param_dict.get('FedOMG_proxy', 16))       # 随机噪声 batch 数（产生随机梯度）

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

        logger.info(f"Round {iter_t+1}; Select clients: {idxs_users}; Start FedOMG")

        results = parallel_executor.run_clients(
            idxs_users, _train_single_client_fedomg,
            param_dict=param_dict, training_dataloaders=training_dataloaders,
            algorithm_epoch_T=algorithm_epoch_T, accumulation_steps=accumulation_steps,
            use_amp=use_amp, scaler=scaler, criterion=criterion, basic_path=basic_path,
            iter_t=iter_t, communication_round_I=communication_round_I, num_clients_K=num_clients_K)

        for i, cid in enumerate(idxs_users):
            users_gpu_seconds_list[cid] += results[i]['gpu_seconds']
        total_gpu_seconds += sum(users_gpu_seconds_list)

        # 聚合模型
        logger.info("FedOMG: Aggregate + On-server Gradient Matching (Inner Product)")
        pre_agg_params = get_parameters(global_model)
        client_model_updates = []
        theta_list = []
        grad_list = []
        for i, cid in enumerate(idxs_users):
            path = os.path.join(basic_path, "client_" + str(cid + 1), 'model.pt')
            sm = torch.load(path, weights_only=False)
            cp = get_parameters(sm)
            theta_list.append(cp)
            grad_list.append(torch.from_numpy(results[i]['target_grad']))
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

        # ===== FedOMG 核心：服务器端无数据梯度内积最大化 =====
        gpu_s = time.time()
        global_model.to(device)
        global_model.train()

        # 加权聚合目标梯度
        target_grad = torch.zeros_like(grad_list[0], device=device)
        for i, g in enumerate(grad_list):
            target_grad += (g.to(device) * float(w_arr[i]))
        target_norm = target_grad.norm(p=2) + 1e-9
        target_grad_unit = target_grad / target_norm

        # 优化器：直接优化全局模型
        server_optim = BERTCLF_Optimizer(method="ADAM", learning_rate=omg_lr, max_grad_norm=0)
        server_optim.set_parameters(list(global_model.named_parameters()))

        def _gen_random_proxy_batch():
            if "SENT_CLF" in param_dict["task"]:
                max_len = param_dict.get('max_len', 128)
                ids = torch.randint(0, 30522, (omg_proxy, max_len), device=device)
                am = torch.ones((omg_proxy, max_len), device=device)
                lb = torch.randint(0, 2, (omg_proxy,), device=device)
                return ids, am, None, lb
            elif "IMG_CLF" in param_dict["task"]:
                inp_ch = 3 if param_dict.get('dataset', '').lower() not in ['fmnist', 'mnist'] else 1
                im = torch.randn(omg_proxy, inp_ch, 32, 32, device=device)
                lb = torch.round(torch.rand(omg_proxy, device=device)).long()
                return None, None, im, lb
            else:
                inp_size = param_dict.get('nn_input_size', 128)
                X = torch.randn(omg_proxy, inp_size, device=device)
                lb = torch.round(torch.rand(omg_proxy, device=device)).long()
                return None, None, X, lb

        for _step in range(omg_steps):
            ids, am, imX, lb = _gen_random_proxy_batch()
            server_optim.zero_grad()
            if "SENT_CLF" in param_dict["task"]:
                _, logits = global_model(input_ids=ids, attention_mask=am)
                loss_rand = torch.sum(criterion(logits, lb)) / omg_proxy
            elif "IMG_CLF" in param_dict["task"]:
                preds, _ = global_model(imX)
                loss_rand = torch.sum(criterion(preds[:, 0], lb.float())) / omg_proxy
            else:
                if "ANN" in str(type(global_model)):
                    preds, _ = global_model(imX)
                else:
                    preds = global_model(imX)
                loss_rand = torch.sum(criterion(preds[:, 0], lb.float())) / omg_proxy

            # 计算当前参数关于随机任务的梯度向量
            trainable = [p for p in global_model.parameters() if p.requires_grad]
            cur_grads = torch.autograd.grad(loss_rand, trainable, create_graph=False, allow_unused=True)
            cur_vec = []
            for g in cur_grads:
                if g is None:
                    continue
                cur_vec.append(g.view(-1))
            cur_vec = torch.cat(cur_vec, dim=0)
            cur_norm = cur_vec.norm(p=2) + 1e-9
            cur_vec_unit = cur_vec / cur_norm

            # 内积最大化损失（使两梯度夹角余弦最大 → 即 -cos）
            cos_sim = (cur_vec_unit * target_grad_unit).sum()
            loss_omg = -cos_sim

            # 反向传播：走 loss_rand 的图，但乘以内积权重（近似梯度方向）
            # 为了让模型朝目标梯度方向走：直接更新 = 目标梯度方向
            scalar = omg_lambda * (1.0 + cos_sim.detach().item())
            (scalar * loss_rand).backward()
            server_optim.step()

        global_model.to("cpu")
        total_gpu_seconds += (time.time() - gpu_s)

        del theta_arr, theta_list, grad_list
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

    logger.info("FedOMG training finished.")
    save_dir = './save_path/'; os.makedirs(save_dir, exist_ok=True)
    torch.save(global_model, os.path.join(save_dir, "global_FedOMG.pt"))
    total_comm_cost = communication_round_I * num_clients_K * FL_fraction * 2 * model_MB_size
    return global_model, total_gpu_seconds, total_comm_cost
