# FAFI: Does One-shot Give the Best Shot? Mitigating Model Inconsistency in One-shot Federated Learning
# https://openreview.net/ (ICML 2025)
# Git Repo: https://github.com/zenghui9977/FAFI_ICML25
# 核心思想: 操作空间为模型空间。One-shot FL 中参数平均模型与函数空间集成存在"模型不一致性"（model
#           inconsistency）。FAFI 通过在参数平均模型与本地模型集成之间做 α-插值，选取使代理数据上
#           蒸馏不一致损失最小的 α*，再将 α*-混合的集成 teacher 蒸馏回参数平均模型，从而缓解不一致。
# Core Idea: Operates in model space. In one-shot FL, the parameter-averaged model is inconsistent with
#            the function-space ensemble of local models ("model inconsistency"). FAFI interpolates
#            between the averaged model and the local-model ensemble with a coefficient alpha, selects
#            the alpha* minimizing the distillation inconsistency loss on proxy data, and distills the
#            alpha*-blended ensemble teacher back into the averaged model.
# 框架适配说明: 原论文为 one-shot；本框架为多轮，每轮重复 "本地训练 -> α 网格搜索 -> 一致性蒸馏"。

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


def _task_forward(model, param_dict, batch, device, criterion):
    if "SENT_CLF" in param_dict["task"]:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        features, logits = model(input_ids=input_ids, attention_mask=attention_mask)
        batch_loss = criterion(logits, labels)
        return batch_loss, features, labels
    elif "IMG_CLF" in param_dict["task"]:
        imgs = batch["img"].to(device)
        labels = batch["labels"].to(device)
        preds, features = model(imgs)
        batch_loss = criterion(preds[:, 0], labels.float())
        return batch_loss, features, labels
    else:
        X = batch["X"].to(device)
        labels = batch["labels"].to(device)
        if "LogisticRegression" in str(type(model)):
            local_prediction = model(X)
            features = X
        else:
            local_prediction, features = model(X)
        batch_loss = criterion(local_prediction[:, 0], labels.float())
        return batch_loss, features, labels


def _noise_inputs(param_dict, device, n):
    if "SENT_CLF" in param_dict["task"]:
        max_len = param_dict.get('max_len', 128)
        return (torch.randint(0, 30522, (n, max_len), device=device),
                torch.ones((n, max_len), device=device))
    elif "IMG_CLF" in param_dict["task"]:
        inp_ch = 3 if param_dict.get('dataset', '').lower() not in ['fmnist', 'mnist'] else 1
        return torch.randn(n, inp_ch, 32, 32, device=device)
    else:
        return torch.randn(n, param_dict.get('nn_input_size', 128), device=device)


def _model_probs(model, param_dict, inputs, device, temperature=1.0):
    with torch.no_grad():
        if "SENT_CLF" in param_dict["task"]:
            input_ids, attention_mask = inputs
            _, logits = model(input_ids=input_ids, attention_mask=attention_mask)
            return torch.softmax(logits / temperature, dim=1)
        elif "IMG_CLF" in param_dict["task"]:
            preds, _ = model(inputs)
            return torch.sigmoid(preds[:, 0] / temperature)
        else:
            if "LogisticRegression" in str(type(model)):
                out = model(inputs)
            else:
                out, _ = model(inputs)
            return torch.sigmoid(out[:, 0] / temperature)


def _train_single_client_fafi(client_id, device, model, param_dict,
                              training_dataloaders, algorithm_epoch_T,
                              accumulation_steps, use_amp, scaler, criterion,
                              basic_path, iter_t, communication_round_I, num_clients_K):
    """FAFI 单客户端训练（标准本地训练，模型存盘作集成成员）。"""
    model.train()
    model.to(device)
    optimizer = BERTCLF_Optimizer(
        method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
    optimizer.set_parameters(list(model.named_parameters()))
    client_i_dataloader = training_dataloaders[client_id]

    gpu_seconds = 0
    for epoch in range(algorithm_epoch_T):
        epoch_total_loss = 0
        epoch_total_size = 0
        for batch_id, batch in enumerate(client_i_dataloader):
            gpu_start_time = time.time()
            with autocast_context(device, use_amp):
                batch_loss, features, labels = _task_forward(model, param_dict, batch, device, criterion)
            true_batch_size = labels.size(0)
            epoch_total_size += true_batch_size
            loss = torch.sum(batch_loss) / true_batch_size
            scale_backward(loss, scaler)
            if (batch_id + 1) % accumulation_steps == 0:
                scaler_step(scaler, optimizer)
                model.zero_grad()
            gpu_seconds += time.time() - gpu_start_time
            epoch_total_loss += loss
            del features, labels
            gc.collect()
        if (batch_id + 1) % accumulation_steps != 0:
            scaler_step(scaler, optimizer)
            model.zero_grad()
        logger.info(f"Round {iter_t + 1}/{communication_round_I}; Client {client_id}/{num_clients_K}; "
                    f"Epoch {epoch + 1}; Avg Loss: {epoch_total_loss / max(epoch_total_size, 1):.4f}")

    client_model_path = os.path.join(basic_path, "client_" + str(client_id + 1), 'model.pt')
    torch.save(model.cpu(), client_model_path)
    return {'gpu_seconds': gpu_seconds}


def FAFI(device,
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

    # FAFI 超参数
    fafi_proxy = int(param_dict.get('FAFI_proxy_num', 256))
    fafi_batch = int(param_dict.get('FAFI_batch', 64))
    fafi_steps = int(param_dict.get('FAFI_steps', 5))
    fafi_lr = float(param_dict.get('FAFI_lr', 0.01))
    fafi_T = float(param_dict.get('FAFI_T', 2.0))
    fafi_alphas = param_dict.get('FAFI_alphas', [0.0, 0.25, 0.5, 0.75, 1.0])  # α 网格

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

        logger.info(f"Round {iter_t + 1}; Select clients: {idxs_users}; Start Local Training (FAFI)")

        results = parallel_executor.run_clients(
            idxs_users, _train_single_client_fafi,
            param_dict=param_dict, training_dataloaders=training_dataloaders,
            algorithm_epoch_T=algorithm_epoch_T, accumulation_steps=accumulation_steps,
            use_amp=use_amp, scaler=scaler, criterion=criterion, basic_path=basic_path,
            iter_t=iter_t, communication_round_I=communication_round_I, num_clients_K=num_clients_K)

        for i, client_id in enumerate(idxs_users):
            users_gpu_seconds_list[client_id] += results[i]['gpu_seconds']
        total_gpu_seconds += sum(users_gpu_seconds_list)

        # ===== FAFI 核心：α-插值缓解模型不一致性 =====
        logger.info("FAFI: alpha-interpolation against model inconsistency")
        pre_agg_params = get_parameters(global_model)
        client_model_updates = []
        client_state_dicts = []
        client_params_list = []

        for i, id in enumerate(idxs_users):
            client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
            selected_model = torch.load(client_model_path, weights_only=False)
            client_state_dicts.append(copy.deepcopy(selected_model.state_dict()))
            client_params_list.append(get_parameters(selected_model))

            updates = {}
            for j, (p_local, p_global) in enumerate(zip(get_parameters(selected_model), pre_agg_params)):
                updates[str(j)] = torch.tensor(p_local) - torch.tensor(p_global)
            client_model_updates.append(updates)
            del selected_model
            gc.collect()

        gpu_s = time.time()
        # 参数平均模型
        weights = np.array([client_datasets_size_list[j] for j in idxs_users], dtype=np.float64)
        weights = weights / weights.sum()
        theta_avg = []
        for j in range(len(client_params_list[0])):
            stacked = np.stack([np.asarray(cp[j], dtype=np.float64) for cp in client_params_list], axis=0)
            theta_avg.append((weights.reshape(-1, *([1] * (stacked.ndim - 1))) * stacked).sum(axis=0).astype(np.float32))

        # 缓存集成成员在噪声池上的概率 + 计算平均模型的概率
        global_model.to(device)
        noise_pool = []
        n_left = fafi_proxy
        while n_left > 0:
            n = min(fafi_batch, n_left)
            noise_pool.append(_noise_inputs(param_dict, device, n))
            n_left -= n

        member_probs = []
        for sd in client_state_dicts:
            global_model.load_state_dict(sd)
            global_model.eval()
            member_probs.append(torch.cat(
                [_model_probs(global_model, param_dict, inp, device, fafi_T) for inp in noise_pool], dim=0).detach())
        ens_prob = torch.stack(member_probs, dim=0).mean(dim=0)  # 函数空间集成

        set_parameters(global_model, theta_avg)
        global_model.eval()
        avg_prob = torch.cat(
            [_model_probs(global_model, param_dict, inp, device, fafi_T) for inp in noise_pool], dim=0).detach()

        # α 网格搜索：teacher_α = (1-α)·avg_prob + α·ens_prob，选使与 avg_prob 差异最小的 α*
        # （即选择"最接近平均模型行为"的一致性 teacher，缓解不一致）
        best_alpha, best_loss = None, float('inf')
        for alpha in fafi_alphas:
            teacher_alpha = (1 - alpha) * avg_prob + alpha * ens_prob
            loss_incons = torch.mean((teacher_alpha - avg_prob) ** 2).item()
            if loss_incons < best_loss:
                best_loss, best_alpha = loss_incons, alpha
        logger.info(f"FAFI: best alpha = {best_alpha} (inconsistency = {best_loss:.6f})")
        teacher_all = ((1 - best_alpha) * avg_prob + best_alpha * ens_prob).detach()

        # 一致性蒸馏：teacher_α* -> 参数平均模型
        global_model.train()
        distill_optimizer = BERTCLF_Optimizer(method="SGD", learning_rate=fafi_lr, max_grad_norm=0)
        distill_optimizer.set_parameters(list(global_model.named_parameters()))
        for _epoch in range(fafi_steps):
            offset = 0
            for inp in noise_pool:
                n = _model_probs(global_model, param_dict, inp, device, fafi_T).size(0)
                teacher = teacher_all[offset:offset + n]
                offset += n
                if "SENT_CLF" in param_dict["task"]:
                    input_ids, attention_mask = inp
                    _, logits = global_model(input_ids=input_ids, attention_mask=attention_mask)
                    student_logp = torch.log_softmax(logits, dim=1)
                    loss_kd = -(teacher * student_logp).sum(dim=1).mean()
                else:
                    if "LogisticRegression" in str(type(global_model)):
                        out = global_model(inp)
                    else:
                        out, _ = global_model(inp)
                    loss_kd = torch.nn.functional.binary_cross_entropy_with_logits(out[:, 0], teacher)
                distill_optimizer.zero_grad()
                loss_kd.backward()
                distill_optimizer.step()

        total_gpu_seconds += time.time() - gpu_s
        global_model.to("cpu")
        del client_state_dicts, member_probs, teacher_all
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

    logger.info("FAFI training finished.")
    save_dir = './save_path/'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(global_model, os.path.join(save_dir, "global_FAFI.pt"))
    total_comm_cost = communication_round_I * num_clients_K * FL_fraction * 2 * model_MB_size
    return global_model, total_gpu_seconds, total_comm_cost
