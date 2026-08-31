# FedBEns: One-Shot Federated Learning based on Bayesian Ensemble
# https://openreview.net/ (ICML 2025)
# 核心思想: 操作空间为模型空间（Bayesian ensemble）。各客户端本地模型视作后验样本，服务器用
#           拉普拉斯近似（对角经验 Fisher）混合各局部后验：先做精度加权（Fisher 大的维度更信任
#           对应客户端），再在聚合中心附近按 1/Fisher 协方差扰动出贝叶斯集成，最后在无标签代理
#           数据上把集成平均预测蒸馏回聚合中心模型。不依赖特定架构。
# Core Idea: Operates in model space (Bayesian ensemble). Local models are posterior samples; the
#            server mixes local posteriors via Laplace approximation (diagonal empirical Fisher):
#            precision-weighted aggregation, then perturbs around the aggregation center with
#            1/Fisher covariance to form a Bayesian ensemble, and distills the ensemble prediction
#            on unlabeled proxy data back into the aggregated model. Architecture-agnostic.
# 框架适配说明: 原论文为 one-shot；本框架为多轮，每轮重复 "本地训练+Fisher -> 混合聚合+集成蒸馏"。

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


def _train_single_client_fedbens(client_id, device, model, param_dict,
                                 training_dataloaders, algorithm_epoch_T,
                                 accumulation_steps, use_amp, scaler, criterion,
                                 basic_path, iter_t, communication_round_I, num_clients_K):
    """FedBEns 单客户端：标准本地训练 + 对角经验 Fisher 计算（拉普拉斯近似的曲率代理）。"""
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

    # ---- 对角经验 Fisher：本地数据上梯度平方的均值（拉普拉斯近似的曲率代理）----
    gpu_start_time = time.time()
    model.eval()
    fisher_acc = [torch.zeros_like(p) for p in model.parameters()]
    n_batches = 0
    for batch in client_i_dataloader:
        batch_loss, _, _ = _task_forward(model, param_dict, batch, device, criterion)
        loss = batch_loss.mean()
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            for fa, p in zip(fisher_acc, model.parameters()):
                if p.grad is not None:
                    fa += p.grad.detach() ** 2
        n_batches += 1
        if n_batches >= 8:  # 限制 Fisher 估计成本
            break
    fisher_diag = [fa.cpu().numpy() / max(n_batches, 1) for fa in fisher_acc]
    gpu_seconds += time.time() - gpu_start_time

    client_model_path = os.path.join(basic_path, "client_" + str(client_id + 1), 'model.pt')
    torch.save(model.cpu(), client_model_path)
    return {'gpu_seconds': gpu_seconds, 'fisher_diag': fisher_diag}


def Fed_BEns(device,
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

    # FedBEns 超参数
    bens_proxy = int(param_dict.get('FedBEns_proxy_num', 256))   # 代理数据样本数
    bens_batch = int(param_dict.get('FedBEns_batch', 64))
    bens_steps = int(param_dict.get('FedBEns_steps', 5))        # 集成蒸馏步数
    bens_lr = float(param_dict.get('FedBEns_lr', 0.01))
    bens_T = float(param_dict.get('FedBEns_T', 2.0))
    bens_members = int(param_dict.get('FedBEns_members', 5))    # 拉普拉斯扰动集成成员数
    bens_eps = float(param_dict.get('FedBEns_eps', 0.05))       # 扰动尺度（相对 1/sqrt(Fisher)）

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

        logger.info(f"Round {iter_t + 1}; Select clients: {idxs_users}; Start Local Training (FedBEns)")

        results = parallel_executor.run_clients(
            idxs_users, _train_single_client_fedbens,
            param_dict=param_dict, training_dataloaders=training_dataloaders,
            algorithm_epoch_T=algorithm_epoch_T, accumulation_steps=accumulation_steps,
            use_amp=use_amp, scaler=scaler, criterion=criterion, basic_path=basic_path,
            iter_t=iter_t, communication_round_I=communication_round_I, num_clients_K=num_clients_K)

        for i, client_id in enumerate(idxs_users):
            users_gpu_seconds_list[client_id] += results[i]['gpu_seconds']
        total_gpu_seconds += sum(users_gpu_seconds_list)

        # ===== FedBEns 核心：拉普拉斯近似混合聚合 + 贝叶斯集成蒸馏 =====
        logger.info("FedBEns: Laplace-approximated posterior mixture aggregation")
        pre_agg_params = get_parameters(global_model)
        client_model_updates = []
        client_params_list = []
        client_fisher_list = []

        for i, id in enumerate(idxs_users):
            client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
            selected_model = torch.load(client_model_path, weights_only=False)
            client_params_list.append(get_parameters(selected_model))
            client_fisher_list.append(results[i]['fisher_diag'])

            updates = {}
            for j, (p_local, p_global) in enumerate(zip(get_parameters(selected_model), pre_agg_params)):
                updates[str(j)] = torch.tensor(p_local) - torch.tensor(p_global)
            client_model_updates.append(updates)
            del selected_model
            gc.collect()

        gpu_s = time.time()
        # ---- Step 1: 精度加权混合（Fisher 大的维度更信任对应客户端；权重 ∝ n_i * Fisher_ij）----
        weights_n = np.array([client_datasets_size_list[j] for j in idxs_users], dtype=np.float64)
        weights_n = weights_n / weights_n.sum()
        agg_params = []
        for j in range(len(client_params_list[0])):
            stacked = np.stack([np.asarray(cp[j], dtype=np.float64) for cp in client_params_list], axis=0)
            fisher_stack = np.stack([np.asarray(cf[j], dtype=np.float64) + 1e-8 for cf in client_fisher_list], axis=0)
            w = weights_n[:, None, None] if stacked.ndim == 3 else weights_n[:, None]
            w = np.broadcast_to(w.reshape(-1, *([1] * (stacked.ndim - 1))), stacked.shape)
            precision_w = w * fisher_stack
            precision_w = precision_w / (precision_w.sum(axis=0, keepdims=True) + 1e-12)
            agg_params.append((precision_w * stacked).sum(axis=0))
        set_parameters(global_model, agg_params)

        # ---- Step 2: 拉普拉斯扰动采样出贝叶斯集成，在代理数据上取平均预测 ----
        global_model.to(device)
        noise_pool = []
        n_left = bens_proxy
        while n_left > 0:
            n = min(bens_batch, n_left)
            noise_pool.append(_noise_inputs(param_dict, device, n))
            n_left -= n

        center_params = [np.asarray(p, dtype=np.float64) for p in agg_params]
        teacher_batches = None
        for m in range(bens_members):
            perturbed = []
            for j, (p_c, name_p) in enumerate(zip(center_params, global_model.parameters())):
                fisher_j = np.maximum(np.asarray(client_fisher_list[0][j], dtype=np.float64), 1e-8)
                for ci in range(1, len(client_fisher_list)):
                    fisher_j = fisher_j + np.asarray(client_fisher_list[ci][j], dtype=np.float64)
                scale = bens_eps / np.sqrt(fisher_j * len(idxs_users))
                perturbed.append((p_c + np.random.randn(*p_c.shape) * scale).astype(np.float32))
            set_parameters(global_model, perturbed)
            global_model.eval()
            probs = torch.cat([_model_probs(global_model, param_dict, inp, device, bens_T) for inp in noise_pool], dim=0)
            teacher_batches = probs.detach() if teacher_batches is None else teacher_batches + probs
        teacher_all = (teacher_batches / bens_members).detach()

        # ---- Step 3: 集成平均预测蒸馏回聚合中心模型 ----
        set_parameters(global_model, [p.astype(np.float32) for p in center_params])
        global_model.train()
        distill_optimizer = BERTCLF_Optimizer(method="SGD", learning_rate=bens_lr, max_grad_norm=0)
        distill_optimizer.set_parameters(list(global_model.named_parameters()))
        for _epoch in range(bens_steps):
            offset = 0
            for inp in noise_pool:
                n = _model_probs(global_model, param_dict, inp, device, bens_T).size(0)
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
        del client_params_list, client_fisher_list, teacher_all
        gc.collect()

        avg_gpu_seconds = total_gpu_seconds / num_clients_K
        if (iter_t + 1) != param_dict['communication_round_I']:
            if "SENT_CLF" in param_dict["task"]:
                acc, DEO, SPD = FL_fairness_and_accuracy_test(global_model, param_dict, testing_dataloader, testing_dataset_len)
                logger.info(f"ACC: {round(float(acc),3)}, DEO: {round(float(DEO),3)}, SPD: {round(float(SPD),3)}")
                log_test_metrics(accuracy=float(acc), DEO=float(DEO), SPD=float(SPD),
                    step=iter_t+1, gpu_seconds=total_gpu_seconds, avg_gpu_seconds=avg_gpu_seconds,
                    communication_cost=(iter_t+1)*len(idxs_users)*3*model_MB_size)
                log_system_metrics(step=iter_t+1, gpu_seconds=total_gpu_seconds,
                    communication_cost=(iter_t+1)*len(idxs_users)*3*model_MB_size,
                    selected_client_count=len(idxs_users), selected_clients=idxs_users.tolist(), model_mb_size=model_MB_size)
                flush()
            elif "IMG_CLF" in param_dict["task"]:
                acc, DEO, SPD = FL_fairness_and_accuracy_test_4_IMG_CLF(global_model, param_dict, testing_dataloader, testing_dataset_len)
                FR = 1 - DEO
                HM = get_HM_by_two_value(acc, FR)
                logger.info(f"ACC: {round(float(acc),3)}, DEO: {round(float(DEO),3)}, SPD: {round(float(SPD),3)}, FR: {round(float(FR),3)}, HM: {round(float(HM),3)}")
                log_test_metrics(accuracy=float(acc), DEO=float(DEO), SPD=float(SPD), FR=float(FR), HM=float(HM),
                    step=iter_t+1, gpu_seconds=total_gpu_seconds, avg_gpu_seconds=avg_gpu_seconds,
                    communication_cost=(iter_t+1)*len(idxs_users)*3*model_MB_size)
                log_system_metrics(step=iter_t+1, gpu_seconds=total_gpu_seconds,
                    communication_cost=(iter_t+1)*len(idxs_users)*3*model_MB_size,
                    selected_client_count=len(idxs_users), selected_clients=idxs_users.tolist(), model_mb_size=model_MB_size)
                flush()
            elif "Tabular_CLF" in param_dict["task"]:
                acc, DEO, SPD = FL_fairness_and_accuracy_test_4_Tabular_CLF(global_model, param_dict, testing_dataloader, testing_dataset_len)
                FR = 1 - DEO
                HM = get_HM_by_two_value(acc, FR)
                logger.info(f"ACC: {round(float(acc),3)}, DEO: {round(float(DEO),3)}, SPD: {round(float(SPD),3)}, FR: {round(float(FR),3)}, HM: {round(float(HM),3)}")
                log_test_metrics(accuracy=float(acc), DEO=float(DEO), SPD=float(SPD), FR=float(FR), HM=float(HM),
                    step=iter_t+1, gpu_seconds=total_gpu_seconds, avg_gpu_seconds=avg_gpu_seconds,
                    communication_cost=(iter_t+1)*len(idxs_users)*3*model_MB_size)
                log_system_metrics(step=iter_t+1, gpu_seconds=total_gpu_seconds,
                    communication_cost=(iter_t+1)*len(idxs_users)*3*model_MB_size,
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

    logger.info("FedBEns training finished.")
    save_dir = './save_path/'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(global_model, os.path.join(save_dir, "global_FedBEns.pt"))
    total_comm_cost = communication_round_I * num_clients_K * FL_fraction * 3 * model_MB_size
    return global_model, total_gpu_seconds, total_comm_cost
