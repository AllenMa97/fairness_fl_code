# FedBE: Making Bayesian Model Ensemble Applicable to Federated Learning
# https://arxiv.org/abs/2009.01974 (ICLR 2021)
# Git Repo: https://github.com/hongyouc/FedBE
# 核心思想: 操作空间为模型空间（Bayesian model ensemble）。将客户端模型视作贝叶斯后验样本组成的集成，
#           先选出与集成函数空间中心最接近的 base 模型（medoid），再在无标签代理数据（随机噪声）上
#           用集成平均预测蒸馏 base 模型，得到兼顾集成性能与单模型部署的全局模型。
# Core Idea: Operates in model space (Bayesian model ensemble). Client models are treated as
#            posterior samples forming an ensemble. The server picks a base (medoid) model closest
#            to the ensemble's function-space center, then distills the ensemble's averaged prediction
#            on unlabeled proxy data (random noise) into the base model.
# 框架适配说明: 原论文为 one-shot 设置；本框架为多轮，故每轮执行 "本地训练 -> medoid 选择 -> 集成蒸馏"。

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
    """统一任务前向，返回 (loss_per_sample, features, labels)。"""
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
    else:  # Tabular_CLF
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
    """为三任务生成服务器端无标签代理输入（随机噪声）。"""
    if "SENT_CLF" in param_dict["task"]:
        max_len = param_dict.get('max_len', 128)
        input_ids = torch.randint(0, 30522, (n, max_len), device=device)
        attention_mask = torch.ones((n, max_len), device=device)
        return (input_ids, attention_mask)
    elif "IMG_CLF" in param_dict["task"]:
        inp_ch = 3 if param_dict.get('dataset', '').lower() not in ['fmnist', 'mnist'] else 1
        return torch.randn(n, inp_ch, 32, 32, device=device)
    else:
        return torch.randn(n, param_dict.get('nn_input_size', 128), device=device)


def _model_logits(model, param_dict, inputs, device):
    """模型在代理输入上的 logits（二分类任务返回单维 logit）。"""
    with torch.no_grad():
        if "SENT_CLF" in param_dict["task"]:
            input_ids, attention_mask = inputs
            _, logits = model(input_ids=input_ids, attention_mask=attention_mask)
            return logits
        elif "IMG_CLF" in param_dict["task"]:
            preds, _ = model(inputs)
            return preds[:, 0]
        else:
            if "LogisticRegression" in str(type(model)):
                out = model(inputs)
            else:
                out, _ = model(inputs)
            return out[:, 0]


def _model_probs(model, param_dict, inputs, device, temperature=1.0):
    """logits -> 概率（统一蒸馏空间）。"""
    logits = _model_logits(model, param_dict, inputs, device)
    if "SENT_CLF" in param_dict["task"]:
        return torch.softmax(logits / temperature, dim=1)
    else:
        return torch.sigmoid(logits / temperature)


def _train_single_client_fedbe(client_id, device, model, param_dict,
                               training_dataloaders, algorithm_epoch_T,
                               accumulation_steps, use_amp, scaler, criterion,
                               basic_path, iter_t, communication_round_I, num_clients_K):
    """FedBE 单客户端训练：标准本地训练，模型存盘作为集成成员。"""
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

        avg_loss = epoch_total_loss / max(epoch_total_size, 1)
        logger.info(f"Round {iter_t + 1}/{communication_round_I}; Client {client_id}/{num_clients_K}; "
                    f"Epoch {epoch + 1}; Avg Loss: {avg_loss:.4f}")

    client_model_path = os.path.join(basic_path, "client_" + str(client_id + 1), 'model.pt')
    torch.save(model.cpu(), client_model_path)
    return {'gpu_seconds': gpu_seconds}


def Fed_BE(device,
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

    # FedBE 超参数
    be_proxy = int(param_dict.get('FedBE_proxy_num', 256))       # 代理数据（噪声）样本数
    be_batch = int(param_dict.get('FedBE_batch', 64))           # 代理数据 batch 大小
    be_steps = int(param_dict.get('FedBE_steps', 5))            # 集成蒸馏 epoch 数
    be_lr = float(param_dict.get('FedBE_lr', 0.01))             # 蒸馏学习率
    be_T = float(param_dict.get('FedBE_T', 2.0))                # 蒸馏温度（SENT 任务 softmax 温度）

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

        logger.info(f"Round {iter_t + 1}; Select clients: {idxs_users}; Start Local Training (FedBE)")

        results = parallel_executor.run_clients(
            idxs_users, _train_single_client_fedbe,
            param_dict=param_dict, training_dataloaders=training_dataloaders,
            algorithm_epoch_T=algorithm_epoch_T, accumulation_steps=accumulation_steps,
            use_amp=use_amp, scaler=scaler, criterion=criterion, basic_path=basic_path,
            iter_t=iter_t, communication_round_I=communication_round_I, num_clients_K=num_clients_K)

        for i, client_id in enumerate(idxs_users):
            users_gpu_seconds_list[client_id] += results[i]['gpu_seconds']
        total_gpu_seconds += sum(users_gpu_seconds_list)

        # ===== FedBE 核心：贝叶斯集成 -> medoid base -> 集成蒸馏 =====
        logger.info("FedBE: Bayesian ensemble selection + ensemble distillation")
        pre_agg_params = get_parameters(global_model)
        client_model_updates = []
        client_state_dicts = []  # 集成成员（state_dict，CPU）

        for i, id in enumerate(idxs_users):
            client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
            selected_model = torch.load(client_model_path, weights_only=False)
            client_state_dicts.append(copy.deepcopy(selected_model.state_dict()))

            client_params = get_parameters(selected_model)
            updates = {}
            for j, (p_local, p_global) in enumerate(zip(client_params, pre_agg_params)):
                updates[str(j)] = torch.tensor(p_local) - torch.tensor(p_global)
            client_model_updates.append(updates)
            del selected_model
            gc.collect()

        gpu_s = time.time()
        # ---- Step 1: 缓存每个集成成员在固定噪声池上的预测 ----
        global_model.to(device)
        noise_pool = []
        n_left = be_proxy
        while n_left > 0:
            n = min(be_batch, n_left)
            noise_pool.append(_noise_inputs(param_dict, device, n))
            n_left -= n

        member_probs = []  # 每个成员：拼接所有 batch 的概率
        for sd in client_state_dicts:
            global_model.load_state_dict(sd)
            global_model.eval()
            probs = torch.cat([_model_probs(global_model, param_dict, inp, device, be_T) for inp in noise_pool], dim=0)
            member_probs.append(probs.detach())

        # ---- Step 2: medoid base 选择（与集成函数中心 L2 距离最小者）----
        mean_prob = torch.stack(member_probs, dim=0).mean(dim=0)
        dists = [torch.sum((mp - mean_prob) ** 2).item() for mp in member_probs]
        base_idx = int(np.argmin(dists))
        logger.info(f"FedBE: medoid base = client {idxs_users[base_idx]} (dist={dists[base_idx]:.4f})")
        global_model.load_state_dict(client_state_dicts[base_idx])

        # ---- Step 3: 集成蒸馏（teacher = 集成平均概率）----
        global_model.train()
        distill_optimizer = BERTCLF_Optimizer(method="SGD", learning_rate=be_lr, max_grad_norm=0)
        distill_optimizer.set_parameters(list(global_model.named_parameters()))
        teacher_all = mean_prob.detach()

        for epoch in range(be_steps):
            offset = 0
            for inp in noise_pool:
                n = _model_logits(global_model, param_dict, inp, device).size(0)
                teacher = teacher_all[offset:offset + n]
                offset += n
                if "SENT_CLF" in param_dict["task"]:
                    _, logits = global_model(input_ids=inp[0], attention_mask=inp[1])
                    student_logp = torch.log_softmax(logits, dim=1)
                    loss_kd = -(teacher * student_logp).sum(dim=1).mean()
                else:
                    if "LogisticRegression" in str(type(global_model)):
                        out = global_model(inp[0] if isinstance(inp, torch.Tensor) else inp)
                    else:
                        out, _ = global_model(inp)
                    student_logit = out[:, 0]
                    loss_kd = torch.nn.functional.binary_cross_entropy_with_logits(
                        student_logit, teacher)
                distill_optimizer.zero_grad()
                loss_kd.backward()
                distill_optimizer.step()

        total_gpu_seconds += time.time() - gpu_s
        global_model.to("cpu")
        del client_state_dicts, member_probs, mean_prob, teacher_all
        gc.collect()

        avg_gpu_seconds = total_gpu_seconds / num_clients_K
        logger.info(f"Total GPU seconds: {total_gpu_seconds:.1f}, Avg: {avg_gpu_seconds:.1f}")

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

    logger.info("FedBE training finished.")
    save_dir = './save_path/'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(global_model, os.path.join(save_dir, "global_FedBE.pt"))
    total_comm_cost = communication_round_I * num_clients_K * FL_fraction * 2 * model_MB_size
    return global_model, total_gpu_seconds, total_comm_cost
