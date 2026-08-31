# FedTMOS: Efficient One-Shot Federated Learning with Tsetlin Machine
# https://openreview.net/ (ICLR 2025)
# 核心思想: 操作空间为模型空间（Tsetlin Machine 集成投票）。将各客户端模型视作 Tsetlin 自动机
#           驱动的"子句"投票器：服务器在代理数据上评估各客户端预测，用 Tsetlin 自动机的
#           奖励/惩罚状态更新规则为每个客户端学习投票权重（与多数共识一致者被奖励），
#           最终用自动机加权的集成预测蒸馏全局模型，实现高效的一次性联邦聚合。
# Core Idea: Operates in model space (Tsetlin-Machine-style ensemble voting). Client models act as
#            clause-like voters driven by Tsetlin automata: the server evaluates each client's
#            prediction on proxy data and learns per-client voting weights via Tsetlin automaton
#            reward/punish state updates (consensus-agreeing voters are rewarded). The automaton-
#            weighted ensemble prediction is then distilled into the global model.
# 框架适配说明: 原论文使用 Tsetlin Machine 作为本地模型；本框架的深度模型被适配为自动机投票器，
#               自动机状态在缓存的代理预测上更新（避免反复前向），one-shot 逻辑保留在服务器端。

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


def _train_single_client_fedtmos(client_id, device, model, param_dict,
                                 training_dataloaders, algorithm_epoch_T,
                                 accumulation_steps, use_amp, scaler, criterion,
                                 basic_path, iter_t, communication_round_I, num_clients_K):
    """FedTMOS 单客户端训练（标准本地训练，模型存盘作投票器）。"""
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


def Fed_TMOS(device,
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

    # FedTMOS 超参数
    tm_proxy = int(param_dict.get('FedTMOS_proxy_num', 256))     # 代理数据样本数
    tm_batch = int(param_dict.get('FedTMOS_batch', 64))
    tm_automata_steps = int(param_dict.get('FedTMOS_automata_steps', 20))  # 自动机状态更新步数
    tm_beta = float(param_dict.get('FedTMOS_beta', 1.0))        # 投票权重 softmax 温度
    tm_distill_steps = int(param_dict.get('FedTMOS_steps', 5))  # 集成蒸馏步数
    tm_lr = float(param_dict.get('FedTMOS_lr', 0.01))
    tm_T = float(param_dict.get('FedTMOS_T', 2.0))

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

        logger.info(f"Round {iter_t + 1}; Select clients: {idxs_users}; Start Local Training (FedTMOS)")

        results = parallel_executor.run_clients(
            idxs_users, _train_single_client_fedtmos,
            param_dict=param_dict, training_dataloaders=training_dataloaders,
            algorithm_epoch_T=algorithm_epoch_T, accumulation_steps=accumulation_steps,
            use_amp=use_amp, scaler=scaler, criterion=criterion, basic_path=basic_path,
            iter_t=iter_t, communication_round_I=communication_round_I, num_clients_K=num_clients_K)

        for i, client_id in enumerate(idxs_users):
            users_gpu_seconds_list[client_id] += results[i]['gpu_seconds']
        total_gpu_seconds += sum(users_gpu_seconds_list)

        # ===== FedTMOS 核心：Tsetlin 自动机加权投票 + 集成蒸馏 =====
        logger.info("FedTMOS: Tsetlin automata voting + ensemble distillation")
        pre_agg_params = get_parameters(global_model)
        client_model_updates = []
        client_state_dicts = []
        theta_list = []

        for i, id in enumerate(idxs_users):
            client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
            selected_model = torch.load(client_model_path, weights_only=False)
            client_state_dicts.append(copy.deepcopy(selected_model.state_dict()))
            theta_list.append(get_parameters(selected_model))

            updates = {}
            for j, (p_local, p_global) in enumerate(zip(get_parameters(selected_model), pre_agg_params)):
                updates[str(j)] = torch.tensor(p_local) - torch.tensor(p_global)
            client_model_updates.append(updates)
            del selected_model
            gc.collect()

        gpu_s = time.time()
        # ---- Step 1: 常规加权聚合（为蒸馏提供合理初始化）----
        theta_arr = np.array(theta_list, dtype=object)
        theta_avg = np.average(theta_arr, axis=0,
                               weights=[client_datasets_size_list[j] for j in idxs_users]).tolist()
        set_parameters(global_model, theta_avg)

        # ---- Step 2: 缓存各投票器（客户端模型）在固定噪声池上的预测 ----
        global_model.to(device)
        noise_pool = []
        n_left = tm_proxy
        while n_left > 0:
            n = min(tm_batch, n_left)
            noise_pool.append(_noise_inputs(param_dict, device, n))
            n_left -= n

        member_probs = []
        for sd in client_state_dicts:
            global_model.load_state_dict(sd)
            global_model.eval()
            member_probs.append(torch.cat(
                [_model_probs(global_model, param_dict, inp, device, tm_T) for inp in noise_pool], dim=0).detach())
        M = torch.stack(member_probs, dim=0)  # [K, N, C]

        # ---- Step 3: Tsetlin 自动机状态更新（奖励与共识一致者，惩罚不一致者）----
        n_members = M.size(0)
        automaton_states = np.zeros(n_members, dtype=np.float64)
        for _s in range(tm_automata_steps):
            vote_w = np.exp(tm_beta * automaton_states)
            vote_w = vote_w / vote_w.sum()
            weighted_vote = (vote_w.reshape(-1, 1, 1) * M.numpy()).sum(axis=0)  # [N, C]
            if "SENT_CLF" in param_dict["task"]:
                majority = weighted_vote.argmax(axis=1)  # [N]
                pred = M.numpy().argmax(axis=2)          # [K, N]
            else:
                majority = (weighted_vote > 0.5).astype(np.float32)
                pred = (M.numpy() > 0.5).astype(np.float32)
            agree = (pred == majority[None, :]).astype(np.float64)
            if "SENT_CLF" in param_dict["task"]:
                per_sample_agree = agree.mean(axis=1)
            else:
                per_sample_agree = agree.mean(axis=1)
            # Tsetlin 风格状态更新：一致 +1，不一致 -1（截断在 [-5, 5]）
            automaton_states = np.clip(automaton_states + 2 * (per_sample_agree - 0.5), -5.0, 5.0)

        vote_w = np.exp(tm_beta * automaton_states)
        vote_w = vote_w / vote_w.sum()
        logger.info(f"FedTMOS: automaton voting weights = {np.round(vote_w, 3).tolist()}")
        teacher_all = torch.tensor((vote_w.reshape(-1, 1, 1) * M.numpy()).sum(axis=0), device=device)

        # ---- Step 4: 自动机加权集成蒸馏 ----
        set_parameters(global_model, theta_avg)
        global_model.train()
        distill_optimizer = BERTCLF_Optimizer(method="SGD", learning_rate=tm_lr, max_grad_norm=0)
        distill_optimizer.set_parameters(list(global_model.named_parameters()))
        for _epoch in range(tm_distill_steps):
            offset = 0
            for inp in noise_pool:
                n = _model_probs(global_model, param_dict, inp, device, tm_T).size(0)
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

    logger.info("FedTMOS training finished.")
    save_dir = './save_path/'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(global_model, os.path.join(save_dir, "global_FedTMOS.pt"))
    total_comm_cost = communication_round_I * num_clients_K * FL_fraction * 2 * model_MB_size
    return global_model, total_gpu_seconds, total_comm_cost
