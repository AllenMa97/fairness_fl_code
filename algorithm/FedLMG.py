# FedLMG: One-Shot Heterogeneous Federated Learning with Local Model-Guided Diffusion Models
# https://proceedings.mlr.press/v267/yang25c.html (ICML 2025, PMLR 267:71157-71176)
# 核心思想: 操作空间为特征空间（guided diffusion sampling）。各客户端模型作为"引导信号"（guidance），
#           服务器在特征空间运行退火朗之万式扩散采样：从高斯先验出发，逐步加噪-去噪，去噪方向由
#           客户端分类头集成的类别条件引导（classifier guidance：最大化集成对目标类的置信度），
#           采样得到的类别条件伪特征用于知识蒸馏训练全局模型，支持异构本地模型。
# Core Idea: Operates in feature space (guided diffusion sampling). Client models serve as guidance
#            signals: the server runs annealed Langevin-style diffusion sampling in feature space --
#            starting from a Gaussian prior, alternating noise injection and denoising, where the
#            denoising direction is class-conditional guidance from the ensemble of client classifier
#            heads (classifier guidance: maximize ensemble confidence on the target class). The
#            sampled class-conditional pseudo features are used for KD to train the global model.
# 框架适配说明: 原论文使用像素空间扩散模型；本框架适配为特征空间扩散采样（无需训练扩散网络），
#               one-shot 逻辑保留在服务器端，框架内每轮重复执行。

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


def _get_clf_module(model, param_dict):
    if "SENT_CLF" in param_dict["task"]:
        return model.out
    elif "IMG_CLF" in param_dict["task"]:
        return model.out_layer
    else:
        if "LogisticRegression" in str(type(model)):
            return model.layer
        return model.out_layer


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


def _train_single_client_fedlmg(client_id, device, model, param_dict,
                                training_dataloaders, algorithm_epoch_T,
                                accumulation_steps, use_amp, scaler, criterion,
                                basic_path, iter_t, communication_round_I, num_clients_K):
    """FedLMG 单客户端：标准本地训练 + 上传分类头（作为扩散采样引导信号）。"""
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

    clf_state = {name: p.detach().cpu() for name, p in _get_clf_module(model, param_dict).state_dict().items()}
    client_model_path = os.path.join(basic_path, "client_" + str(client_id + 1), 'model.pt')
    torch.save(model.cpu(), client_model_path)
    return {'gpu_seconds': gpu_seconds, 'clf_state': clf_state}


def Fed_LMG(device,
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

    # FedLMG 超参数
    lmg_gen_num = int(param_dict.get('FedLMG_gen_num', 256))     # 采样伪特征数
    lmg_diff_steps = int(param_dict.get('FedLMG_diff_steps', 30))  # 退火朗之万扩散步数
    lmg_guidance_lr = float(param_dict.get('FedLMG_guidance_lr', 0.1))  # 引导步长
    lmg_noise_decay = float(param_dict.get('FedLMG_noise_decay', 0.9))  # 噪声衰减系数
    lmg_kd_epochs = int(param_dict.get('FedLMG_kd_epochs', 5))  # 全局头蒸馏轮数
    lmg_kd_lr = float(param_dict.get('FedLMG_kd_lr', 0.01))
    lmg_T = float(param_dict.get('FedLMG_T', 2.0))

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

        logger.info(f"Round {iter_t + 1}; Select clients: {idxs_users}; Start Local Training (FedLMG)")

        results = parallel_executor.run_clients(
            idxs_users, _train_single_client_fedlmg,
            param_dict=param_dict, training_dataloaders=training_dataloaders,
            algorithm_epoch_T=algorithm_epoch_T, accumulation_steps=accumulation_steps,
            use_amp=use_amp, scaler=scaler, criterion=criterion, basic_path=basic_path,
            iter_t=iter_t, communication_round_I=communication_round_I, num_clients_K=num_clients_K)

        for i, client_id in enumerate(idxs_users):
            users_gpu_seconds_list[client_id] += results[i]['gpu_seconds']
        total_gpu_seconds += sum(users_gpu_seconds_list)

        # ---- 常规 FedAvg 聚合（骨干网络）----
        logger.info("FedLMG: Parameter aggregation")
        pre_agg_params = get_parameters(global_model)
        client_model_updates = []
        theta_list = []
        for i, id in enumerate(idxs_users):
            client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
            selected_model = torch.load(client_model_path, weights_only=False)
            theta_list.append(get_parameters(selected_model))
            updates = {}
            for j, (p_local, p_global) in enumerate(zip(get_parameters(selected_model), pre_agg_params)):
                updates[str(j)] = torch.tensor(p_local) - torch.tensor(p_global)
            client_model_updates.append(updates)
            del selected_model
            gc.collect()
        theta_list = np.array(theta_list, dtype=object)
        weights = [client_datasets_size_list[j] for j in idxs_users]
        theta_avg = np.average(theta_list, axis=0, weights=weights).tolist()
        set_parameters(global_model, theta_avg)

        # ===== FedLMG 核心：本地模型引导的扩散式特征采样 + KD =====
        gpu_s = time.time()
        global_model.to(device)

        # Step 1: 重建客户端分类头（引导集成）
        client_clfs = []
        for i in range(len(idxs_users)):
            clf_template = _get_clf_module(global_model, param_dict)
            clf = copy.deepcopy(clf_template).to(device)
            clf.load_state_dict({k: v.to(device) for k, v in results[i]['clf_state'].items()})
            clf.eval()
            client_clfs.append(clf)

        head = _get_clf_module(global_model, param_dict)
        feat_dim = head.in_features
        noise_std = 1.0  # 初始噪声尺度（随扩散步衰减）

        # Step 2: 类别平衡采样目标标签
        n_gen_half = lmg_gen_num // 2
        target_labels = torch.cat([
            torch.zeros(n_gen_half, dtype=torch.long, device=device),
            torch.ones(lmg_gen_num - n_gen_half, dtype=torch.long, device=device)])
        perm = torch.randperm(lmg_gen_num, device=device)
        target_labels = target_labels[perm]

        # Step 3: 退火朗之万扩散采样（高斯先验 -> 类条件伪特征）
        feats = torch.randn(lmg_gen_num, feat_dim, device=device)
        cur_noise = noise_std
        for _step in range(lmg_diff_steps):
            feats = feats + cur_noise * torch.randn_like(feats)  # 加噪
            feats = feats.detach().requires_grad_(True)
            logit_sum = 0
            for clf in client_clfs:
                logit_sum = logit_sum + clf(feats)
            logit_avg = logit_sum / len(client_clfs)
            if "SENT_CLF" in param_dict["task"]:
                guidance_loss = torch.nn.functional.cross_entropy(logit_avg, target_labels)
            else:
                guidance_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logit_avg[:, 0], target_labels.float())
            grad = torch.autograd.grad(guidance_loss, feats)[0]
            with torch.no_grad():
                feats = feats - lmg_guidance_lr * grad  # 去噪（类别条件引导）
            feats = feats.detach()
            cur_noise *= lmg_noise_decay

        # Step 4: 集成 teacher 软标签
        teacher_probs = []
        with torch.no_grad():
            for clf in client_clfs:
                logit = clf(feats)
                if "SENT_CLF" in param_dict["task"]:
                    teacher_probs.append(torch.softmax(logit / lmg_T, dim=1))
                else:
                    teacher_probs.append(torch.sigmoid(logit / lmg_T))
        teacher_mean = torch.stack(teacher_probs, dim=0).mean(dim=0).detach()

        # Step 5: KD 训练全局分类头（冻结骨干）
        for p in global_model.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = True
        head_opt = torch.optim.SGD(head.parameters(), lr=lmg_kd_lr)
        head.train()
        for _epoch in range(lmg_kd_epochs):
            if "SENT_CLF" in param_dict["task"]:
                logits = head(feats)
                logp = torch.log_softmax(logits, dim=1)
                loss_kd = -(teacher_mean * logp).sum(dim=1).mean()
            else:
                logit = head(feats)[:, 0]
                loss_kd = torch.nn.functional.binary_cross_entropy_with_logits(logit, teacher_mean[:, 0])
            head_opt.zero_grad()
            loss_kd.backward()
            head_opt.step()
        for p in global_model.parameters():
            p.requires_grad = True

        total_gpu_seconds += time.time() - gpu_s
        global_model.to("cpu")
        del client_clfs, feats, teacher_mean
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

    logger.info("FedLMG training finished.")
    save_dir = './save_path/'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(global_model, os.path.join(save_dir, "global_FedLMG.pt"))
    total_comm_cost = communication_round_I * num_clients_K * FL_fraction * 2 * model_MB_size
    return global_model, total_gpu_seconds, total_comm_cost
