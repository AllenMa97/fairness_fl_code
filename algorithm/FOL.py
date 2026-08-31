# FOL: Federated Oriented Learning: A Practical One-Shot Personalized Federated Learning Framework
# https://openreview.net/ (ICML 2025)
# 核心思想: 操作空间为特征空间（orientation alignment）。各客户端上传本地"方向向量"（逐类特征
#           中心/取向）与分类头；服务器将各客户端的方向向量数据量加权聚合成全局取向（orientation），
#           在"取向引导"的合成特征（类中心 + 噪声）上，用客户端分类头集成做 teacher，定向对齐全局
#           分类头，实现一次通信下的个性化知识聚合。
# Core Idea: Operates in feature space (orientation alignment). Clients upload local "orientation
#            vectors" (per-class feature centroids) and classifier heads; the server aggregates the
#            orientations into a global orientation, synthesizes orientation-guided features
#            (centroid + noise), and aligns the global classifier head with the ensemble of client
#            heads on these features (one-shot personalized knowledge aggregation).
# 框架适配说明: 原论文为 one-shot 个性化 FL；本框架为多轮，每轮重复 "本地训练+取向提取 -> 服务器定向对齐"。

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


def _forward_features(model, param_dict, batch, device):
    with torch.no_grad():
        if "SENT_CLF" in param_dict["task"]:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            return model.only_PLM_forward(input_ids=input_ids, attention_mask=attention_mask)
        elif "IMG_CLF" in param_dict["task"]:
            imgs = batch["img"].to(device)
            return model.only_backbone_forward(imgs)
        else:
            X = batch["X"].to(device)
            if "LogisticRegression" in str(type(model)):
                return X
            return model.only_backbone_forward(X)


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


def _train_single_client_fol(client_id, device, model, param_dict,
                             training_dataloaders, algorithm_epoch_T,
                             accumulation_steps, use_amp, scaler, criterion,
                             basic_path, iter_t, communication_round_I, num_clients_K,
                             fol_max_samples):
    """FOL 单客户端：标准本地训练 + 逐类特征中心（取向向量）与分类头提取。"""
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

    # ---- 提取逐类特征中心（取向向量）与分类头 ----
    gpu_start_time = time.time()
    model.eval()
    cls_sum, cls_cnt = {}, {}
    collected = 0
    for batch in client_i_dataloader:
        feats = _forward_features(model, param_dict, batch, device).detach()
        labels = batch["labels"].to(device)
        for y in torch.unique(labels):
            yv = int(y.item())
            mask = (labels == y)
            if yv in cls_sum:
                cls_sum[yv] = cls_sum[yv] + feats[mask].sum(dim=0)
                cls_cnt[yv] += int(mask.sum().item())
            else:
                cls_sum[yv] = feats[mask].sum(dim=0)
                cls_cnt[yv] = int(mask.sum().item())
        collected += feats.size(0)
        if collected >= fol_max_samples:
            break

    centroids = {yv: (cls_sum[yv] / max(cls_cnt[yv], 1)).cpu().numpy() for yv in cls_sum}
    counts = dict(cls_cnt)
    clf_state = {name: p.detach().cpu() for name, p in _get_clf_module(model, param_dict).state_dict().items()}
    gpu_seconds += time.time() - gpu_start_time

    client_model_path = os.path.join(basic_path, "client_" + str(client_id + 1), 'model.pt')
    torch.save(model.cpu(), client_model_path)
    return {'gpu_seconds': gpu_seconds, 'centroids': centroids, 'counts': counts, 'clf_state': clf_state}


def FOL(device,
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

    # FOL 超参数
    fol_max_samples = int(param_dict.get('FOL_max_samples', 512))
    fol_gen_num = int(param_dict.get('FOL_gen_num', 256))       # 取向引导合成特征数
    fol_noise_std = float(param_dict.get('FOL_noise_std', 0.5))  # 类中心扰动噪声标准差
    fol_steps = int(param_dict.get('FOL_steps', 5))             # 全局头定向对齐步数
    fol_lr = float(param_dict.get('FOL_lr', 0.01))
    fol_T = float(param_dict.get('FOL_T', 2.0))                 # 集成 KD 温度
    fol_lambda_kd = float(param_dict.get('FOL_lambda_kd', 0.5))  # 集成 KD 损失权重

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

        logger.info(f"Round {iter_t + 1}; Select clients: {idxs_users}; Start Local Training (FOL)")

        results = parallel_executor.run_clients(
            idxs_users, _train_single_client_fol,
            param_dict=param_dict, training_dataloaders=training_dataloaders,
            algorithm_epoch_T=algorithm_epoch_T, accumulation_steps=accumulation_steps,
            use_amp=use_amp, scaler=scaler, criterion=criterion, basic_path=basic_path,
            iter_t=iter_t, communication_round_I=communication_round_I, num_clients_K=num_clients_K,
            fol_max_samples=fol_max_samples)

        for i, client_id in enumerate(idxs_users):
            users_gpu_seconds_list[client_id] += results[i]['gpu_seconds']
        total_gpu_seconds += sum(users_gpu_seconds_list)

        # ---- 常规 FedAvg 聚合（骨干网络）----
        logger.info("FOL: Parameter aggregation")
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

        # ===== FOL 核心：全局取向聚合 + 取向引导特征 + 定向对齐 =====
        gpu_s = time.time()
        global_model.to(device)

        # Step 1: 数据量加权聚合各客户端的取向向量（逐类特征中心）
        global_centroids = {}
        total_w = sum(weights)
        for i, res in enumerate(results):
            w_i = weights[i] / total_w
            for yv, c in res['centroids'].items():
                t = torch.tensor(c, dtype=torch.float32, device=device) * w_i
                global_centroids[yv] = global_centroids.get(yv, torch.zeros_like(t)) + t
        if len(global_centroids) == 0:
            global_centroids = {0: torch.zeros(1, device=device)}

        # Step 2: 重建各客户端分类头（teacher 集成）
        client_clfs = []
        for i in range(len(idxs_users)):
            clf_template = _get_clf_module(global_model, param_dict)
            clf = copy.deepcopy(clf_template).to(device)
            clf.load_state_dict({k: v.to(device) for k, v in results[i]['clf_state'].items()})
            clf.eval()
            client_clfs.append(clf)

        # Step 3: 取向引导合成特征（类中心 + 高斯噪声），类别按各客户端样本计数加权采样
        all_y, all_cnt = list(results[0]['counts'].keys()), None
        count_pool = {}
        for res in results:
            for yv, c in res['counts'].items():
                count_pool[yv] = count_pool.get(yv, 0) + c
        ys = list(count_pool.keys())
        ps = np.array([count_pool[y] for y in ys], dtype=np.float64)
        ps = ps / ps.sum()

        head = _get_clf_module(global_model, param_dict)
        feat_dim = head.in_features
        synth_feats, synth_labels = [], []
        for _ in range(fol_gen_num):
            yv = int(np.random.choice(ys, p=ps))
            anchor = global_centroids.get(yv)
            if anchor is None or anchor.numel() != feat_dim:
                anchor = torch.randn(feat_dim, device=device) * 0.1
            f = anchor + fol_noise_std * torch.randn(feat_dim, device=device)
            synth_feats.append(f)
            synth_labels.append(yv)
        synth_feats = torch.stack(synth_feats, dim=0)
        synth_labels = torch.tensor(synth_labels, device=device)

        # Step 4: 集成 teacher 软标签（在合成特征上）
        teacher_probs = []
        with torch.no_grad():
            for clf in client_clfs:
                logit = clf(synth_feats)
                if "SENT_CLF" in param_dict["task"]:
                    teacher_probs.append(torch.softmax(logit / fol_T, dim=1))
                else:
                    teacher_probs.append(torch.sigmoid(logit / fol_T))
        teacher_mean = torch.stack(teacher_probs, dim=0).mean(dim=0).detach()

        # Step 5: 定向对齐全局分类头（CE + 集成 KD）
        for p in global_model.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = True
        head_opt = torch.optim.SGD(head.parameters(), lr=fol_lr)
        head.train()
        for _step in range(fol_steps):
            if "SENT_CLF" in param_dict["task"]:
                logits = head(synth_feats)
                logp = torch.log_softmax(logits, dim=1)
                loss_ce = torch.nn.functional.nll_loss(logp, synth_labels)
                loss_kd = -(teacher_mean * logp).sum(dim=1).mean()
            else:
                logit = head(synth_feats)[:, 0]
                loss_ce = torch.nn.functional.binary_cross_entropy_with_logits(logit, synth_labels.float())
                loss_kd = torch.nn.functional.binary_cross_entropy_with_logits(logit, teacher_mean[:, 0])
            loss = loss_ce + fol_lambda_kd * loss_kd
            head_opt.zero_grad()
            loss.backward()
            head_opt.step()
        for p in global_model.parameters():
            p.requires_grad = True

        total_gpu_seconds += time.time() - gpu_s
        global_model.to("cpu")
        del client_clfs, synth_feats, teacher_mean
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

    logger.info("FOL training finished.")
    save_dir = './save_path/'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(global_model, os.path.join(save_dir, "global_FOL.pt"))
    total_comm_cost = communication_round_I * num_clients_K * FL_fraction * 2 * model_MB_size
    return global_model, total_gpu_seconds, total_comm_cost
