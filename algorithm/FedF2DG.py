# FedF²DG: Data-free knowledge distillation via generator-free data generation for Non-IID federated learning
# Neural Networks, 2024 | https://www.sciencedirect.com/science/article/pii/S0893608024005513
# 核心思想: 操作空间为输入空间（模型反演优化伪输入，无生成器）。
# 不依赖生成器，直接通过最大化各客户端分类头在伪输入上的输出（DeepInversion 风格）
# 来合成高置信伪样本，然后利用伪样本进行知识蒸馏。
# Core Idea: Operates in input space (model-inversion optimized pseudo-input, NO generator).
# Generator-free: directly optimizes pseudo-inputs by maximizing client classifier
# outputs on them (DeepInversion-style), then performs KD on synthetic samples.

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


def _get_clf(model, param_dict):
    if "SENT_CLF" in param_dict["task"]:
        return model.out
    elif "IMG_CLF" in param_dict["task"]:
        return model.out_layer
    else:
        if "LogisticRegression" in str(type(model)):
            return model.layer
        return model.out_layer


def _get_backbone(model, param_dict):
    """提取 backbone 用于从伪输入提取特征。"""
    if "SENT_CLF" in param_dict["task"]:
        def _bb(**kwargs):
            return model.only_PLM_forward(**kwargs)
        return _bb
    elif "IMG_CLF" in param_dict["task"]:
        def _bb(imgs):
            return model.only_backbone_forward(imgs)
        return _bb
    else:
        if "ANN" in str(type(model)):
            def _bb(X):
                return model.only_backbone_forward(X)
            return _bb
        else:
            def _bb(X):
                return X  # LR: 特征即输入
            return _bb


def _deepinversion_synthesize(device, param_dict, client_models_list, num_samples=32, steps=200, lr=0.1):
    """无生成器 DeepInversion：在输入空间直接优化伪样本，使各客户端分类头输出高置信度。"""
    # 1) 初始化伪输入
    if "SENT_CLF" in param_dict["task"]:
        max_len = param_dict.get('max_len', 128)
        emb_dim = param_dict.get('emb_dim', 768)
        # 对 SENT 直接优化 embeddings 空间代替离散 token（连续松弛）
        pseudo_emb = torch.randn(num_samples, max_len, emb_dim, device=device, requires_grad=True)
        am = torch.ones((num_samples, max_len), device=device)
        optim = torch.optim.Adam([pseudo_emb], lr=lr)
    elif "IMG_CLF" in param_dict["task"]:
        inp_ch = 3 if param_dict.get('dataset', '').lower() not in ['fmnist', 'mnist'] else 1
        pseudo_inp = torch.randn(num_samples, inp_ch, 32, 32, device=device, requires_grad=True)
        optim = torch.optim.Adam([pseudo_inp], lr=lr)
    else:  # Tabular
        inp_size = param_dict.get('nn_input_size', 128)
        pseudo_inp = torch.randn(num_samples, inp_size, device=device, requires_grad=True)
        optim = torch.optim.Adam([pseudo_inp], lr=lr)

    for s in range(steps):
        optim.zero_grad()

        # 前向：所有客户端分类头在伪输入上的输出
        logits_list = []
        for cm in client_models_list:
            try:
                if "SENT_CLF" in param_dict["task"]:
                    # 直接从伪 embedding 走分类头
                    feats = cm.bert.encoder(pseudo_emb, attention_mask=am.unsqueeze(1).unsqueeze(2))[0][:, 0]
                    lg = cm.out(feats)
                elif "IMG_CLF" in param_dict["task"]:
                    feats = cm.only_backbone_forward(pseudo_inp)
                    lg = cm.out_layer(feats)
                else:
                    if "ANN" in str(type(cm)):
                        feats = cm.only_backbone_forward(pseudo_inp)
                        lg = cm.out_layer(feats)
                    else:
                        lg = cm.layer(pseudo_inp)
                logits_list.append(lg)
            except Exception:
                continue

        if not logits_list:
            break

        stacked = torch.stack(logits_list, dim=0).mean(dim=0)  # 集成平均
        # 目标 1: 高置信度（大 logit 绝对值）
        if "SENT_CLF" in param_dict["task"]:
            p = F.softmax(stacked, dim=1)
            entropy = -(p * torch.log(p + 1e-9)).sum(dim=1).mean()
            loss_inv = entropy  # 最小化熵 → 高置信度
        else:
            p = torch.sigmoid(stacked)
            entropy = -(p * torch.log(p + 1e-9) + (1 - p) * torch.log(1 - p + 1e-9)).mean()
            loss_inv = entropy

        # 目标 2: 输入正则（L2 / BN 统计）
        if "IMG_CLF" in param_dict["task"]:
            loss_reg = pseudo_inp.norm(p=2) ** 2 / pseudo_inp.numel()
            total_loss = loss_inv + 0.001 * loss_reg
        elif "SENT_CLF" in param_dict["task"]:
            loss_reg = pseudo_emb.norm(p=2) ** 2 / pseudo_emb.numel()
            total_loss = loss_inv + 0.001 * loss_reg
        else:
            loss_reg = pseudo_inp.norm(p=2) ** 2 / pseudo_inp.numel()
            total_loss = loss_inv + 0.001 * loss_reg

        total_loss.backward()
        optim.step()

    # 返回最终伪输入（detach）
    if "SENT_CLF" in param_dict["task"]:
        return {'type': 'emb', 'emb': pseudo_emb.detach(), 'attention_mask': am}
    elif "IMG_CLF" in param_dict["task"]:
        return {'type': 'img', 'img': pseudo_inp.detach()}
    else:
        return {'type': 'tab', 'X': pseudo_inp.detach()}


def _train_single_client_fedf2dg(client_id, device, model, param_dict,
                                   training_dataloaders, algorithm_epoch_T,
                                   accumulation_steps, use_amp, scaler, criterion,
                                   basic_path, iter_t, communication_round_I, num_clients_K):
    """FedF²DG 本地训练（标准 CE/BCE），分类头随模型一起上传供 DeepInversion 使用。"""
    model.train()
    model.to(device)
    optimizer = BERTCLF_Optimizer(
        method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
    optimizer.set_parameters(list(model.named_parameters()))
    client_i_dataloader = training_dataloaders[client_id]
    gpu_seconds = 0

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
                preds, _ = model(im)
                bl = criterion(preds[:, 0], lb.float())
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

    client_model_path = os.path.join(basic_path, "client_" + str(client_id + 1), 'model.pt')
    torch.save(model.cpu(), client_model_path)
    return {'gpu_seconds': gpu_seconds}


def Fed_F2DG(device,
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

    # FedF²DG 超参数
    f2dg_n = int(param_dict.get('FedF2DG_n', 32))       # 每轮合成伪样本数
    f2dg_s = int(param_dict.get('FedF2DG_steps', 150))  # DeepInversion 步数
    f2dg_lr = float(param_dict.get('FedF2DG_lr', 0.05))  # 伪输入优化 LR
    f2dg_kd = float(param_dict.get('FedF2DG_kd', 1.0))   # KD 权重
    f2dg_kd_epochs = int(param_dict.get('FedF2DG_kd_epochs', 5))  # 全局蒸馏 epoch

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

        logger.info(f"Round {iter_t+1}; Select clients: {idxs_users}; Start FedF²DG")

        results = parallel_executor.run_clients(
            idxs_users, _train_single_client_fedf2dg,
            param_dict=param_dict, training_dataloaders=training_dataloaders,
            algorithm_epoch_T=algorithm_epoch_T, accumulation_steps=accumulation_steps,
            use_amp=use_amp, scaler=scaler, criterion=criterion, basic_path=basic_path,
            iter_t=iter_t, communication_round_I=communication_round_I, num_clients_K=num_clients_K)

        for i, cid in enumerate(idxs_users):
            users_gpu_seconds_list[cid] += results[i]['gpu_seconds']
        total_gpu_seconds += sum(users_gpu_seconds_list)

        # 聚合模型
        logger.info("FedF²DG: Aggregate + Generator-free DeepInversion + KD")
        pre_agg_params = get_parameters(global_model)
        client_model_updates = []
        theta_list = []
        client_models_loaded = []
        for i, cid in enumerate(idxs_users):
            path = os.path.join(basic_path, "client_" + str(cid + 1), 'model.pt')
            sm = torch.load(path, weights_only=False)
            cp = get_parameters(sm)
            theta_list.append(cp)
            client_models_loaded.append(sm)
            updates = {}
            for j, (pl, pg) in enumerate(zip(cp, pre_agg_params)):
                updates[str(j)] = torch.tensor(pl) - torch.tensor(pg)
            client_model_updates.append(updates)

        weights = [client_datasets_size_list[j] for j in idxs_users]
        w_arr = np.array(weights, dtype=np.float64); w_arr = w_arr / w_arr.sum()
        theta_arr = np.array(theta_list, dtype=object)
        theta_avg = np.average(theta_arr, axis=0, weights=w_arr).tolist()
        set_parameters(global_model, theta_avg)

        # ===== FedF²DG 核心：无生成器合成 + KD =====
        gpu_s = time.time()
        try:
            for m in client_models_loaded:
                m.to(device).eval()

            # Step A: DeepInversion 合成伪样本
            pseudo = _deepinversion_synthesize(
                device, param_dict, client_models_loaded,
                num_samples=f2dg_n, steps=f2dg_s, lr=f2dg_lr)

            # Step B: 计算 ensemble teacher（在伪样本上）
            global_model.to(device)
            global_model.train()
            teacher_logits_list = []
            with torch.no_grad():
                for cm in client_models_loaded:
                    if pseudo['type'] == 'emb':  # SENT
                        feats = cm.bert.encoder(pseudo['emb'], attention_mask=pseudo['attention_mask'].unsqueeze(1).unsqueeze(2))[0][:, 0]
                        lg = cm.out(feats)
                    elif pseudo['type'] == 'img':
                        feats = cm.only_backbone_forward(pseudo['img'])
                        lg = cm.out_layer(feats)
                    else:
                        if "ANN" in str(type(cm)):
                            feats = cm.only_backbone_forward(pseudo['X'])
                            lg = cm.out_layer(feats)
                        else:
                            lg = cm.layer(pseudo['X'])
                    teacher_logits_list.append(lg)

            if teacher_logits_list:
                teacher_avg = torch.stack(teacher_logits_list, dim=0).mean(dim=0).detach()

                # Step C: KD 训练全局模型若干 epoch
                distill_optim = BERTCLF_Optimizer(method="ADAM", learning_rate=param_dict['learning_rate']*0.1, max_grad_norm=0)
                distill_optim.set_parameters(list(global_model.named_parameters()))
                T = float(param_dict.get('FedF2DG_T', 3.0))

                for _ep in range(f2dg_kd_epochs):
                    distill_optim.zero_grad()
                    if pseudo['type'] == 'emb':
                        feats_g = global_model.bert.encoder(pseudo['emb'], attention_mask=pseudo['attention_mask'].unsqueeze(1).unsqueeze(2))[0][:, 0]
                        lg_g = global_model.out(feats_g)
                        s_soft = torch.log_softmax(lg_g / T, dim=1)
                        t_soft = torch.softmax(teacher_avg / T, dim=1)
                        loss_kd = -torch.sum(t_soft * s_soft, dim=1).mean() * (T ** 2)
                    elif pseudo['type'] == 'img':
                        feats_g = global_model.only_backbone_forward(pseudo['img'])
                        lg_g = global_model.out_layer(feats_g)
                        sp = torch.sigmoid(lg_g / T)
                        tp = torch.sigmoid(teacher_avg / T)
                        loss_kd = F.mse_loss(sp, tp) * (T ** 2)
                    else:
                        if "ANN" in str(type(global_model)):
                            feats_g = global_model.only_backbone_forward(pseudo['X'])
                            lg_g = global_model.out_layer(feats_g)
                        else:
                            lg_g = global_model.layer(pseudo['X'])
                        sp = torch.sigmoid(lg_g / T)
                        tp = torch.sigmoid(teacher_avg / T)
                        loss_kd = F.mse_loss(sp, tp) * (T ** 2)

                    (f2dg_kd * loss_kd).backward()
                    distill_optim.step()

        except Exception as e:
            logger.warning(f"FedF²DG DeepInversion+KD skipped due to: {e}")

        for m in client_models_loaded:
            m.to("cpu"); del m
        client_models_loaded.clear()
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

    logger.info("FedF²DG training finished.")
    save_dir = './save_path/'; os.makedirs(save_dir, exist_ok=True)
    torch.save(global_model, os.path.join(save_dir, "global_FedF2DG.pt"))
    total_comm_cost = communication_round_I * num_clients_K * FL_fraction * 2 * model_MB_size
    return global_model, total_gpu_seconds, total_comm_cost
