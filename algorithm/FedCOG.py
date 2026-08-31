# FedCOG: Fake It Till Make It: Federated Learning with Consensus-Oriented Generation
# ICLR 2024
# 核心思想: 操作空间为输入/特征空间（从全局模型生成共识数据补充本地数据集）。
# 服务器端训练一个"共识生成器"：给定全局模型和若干客户端模型，
# 生成让所有客户端分类头输出一致的"共识样本"，分发给各客户端补充本地训练。
# Core Idea: Operates in input/feature space (consensus-oriented data generation
# from global model to augment local datasets). Server trains a consensus generator:
# given global & client models, it produces samples where all client classifiers
# agree (consensus); these samples are distributed to augment local training.

import copy
import os
import gc
import time
import torch
import torch.nn as nn
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


# ---------- FedCOG 轻量共识生成器（直接在特征空间合成，避免输入空间离散化问题）----------
class ConsensusFeatGenerator(nn.Module):
    """在特征空间生成"共识"特征（绕开文本/图像离散输入空间）。"""
    def __init__(self, noise_dim, feat_dim, num_classes=2, hidden=256):
        super().__init__()
        self.nc = num_classes
        self.net = nn.Sequential(
            nn.Linear(noise_dim + num_classes, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, feat_dim),
        )

    def forward(self, z, y_onehot):
        return self.net(torch.cat([z, y_onehot], dim=1))

    def sample(self, n, device):
        z = torch.randn(n, self.net[0].in_features - self.nc, device=device)
        y = torch.randint(0, self.nc, (n,), device=device)
        y_oh = torch.zeros(n, self.nc, device=device).scatter_(1, y.unsqueeze(1), 1.0)
        return self.forward(z, y_oh), y


def _clf_forward(clf_module, feat, param_dict):
    if "SENT_CLF" in param_dict["task"]:
        return clf_module(feat)
    else:
        return clf_module(feat)


def _extract_clf(model, param_dict):
    if "SENT_CLF" in param_dict["task"]:
        return model.out
    elif "IMG_CLF" in param_dict["task"]:
        return model.out_layer
    else:
        if "LogisticRegression" in str(type(model)):
            return model.layer
        return model.out_layer


def _train_single_client_fedcog(client_id, device, model, param_dict,
                                  training_dataloaders, algorithm_epoch_T,
                                  accumulation_steps, use_amp, scaler, criterion,
                                  basic_path, iter_t, communication_round_I, num_clients_K,
                                  consensus_feats, consensus_labels):
    """FedCOG 单客户端：本地训练 + 在共识样本上做知识对齐。"""
    model.train()
    model.to(device)
    optimizer = BERTCLF_Optimizer(
        method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
    optimizer.set_parameters(list(model.named_parameters()))
    client_i_dataloader = training_dataloaders[client_id]
    gpu_seconds = 0
    clf = _extract_clf(model, param_dict)

    cog_w = float(param_dict.get('FedCOG_consensus_weight', 0.3))

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
                loss_task = torch.sum(bl) / bs

                # 在共识特征上对齐分类器（用 consensus 样本补充训练）
                loss_consensus = 0.0
                if consensus_feats is not None and cog_w > 0:
                    cf = consensus_feats.to(device)
                    cl = consensus_labels.to(device)
                    lg_c = _clf_forward(clf, cf, param_dict)
                    if "SENT_CLF" in param_dict["task"]:
                        bl_c = F.cross_entropy(lg_c, cl, reduction='mean')
                    else:
                        bl_c = F.binary_cross_entropy_with_logits(lg_c[:, 0], cl.float(), reduction='mean')
                    loss_consensus = bl_c

                loss = loss_task + cog_w * loss_consensus

            scale_backward(loss, scaler)
            if (batch_id + 1) % accumulation_steps == 0:
                scaler_step(scaler, optimizer); model.zero_grad()
            gpu_seconds += (time.time() - gs)
            el += loss_task
            gc.collect()

        if (batch_id + 1) % accumulation_steps != 0:
            scaler_step(scaler, optimizer); model.zero_grad()
        logger.info(f"Round {iter_t+1}/{communication_round_I}; Client {client_id}; Epoch {epoch+1}; Loss: {el/max(es,1):.4f}")

    client_model_path = os.path.join(basic_path, "client_" + str(client_id + 1), 'model.pt')
    torch.save(model.cpu(), client_model_path)
    return {'gpu_seconds': gpu_seconds}


def Fed_COG(device,
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

    # FedCOG 超参数
    cog_n = int(param_dict.get('FedCOG_samples', 64))    # 共识特征数
    cog_steps = int(param_dict.get('FedCOG_gen_steps', 50))  # 共识生成器优化步数
    cog_lr = float(param_dict.get('FedCOG_gen_lr', 0.01))
    cog_agree_w = float(param_dict.get('FedCOG_agree_lambda', 1.0))  # 分类一致性权重

    # feature dim
    if "SENT_CLF" in param_dict["task"]:
        feat_dim = param_dict.get('emb_dim', 768)
    elif "IMG_CLF" in param_dict["task"]:
        feat_dim = param_dict.get('emb_dim', 512)
    else:
        feat_dim = param_dict.get('nn_input_size', 128)
    noise_dim = 32

    basic_path = param_dict['model_path']
    for k in range(param_dict["num_clients_K"]):
        full_path = os.path.join(basic_path, "client_" + str(k + 1), 'model.pt')
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        torch.save(global_model, full_path)

    if "SENT_CLF" in param_dict["task"]:
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
    else:
        criterion = torch.nn.BCELoss(reduction='none').to(device)

    # 初始化共识生成器
    gen = ConsensusFeatGenerator(noise_dim, feat_dim, num_classes=2).to(device)

    total_gpu_seconds = 0
    users_gpu_seconds_list = [0] * num_clients_K
    model_MB_size = sum(p.numel() for p in global_model.parameters()) * 4 / (1024 * 1024)
    gen_MB_size = sum(p.numel() for p in gen.parameters()) * 4 / (1024 * 1024)
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # 初始共识特征（None → 第一轮不使用）
    consensus_feats = None
    consensus_labels = None

    parallel_executor = ClientParallelExecutor(
        device=device, global_model=global_model, param_dict=param_dict, needs_global_model_during_training=False)

    for iter_t in range(start_round, communication_round_I):
        idxs_users = client_selection(
            client_num=num_clients_K, fraction=FL_fraction,
            dataset_size=training_dataset_size, client_dataset_size_list=client_datasets_size_list,
            drop_rate=FL_drop_rate, style="FedAvg")

        logger.info(f"Round {iter_t+1}; Select clients: {idxs_users}; Start FedCOG")

        results = parallel_executor.run_clients(
            idxs_users, _train_single_client_fedcog,
            param_dict=param_dict, training_dataloaders=training_dataloaders,
            algorithm_epoch_T=algorithm_epoch_T, accumulation_steps=accumulation_steps,
            use_amp=use_amp, scaler=scaler, criterion=criterion, basic_path=basic_path,
            iter_t=iter_t, communication_round_I=communication_round_I, num_clients_K=num_clients_K,
            consensus_feats=consensus_feats, consensus_labels=consensus_labels)

        for i, cid in enumerate(idxs_users):
            users_gpu_seconds_list[cid] += results[i]['gpu_seconds']
        total_gpu_seconds += sum(users_gpu_seconds_list)

        # 聚合模型
        logger.info("FedCOG: Aggregate + Train Consensus Generator")
        pre_agg_params = get_parameters(global_model)
        client_model_updates = []
        theta_list = []
        client_clfs = []
        for i, cid in enumerate(idxs_users):
            path = os.path.join(basic_path, "client_" + str(cid + 1), 'model.pt')
            sm = torch.load(path, weights_only=False)
            cp = get_parameters(sm)
            theta_list.append(cp)
            client_clfs.append(copy.deepcopy(_extract_clf(sm, param_dict)))
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

        # ===== FedCOG 核心：训练共识生成器（让各分类头在生成特征上输出一致）=====
        gpu_s = time.time()
        global_model.to(device)
        global_clf = _extract_clf(global_model, param_dict)
        for c in client_clfs:
            c.to(device).eval()
        global_clf.eval()

        gen.train()
        gen_opt = torch.optim.Adam(gen.parameters(), lr=cog_lr)

        for _s in range(cog_steps):
            feats, y = gen.sample(cog_n, device)
            # 各客户端分类头 + 全局分类头输出
            outputs = []
            for c in client_clfs:
                lg = _clf_forward(c, feats, param_dict)
                outputs.append(lg)
            # 全局分类头
            lg_g = _clf_forward(global_clf, feats, param_dict)
            outputs.append(lg_g)
            stacked = torch.stack(outputs, dim=0)  # [M, N, C]

            # 共识损失 1：输出方差最小（分类器间一致性）
            mean_lg = stacked.mean(dim=0, keepdim=True)
            loss_agree = ((stacked - mean_lg) ** 2).mean()

            # 共识损失 2：高置信度（高信息 → 有用样本）
            avg_lg = mean_lg.squeeze(0)
            if "SENT_CLF" in param_dict["task"]:
                p = F.softmax(avg_lg, dim=1)
                ent = -(p * torch.log(p + 1e-9)).sum(dim=1).mean()
                loss_conf = ent
            else:
                p = torch.sigmoid(avg_lg)
                ent = -(p * torch.log(p + 1e-9) + (1 - p) * torch.log(1 - p + 1e-9)).mean()
                loss_conf = ent

            loss_gen = cog_agree_w * loss_agree + loss_conf
            gen_opt.zero_grad()
            loss_gen.backward()
            gen_opt.step()

        # 生成共识样本，下发给下一轮客户端
        gen.eval()
        with torch.no_grad():
            feats_final, y_final = gen.sample(cog_n * 2, device)
            consensus_feats = feats_final.cpu()
            consensus_labels = y_final.cpu()

        for c in client_clfs:
            c.to("cpu"); del c
        client_clfs.clear()
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

    logger.info("FedCOG training finished.")
    save_dir = './save_path/'; os.makedirs(save_dir, exist_ok=True)
    torch.save(global_model, os.path.join(save_dir, "global_FedCOG.pt"))
    total_comm_cost = communication_round_I * num_clients_K * FL_fraction * (2 * model_MB_size + gen_MB_size)
    return global_model, total_gpu_seconds, total_comm_cost
