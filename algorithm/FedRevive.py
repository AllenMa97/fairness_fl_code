# FedRevive: Reviving Stale Updates: Data-Free Knowledge Distillation for Asynchronous Federated Learning
# https://arxiv.org/abs/2511.00655, 2025
# 核心思想: 操作空间为输入空间（meta-learned generator 合成伪样本）。
# 针对异步联邦学习的"陈旧更新（stale updates）"问题：
# 客户端上传后，服务器用元学习生成器合成伪样本，把陈旧客户端模型的知识"复活"（蒸馏）进最新全局模型，
# 从而让过期更新仍然贡献价值。此处实现其核心：meta-generator + KD 融合。
# Core Idea: Operates in input space (meta-learned generator synthesizes pseudo samples).
# Targets stale updates in asynchronous FL: after clients upload, the server uses a
# meta-learned generator to synthesize pseudo samples, which distill stale client
# knowledge into the latest global model—reviving outdated updates.
# This implementation focuses on the core meta-generator + KD fusion mechanism.

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


# ---------- FedRevive Meta Generator (feature space generator 以支持所有模态) ----------
class MetaFeatGenerator(nn.Module):
    """Meta-learned generator: 输入噪声 z + stale 程度 tau，输出伪特征向量（跨模态统一在特征空间）。"""
    def __init__(self, noise_dim, feat_dim, num_classes=2, tau_emb_dim=8, hidden=256):
        super().__init__()
        self.nc = num_classes
        self.tau_embed = nn.Sequential(
            nn.Linear(1, tau_emb_dim),
            nn.ReLU(inplace=True),
        )
        in_dim = noise_dim + num_classes + tau_emb_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, feat_dim),
        )

    def forward(self, z, y_onehot, tau):
        tau_emb = self.tau_embed(tau.view(-1, 1))
        x = torch.cat([z, y_onehot, tau_emb], dim=1)
        return self.net(x)

    def sample(self, n, device, tau_val=0.0):
        noise_dim = self.net[0].in_features - self.nc - self.tau_embed[0].out_features
        z = torch.randn(n, noise_dim, device=device)
        y = torch.randint(0, self.nc, (n,), device=device)
        y_oh = torch.zeros(n, self.nc, device=device).scatter_(1, y.unsqueeze(1), 1.0)
        tau = torch.full((n,), tau_val, device=device)
        return self.forward(z, y_oh, tau), y, tau


def _extract_clf_and_backbone(model, param_dict):
    """返回 (backbone_fn, clf_module)。"""
    if "SENT_CLF" in param_dict["task"]:
        def bb(**kwargs): return model.only_PLM_forward(**kwargs)
        return bb, model.out
    elif "IMG_CLF" in param_dict["task"]:
        def bb(im): return model.only_backbone_forward(im)
        return bb, model.out_layer
    else:
        if "ANN" in str(type(model)):
            def bb(X): return model.only_backbone_forward(X)
            return bb, model.out_layer
        else:
            def bb(X): return X
            return bb, model.layer


def _clf_logits(clf, feats, param_dict):
    return clf(feats)


def _train_single_client_fedrevive(client_id, device, model, param_dict,
                                     training_dataloaders, algorithm_epoch_T,
                                     accumulation_steps, use_amp, scaler, criterion,
                                     basic_path, iter_t, communication_round_I, num_clients_K):
    """FedRevive 单客户端：标准本地训练，记录'轮次'作为陈旧度依据。"""
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
    return {'gpu_seconds': gpu_seconds, 'upload_round': iter_t}


def Fed_Revive(device,
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

    # FedRevive 超参数
    rv_n = int(param_dict.get('FedRevive_n', 64))       # 每轮 meta 合成样本数
    rv_meta_steps = int(param_dict.get('FedRevive_meta_steps', 40))  # meta 更新步数
    rv_kd_epochs = int(param_dict.get('FedRevive_kd_epochs', 5))     # KD epoch
    rv_meta_lr = float(param_dict.get('FedRevive_meta_lr', 0.01))
    rv_T = float(param_dict.get('FedRevive_T', 3.0))                 # KD 温度

    if "SENT_CLF" in param_dict["task"]:
        feat_dim = param_dict.get('emb_dim', 768)
    elif "IMG_CLF" in param_dict["task"]:
        feat_dim = param_dict.get('emb_dim', 512)
    else:
        feat_dim = param_dict.get('nn_input_size', 128)

    basic_path = param_dict['model_path']
    for k in range(param_dict["num_clients_K"]):
        full_path = os.path.join(basic_path, "client_" + str(k + 1), 'model.pt')
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        torch.save(global_model, full_path)

    if "SENT_CLF" in param_dict["task"]:
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
    else:
        criterion = torch.nn.BCELoss(reduction='none').to(device)

    # 初始化 Meta Generator + stale buffer（保存各客户端最近上传的分类头与上传轮次）
    meta_gen = MetaFeatGenerator(32, feat_dim, num_classes=2).to(device)
    stale_clfs_buffer = {}     # client_id -> {'clf_state': dict, 'upload_round': int}
    stale_max_keep = int(param_dict.get('FedRevive_stale_keep', 5))  # 最多保留的陈旧版本数

    total_gpu_seconds = 0
    users_gpu_seconds_list = [0] * num_clients_K
    model_MB_size = sum(p.numel() for p in global_model.parameters()) * 4 / (1024 * 1024)
    gen_MB_size = sum(p.numel() for p in meta_gen.parameters()) * 4 / (1024 * 1024)
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    parallel_executor = ClientParallelExecutor(
        device=device, global_model=global_model, param_dict=param_dict, needs_global_model_during_training=False)

    for iter_t in range(start_round, communication_round_I):
        idxs_users = client_selection(
            client_num=num_clients_K, fraction=FL_fraction,
            dataset_size=training_dataset_size, client_dataset_size_list=client_datasets_size_list,
            drop_rate=FL_drop_rate, style="FedAvg")

        logger.info(f"Round {iter_t+1}; Select clients: {idxs_users}; Start FedRevive (Stale Update Revival)")

        results = parallel_executor.run_clients(
            idxs_users, _train_single_client_fedrevive,
            param_dict=param_dict, training_dataloaders=training_dataloaders,
            algorithm_epoch_T=algorithm_epoch_T, accumulation_steps=accumulation_steps,
            use_amp=use_amp, scaler=scaler, criterion=criterion, basic_path=basic_path,
            iter_t=iter_t, communication_round_I=communication_round_I, num_clients_K=num_clients_K)

        for i, cid in enumerate(idxs_users):
            users_gpu_seconds_list[cid] += results[i]['gpu_seconds']
        total_gpu_seconds += sum(users_gpu_seconds_list)

        # 聚合模型
        logger.info("FedRevive: Aggregate + Meta-Generator Train + Stale KD Revival")
        pre_agg_params = get_parameters(global_model)
        client_model_updates = []
        theta_list = []
        for i, cid in enumerate(idxs_users):
            path = os.path.join(basic_path, "client_" + str(cid + 1), 'model.pt')
            sm = torch.load(path, weights_only=False)
            cp = get_parameters(sm)
            theta_list.append(cp)

            # 存入 stale buffer（分类头状态 + 上传轮次）
            _, clf = _extract_clf_and_backbone(sm, param_dict)
            clf_state = {k: v.detach().cpu().numpy() for k, v in clf.state_dict().items()}
            if cid not in stale_clfs_buffer:
                stale_clfs_buffer[cid] = []
            stale_clfs_buffer[cid].append({'clf_state': clf_state, 'upload_round': iter_t})
            # 保留最近 N 个版本
            if len(stale_clfs_buffer[cid]) > stale_max_keep:
                stale_clfs_buffer[cid] = stale_clfs_buffer[cid][-stale_max_keep:]

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

        # ===== FedRevive 核心：Meta-Generator（联合当前+陈旧分类头训练）+ 全局 KD =====
        gpu_s = time.time()
        global_model.to(device)
        _, global_clf = _extract_clf_and_backbone(global_model, param_dict)

        # 构建 stale 版本分类头集合（用于 meta 训练）
        all_stale_clfs = []
        all_stale_taus = []  # 陈旧度：当前轮 - 上传轮（越大越陈旧）
        for cid, history in stale_clfs_buffer.items():
            for item in history:
                # 重建临时分类头
                clf_ref = copy.deepcopy(global_clf)
                try:
                    sd = {k: torch.tensor(v) for k, v in item['clf_state'].items()}
                    clf_ref.load_state_dict(sd)
                    all_stale_clfs.append(clf_ref.to(device).eval())
                    tau = float(iter_t - item['upload_round'])
                    all_stale_taus.append(max(tau, 0.0))
                except Exception:
                    pass

        meta_gen.train()
        gen_opt = torch.optim.Adam(meta_gen.parameters(), lr=rv_meta_lr)
        distill_optim = BERTCLF_Optimizer(method="ADAM", learning_rate=param_dict['learning_rate']*0.1, max_grad_norm=0)
        distill_optim.set_parameters(list(global_clf.named_parameters()))

        # --- Step 1: Meta-train Generator 让 stale 分类器与 global 分类器在生成特征上输出一致（revival）---
        if len(all_stale_clfs) > 0:
            for _ms in range(rv_meta_steps):
                gen_opt.zero_grad()
                # 随机选择若干 stale 分类器 + 随机 tau（陈旧度）
                K = min(4, len(all_stale_clfs))
                pick_idx = np.random.choice(len(all_stale_clfs), K, replace=False)
                taus_b = []
                feats_all = []
                y_all = []
                for pick_i in pick_idx:
                    clf_pick = all_stale_clfs[pick_i]
                    tau_pick = all_stale_taus[pick_i]
                    feats, y, tau_vec = meta_gen.sample(rv_n, device, tau_val=tau_pick / max(iter_t + 1, 1))
                    feats_all.append(feats)
                    y_all.append(y)
                    taus_b.append(tau_vec[0].item())

                feats_cat = torch.cat(feats_all, dim=0)
                y_cat = torch.cat(y_all, dim=0)

                # global + stale 的输出一致性（revival 目标：stale == global → 陈旧知识被复活）
                lg_g = _clf_logits(global_clf, feats_cat, param_dict).detach()
                total_meta_loss = 0.0
                for kk, pick_i in enumerate(pick_idx):
                    clf_pick = all_stale_clfs[pick_i]
                    sl = slice(kk * rv_n, (kk + 1) * rv_n)
                    lg_s = _clf_logits(clf_pick, feats_cat[sl], param_dict)
                    lg_g_sub = lg_g[sl]
                    # MSE 对齐（即让 stale 与 global 之间尽可能一致 → generator 合成"两者都能识别"的特征）
                    if "SENT_CLF" in param_dict["task"]:
                        ps = F.softmax(lg_s / rv_T, dim=1)
                        pg = F.softmax(lg_g_sub / rv_T, dim=1)
                        total_meta_loss += F.mse_loss(ps, pg) * (rv_T ** 2)
                    else:
                        ps = torch.sigmoid(lg_s / rv_T)
                        pg = torch.sigmoid(lg_g_sub / rv_T)
                        total_meta_loss += F.mse_loss(ps, pg) * (rv_T ** 2)

                (total_meta_loss / K).backward()
                gen_opt.step()

            # --- Step 2: 使用 meta-generator 合成特征 → KD 更新全局分类器（融合陈旧知识）---
            global_clf.train()
            for _ep in range(rv_kd_epochs):
                # 从各 stale 分类器按陈旧度采样生成
                feats_list, teacher_outs_list = [], []
                for _ss in range(max(1, len(all_stale_clfs) // 2 + 1)):
                    pick_i = np.random.choice(len(all_stale_clfs))
                    tau_pick = all_stale_taus[pick_i]
                    clf_pick = all_stale_clfs[pick_i]
                    feats_s, y_s, _ = meta_gen.sample(rv_n // 2, device, tau_val=tau_pick / max(iter_t + 1, 1))
                    with torch.no_grad():
                        lg_s = _clf_logits(clf_pick, feats_s, param_dict)
                    feats_list.append(feats_s)
                    teacher_outs_list.append(lg_s)

                if not feats_list:
                    break

                feats_batch = torch.cat(feats_list, dim=0)
                teacher_batch = torch.cat(teacher_outs_list, dim=0).detach()
                distill_optim.zero_grad()
                lg_student = _clf_logits(global_clf, feats_batch, param_dict)

                if "SENT_CLF" in param_dict["task"]:
                    s_soft = torch.log_softmax(lg_student / rv_T, dim=1)
                    t_soft = torch.softmax(teacher_batch / rv_T, dim=1)
                    loss_kd = -torch.sum(t_soft * s_soft, dim=1).mean() * (rv_T ** 2)
                else:
                    sp = torch.sigmoid(lg_student / rv_T)
                    tp = torch.sigmoid(teacher_batch / rv_T)
                    loss_kd = F.mse_loss(sp, tp) * (rv_T ** 2)
                loss_kd.backward()
                distill_optim.step()

        # 清理临时分类头
        for c in all_stale_clfs:
            c.to("cpu"); del c
        all_stale_clfs.clear()
        global_model.to("cpu")

        total_gpu_seconds += (time.time() - gpu_s)
        del theta_arr, theta_list
        gc.collect()

        avg_gpu_seconds = total_gpu_seconds / num_clients_K
        logger.info(f"Round {iter_t+1} Test; GPU: {total_gpu_seconds:.1f}s, Avg: {avg_gpu_seconds:.1f}s, Stale buffers: {sum(len(v) for v in stale_clfs_buffer.values())}")

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

    logger.info("FedRevive training finished.")
    save_dir = './save_path/'; os.makedirs(save_dir, exist_ok=True)
    torch.save(global_model, os.path.join(save_dir, "global_FedRevive.pt"))
    total_comm_cost = communication_round_I * num_clients_K * FL_fraction * (2 * model_MB_size + gen_MB_size)
    return global_model, total_gpu_seconds, total_comm_cost
