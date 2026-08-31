# GeFL / GeFL-F: Model-Agnostic Federated Learning with Generative Models
# https://arxiv.org/abs/2412.18460 (IEEE Transactions on Computers, 2024)
# Git Repo: https://github.com/HongguKang/GeFL
# 核心思想: 操作空间为输入空间（GeFL）/特征空间（GeFL-F）。服务器训练一个生成模型配合判别器
#           （GAN 式对抗）：生成器合成伪样本/伪特征以"欺骗"在客户端知识上构建的判别器，
#           同时用客户端模型集成给伪样本打软标签做知识蒸馏；模型无关（model-agnostic），
#           GeFL-F 在特征层面生成，不重构原始数据，隐私性更好。
# Core Idea: Operates in input space (GeFL) / feature space (GeFL-F). The server trains a generative
#            model with a discriminator (GAN-style adversarial): the generator synthesizes pseudo
#            samples/features to "fool" a discriminator built on client knowledge, while the client
#            model ensemble soft-labels the pseudo data for knowledge distillation. Model-agnostic;
#            GeFL-F generates at the feature level without reconstructing raw data (better privacy).
# 框架适配说明: 原论文客户端上传本地生成模型；本框架适配为客户端上传分类头 + 逐类特征中心，
#               服务器端对抗信号由"判别器 vs 类中心高斯混合"构建；每轮重复执行。

import copy
import os
import gc
import time
import torch
import torch.nn as nn
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


def _train_single_client_gefl(client_id, device, model, param_dict,
                              training_dataloaders, algorithm_epoch_T,
                              accumulation_steps, use_amp, scaler, criterion,
                              basic_path, iter_t, communication_round_I, num_clients_K,
                              gefl_max_samples):
    """GeFL 单客户端：标准本地训练 + 上传分类头与逐类特征中心。"""
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

    # ---- 提取分类头 + 逐类特征中心（GeFL-F 判别器的"真实分布"代理）----
    gpu_start_time = time.time()
    model.eval()
    cls_sum, cls_cnt = {}, {}
    collected = 0
    for batch in client_i_dataloader:
        with torch.no_grad():
            if "SENT_CLF" in param_dict["task"]:
                feats = model.only_PLM_forward(input_ids=batch["input_ids"].to(device),
                                               attention_mask=batch["attention_mask"].to(device)).detach()
            elif "IMG_CLF" in param_dict["task"]:
                feats = model.only_backbone_forward(batch["img"].to(device)).detach()
            else:
                X = batch["X"].to(device)
                feats = X if "LogisticRegression" in str(type(model)) else model.only_backbone_forward(X).detach()
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
        if collected >= gefl_max_samples:
            break

    centroids = {yv: (cls_sum[yv] / max(cls_cnt[yv], 1)).cpu().numpy() for yv in cls_sum}
    clf_state = {name: p.detach().cpu() for name, p in _get_clf_module(model, param_dict).state_dict().items()}
    gpu_seconds += time.time() - gpu_start_time

    client_model_path = os.path.join(basic_path, "client_" + str(client_id + 1), 'model.pt')
    torch.save(model.cpu(), client_model_path)
    return {'gpu_seconds': gpu_seconds, 'clf_state': clf_state, 'centroids': centroids}


class FeatGenerator(nn.Module):
    """条件特征生成器 G(z, y) -> 伪特征。"""
    def __init__(self, latent_dim, feat_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim))

    def forward(self, z, y):
        return self.net(torch.cat([z, y.unsqueeze(1).float()], dim=1))


class FeatDiscriminator(nn.Module):
    """特征判别器 D(f) -> 真实概率。"""
    def __init__(self, feat_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1))

    def forward(self, f):
        return self.net(f)


def _gefl_core(device, global_model, algorithm_epoch_T, num_clients_K, communication_round_I,
               FL_fraction, FL_drop_rate, training_dataloaders, training_dataset,
               client_dataset_list, param_dict, testing_dataloader, testing_dataset_len,
               start_round, mode):
    """GeFL / GeFL-F 共享主循环；mode ∈ {"GeFL", "GeFL_F"}。"""
    accumulation_steps = max(1, int(256 / param_dict['batch_size']))
    use_amp = param_dict.get('use_amp', False)
    scaler = get_scaler(device, use_amp)

    training_dataset_size = len(training_dataset.labels) if hasattr(training_dataset, 'labels') else len(training_dataset)
    client_datasets_size_list = [len(_) for _ in client_dataset_list]
    del training_dataset, client_dataset_list
    gc.collect()

    # 超参数
    gefl_max_samples = int(param_dict.get('GeFL_max_samples', 512))
    gl_latent_dim = int(param_dict.get('GeFL_latent_dim', 32))
    gl_adv_steps = int(param_dict.get('GeFL_adv_steps', 50))     # GAN 对抗训练步数
    gl_gen_num = int(param_dict.get('GeFL_gen_num', 256))        # 最终蒸馏伪样本数
    gl_kd_epochs = int(param_dict.get('GeFL_kd_epochs', 5))
    gl_kd_lr = float(param_dict.get('GeFL_kd_lr', 0.01))
    gl_T = float(param_dict.get('GeFL_T', 2.0))
    gl_inv_steps = int(param_dict.get('GeFL_inv_steps', 20))     # GeFL 输入空间反演步数
    gl_inv_lr = float(param_dict.get('GeFL_inv_lr', 0.05))

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

        logger.info(f"Round {iter_t + 1}; Select clients: {idxs_users}; Start Local Training ({mode})")

        results = parallel_executor.run_clients(
            idxs_users, _train_single_client_gefl,
            param_dict=param_dict, training_dataloaders=training_dataloaders,
            algorithm_epoch_T=algorithm_epoch_T, accumulation_steps=accumulation_steps,
            use_amp=use_amp, scaler=scaler, criterion=criterion, basic_path=basic_path,
            iter_t=iter_t, communication_round_I=communication_round_I, num_clients_K=num_clients_K,
            gefl_max_samples=gefl_max_samples)

        for i, client_id in enumerate(idxs_users):
            users_gpu_seconds_list[client_id] += results[i]['gpu_seconds']
        total_gpu_seconds += sum(users_gpu_seconds_list)

        # ---- 常规 FedAvg 聚合 ----
        logger.info(f"{mode}: Parameter aggregation")
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

        gpu_s = time.time()
        global_model.to(device)

        # 重建客户端分类头集成（teacher）
        client_clfs = []
        for i in range(len(idxs_users)):
            clf_template = _get_clf_module(global_model, param_dict)
            clf = copy.deepcopy(clf_template).to(device)
            clf.load_state_dict({k: v.to(device) for k, v in results[i]['clf_state'].items()})
            clf.eval()
            client_clfs.append(clf)

        head = _get_clf_module(global_model, param_dict)
        feat_dim = head.in_features

        # ===== 模式分支：伪数据生成 =====
        if mode == "GeFL_F":
            # ---- GeFL-F：特征空间 GAN + 集成 KD ----
            # "真实"分布代理：客户端逐类特征中心 + 高斯噪声（不接触原始数据，隐私友好）
            global_centroids = {}
            total_w = sum(weights)
            for i, res in enumerate(results):
                w_i = weights[i] / total_w
                for yv, c in res['centroids'].items():
                    t = torch.tensor(c, dtype=torch.float32, device=device) * w_i
                    global_centroids[yv] = global_centroids.get(yv, torch.zeros_like(t)) + t
            ys = list(global_centroids.keys())
            gen = FeatGenerator(gl_latent_dim, feat_dim).to(device).train()
            disc = FeatDiscriminator(feat_dim).to(device).train()
            opt_g = torch.optim.Adam(gen.parameters(), lr=0.002, betas=(0.5, 0.999))
            opt_d = torch.optim.Adam(disc.parameters(), lr=0.002, betas=(0.5, 0.999))
            bce = torch.nn.BCEWithLogitsLoss()

            def _real_batch(n):
                fl, ll = [], []
                for _ in range(n):
                    yv = int(ys[int(np.random.randint(0, len(ys)))])
                    anchor = global_centroids[yv]
                    fl.append(anchor + 0.3 * torch.randn(feat_dim, device=device))
                    ll.append(1.0)
                return torch.stack(fl), torch.tensor(ll, device=device)

            # 对抗训练
            for _step in range(gl_adv_steps):
                # D 步
                f_real, l_real = _real_batch(32)
                z = torch.randn(32, gl_latent_dim, device=device)
                y_fake = torch.randint(0, 2, (32,), device=device)
                with torch.no_grad():
                    f_fake = gen(z, y_fake)
                d_loss = bce(disc(f_real).squeeze(1), l_real) + \
                         bce(disc(f_fake).squeeze(1), torch.zeros(32, device=device))
                opt_d.zero_grad(); d_loss.backward(); opt_d.step()
                # G 步（欺骗 D + 集成高置信）
                z = torch.randn(32, gl_latent_dim, device=device)
                y_fake = torch.randint(0, 2, (32,), device=device)
                f_fake = gen(z, y_fake)
                g_adv = bce(disc(f_fake).squeeze(1), torch.ones(32, device=device))
                logit_ens = sum(clf(f_fake) for clf in client_clfs) / len(client_clfs)
                if "SENT_CLF" in param_dict["task"]:
                    g_conf = torch.nn.functional.cross_entropy(
                        logit_ens, y_fake, reduction='mean')
                else:
                    g_conf = torch.nn.functional.binary_cross_entropy_with_logits(
                        logit_ens[:, 0], y_fake.float())
                g_loss = g_adv + g_conf
                opt_g.zero_grad(); g_loss.backward(); opt_g.step()

            # 生成蒸馏数据 + 集成软标签
            z = torch.randn(gl_gen_num, gl_latent_dim, device=device)
            y_lbl = torch.randint(0, 2, (gl_gen_num,), device=device)
            with torch.no_grad():
                pseudo = gen(z, y_lbl)
            teacher_probs = []
            with torch.no_grad():
                for clf in client_clfs:
                    logit = clf(pseudo)
                    teacher_probs.append(torch.softmax(logit / gl_T, dim=1)
                                         if "SENT_CLF" in param_dict["task"]
                                         else torch.sigmoid(logit / gl_T))
            teacher_mean = torch.stack(teacher_probs, dim=0).mean(dim=0).detach()

            # KD 训练全局头（冻结骨干）
            for p in global_model.parameters():
                p.requires_grad = False
            for p in head.parameters():
                p.requires_grad = True
            head_opt = torch.optim.SGD(head.parameters(), lr=gl_kd_lr)
            head.train()
            for _epoch in range(gl_kd_epochs):
                if "SENT_CLF" in param_dict["task"]:
                    logp = torch.log_softmax(head(pseudo), dim=1)
                    loss_kd = -(teacher_mean * logp).sum(dim=1).mean()
                else:
                    logit = head(pseudo)[:, 0]
                    loss_kd = torch.nn.functional.binary_cross_entropy_with_logits(logit, teacher_mean[:, 0])
                head_opt.zero_grad(); loss_kd.backward(); head_opt.step()
            for p in global_model.parameters():
                p.requires_grad = True
            del gen, disc, pseudo, teacher_mean
        else:
            # ---- GeFL：输入空间模型反演 + 集成 KD（全模型蒸馏）----
            # 反演目标：使"全局模型特征 -> 客户端头集成"的预测对随机标签高置信（ensemble-guided inversion）
            noise_pool = [_noise_inputs(param_dict, device, 32) for _ in range(4)]
            if "SENT_CLF" in param_dict["task"]:
                pseudo_inputs = [inp[0].clone().float().requires_grad_(True) for inp in noise_pool]
                masks = [inp[1] for inp in noise_pool]
            else:
                pseudo_inputs = [inp.clone().requires_grad_(True) for inp in noise_pool]

            # 类别平衡随机目标标签
            def _rand_labels(n):
                half = n // 2
                lab = torch.cat([torch.zeros(half, device=device),
                                 torch.ones(n - half, device=device)])
                return lab[torch.randperm(n, device=device)]

            def _ens_logits(pi, mask_i):
                """全局模型特征 -> 客户端头集成 logits。"""
                if "SENT_CLF" in param_dict["task"]:
                    feats_g, _ = global_model(input_ids=pi.long().clamp(0, 30521), attention_mask=mask_i)
                elif "IMG_CLF" in param_dict["task"]:
                    _, feats_g = global_model(pi)
                else:
                    if "LogisticRegression" in str(type(global_model)):
                        feats_g = pi
                    else:
                        _, feats_g = global_model(pi)
                return sum(clf(feats_g) for clf in client_clfs) / len(client_clfs)

            for _step in range(gl_inv_steps):
                for bi, pi in enumerate(pseudo_inputs):
                    y_t = _rand_labels(pi.size(0))
                    logit_ens = _ens_logits(pi, masks[bi] if "SENT_CLF" in param_dict["task"] else None)
                    if "SENT_CLF" in param_dict["task"]:
                        conf_loss = torch.nn.functional.cross_entropy(logit_ens, y_t.long())
                    else:
                        conf_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            logit_ens[:, 0], y_t)
                    reg = 0.001 * (pi ** 2).mean()
                    inv_loss = conf_loss + reg
                    grad = torch.autograd.grad(inv_loss, pi, allow_unused=True)[0]
                    if grad is not None:
                        with torch.no_grad():
                            pi -= gl_inv_lr * grad

            # 集成软标签（client heads ensemble）+ 全模型 KD
            global_model.eval()
            teacher_batches = []
            with torch.no_grad():
                for bi, pi in enumerate(pseudo_inputs):
                    logits_ens = _ens_logits(pi, masks[bi] if "SENT_CLF" in param_dict["task"] else None)
                    teacher_batches.append(
                        torch.softmax(logits_ens / gl_T, dim=1)
                        if "SENT_CLF" in param_dict["task"]
                        else torch.sigmoid(logits_ens / gl_T))

            global_model.train()
            kd_optimizer = BERTCLF_Optimizer(method="SGD", learning_rate=gl_kd_lr, max_grad_norm=0)
            kd_optimizer.set_parameters(list(global_model.named_parameters()))
            for _epoch in range(gl_kd_epochs):
                for pi, teacher in zip(pseudo_inputs, teacher_batches):
                    if "SENT_CLF" in param_dict["task"]:
                        _, logits_s = global_model(input_ids=pi.long().clamp(0, 30521), attention_mask=masks[0][:pi.size(0)])
                        logp = torch.log_softmax(logits_s, dim=1)
                        loss_kd = -(teacher * logp).sum(dim=1).mean()
                    elif "IMG_CLF" in param_dict["task"]:
                        preds_s, _ = global_model(pi)
                        loss_kd = torch.nn.functional.binary_cross_entropy_with_logits(preds_s[:, 0], teacher)
                    else:
                        out_s, _ = global_model(pi)
                        loss_kd = torch.nn.functional.binary_cross_entropy_with_logits(out_s[:, 0], teacher)
                    kd_optimizer.zero_grad()
                    loss_kd.backward()
                    kd_optimizer.step()
            del pseudo_inputs, teacher_batches

        total_gpu_seconds += time.time() - gpu_s
        global_model.to("cpu")
        del client_clfs
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

    logger.info(f"{mode} training finished.")
    save_dir = './save_path/'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(global_model, os.path.join(save_dir, f"global_{mode}.pt"))
    total_comm_cost = communication_round_I * num_clients_K * FL_fraction * 2 * model_MB_size
    return global_model, total_gpu_seconds, total_comm_cost


def GeFL(device, global_model, algorithm_epoch_T, num_clients_K, communication_round_I,
         FL_fraction, FL_drop_rate, training_dataloaders, training_dataset,
         client_dataset_list, param_dict, testing_dataloader, testing_dataset_len, start_round=0):
    """GeFL：输入空间（模型反演伪输入 + 集成 KD）。"""
    return _gefl_core(device, global_model, algorithm_epoch_T, num_clients_K, communication_round_I,
                      FL_fraction, FL_drop_rate, training_dataloaders, training_dataset,
                      client_dataset_list, param_dict, testing_dataloader, testing_dataset_len,
                      start_round, mode="GeFL")


def GeFL_F(device, global_model, algorithm_epoch_T, num_clients_K, communication_round_I,
           FL_fraction, FL_drop_rate, training_dataloaders, training_dataset,
           client_dataset_list, param_dict, testing_dataloader, testing_dataset_len, start_round=0):
    """GeFL-F：特征空间（GAN 生成器 + 集成 KD，隐私友好）。"""
    return _gefl_core(device, global_model, algorithm_epoch_T, num_clients_K, communication_round_I,
                      FL_fraction, FL_drop_rate, training_dataloaders, training_dataset,
                      client_dataset_list, param_dict, testing_dataloader, testing_dataset_len,
                      start_round, mode="GeFL_F")
