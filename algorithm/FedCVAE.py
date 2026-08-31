# FedCVAE-Ens / FedCVAE-KD: Data-Free One-Shot Federated Learning Under Very High Statistical Heterogeneity
# https://openreview.net/ (ICLR 2026 Poster)
# 核心思想: 操作空间为 latent/特征空间。各客户端在本地特征上训练一个条件变分自编码器（CVAE）重构本地任务
#           分布并上传解码器；服务器从客户端 CVAE 混合分布中采样伪特征，用客户端分类头集成打伪标签，
#           再训练全局模型。明确支持异构本地模型（data-free，不传输原始数据）。
#           - FedCVAE-Ens: 集成硬投票伪标签 + 交叉熵蒸馏
#           - FedCVAE-KD:  集成软标签 + 带温度的知识蒸馏
# Core Idea: Operates in latent/feature space. Each client trains a conditional VAE (CVAE) to reconstruct
#            its local task distribution and uploads the decoder; the server samples pseudo features from
#            the mixture of client CVAEs, pseudo-labels them with the ensemble of client classifier heads,
#            and trains the global model (data-free, supports heterogeneous local models).
# 框架适配说明: 原论文为 one-shot 设置；本框架为多轮，每轮重复 "本地训练+CVAE -> 服务器采样+蒸馏"。

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
    """获取各任务的分类头（输出层）模块。"""
    if "SENT_CLF" in param_dict["task"]:
        return model.out
    elif "IMG_CLF" in param_dict["task"]:
        return model.out_layer
    else:
        if "LogisticRegression" in str(type(model)):
            return model.layer
        return model.out_layer


def _forward_features(model, param_dict, batch, device):
    """冻结骨干提取特征（不经过分类头）。"""
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


class FedCVAE(nn.Module):
    """轻量条件变分自编码器（作用于特征空间）：q(z|x,y) / p(x|z,y)。"""
    def __init__(self, feat_dim, latent_dim=32, hidden_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.label_emb = nn.Embedding(2, latent_dim)  # 本框架为二分类任务（labels ∈ {0,1}）
        self.enc = nn.Sequential(
            nn.Linear(feat_dim + latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2))
        self.dec = nn.Sequential(
            nn.Linear(latent_dim + latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim))

    def encode(self, x, y):
        h = torch.cat([x, self.label_emb(y)], dim=1)
        mu_logvar = self.enc(h)
        mu, logvar = mu_logvar[:, :self.latent_dim], mu_logvar[:, self.latent_dim:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z, y):
        return self.dec(torch.cat([z, self.label_emb(y)], dim=1))

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        x_rec = self.decode(z, y)
        return x_rec, mu, logvar


def _train_single_client_fedcvae(client_id, device, model, param_dict,
                                 training_dataloaders, algorithm_epoch_T,
                                 accumulation_steps, use_amp, scaler, criterion,
                                 basic_path, iter_t, communication_round_I, num_clients_K,
                                 cvae_epochs, cvae_lr, cvae_max_samples):
    """FedCVAE 单客户端：标准本地训练 + 本地特征上训练 CVAE，上传解码器与分类头。"""
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

    # ---- 本地 CVAE 训练（在本地特征上重构任务分布）----
    gpu_start_time = time.time()
    model.eval()
    feats_list, labels_list = [], []
    collected = 0
    for batch in client_i_dataloader:
        feats = _forward_features(model, param_dict, batch, device).detach()
        labels = batch["labels"].to(device)
        feats_list.append(feats)
        labels_list.append(labels)
        collected += feats.size(0)
        if collected >= cvae_max_samples:
            break
    feats_all = torch.cat(feats_list, dim=0)[:cvae_max_samples]
    labels_all = torch.cat(labels_list, dim=0)[:cvae_max_samples].long()

    feat_dim = feats_all.size(1)
    cvae = FedCVAE(feat_dim).to(device)
    if len(feats_all) >= 2:
        cvae_opt = torch.optim.Adam(cvae.parameters(), lr=cvae_lr)
        for _e in range(cvae_epochs):
            x_rec, mu, logvar = cvae(feats_all, labels_all)
            recon = torch.nn.functional.mse_loss(x_rec, feats_all)
            kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            cvae_loss = recon + 0.01 * kld
            cvae_opt.zero_grad()
            cvae_loss.backward()
            cvae_opt.step()

    # 上传解码器 + label embedding + 分类头
    clf = _get_clf_module(model, param_dict)
    upload_state = {}
    for name, p in cvae.dec.state_dict().items():
        upload_state["dec." + name] = p.detach().cpu()
    for name, p in cvae.label_emb.state_dict().items():
        upload_state["emb." + name] = p.detach().cpu()
    for name, p in clf.state_dict().items():
        upload_state["clf." + name] = p.detach().cpu()
    gpu_seconds += time.time() - gpu_start_time

    client_model_path = os.path.join(basic_path, "client_" + str(client_id + 1), 'model.pt')
    torch.save(model.cpu(), client_model_path)
    return {'gpu_seconds': gpu_seconds, 'cvae_state': upload_state, 'feat_dim': feat_dim}


def _fedcvae_core(device, global_model, algorithm_epoch_T, num_clients_K, communication_round_I,
                  FL_fraction, FL_drop_rate, training_dataloaders, training_dataset,
                  client_dataset_list, param_dict, testing_dataloader, testing_dataset_len,
                  start_round, mode):
    """FedCVAE-Ens / FedCVAE-KD 共享主循环；mode ∈ {"Ens", "KD"}。"""
    accumulation_steps = max(1, int(256 / param_dict['batch_size']))
    use_amp = param_dict.get('use_amp', False)
    scaler = get_scaler(device, use_amp)

    training_dataset_size = len(training_dataset.labels) if hasattr(training_dataset, 'labels') else len(training_dataset)
    client_datasets_size_list = [len(_) for _ in client_dataset_list]
    del training_dataset, client_dataset_list
    gc.collect()

    # 超参数
    cvae_epochs = int(param_dict.get('FedCVAE_epochs', 10))
    cvae_lr = float(param_dict.get('FedCVAE_lr', 0.002))
    cvae_max_samples = int(param_dict.get('FedCVAE_max_samples', 512))
    fc_gen_num = int(param_dict.get('FedCVAE_gen_num', 256))     # 每轮生成的伪特征数
    fc_steps = int(param_dict.get('FedCVAE_steps', 5))           # 全局头蒸馏步数
    fc_lr = float(param_dict.get('FedCVAE_kd_lr', 0.01))
    fc_T = float(param_dict.get('FedCVAE_T', 2.0))               # KD 温度

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

        logger.info(f"Round {iter_t + 1}; Select clients: {idxs_users}; Start Local Training (FedCVAE-{mode})")

        results = parallel_executor.run_clients(
            idxs_users, _train_single_client_fedcvae,
            param_dict=param_dict, training_dataloaders=training_dataloaders,
            algorithm_epoch_T=algorithm_epoch_T, accumulation_steps=accumulation_steps,
            use_amp=use_amp, scaler=scaler, criterion=criterion, basic_path=basic_path,
            iter_t=iter_t, communication_round_I=communication_round_I, num_clients_K=num_clients_K,
            cvae_epochs=cvae_epochs, cvae_lr=cvae_lr, cvae_max_samples=cvae_max_samples)

        for i, client_id in enumerate(idxs_users):
            users_gpu_seconds_list[client_id] += results[i]['gpu_seconds']
        total_gpu_seconds += sum(users_gpu_seconds_list)

        # ---- 常规 FedAvg 聚合（骨干网络）----
        logger.info("FedCVAE: Parameter aggregation")
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

        # ===== FedCVAE 核心：CVAE 混合采样 + 客户端头集成伪标签 + 全局头蒸馏 =====
        gpu_s = time.time()
        global_model.to(device)

        # 重建各客户端解码器与分类头
        client_decoders, client_clfs = [], []
        for i, id in enumerate(idxs_users):
            up = results[i]['cvae_state']
            feat_dim = results[i]['feat_dim']
            dec = FedCVAE(feat_dim).to(device)
            dec_sd = {k[len("dec."):]: v.to(device) for k, v in up.items() if k.startswith("dec.")}
            dec.dec.load_state_dict(dec_sd)
            dec.label_emb.load_state_dict({k[len("emb."):]: v.to(device) for k, v in up.items() if k.startswith("emb.")})
            dec.eval()
            client_decoders.append(dec)

            clf_template = _get_clf_module(global_model, param_dict)
            clf = copy.deepcopy(clf_template).to(device)
            clf.load_state_dict({k[len("clf."):]: v.to(device) for k, v in up.items() if k.startswith("clf.")})
            clf.eval()
            client_clfs.append(clf)

        n_clf_out = _get_clf_module(global_model, param_dict).out_features

        # 采样伪特征（从客户端 CVAE 混合分布中均匀抽取）+ 集成软标签
        pseudo_feats, pseudo_labels, teacher_probs = [], [], []
        for _ in range(fc_gen_num):
            cid = int(np.random.randint(0, len(idxs_users)))
            y = int(np.random.randint(0, 2))
            z = torch.randn(1, client_decoders[cid].latent_dim, device=device)
            y_t = torch.tensor([y], device=device)
            with torch.no_grad():
                f = client_decoders[cid].decode(z, y_t)
                probs_c = []
                for clf in client_clfs:
                    logit = clf(f)
                    if "SENT_CLF" in param_dict["task"]:
                        probs_c.append(torch.softmax(logit / fc_T, dim=1))
                    else:
                        probs_c.append(torch.sigmoid(logit / fc_T))
                p_mean = torch.stack(probs_c, dim=0).mean(dim=0)
            pseudo_feats.append(f.squeeze(0))
            teacher_probs.append(p_mean.squeeze(0))
            if "SENT_CLF" in param_dict["task"]:
                pseudo_labels.append(p_mean.squeeze(0).argmax())
            else:
                pseudo_labels.append((p_mean.squeeze(0) > 0.5).float())

        pseudo_feats = torch.stack(pseudo_feats, dim=0)
        pseudo_labels = torch.stack(pseudo_labels, dim=0)
        teacher_probs = torch.stack(teacher_probs, dim=0)

        # 冻结骨干，只训练全局分类头
        head = _get_clf_module(global_model, param_dict)
        for p in global_model.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = True
        head_opt = torch.optim.SGD(head.parameters(), lr=fc_lr)
        head.train()
        for _step in range(fc_steps):
            if "SENT_CLF" in param_dict["task"]:
                logits = head(pseudo_feats)
                logp = torch.log_softmax(logits, dim=1)
                if mode == "Ens":
                    loss_kd = torch.nn.functional.nll_loss(logp, pseudo_labels)
                else:  # KD: 软标签蒸馏
                    loss_kd = -(teacher_probs * logp).sum(dim=1).mean()
            else:
                logit = head(pseudo_feats)[:, 0]
                if mode == "Ens":
                    loss_kd = torch.nn.functional.binary_cross_entropy_with_logits(logit, pseudo_labels)
                else:
                    loss_kd = torch.nn.functional.binary_cross_entropy_with_logits(logit, teacher_probs[:, 0])
            head_opt.zero_grad()
            loss_kd.backward()
            head_opt.step()
        for p in global_model.parameters():
            p.requires_grad = True

        total_gpu_seconds += time.time() - gpu_s
        global_model.to("cpu")
        del client_decoders, client_clfs, pseudo_feats, teacher_probs
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

    logger.info(f"FedCVAE-{mode} training finished.")
    save_dir = './save_path/'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(global_model, os.path.join(save_dir, f"global_FedCVAE_{mode}.pt"))
    total_comm_cost = communication_round_I * num_clients_K * FL_fraction * 2 * model_MB_size
    return global_model, total_gpu_seconds, total_comm_cost


def Fed_CVAE_Ens(device, global_model, algorithm_epoch_T, num_clients_K, communication_round_I,
                 FL_fraction, FL_drop_rate, training_dataloaders, training_dataset,
                 client_dataset_list, param_dict, testing_dataloader, testing_dataset_len, start_round=0):
    """FedCVAE-Ens：集成硬投票伪标签 + 交叉熵蒸馏。"""
    return _fedcvae_core(device, global_model, algorithm_epoch_T, num_clients_K, communication_round_I,
                         FL_fraction, FL_drop_rate, training_dataloaders, training_dataset,
                         client_dataset_list, param_dict, testing_dataloader, testing_dataset_len,
                         start_round, mode="Ens")


def Fed_CVAE_KD(device, global_model, algorithm_epoch_T, num_clients_K, communication_round_I,
                FL_fraction, FL_drop_rate, training_dataloaders, training_dataset,
                client_dataset_list, param_dict, testing_dataloader, testing_dataset_len, start_round=0):
    """FedCVAE-KD：集成软标签 + 带温度知识蒸馏。"""
    return _fedcvae_core(device, global_model, algorithm_epoch_T, num_clients_K, communication_round_I,
                         FL_fraction, FL_drop_rate, training_dataloaders, training_dataset,
                         client_dataset_list, param_dict, testing_dataloader, testing_dataset_len,
                         start_round, mode="KD")
